import os
import re
import csv
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================
# Helpers
# =========================

def get_scene_key(filename: str) -> str:
    """
    Remove extension and _pre_disaster / _post_disaster suffix.
    Example:
      hurricane-harvey_00000350_pre_disaster.jpeg
      -> hurricane-harvey_00000350
    """
    stem = Path(filename).stem
    stem = stem.replace("_pre_disaster", "").replace("_post_disaster", "")
    return stem


def is_pre_disaster(filename: str) -> bool:
    return "_pre_disaster" in Path(filename).stem


def is_post_disaster(filename: str) -> bool:
    return "_post_disaster" in Path(filename).stem


def parse_wkt_polygon(wkt: str):
    """
    Parse a WKT polygon string like:
      POLYGON ((x1 y1, x2 y2, ...))
    into a list of (x, y) tuples.

    This is intentionally simple and meant for the xView2-style polygons.
    """
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        return []

    # Pull out the inside of POLYGON ((...))
    match = re.search(r"POLYGON\s*\(\((.*)\)\)", wkt)
    if match is None:
        return []

    coords_text = match.group(1)
    points = []

    for pair in coords_text.split(","):
        pair = pair.strip()
        pieces = pair.split()
        if len(pieces) < 2:
            continue

        x = float(pieces[0])
        y = float(pieces[1])
        points.append((x, y))

    return points


def json_to_mask(json_path, image_size=(1024, 1024)):
    """
    Convert xView2-style JSON polygon annotations into a binary mask.
    Uses features.xy polygons if present.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    features = data.get("features", {})
    xy_features = features.get("xy", [])

    for feat in xy_features:
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue

        wkt = feat.get("wkt", "")
        polygon = parse_wkt_polygon(wkt)

        if len(polygon) >= 3:
            draw.polygon(polygon, outline=1, fill=1)

    return mask


# =========================
# Dataset
# =========================

class XView2SiameseDataset(Dataset):
    def __init__(self, split_root, image_size=256):
        """
        split_root should be something like:
          ../../siamese_project/data/xview2_jpg/tier1
        or
          ../../siamese_project/data/xview2_jpg/hold
        """
        self.split_root = Path(split_root)
        self.images_dir = self.split_root / "images_jpeg"
        self.labels_dir = self.split_root / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images_jpeg folder: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing labels folder: {self.labels_dir}")

        image_files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file()])
        json_files = sorted([p.name for p in self.labels_dir.iterdir() if p.is_file() and p.suffix == ".json"])

        self.image_map = {Path(f).stem: self.images_dir / f for f in image_files}
        self.json_map = {Path(f).stem: self.labels_dir / f for f in json_files}

        # group by scene key
        scenes = {}
        for stem in self.image_map.keys():
            key = get_scene_key(stem)
            scenes.setdefault(key, {})
            if is_pre_disaster(stem):
                scenes[key]["pre_img"] = self.image_map[stem]
            elif is_post_disaster(stem):
                scenes[key]["post_img"] = self.image_map[stem]

        for stem in self.json_map.keys():
            key = get_scene_key(stem)
            scenes.setdefault(key, {})
            if is_pre_disaster(stem):
                scenes[key]["pre_json"] = self.json_map[stem]
            elif is_post_disaster(stem):
                scenes[key]["post_json"] = self.json_map[stem]

        # keep only valid paired scenes
        self.samples = []
        for key, item in scenes.items():
            if "pre_img" in item and "post_img" in item:
                # prefer post json, fallback to pre json
                if "post_json" in item:
                    item["target_json"] = item["post_json"]
                elif "pre_json" in item:
                    item["target_json"] = item["pre_json"]
                else:
                    continue
                self.samples.append((key, item))

        if len(self.samples) == 0:
            raise ValueError(f"No valid pre/post pairs found in {self.split_root}")

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, item = self.samples[idx]

        pre_img_path = item["pre_img"]
        post_img_path = item["post_img"]
        target_json_path = item["target_json"]

        pre_img = Image.open(pre_img_path).convert("RGB")
        post_img = Image.open(post_img_path).convert("RGB")

        # Make mask from JSON polygons using original image size
        original_size = pre_img.size  # (width, height)
        mask = json_to_mask(target_json_path, image_size=original_size)

        pre_img = self.img_tf(pre_img)
        post_img = self.img_tf(post_img)
        mask = self.mask_tf(mask)

        mask = (mask > 0.5).float()

        return pre_img, post_img, mask, key


# =========================
# Model
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base * 2)
        self.e3 = EncoderBlock(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b = self.bottleneck(p3)
        return s1, s2, s3, b


class BaselineSiameseUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, base)

        self.bottleneck_reduce = DoubleConv(base * 16, base * 8)
        self.skip3_reduce = DoubleConv(base * 8, base * 4)
        self.skip2_reduce = DoubleConv(base * 4, base * 2)
        self.skip1_reduce = DoubleConv(base * 2, base)

        self.d3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.d2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.d1 = DecoderBlock(base * 2, base, base)

        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, before, after):
        b1, b2, b3, bb = self.encoder(before)
        a1, a2, a3, ab = self.encoder(after)

        x = torch.cat([bb, ab], dim=1)
        x = self.bottleneck_reduce(x)

        s3 = self.skip3_reduce(torch.cat([b3, a3], dim=1))
        s2 = self.skip2_reduce(torch.cat([b2, a2], dim=1))
        s1 = self.skip1_reduce(torch.cat([b1, a1], dim=1))

        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)

        return self.out(x)


# =========================
# Metrics
# =========================

def dice_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()


def iou_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for before, after, mask, _ in loader:
        before = before.to(device)
        after = after.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(before, after)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * before.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for before, after, mask, _ in loader:
        before = before.to(device)
        after = after.to(device)
        mask = mask.to(device)

        logits = model(before, after)
        loss = criterion(logits, mask)

        bs = before.size(0)
        total_loss += loss.item() * bs
        total_dice += dice_score(logits, mask) * bs
        total_iou += iou_score(logits, mask) * bs

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="baseline_xview2_run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Train root: {args.train_root}")
    print(f"Val root: {args.val_root}")
    print(f"Save dir: {args.save_dir}")

    train_ds = XView2SiameseDataset(args.train_root, image_size=args.image_size)
    val_ds = XView2SiameseDataset(args.val_root, image_size=args.image_size)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = BaselineSiameseUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    best_model_path = os.path.join(args.save_dir, "best_model.pt")

    best_dice = -1.0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_dice:.4f} | "
                f"val_iou={val_iou:.4f}"
            )

            writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou])
            f.flush()

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    print("Finished training.")
    print(f"Best val dice: {best_dice:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()