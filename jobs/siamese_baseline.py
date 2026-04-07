import os
import re
import csv
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================
# Helpers
# =========================

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

IDX_TO_DAMAGE = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}


def get_scene_key(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.replace("_pre_disaster", "").replace("_post_disaster", "")
    return stem


def is_pre_disaster(filename: str) -> bool:
    return "_pre_disaster" in Path(filename).stem


def is_post_disaster(filename: str) -> bool:
    return "_post_disaster" in Path(filename).stem


def parse_wkt_polygon(wkt: str):
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        return []

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


def json_to_damage_mask_and_label(json_path, image_size=(1024, 1024)):
    """
    Build a multiclass mask from post-disaster JSON.

    Classes:
      0 = background
      1 = no-damage
      2 = minor-damage
      3 = major-damage
      4 = destroyed

    Also returns an image-level class label:
      max damage severity present in the tile
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    width, height = image_size

    # IMPORTANT:
    # Use mode "L" and fill with integer class IDs directly
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    max_class = 0

    features = data.get("features", {})
    xy_features = features.get("xy", [])

    for feat in xy_features:
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue

        subtype = props.get("subtype", "no-damage")
        class_id = DAMAGE_MAP.get(subtype, 1)

        wkt = feat.get("wkt", "")
        polygon = parse_wkt_polygon(wkt)

        if len(polygon) >= 3:
            draw.polygon(polygon, outline=class_id, fill=class_id)
            max_class = max(max_class, class_id)

    return mask, max_class


# =========================
# Dataset
# =========================

class XView2SiameseDataset(Dataset):
    def __init__(self, split_root, image_size=256, debug_samples=0):
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

        self.samples = []
        for key, item in scenes.items():
            # For real damage prediction, require post-disaster json
            if "pre_img" in item and "post_img" in item and "post_json" in item:
                self.samples.append((key, item))

        if len(self.samples) == 0:
            raise ValueError(f"No valid pre/post pairs with post_json found in {self.split_root}")

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # For multiclass mask:
        # do resize only, then convert with torch.tensor instead of ToTensor
        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=Image.NEAREST
        )

        self.debug_samples = debug_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, item = self.samples[idx]

        pre_img_path = item["pre_img"]
        post_img_path = item["post_img"]
        post_json_path = item["post_json"]

        pre_img = Image.open(pre_img_path).convert("RGB")
        post_img = Image.open(post_img_path).convert("RGB")

        original_size = pre_img.size  # (width, height)
        damage_mask, class_label = json_to_damage_mask_and_label(post_json_path, image_size=original_size)

        pre_img = self.img_tf(pre_img)
        post_img = self.img_tf(post_img)

        damage_mask = self.mask_resize(damage_mask)
        damage_mask = torch.tensor(list(damage_mask.getdata()), dtype=torch.long).view(damage_mask.size[1], damage_mask.size[0])

        class_label = torch.tensor(class_label, dtype=torch.long)

        if idx < self.debug_samples:
            print(
                f"[DEBUG] {key} | "
                f"mask unique={torch.unique(damage_mask)} | "
                f"mask max={damage_mask.max().item()} | "
                f"class_label={class_label.item()} ({IDX_TO_DAMAGE[class_label.item()]})"
            )

        return pre_img, post_img, damage_mask, class_label, key


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


class BaselineSiameseUNetWithClassifier(nn.Module):
    def __init__(self, in_ch=3, seg_out_ch=5, base=32):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, base)

        # segmentation branch
        self.bottleneck_reduce = DoubleConv(base * 16, base * 8)
        self.skip3_reduce = DoubleConv(base * 8, base * 4)
        self.skip2_reduce = DoubleConv(base * 4, base * 2)
        self.skip1_reduce = DoubleConv(base * 2, base)

        self.d3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.d2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.d1 = DecoderBlock(base * 2, base, base)
        self.seg_out = nn.Conv2d(base, seg_out_ch, kernel_size=1)

        # classification branch: 5 classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)
        )

    def forward(self, before, after):
        b1, b2, b3, bb = self.encoder(before)
        a1, a2, a3, ab = self.encoder(after)

        fused_bottleneck = torch.cat([bb, ab], dim=1)

        x = self.bottleneck_reduce(fused_bottleneck)
        s3 = self.skip3_reduce(torch.cat([b3, a3], dim=1))
        s2 = self.skip2_reduce(torch.cat([b2, a2], dim=1))
        s1 = self.skip1_reduce(torch.cat([b1, a1], dim=1))

        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)

        seg_logits = self.seg_out(x)
        cls_logits = self.classifier(fused_bottleneck)

        return seg_logits, cls_logits


# =========================
# Metrics
# =========================

def multiclass_dice_score(logits, targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    dices = []

    for cls in range(1, num_classes):  # skip background
        pred_c = (preds == cls).float()
        targ_c = (targets == cls).float()

        inter = (pred_c * targ_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))

        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice.mean().item())

    return sum(dices) / len(dices)


def multiclass_iou_score(logits, targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    ious = []

    for cls in range(1, num_classes):  # skip background
        pred_c = (preds == cls).float()
        targ_c = (targets == cls).float()

        inter = (pred_c * targ_c).sum(dim=(1, 2))
        union = (pred_c + targ_c - pred_c * targ_c).sum(dim=(1, 2))

        iou = (inter + eps) / (union + eps)
        ious.append(iou.mean().item())

    return sum(ious) / len(ious)


def classification_metrics(cls_logits, cls_targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(cls_logits, dim=1)

    acc = (preds == cls_targets).float().mean().item()

    f1s = []
    for cls in range(1, num_classes):  # skip background as main target interest
        pred_c = (preds == cls)
        targ_c = (cls_targets == cls)

        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)

    return acc, sum(f1s) / len(f1s)


# =========================
# Plotting
# =========================

def save_training_plots(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_seg_loss"], label="Train Seg Loss")
    plt.plot(epochs, history["val_seg_loss"], label="Val Seg Loss")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Segmentation Performance Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "segmentation_metrics.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_cls_loss"], label="Train Cls Loss")
    plt.plot(epochs, history["val_cls_loss"], label="Val Cls Loss")
    plt.plot(epochs, history["val_cls_acc"], label="Val Cls Accuracy")
    plt.plot(epochs, history["val_cls_f1"], label="Val Cls F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Classification Performance Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classification_metrics.png"))
    plt.close()


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, seg_criterion, cls_criterion, optimizer, device, cls_weight=1.0):
    model.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0

    for before, after, mask, class_label, _ in loader:
        before = before.to(device)
        after = after.to(device)
        mask = mask.to(device)
        class_label = class_label.to(device)

        optimizer.zero_grad()

        seg_logits, cls_logits = model(before, after)

        seg_loss = seg_criterion(seg_logits, mask)
        cls_loss = cls_criterion(cls_logits, class_label)

        loss = seg_loss + cls_weight * cls_loss
        loss.backward()
        optimizer.step()

        bs = before.size(0)
        total_loss += loss.item() * bs
        total_seg_loss += seg_loss.item() * bs
        total_cls_loss += cls_loss.item() * bs

    n = len(loader.dataset)
    return total_loss / n, total_seg_loss / n, total_cls_loss / n


@torch.no_grad()
def evaluate(model, loader, seg_criterion, cls_criterion, device, cls_weight=1.0):
    model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_cls_acc = 0.0
    total_cls_f1 = 0.0

    for before, after, mask, class_label, _ in loader:
        before = before.to(device)
        after = after.to(device)
        mask = mask.to(device)
        class_label = class_label.to(device)

        seg_logits, cls_logits = model(before, after)

        seg_loss = seg_criterion(seg_logits, mask)
        cls_loss = cls_criterion(cls_logits, class_label)
        loss = seg_loss + cls_weight * cls_loss

        bs = before.size(0)
        total_loss += loss.item() * bs
        total_seg_loss += seg_loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_dice += multiclass_dice_score(seg_logits, mask) * bs
        total_iou += multiclass_iou_score(seg_logits, mask) * bs

        acc, f1 = classification_metrics(cls_logits, class_label)
        total_cls_acc += acc * bs
        total_cls_f1 += f1 * bs

    n = len(loader.dataset)
    return {
        "total_loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "cls_acc": total_cls_acc / n,
        "cls_f1": total_cls_f1 / n,
    }


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="baseline_xview2_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--debug_samples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    print(f"Train root: {args.train_root}")
    print(f"Val root: {args.val_root}")
    print(f"Save dir: {args.save_dir}")

    train_ds = XView2SiameseDataset(
        args.train_root,
        image_size=args.image_size,
        debug_samples=args.debug_samples
    )
    val_ds = XView2SiameseDataset(
        args.val_root,
        image_size=args.image_size,
        debug_samples=args.debug_samples
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    sample_pre, sample_post, sample_mask, sample_cls, sample_key = train_ds[0]
    print(f"[SANITY CHECK] sample key: {sample_key}")
    print(f"[SANITY CHECK] mask unique values: {torch.unique(sample_mask)}")
    print(f"[SANITY CHECK] mask max class: {sample_mask.max().item()} ({IDX_TO_DAMAGE[int(sample_mask.max().item())]})")
    print(f"[SANITY CHECK] class label: {sample_cls.item()} ({IDX_TO_DAMAGE[int(sample_cls.item())]})")

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

    model = BaselineSiameseUNetWithClassifier().to(device)
    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    plot_dir = os.path.join(args.save_dir, "plots")

    best_dice = -1.0

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_seg_loss": [],
        "train_cls_loss": [],
        "val_total_loss": [],
        "val_seg_loss": [],
        "val_cls_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_cls_acc": [],
        "val_cls_f1": [],
    }

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_seg_loss",
            "train_cls_loss",
            "val_total_loss",
            "val_seg_loss",
            "val_cls_loss",
            "val_dice",
            "val_iou",
            "val_cls_acc",
            "val_cls_f1",
        ])

        for epoch in range(1, args.epochs + 1):
            train_total_loss, train_seg_loss, train_cls_loss = train_one_epoch(
                model, train_loader, seg_criterion, cls_criterion, optimizer, device, cls_weight=args.cls_weight
            )

            val_metrics = evaluate(
                model, val_loader, seg_criterion, cls_criterion, device, cls_weight=args.cls_weight
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_total={train_total_loss:.4f} | "
                f"train_seg={train_seg_loss:.4f} | "
                f"train_cls={train_cls_loss:.4f} | "
                f"val_total={val_metrics['total_loss']:.4f} | "
                f"val_seg={val_metrics['seg_loss']:.4f} | "
                f"val_cls={val_metrics['cls_loss']:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_cls_acc={val_metrics['cls_acc']:.4f} | "
                f"val_cls_f1={val_metrics['cls_f1']:.4f}"
            )

            writer.writerow([
                epoch,
                train_total_loss,
                train_seg_loss,
                train_cls_loss,
                val_metrics["total_loss"],
                val_metrics["seg_loss"],
                val_metrics["cls_loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["cls_acc"],
                val_metrics["cls_f1"],
            ])
            f.flush()

            history["epoch"].append(epoch)
            history["train_total_loss"].append(train_total_loss)
            history["train_seg_loss"].append(train_seg_loss)
            history["train_cls_loss"].append(train_cls_loss)
            history["val_total_loss"].append(val_metrics["total_loss"])
            history["val_seg_loss"].append(val_metrics["seg_loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_cls_acc"].append(val_metrics["cls_acc"])
            history["val_cls_f1"].append(val_metrics["cls_f1"])

            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    save_training_plots(history, plot_dir)

    print("Finished training.")
    print(f"Best val dice: {best_dice:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()