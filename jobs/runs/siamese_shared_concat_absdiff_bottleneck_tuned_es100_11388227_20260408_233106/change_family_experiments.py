import os
import re
import csv
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
        pieces = pair.strip().split()
        if len(pieces) < 2:
            continue
        points.append((float(pieces[0]), float(pieces[1])))
    return points


def json_to_damage_mask_and_label(json_path, image_size=(1024, 1024)):
    with open(json_path, "r") as f:
        data = json.load(f)

    width, height = image_size
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
        polygon = parse_wkt_polygon(feat.get("wkt", ""))

        if len(polygon) >= 3:
            draw.polygon(polygon, outline=class_id, fill=class_id)
            max_class = max(max_class, class_id)

    return mask, max_class


class XView2ChangeDataset(Dataset):
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
            if is_post_disaster(stem):
                scenes[key]["post_json"] = self.json_map[stem]

        self.samples = []
        for key, item in scenes.items():
            if "pre_img" in item and "post_img" in item and "post_json" in item:
                self.samples.append((key, item))

        if len(self.samples) == 0:
            raise ValueError(f"No valid pre/post pairs with post_json found in {self.split_root}")

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)
        self.debug_samples = debug_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, item = self.samples[idx]

        pre_img = Image.open(item["pre_img"]).convert("RGB")
        post_img = Image.open(item["post_img"]).convert("RGB")
        damage_mask, class_label = json_to_damage_mask_and_label(item["post_json"], image_size=pre_img.size)

        pre_img = self.img_tf(pre_img)
        post_img = self.img_tf(post_img)

        damage_mask = self.mask_resize(damage_mask)
        damage_mask = torch.tensor(list(damage_mask.getdata()), dtype=torch.long).view(
            damage_mask.size[1], damage_mask.size[0]
        )
        class_label = torch.tensor(class_label, dtype=torch.long)

        if idx < self.debug_samples:
            print(
                f"[DEBUG] {key} | "
                f"mask unique={torch.unique(damage_mask)} | "
                f"mask max={damage_mask.max().item()} | "
                f"class_label={class_label.item()} ({IDX_TO_DAMAGE[class_label.item()]})"
            )

        return pre_img, post_img, damage_mask, class_label, key


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Encoder(nn.Module):
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


def fused_channels(channels, input_mode, fusion_strategy):
    if input_mode == "after_only":
        return channels
    if fusion_strategy == "concat":
        return channels * 2
    if fusion_strategy == "absdiff":
        return channels
    if fusion_strategy == "concat_absdiff":
        return channels * 3
    raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")


def fuse_features(before_feat, after_feat, input_mode, fusion_strategy):
    if input_mode == "after_only":
        return after_feat
    if fusion_strategy == "concat":
        return torch.cat([before_feat, after_feat], dim=1)
    if fusion_strategy == "absdiff":
        return torch.abs(before_feat - after_feat)
    if fusion_strategy == "concat_absdiff":
        return torch.cat([before_feat, after_feat, torch.abs(before_feat - after_feat)], dim=1)
    raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")


class ConfigurableChangeModel(nn.Module):
    def __init__(
        self,
        input_mode="siamese",
        encoder_mode="shared",
        fusion_strategy="concat",
        fusion_stage="bottleneck_only",
        base_channels=32,
        seg_out_ch=5,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.encoder_mode = encoder_mode
        self.fusion_strategy = fusion_strategy
        self.fusion_stage = fusion_stage

        self.after_encoder = Encoder(in_ch=3, base=base_channels)
        if input_mode == "siamese" and encoder_mode == "separate":
            self.before_encoder = Encoder(in_ch=3, base=base_channels)
        else:
            self.before_encoder = self.after_encoder

        bottleneck_in = fused_channels(base_channels * 8, input_mode, fusion_strategy)
        self.bottleneck_reduce = DoubleConv(bottleneck_in, base_channels * 8)

        if input_mode == "siamese" and fusion_stage == "multiscale":
            self.skip3_reduce = DoubleConv(
                fused_channels(base_channels * 4, input_mode, fusion_strategy),
                base_channels * 4,
            )
            self.skip2_reduce = DoubleConv(
                fused_channels(base_channels * 2, input_mode, fusion_strategy),
                base_channels * 2,
            )
            self.skip1_reduce = DoubleConv(
                fused_channels(base_channels, input_mode, fusion_strategy),
                base_channels,
            )
        else:
            self.skip3_reduce = None
            self.skip2_reduce = None
            self.skip1_reduce = None

        self.d3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.d2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.d1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        self.seg_out = nn.Conv2d(base_channels, seg_out_ch, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, seg_out_ch),
        )

    def forward(self, before, after):
        a1, a2, a3, ab = self.after_encoder(after)

        if self.input_mode == "after_only":
            fused_bottleneck = ab
            s1, s2, s3 = a1, a2, a3
        else:
            b1, b2, b3, bb = self.before_encoder(before)
            fused_bottleneck = fuse_features(bb, ab, self.input_mode, self.fusion_strategy)

            if self.fusion_stage == "multiscale":
                s3 = self.skip3_reduce(fuse_features(b3, a3, self.input_mode, self.fusion_strategy))
                s2 = self.skip2_reduce(fuse_features(b2, a2, self.input_mode, self.fusion_strategy))
                s1 = self.skip1_reduce(fuse_features(b1, a1, self.input_mode, self.fusion_strategy))
            else:
                s3, s2, s1 = a3, a2, a1

        x = self.bottleneck_reduce(fused_bottleneck)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)

        return self.seg_out(x), self.classifier(fused_bottleneck)


def multiclass_dice_score(logits, targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    dices = []
    for cls in range(1, num_classes):
        pred_c = (preds == cls).float()
        targ_c = (targets == cls).float()
        inter = (pred_c * targ_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))
        dices.append(((2 * inter + eps) / (denom + eps)).mean().item())
    return sum(dices) / len(dices)


def multiclass_iou_score(logits, targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for cls in range(1, num_classes):
        pred_c = (preds == cls).float()
        targ_c = (targets == cls).float()
        inter = (pred_c * targ_c).sum(dim=(1, 2))
        union = (pred_c + targ_c - pred_c * targ_c).sum(dim=(1, 2))
        ious.append(((inter + eps) / (union + eps)).mean().item())
    return sum(ious) / len(ious)


def classification_metrics(cls_logits, cls_targets, num_classes=5, eps=1e-6):
    preds = torch.argmax(cls_logits, dim=1)
    acc = (preds == cls_targets).float().mean().item()

    f1s = []
    for cls in range(1, num_classes):
        pred_c = preds == cls
        targ_c = cls_targets == cls
        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s.append(2 * precision * recall / (precision + recall + eps))

    return acc, sum(f1s) / len(f1s)


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

        batch_size = before.size(0)
        total_loss += loss.item() * batch_size
        total_seg_loss += seg_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size

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

        batch_size = before.size(0)
        total_loss += loss.item() * batch_size
        total_seg_loss += seg_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_dice += multiclass_dice_score(seg_logits, mask) * batch_size
        total_iou += multiclass_iou_score(seg_logits, mask) * batch_size

        acc, f1 = classification_metrics(cls_logits, class_label)
        total_cls_acc += acc * batch_size
        total_cls_f1 += f1 * batch_size

    n = len(loader.dataset)
    return {
        "total_loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "cls_acc": total_cls_acc / n,
        "cls_f1": total_cls_f1 / n,
        "joint_score": 0.5 * ((total_dice / n) + (total_cls_f1 / n)),
    }


def build_experiment_name(args):
    if args.exp_name:
        return args.exp_name
    if args.input_mode == "after_only":
        return "after_only_baseline"
    return "_".join(["siamese", args.encoder_mode, args.fusion_strategy, args.fusion_stage])


def save_training_plots(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_total_loss"], label="Train Total Loss")
    plt.plot(epochs, history["val_total_loss"], label="Val Total Loss")
    plt.plot(epochs, history["train_seg_loss"], label="Train Seg Loss")
    plt.plot(epochs, history["val_seg_loss"], label="Val Seg Loss")
    plt.plot(epochs, history["train_cls_loss"], label="Train Cls Loss")
    plt.plot(epochs, history["val_cls_loss"], label="Val Cls Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curves.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Segmentation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "segmentation_metrics.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_cls_acc"], label="Val Cls Accuracy")
    plt.plot(epochs, history["val_cls_f1"], label="Val Cls F1")
    plt.plot(epochs, history["val_joint_score"], label="Val Joint Score")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Classification And Joint Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "classification_metrics.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--selection_metric", type=str, default="joint", choices=["joint", "dice", "cls_f1", "cls_acc"])
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)

    parser.add_argument("--input_mode", type=str, default="siamese", choices=["after_only", "siamese"])
    parser.add_argument("--encoder_mode", type=str, default="shared", choices=["shared", "separate"])
    parser.add_argument("--fusion_strategy", type=str, default="concat", choices=["concat", "absdiff", "concat_absdiff"])
    parser.add_argument("--fusion_stage", type=str, default="bottleneck_only", choices=["bottleneck_only", "multiscale"])
    args = parser.parse_args()

    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be >= 0")

    if args.input_mode == "after_only":
        args.encoder_mode = "shared"
        args.fusion_stage = "bottleneck_only"

    experiment_name = build_experiment_name(args)
    run_name = args.run_name if args.run_name else experiment_name
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"resolved_experiment_name": experiment_name, "resolved_run_name": run_name}, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Experiment name: {experiment_name}")
    print(f"Train root: {args.train_root}")
    print(f"Val root: {args.val_root}")
    print(f"Run dir: {run_dir}")
    print(
        f"Config | input_mode={args.input_mode} | "
        f"encoder_mode={args.encoder_mode} | "
        f"fusion_strategy={args.fusion_strategy} | "
        f"fusion_stage={args.fusion_stage}"
    )
    if args.early_stopping_patience > 0:
        print(
            f"Early stopping | patience={args.early_stopping_patience} | "
            f"min_delta={args.early_stopping_min_delta}"
        )
    else:
        print("Early stopping | disabled")

    train_ds = XView2ChangeDataset(args.train_root, image_size=args.image_size, debug_samples=args.debug_samples)
    val_ds = XView2ChangeDataset(args.val_root, image_size=args.image_size, debug_samples=args.debug_samples)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ConfigurableChangeModel(
        input_mode=args.input_mode,
        encoder_mode=args.encoder_mode,
        fusion_strategy=args.fusion_strategy,
        fusion_stage=args.fusion_stage,
        base_channels=args.base_channels,
    ).to(device)

    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    plot_dir = os.path.join(run_dir, "plots")

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
        "val_joint_score": [],
    }

    selection_key = {
        "joint": "joint_score",
        "dice": "dice",
        "cls_f1": "cls_f1",
        "cls_acc": "cls_acc",
    }[args.selection_metric]
    best_value = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

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
            "val_joint_score",
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
                f"val_cls_f1={val_metrics['cls_f1']:.4f} | "
                f"val_joint={val_metrics['joint_score']:.4f}"
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
                val_metrics["joint_score"],
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
            history["val_joint_score"].append(val_metrics["joint_score"])

            current_value = val_metrics[selection_key]
            if current_value > best_value + args.early_stopping_min_delta:
                best_value = current_value
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
            else:
                epochs_without_improvement += 1

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                print(
                    f"Early stopping triggered at epoch {epoch} after "
                    f"{epochs_without_improvement} epochs without "
                    f"{selection_key} improvement greater than "
                    f"{args.early_stopping_min_delta}."
                )
                break

    save_training_plots(history, plot_dir)
    print("Finished training.")
    print(f"Best {selection_key}: {best_value:.4f} at epoch {best_epoch}")
    if stopped_early:
        print(f"Training stopped early at epoch {history['epoch'][-1]}.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
