import os
import json
import csv
import math
import copy
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn

from transformers import SegformerForSemanticSegmentation


# Global class metadata for xView2 damage labels
CLASS_NAMES = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
NUM_CLASSES = 5


# Reproducibility setup
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# Dataset for loading pre/post disaster imagery and masks
class XView2Dataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.files = sorted(
            [
                f.replace("_post_disaster.jpg", "").replace("_post_disaster.png", "")
                for f in os.listdir(image_dir)
                if "_post_disaster" in f
            ]
        )

        self.damage_map = {
            "un-classified": 0,
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
        }

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_post_mask(self, json_path: str, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(json_path, "r") as f:
            data = json.load(f)

        for feature in data["features"]["xy"]:
            props = feature["properties"]
            if "subtype" not in props:
                continue

            damage = props["subtype"]
            class_id = self.damage_map[damage]

            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])

            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)

        return mask

    def _load_pre_mask(self, json_path: str, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(json_path, "r") as f:
            data = json.load(f)

        for feature in data["features"]["xy"]:
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])

            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)

        return mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = self.files[idx]

        pre_path = None
        post_path = None
        for ext in [".jpg", ".png"]:
            candidate_pre = os.path.join(self.image_dir, base + "_pre_disaster" + ext)
            candidate_post = os.path.join(self.image_dir, base + "_post_disaster" + ext)
            if os.path.exists(candidate_pre) and os.path.exists(candidate_post):
                pre_path = candidate_pre
                post_path = candidate_post
                break

        if pre_path is None or post_path is None:
            raise FileNotFoundError(f"Could not find pre/post image pair for base name {base}")

        pre_img = self._load_image(pre_path)
        post_img = self._load_image(post_path)

        image = np.concatenate([pre_img, post_img], axis=2)

        pre_label_path = os.path.join(self.label_dir, base + "_pre_disaster.json")
        post_label_path = os.path.join(self.label_dir, base + "_post_disaster.json")

        pre_mask = self._load_pre_mask(pre_label_path, pre_img.shape[:2])
        post_mask = self._load_post_mask(post_label_path, pre_img.shape[:2])

        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "pre_mask": torch.tensor(pre_mask).long(),
            "post_mask": torch.tensor(post_mask).long(),
        }


# Build train/val/test datasets from xView2 split folders
def build_datasets(data_root: str):
    base_folder = os.path.join(data_root, "xview2_jpeg")

    train_dataset = XView2Dataset(
        os.path.join(base_folder, "tier1/images_jpeg"),
        os.path.join(base_folder, "tier1/labels"),
    )
    val_dataset = XView2Dataset(
        os.path.join(base_folder, "hold/images_jpeg"),
        os.path.join(base_folder, "hold/labels"),
    )
    test_dataset = XView2Dataset(
        os.path.join(base_folder, "test/images_jpeg"),
        os.path.join(base_folder, "test/labels"),
    )

    return train_dataset, val_dataset, test_dataset


# Wrap datasets in PyTorch dataloaders
def build_loaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, pin_memory: bool):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# Compute class distribution on building pixels only
def compute_building_class_distribution(loader, num_classes: int):
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for batch in tqdm(loader, desc="Computing class distribution", leave=False):
        pre_mask = batch["pre_mask"]
        post_mask = batch["post_mask"]
        building_mask = pre_mask > 0
        labels = post_mask[building_mask]

        for cls in range(num_classes):
            class_counts[cls] += (labels == cls).sum().item()

    total = class_counts.sum()
    class_percentages = class_counts / total.clamp(min=1.0)
    return class_counts, class_percentages


# Save class distribution table and figure
def save_class_distribution(train_counts, train_perc, val_counts, val_perc, test_counts, test_perc, output_dir: Path):
    rows = []
    for i, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "Class": class_name,
                "Train Count": int(train_counts[i].item()),
                "Train %": float(train_perc[i].item() * 100),
                "Val Count": int(val_counts[i].item()),
                "Val %": float(val_perc[i].item() * 100),
                "Test Count": int(test_counts[i].item()),
                "Test %": float(test_perc[i].item() * 100),
            }
        )

    csv_path = output_dir / "class_distribution_building_pixels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, train_perc.numpy() * 100, width, label="Train")
    plt.bar(x, val_perc.numpy() * 100, width, label="Val")
    plt.bar(x + width, test_perc.numpy() * 100, width, label="Test")
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.ylabel("Percent of Building Pixels")
    plt.title("Damage Class Distribution on Building Pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_building_pixels.png", dpi=200)
    plt.close()


# Build SegFormer and modify first conv layer to accept 6 channels
def build_segformer(device: torch.device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    original_conv = model.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=True,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = original_conv.weight
        new_conv.weight[:, 3:] = original_conv.weight
        new_conv.bias.copy_(original_conv.bias)

    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    model = model.to(device)
    return model


# Build weighted cross-entropy loss to address class imbalance
def build_weighted_criterion(train_counts: torch.Tensor, device: torch.device):
    counts_for_weights = train_counts.clone()
    counts_for_weights[counts_for_weights == 0] = 1.0

    class_weights = 1.0 / counts_for_weights
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    return criterion, class_weights


# Compute masked loss on building pixels only
def compute_masked_loss(logits, post_mask, pre_mask, criterion):
    pixel_loss = criterion(logits, post_mask)
    building_mask = (pre_mask > 0).float()
    loss = (pixel_loss * building_mask).sum() / (building_mask.sum() + 1e-8)
    return loss


# Compute per-class IoU, Dice, and F1 on building pixels only
def compute_per_class_metrics(logits, labels, pre_mask, num_classes: int):
    preds = torch.argmax(logits, dim=1)
    building_mask = pre_mask > 0

    preds = preds[building_mask]
    labels = labels[building_mask]

    per_class_iou = []
    per_class_dice = []
    per_class_f1 = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls

        tp = (pred_cls & label_cls).sum().item()
        fp = (pred_cls & ~label_cls).sum().item()
        fn = (~pred_cls & label_cls).sum().item()

        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        per_class_iou.append(iou)
        per_class_dice.append(dice)
        per_class_f1.append(f1)

    return per_class_iou, per_class_dice, per_class_f1


# Macro metric including all classes
def macro_all_classes(values):
    return float(np.mean(values))


# Macro metric excluding background
def macro_without_background(values):
    return float(np.mean(values[1:]))


# One training epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        pre_mask = batch["pre_mask"].to(device, non_blocking=True)
        post_mask = batch["post_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=post_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = compute_masked_loss(logits, post_mask, pre_mask, criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Validation or test evaluation pass
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_per_class_iou = []
    all_per_class_dice = []
    all_per_class_f1 = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            pre_mask = batch["pre_mask"].to(device, non_blocking=True)
            post_mask = batch["post_mask"].to(device, non_blocking=True)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=post_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = compute_masked_loss(logits, post_mask, pre_mask, criterion)
            total_loss += loss.item()

            per_class_iou, per_class_dice, per_class_f1 = compute_per_class_metrics(
                logits, post_mask, pre_mask, NUM_CLASSES
            )

            all_per_class_iou.append(per_class_iou)
            all_per_class_dice.append(per_class_dice)
            all_per_class_f1.append(per_class_f1)

    mean_per_class_iou = np.mean(np.array(all_per_class_iou), axis=0)
    mean_per_class_dice = np.mean(np.array(all_per_class_dice), axis=0)
    mean_per_class_f1 = np.mean(np.array(all_per_class_f1), axis=0)

    return {
        "loss": total_loss / len(loader),
        "per_class_iou": mean_per_class_iou,
        "per_class_dice": mean_per_class_dice,
        "per_class_f1": mean_per_class_f1,
        "macro_iou_all": macro_all_classes(mean_per_class_iou),
        "macro_dice_all": macro_all_classes(mean_per_class_dice),
        "macro_f1_all": macro_all_classes(mean_per_class_f1),
        "macro_iou_damage": macro_without_background(mean_per_class_iou),
        "macro_dice_damage": macro_without_background(mean_per_class_dice),
        "macro_f1_damage": macro_without_background(mean_per_class_f1),
    }


# Save training curves
def save_training_curves(history: dict, output_dir: Path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_losses"], label="Train Loss")
    plt.plot(history["val_losses"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["val_iou_all"], label="IoU All")
    plt.plot(history["val_iou_damage"], label="IoU Damage")
    plt.title("IoU")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1_all"], label="F1 All")
    plt.plot(history["val_f1_damage"], label="F1 Damage")
    plt.title("F1")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


# Save per-class IoU curves
def save_per_class_iou_curves(per_class_iou_history, output_dir: Path):
    plt.figure(figsize=(8, 5))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        plt.plot(
            [epoch_vals[cls_idx] for epoch_vals in per_class_iou_history],
            label=cls_name,
        )

    plt.title("Per-Class IoU Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_iou_over_epochs.png", dpi=200)
    plt.close()


# Save epoch-level metrics
def save_results_csv(history: dict, output_dir: Path):
    rows = []
    epochs = len(history["train_losses"])

    for i in range(epochs):
        rows.append(
            {
                "Epoch": i + 1,
                "Train Loss": history["train_losses"][i],
                "Val Loss": history["val_losses"][i],
                "Val IoU All": history["val_iou_all"][i],
                "Val Dice All": history["val_dice_all"][i],
                "Val F1 All": history["val_f1_all"][i],
                "Val IoU Damage": history["val_iou_damage"][i],
                "Val Dice Damage": history["val_dice_damage"][i],
                "Val F1 Damage": history["val_f1_damage"][i],
            }
        )

    csv_path = output_dir / "segformer_training_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Save final per-class test metrics
def save_final_per_class_metrics(metrics_dict: dict, output_dir: Path):
    rows = []
    for i, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "Class": class_name,
                "IoU": float(metrics_dict["per_class_iou"][i]),
                "Dice": float(metrics_dict["per_class_dice"][i]),
                "F1": float(metrics_dict["per_class_f1"][i]),
            }
        )

    csv_path = output_dir / "final_per_class_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Save one qualitative prediction example
def save_prediction_visualization(model, loader, device, output_dir: Path):
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    pre_mask = batch["pre_mask"].to(device)
    post_mask = batch["post_mask"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=post_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.argmax(logits, dim=1)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(images[0][:3].cpu().permute(1, 2, 0))
    plt.title("Pre Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(images[0][3:6].cpu().permute(1, 2, 0))
    plt.title("Post Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(post_mask[0].cpu(), vmin=0, vmax=4)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(preds[0].cpu(), vmin=0, vmax=4)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_example.png", dpi=200)
    plt.close()


# Save building-pixel confusion matrix
def save_confusion_matrix(model, loader, device, output_dir: Path):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    pre_mask = batch["pre_mask"].to(device)
    post_mask = batch["post_mask"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=post_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.argmax(logits, dim=1)

    building_mask = pre_mask[0] > 0
    preds_flat = preds[0][building_mask].cpu().numpy().flatten()
    masks_flat = post_mask[0][building_mask].cpu().numpy().flatten()

    cm = confusion_matrix(masks_flat, preds_flat, labels=np.arange(NUM_CLASSES))

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix on Building Pixels")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_building_pixels.png", dpi=200)
    plt.close()


# Save final run metadata and summary metrics
def save_run_summary(args, device, train_dataset, val_dataset, test_dataset, class_weights, best_epoch, test_metrics, output_dir: Path):
    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "early_stopping_patience": args.patience,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_names": CLASS_NAMES,
        "class_weights": [float(x) for x in class_weights.detach().cpu().numpy()],
        "best_epoch_by_val_damage_iou": int(best_epoch),
        "test_loss": float(test_metrics["loss"]),
        "test_macro_iou_all": float(test_metrics["macro_iou_all"]),
        "test_macro_dice_all": float(test_metrics["macro_dice_all"]),
        "test_macro_f1_all": float(test_metrics["macro_f1_all"]),
        "test_macro_iou_damage": float(test_metrics["macro_iou_damage"]),
        "test_macro_dice_damage": float(test_metrics["macro_dice_damage"]),
        "test_macro_f1_damage": float(test_metrics["macro_f1_damage"]),
        "test_per_class_iou": [float(x) for x in test_metrics["per_class_iou"]],
        "test_per_class_dice": [float(x) for x in test_metrics["per_class_dice"]],
        "test_per_class_f1": [float(x) for x in test_metrics["per_class_f1"]],
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# Command-line arguments for Slurm execution
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Main training and evaluation pipeline
def main():
    args = parse_args()
    setup_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Device:", device)

    train_dataset, val_dataset, test_dataset = build_datasets(args.data_root)
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample = train_dataset[0]
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Sample image shape:", sample["image"].shape)
    print("Sample pre-mask shape:", sample["pre_mask"].shape)
    print("Sample post-mask shape:", sample["post_mask"].shape)
    print("Sample post-mask unique values:", torch.unique(sample["post_mask"]))

    train_counts, train_perc = compute_building_class_distribution(train_loader, NUM_CLASSES)
    val_counts, val_perc = compute_building_class_distribution(val_loader, NUM_CLASSES)
    test_counts, test_perc = compute_building_class_distribution(test_loader, NUM_CLASSES)

    save_class_distribution(
        train_counts=train_counts,
        train_perc=train_perc,
        val_counts=val_counts,
        val_perc=val_perc,
        test_counts=test_counts,
        test_perc=test_perc,
        output_dir=output_dir,
    )

    model = build_segformer(device)
    criterion, class_weights = build_weighted_criterion(train_counts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_losses": [],
        "val_losses": [],
        "val_iou_all": [],
        "val_dice_all": [],
        "val_f1_all": [],
        "val_iou_damage": [],
        "val_dice_damage": [],
        "val_f1_damage": [],
    }
    per_class_iou_history = []
    per_class_dice_history = []
    per_class_f1_history = []

    best_epoch = 0
    best_val_iou_damage = -math.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_metrics["loss"])
        history["val_iou_all"].append(val_metrics["macro_iou_all"])
        history["val_dice_all"].append(val_metrics["macro_dice_all"])
        history["val_f1_all"].append(val_metrics["macro_f1_all"])
        history["val_iou_damage"].append(val_metrics["macro_iou_damage"])
        history["val_dice_damage"].append(val_metrics["macro_dice_damage"])
        history["val_f1_damage"].append(val_metrics["macro_f1_damage"])

        per_class_iou_history.append(val_metrics["per_class_iou"])
        per_class_dice_history.append(val_metrics["per_class_dice"])
        per_class_f1_history.append(val_metrics["per_class_f1"])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val IoU All: {val_metrics['macro_iou_all']:.4f}")
        print(f"Val Dice All: {val_metrics['macro_dice_all']:.4f}")
        print(f"Val F1 All: {val_metrics['macro_f1_all']:.4f}")
        print(f"Val IoU Damage: {val_metrics['macro_iou_damage']:.4f}")
        print(f"Val Dice Damage: {val_metrics['macro_dice_damage']:.4f}")
        print(f"Val F1 Damage: {val_metrics['macro_f1_damage']:.4f}")

        if val_metrics["macro_iou_damage"] > best_val_iou_damage:
            best_val_iou_damage = val_metrics["macro_iou_damage"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_segformer.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Early stopping counter: {epochs_without_improvement}/{args.patience}")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Epoch by Validation Damage IoU: {best_epoch}")

    save_results_csv(history, output_dir)
    save_training_curves(history, output_dir)
    save_per_class_iou_curves(per_class_iou_history, output_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nTest Metrics")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test IoU All: {test_metrics['macro_iou_all']:.4f}")
    print(f"Test Dice All: {test_metrics['macro_dice_all']:.4f}")
    print(f"Test F1 All: {test_metrics['macro_f1_all']:.4f}")
    print(f"Test IoU Damage: {test_metrics['macro_iou_damage']:.4f}")
    print(f"Test Dice Damage: {test_metrics['macro_dice_damage']:.4f}")
    print(f"Test F1 Damage: {test_metrics['macro_f1_damage']:.4f}")

    save_final_per_class_metrics(test_metrics, output_dir)
    save_prediction_visualization(model, val_loader, device, output_dir)
    save_confusion_matrix(model, val_loader, device, output_dir)
    save_run_summary(
        args=args,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_weights=class_weights,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()