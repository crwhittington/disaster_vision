import os
import csv
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from siamese_baseline import (
    IDX_TO_DAMAGE,
    XView2SiameseDataset,
    BaselineSiameseUNetWithClassifier,
    train_one_epoch,
    evaluate,
    save_training_plots,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="baseline_xview2_v2_run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--debug_samples", type=int, default=3)
    parser.add_argument("--selection_metric", type=str, default="joint", choices=["joint", "dice", "cls_f1", "cls_acc"])
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
    print(f"Training profile: v2 | lr={args.lr} | epochs={args.epochs} | selection_metric={args.selection_metric}")

    train_ds = XView2SiameseDataset(
        args.train_root,
        image_size=args.image_size,
        debug_samples=args.debug_samples,
    )
    val_ds = XView2SiameseDataset(
        args.val_root,
        image_size=args.image_size,
        debug_samples=args.debug_samples,
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = BaselineSiameseUNetWithClassifier().to(device)
    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    plot_dir = os.path.join(args.save_dir, "plots")
    config_path = os.path.join(args.save_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

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

    metric_key = {
        "joint": "joint_score",
        "dice": "dice",
        "cls_f1": "cls_f1",
        "cls_acc": "cls_acc",
    }[args.selection_metric]
    best_metric = -1.0

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
                model,
                train_loader,
                seg_criterion,
                cls_criterion,
                optimizer,
                device,
                cls_weight=args.cls_weight,
            )

            val_metrics = evaluate(
                model,
                val_loader,
                seg_criterion,
                cls_criterion,
                device,
                cls_weight=args.cls_weight,
            )
            val_joint_score = 0.5 * (val_metrics["dice"] + val_metrics["cls_f1"])

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
                f"val_joint={val_joint_score:.4f}"
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
                val_joint_score,
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
            history["val_joint_score"].append(val_joint_score)

            current_value = val_joint_score if metric_key == "joint_score" else val_metrics[metric_key]
            if current_value > best_metric:
                best_metric = current_value
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    save_training_plots(history, plot_dir)

    print("Finished training.")
    print(f"Best {args.selection_metric}: {best_metric:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plot_dir}")
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
