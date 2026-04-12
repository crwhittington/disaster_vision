import numpy as np
import torch
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------
# Flatten helper
# -------------------------
def _flatten(y_true, y_pred):
    # Convert PyTorch → NumPy if needed
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert y_true.shape == y_pred.shape, f"{y_true.shape} vs {y_pred.shape}"
    return y_true.reshape(-1), y_pred.reshape(-1)

def normalize_pred(y_pred):
    y_pred = y_pred.detach()

    # Case: batched logits (B, C, H, W)
    if y_pred.ndim == 4:
        y_pred = torch.argmax(y_pred, dim=1)

    # Case: single image logits (C, H, W)
    elif y_pred.ndim == 3 and y_pred.shape[0] > 1:
        # likely logits, not mask
        y_pred = torch.argmax(y_pred, dim=0)

    # Case: (1, H, W)
    elif y_pred.ndim == 3 and y_pred.shape[0] == 1:
        y_pred = y_pred.squeeze(0)

    return y_pred

# -------------------------
# Localization F1
# -------------------------
def localization_f1(y_true, y_pred):
    y_true_flat, y_pred_flat = _flatten(y_true, y_pred)

    y_true_bin = (y_true_flat > 0).astype(int)
    y_pred_bin = (y_pred_flat > 0).astype(int)

    return f1_score(y_true_bin, y_pred_bin)


# -------------------------
# Damage F1 (macro + per-class)
# -------------------------
def damage_f1_with_breakdown(y_true, y_pred):
    y_true_flat, y_pred_flat = _flatten(y_true, y_pred)

    mask = y_true_flat > 0
    if mask.sum() == 0:
        return {
            "macro_f1": 0.0,
            "per_class_f1": {c: 0.0 for c in [1,2,3,4]}
        }

    y_true_build = y_true_flat[mask]
    y_pred_build = y_pred_flat[mask]

    labels = [1, 2, 3, 4]

    # Per-class F1
    per_class = f1_score(
        y_true_build,
        y_pred_build,
        labels=labels,
        average=None,
        zero_division=0
    )

    per_class_dict = {cls: score for cls, score in zip(labels, per_class)}

    # Macro F1
    macro = f1_score(
        y_true_build,
        y_pred_build,
        labels=labels,
        average="macro",
        zero_division=0
    )

    return {
        "macro_f1": macro,
        "per_class_f1": per_class_dict
    }


# -------------------------
# Single image score
# -------------------------
def xview2_score_single(y_true, y_pred, w_loc=0.3, w_dmg=0.7):
    loc = localization_f1(y_true, y_pred)
    dmg_dict = damage_f1_with_breakdown(y_true, y_pred)

    final = w_loc * loc + w_dmg * dmg_dict["macro_f1"]

    return {
        "localization_f1": loc,
        "damage_f1": dmg_dict["macro_f1"],
        "damage_per_class": dmg_dict["per_class_f1"],
        "final_score": final
    }


# -------------------------
# FULL DATASET EVALUATION
# -------------------------
def xview2_score_dataset(y_true_list, y_pred_list, w_loc=0.3, w_dmg=0.7):
    # -------------------------
    # Accumulators
    # -------------------------
    tp = fp = fn = 0  # for localization

    # confusion matrix for damage (classes 1–4)
    num_classes = 4
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    # -------------------------
    # Loop once over dataset
    # -------------------------
    for yt, yp in zip(y_true_list, y_pred_list):
        yt = np.squeeze(yt)
        yp = np.squeeze(yp)

        # -------------------------
        # Localization (binary)
        # -------------------------
        yt_bin = yt > 0
        yp_bin = yp > 0

        tp += np.sum(yt_bin & yp_bin)
        fp += np.sum(~yt_bin & yp_bin)
        fn += np.sum(yt_bin & ~yp_bin)

        # -------------------------
        # Damage (only GT buildings)
        # -------------------------
        mask = yt_bin
        if np.any(mask):
            yt_build = yt[mask] - 1  # shift to 0–3
            yp_build = yp[mask] - 1

            # filter invalid preds
            valid = (yp_build >= 0) & (yp_build < num_classes)
            yt_build = yt_build[valid]
            yp_build = yp_build[valid]

            # update confusion matrix
            idx = yt_build * num_classes + yp_build
            bincount = np.bincount(idx, minlength=num_classes**2)
            conf += bincount.reshape(num_classes, num_classes)

    # -------------------------
    # Localization F1
    # -------------------------
    loc_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    # -------------------------
    # Damage F1 (per class)
    # -------------------------
    per_class_f1 = {}

    for c in range(num_classes):
        TP = conf[c, c]
        FP = conf[:, c].sum() - TP
        FN = conf[c, :].sum() - TP

        denom = (2 * TP + FP + FN)
        f1 = (2 * TP / denom) if denom > 0 else 0.0

        per_class_f1[c + 1] = f1  # shift back to 1–4

    dmg_macro = np.mean(list(per_class_f1.values()))

    # -------------------------
    # Final score
    # -------------------------
    final = w_loc * loc_f1 + w_dmg * dmg_macro

    return {
        "localization_f1": loc_f1,
        "damage_f1": dmg_macro,
        "damage_per_class": per_class_f1,
        "final_score": final
    }

import torch


class XView2Evaluator:
    def __init__(self, num_classes=4, device="cuda"):
        self.num_classes = num_classes
        self.device = device

        # localization accumulation (building segmentation)
        self.tp = torch.tensor(0.0, device=device)
        self.tn = torch.tensor(0.0, device=device)
        self.fp = torch.tensor(0.0, device=device)
        self.fn = torch.tensor(0.0, device=device)

        # Damage confusion matrix (4x4)
        self.conf = torch.zeros((num_classes, num_classes), device=device)

    @torch.no_grad()
    def update(self, y_true, y_pred):
        """
        y_true: (H, W) tensor
        y_pred: (H, W) tensor OR (C, H, W) logits
        """

        # -------------------------
        # Preprocess
        # -------------------------
        # y_true, y_pred = _flatten(y_true, y_pred)

        y_true = y_true.squeeze().to(self.device)

        y_pred = normalize_pred(y_pred)

        y_pred = y_pred.squeeze().to(self.device)

        # print("GT unique:", torch.unique(y_true))
        # print("Pred unique:", torch.unique(y_pred))

        # print("GT buildings:", (y_true > 0).sum().item())
        # print("Pred buildings:", (y_pred > 0).sum().item())

        # print("Overlap (TP):", ((y_true > 0) & (y_pred > 0)).sum().item())

        # -------------------------
        # Localization
        # -------------------------
        yt_bin = y_true > 0
        yp_bin = y_pred > 0

        self.tp += torch.sum(yt_bin & yp_bin)
        self.tn += torch.sum(~yt_bin & ~yp_bin)
        self.fp += torch.sum(~yt_bin & yp_bin)
        self.fn += torch.sum(yt_bin & ~yp_bin)

        # -------------------------
        # Damage (only GT buildings)
        # -------------------------
        mask = y_true > 0

        if mask.sum() > 0:
            yt_build = y_true[mask].to(torch.long) - 1
            yp_build = y_pred[mask].to(torch.long) - 1

            valid = (yp_build >= 0) & (yp_build < self.num_classes)
            yt_build = yt_build[valid]
            yp_build = yp_build[valid]

            if yt_build.numel() > 0:
                idx = yt_build * self.num_classes + yp_build
                idx = idx.to(torch.long)

                bincount = torch.bincount(
                    idx,
                    minlength=self.num_classes ** 2
                )

                self.conf += bincount.view(self.num_classes, self.num_classes)

    def compute(self, w_loc=0.3, w_dmg=0.7):
        eps = 1e-8

        # -------------------------
        # Localization F1
        # -------------------------
        loc_f1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)

        # -------------------------
        # Damage F1
        # -------------------------
        per_class_f1 = []

        for c in range(self.num_classes):
            TP = self.conf[c, c]
            FP = self.conf[:, c].sum() - TP
            FN = self.conf[c, :].sum() - TP

            denom = (2 * TP + FP + FN)
            f1 = torch.where(denom > 0, 2 * TP / denom, torch.tensor(0.0, device=self.device))

            per_class_f1.append(f1)

        per_class_f1 = torch.stack(per_class_f1)
        dmg_macro = per_class_f1.mean()

        # -------------------------
        # Final score
        # -------------------------
        final = w_loc * loc_f1 + w_dmg * dmg_macro

        return {
            "localization_f1": loc_f1.item(),
            "damage_f1": dmg_macro.item(),
            "damage_per_class": {
                i + 1: per_class_f1[i].item() for i in range(self.num_classes)
            },
            "final_score": final.item(),
            "damage_cm": self.conf.cpu().tolist(),
            "building_cm": torch.tensor([
                [self.tn, self.fp],
                [self.fn, self.tp]
            ], device=self.device).cpu().tolist()
        }

def normalize_confusion_matrix(conf, row_wise=True):
    conf = conf.float()

    if row_wise:
        row_sums = conf.sum(dim=1, keepdim=True)
        norm_conf = conf / (row_sums + 1e-8)
    else:
        col_sums = conf.sum(dim=0, keepdim=True)
        norm_conf = conf / (col_sums + 1e-8)

    return norm_conf

def plot_confusion(conf, row_wise=True, labels=None):
    conf = normalize_confusion_matrix(conf, row_wise=row_wise)

    plt.figure(figsize=(6,5))
    if labels:
        sns.heatmap(conf, annot=True, fmt=".3f",
                xticklabels=labels,
                yticklabels=labels)
    else:
        sns.heatmap(conf, annot=True, fmt=".3f")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()