import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              precision_recall_curve, roc_curve, average_precision_score)

from src.data.dataset import DRDataset
from src.preprocessing.transforms import get_transforms
from src.classification.model import get_model
from torch.utils.data import DataLoader

DR_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true, y_pred, y_prob, referable_th=2, out_dir="outputs/reports"):
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class = []
    for cls in range(5):
        bin_true = (y_true == cls).astype(int)
        bin_pred = (y_pred == cls).astype(int)
        per_class.append({
            "class": cls,
            "label": DR_LABELS[cls],
            "precision": float(precision_score(bin_true, bin_pred, zero_division=0)),
            "recall": float(recall_score(bin_true, bin_pred, zero_division=0)),
            "f1": float(f1_score(bin_true, bin_pred, zero_division=0)),
            "support": int(np.sum(bin_true)),
        })
    per_class_df = pd.DataFrame(per_class)
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    # Referable DR (binary) metrics
    ref_true = (y_true >= referable_th).astype(int)
    ref_pred = (y_pred >= referable_th).astype(int)
    cm_ref = confusion_matrix(ref_true, ref_pred)
    tn, fp, fn, tp = cm_ref.ravel()
    metrics["referable_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["referable_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["referable_ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["referable_npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    metrics["referable_tp"] = int(tp)
    metrics["referable_tn"] = int(tn)
    metrics["referable_fp"] = int(fp)
    metrics["referable_fn"] = int(fn)

    # ROC-AUC (multi-class OvR)
    try:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        metrics["roc_auc_ovr"] = None
        print(f"[WARN] ROC-AUC failed: {e}")

    # PR-AUC (referable binary)
    ref_prob = y_prob[:, referable_th:].sum(axis=1)
    try:
        metrics["pr_auc_referable"] = float(average_precision_score(ref_true, ref_prob))
    except Exception as e:
        metrics["pr_auc_referable"] = None

    # Save JSON
    with open(os.path.join(out_dir, "evaluation_report.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "evaluation_report.csv"), index=False)

    # ── Confusion Matrix ────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(DR_LABELS.values()), yticklabels=list(DR_LABELS.values()))
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC Curve (referable DR binary) ────────────────────────────────────────
    try:
        fpr, tpr, _ = roc_curve(ref_true, ref_prob)
        auc_val = metrics.get("pr_auc_referable") or 0.0
        roc_auc = metrics.get("roc_auc_ovr") or 0.0
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve – Referable DR")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] ROC curve failed: {e}")

    # ── Precision-Recall Curve ──────────────────────────────────────────────────
    try:
        prec, rec, _ = precision_recall_curve(ref_true, ref_prob)
        pr_auc = metrics.get("pr_auc_referable") or 0.0
        plt.figure(figsize=(7, 5))
        plt.plot(rec, prec, color="darkorange", lw=2, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve – Referable DR")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] PR curve failed: {e}")

    return metrics


def evaluate(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    split_dir = "data/splits"
    if not os.path.exists(os.path.join(split_dir, "test.csv")):
        print("Splits not found. Please run prepare_dataset.py first.")
        return

    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    dataset_root = config["dataset"]["root"]
    image_dir = os.path.join(dataset_root, "train_images")

    # Use "val" transform mode — same as test (no augmentation)
    test_transform = get_transforms(image_size=config["image"]["size"], mode="val")
    test_dataset = DRDataset(test_df, image_dir, transform=test_transform)
    # num_workers=0 for Windows compatibility
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    model = get_model(num_classes=config["classification"]["num_classes"],
                      model_name=config["classification"]["model"])
    model_path = "models/classification/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("[WARNING] Model checkpoint not found. Evaluating with untrained weights.")
        print("Results will NOT be meaningful. Train the model first.")

    model = model.to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs,
                                referable_th=config["referable"]["minimum_grade"])

    print("\n═══════════════════════════════════════════")
    print("           EVALUATION RESULTS")
    print("═══════════════════════════════════════════")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Macro:            {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC (OvR):       {metrics.get('roc_auc_ovr', 'N/A')}")
    print(f"PR-AUC (Referable):  {metrics.get('pr_auc_referable', 'N/A')}")
    print("─── Referable DR (binary) ─────────────────")
    print(f"Sensitivity:         {metrics['referable_sensitivity']:.4f}")
    print(f"Specificity:         {metrics['referable_specificity']:.4f}")
    print(f"PPV:                 {metrics['referable_ppv']:.4f}")
    print(f"NPV:                 {metrics['referable_npv']:.4f}")
    print("─── Target ────────────────────────────────")
    print("  Sensitivity > 90%:", "✓ MET" if metrics["referable_sensitivity"] >= 0.9 else "✗ NOT MET")
    print("  Specificity > 85%:", "✓ MET" if metrics["referable_specificity"] >= 0.85 else "✗ NOT MET")
    print("═══════════════════════════════════════════")
    print("Outputs saved to outputs/reports/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DR Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
