"""
scripts/validate_dataset.py — Comprehensive dataset validation
============================================================
Usage:
    python scripts/validate_dataset.py --config configs/config.yaml

Generates:
    outputs/reports/dataset_report.json
    outputs/reports/duplicate_report.json
    outputs/reports/leakage_report.json
    outputs/reports/class_distribution.csv
    outputs/reports/dataset_summary.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.deduplication import DuplicateDetector
from src.data.leakage import LeakageDetector
from src.data.split import create_stratified_splits


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_image_file(img_path):
    """Check if a single image is readable and valid. Returns dict of issues."""
    issues = []
    if not os.path.exists(img_path):
        return {"path": img_path, "issues": ["file_not_found"]}

    try:
        img = cv2.imread(img_path)
        if img is None:
            issues.append("unreadable")
        else:
            h, w = img.shape[:2]
            if h < 64 or w < 64:
                issues.append("too_small")
            if h > 5000 or w > 5000:
                issues.append("extremely_large")
    except Exception as e:
        issues.append(f"error: {str(e)}")

    return {"path": img_path, "issues": issues}


def validate_labels(df, label_col="diagnosis", num_classes=5):
    """Check for missing or invalid labels."""
    issues = []
    if label_col not in df.columns:
        return [f"Label column '{label_col}' not found in CSV"]

    null_labels = df[label_col].isnull().sum()
    if null_labels > 0:
        issues.append(f"{null_labels} rows with missing labels")

    valid_range = set(range(num_classes))
    invalid = df[~df[label_col].isin(valid_range)]
    if len(invalid) > 0:
        issues.append(f"{len(invalid)} rows with invalid labels: {df[label_col].unique().tolist()}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate DR dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Skip per-image validation (faster for large datasets)")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = config["dataset"]["root"]
    os.makedirs("outputs/reports", exist_ok=True)

    print("═══════════════════════════════════════════════")
    print("         DR Dataset Validation Report")
    print("═══════════════════════════════════════════════")

    csv_path = os.path.join(dataset_root, "train.csv")
    image_dir = os.path.join(dataset_root, "train_images")

    if not os.path.exists(csv_path):
        print(f"[ERROR] train.csv not found at: {csv_path}")
        return
    if not os.path.exists(image_dir):
        print(f"[ERROR] train_images directory not found at: {image_dir}")
        return

    df = pd.read_csv(csv_path)
    print(f"\nDataset: {config['dataset']['name']} | Records: {len(df)}")

    # ── Label validation ────────────────────────────────────────────────────
    print("\n[1/5] Validating labels...")
    label_issues = validate_labels(df)
    if label_issues:
        for issue in label_issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ All labels valid (0–4)")

    # ── Class distribution ──────────────────────────────────────────────────
    print("\n[2/5] Class distribution...")
    class_dist = df["diagnosis"].value_counts().sort_index()
    label_names = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
    for cls, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {cls} ({label_names.get(cls, '?')}): {count:5d} ({pct:.1f}%)")

    class_dist_df = class_dist.reset_index()
    class_dist_df.columns = ["diagnosis", "count"]
    class_dist_df["label"] = class_dist_df["diagnosis"].map(label_names)
    class_dist_df.to_csv("outputs/reports/class_distribution.csv", index=False)

    # Check class imbalance (max/min ratio)
    max_cls, min_cls = class_dist.max(), class_dist.min()
    imbalance_ratio = max_cls / min_cls if min_cls > 0 else float("inf")
    if imbalance_ratio > 10:
        print(f"  ⚠ Severe class imbalance: ratio = {imbalance_ratio:.1f}x")
    else:
        print(f"  ✓ Class imbalance ratio: {imbalance_ratio:.1f}x")

    # ── Per-image file validation ───────────────────────────────────────────
    image_paths = []
    missing_files = []
    for img_id in df["id_code"]:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                image_paths.append(p)
                found = True
                break
        if not found:
            missing_files.append(img_id)

    print(f"\n[3/5] File existence check...")
    print(f"  Found: {len(image_paths)} / {len(df)} | Missing: {len(missing_files)}")
    if missing_files:
        print(f"  ⚠ First 5 missing: {missing_files[:5]}")

    if not args.quick and len(image_paths) > 0:
        print(f"  Checking readability of {min(len(image_paths), 500)} images...")
        corrupted = []
        check_n = min(len(image_paths), 500)
        for p in image_paths[:check_n]:
            result = check_image_file(p)
            if result["issues"]:
                corrupted.append(result)
        print(f"  Corrupted/unreadable: {len(corrupted)}")
    else:
        corrupted = []
        if args.quick:
            print("  Skipped (--quick mode)")

    # ── Duplicate detection ─────────────────────────────────────────────────
    print(f"\n[4/5] Duplicate detection (SHA-256 + pHash)...")
    dup_report = {"exact_duplicates": [], "near_duplicates": [],
                  "note": "Skipped — no image files found" if not image_paths else ""}

    if image_paths:
        # Limit to 2000 for speed
        check_paths = image_paths[:2000]
        detector = DuplicateDetector(check_paths)
        dup_report = detector.find_duplicates(report_path="outputs/reports/duplicate_report.json")
        print(f"  Exact duplicates: {len(dup_report['exact_duplicates'])}")
        print(f"  Near duplicates (pHash):  {len(dup_report['near_duplicates'])}")
    else:
        with open("outputs/reports/duplicate_report.json", "w") as f:
            json.dump(dup_report, f, indent=4)

    # ── Splits + leakage check ────────────────────────────────────────────
    print(f"\n[5/5] Creating splits and checking leakage...")
    # Remove exact duplicates first
    if dup_report["exact_duplicates"]:
        dup_ids = {os.path.basename(d["duplicate"]).split(".")[0]
                   for d in dup_report["exact_duplicates"]}
        df = df[~df["id_code"].isin(dup_ids)]
        print(f"  Removed {len(dup_ids)} exact duplicates before splitting.")

    train_df, val_df, test_df = create_stratified_splits(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    leak_detector = LeakageDetector(df)
    leak_report = leak_detector.check_splits(
        train_df, val_df, test_df,
        report_path="outputs/reports/leakage_report.json"
    )
    if leak_report["image_leakage"]:
        print(f"  ⚠ Leakage: {len(leak_report['image_leakage'])} images in multiple splits!")
    else:
        print("  ✓ No data leakage detected")

    # ── Final report ──────────────────────────────────────────────────────
    dataset_report = {
        "dataset": config["dataset"]["name"],
        "total_records": len(df),
        "image_dir": image_dir,
        "files_found": len(image_paths),
        "files_missing": len(missing_files),
        "corrupted_images": len(corrupted),
        "exact_duplicates": len(dup_report["exact_duplicates"]),
        "near_duplicates": len(dup_report["near_duplicates"]),
        "class_distribution": class_dist.to_dict(),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "label_issues": label_issues,
        "leakage_detected": bool(leak_report["image_leakage"]),
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }

    with open("outputs/reports/dataset_report.json", "w") as f:
        json.dump(dataset_report, f, indent=4)

    summary_df = pd.DataFrame([{
        k: v for k, v in dataset_report.items()
        if not isinstance(v, (dict, list))
    }])
    summary_df.to_csv("outputs/reports/dataset_summary.csv", index=False)

    print("\n═══════════════════════════════════════════════")
    print("OUTPUTS:")
    print("  outputs/reports/dataset_report.json")
    print("  outputs/reports/duplicate_report.json")
    print("  outputs/reports/leakage_report.json")
    print("  outputs/reports/class_distribution.csv")
    print("  outputs/reports/dataset_summary.csv")
    print("═══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
