import yaml
import os
import json
import argparse
import pandas as pd
from src.preprocessing.deduplication import DuplicateDetector
from src.data.leakage import LeakageDetector
from src.data.split import create_stratified_splits


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config["dataset"]["name"]
    dataset_root = config["dataset"]["root"]

    print(f"Preparing dataset: {dataset_name} at {dataset_root}")

    csv_path = os.path.join(dataset_root, "train.csv")
    image_dir = os.path.join(dataset_root, "train_images")

    if not os.path.exists(csv_path) or not os.path.exists(image_dir):
        print(f"Dataset not found at {dataset_root}. Expected structure:")
        print(f"  {dataset_root}/train.csv")
        print(f"  {dataset_root}/train_images/")
        print("Please place the dataset and run again.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    os.makedirs("outputs/reports", exist_ok=True)

    # ── Class distribution ─────────────────────────────────────────────────────
    class_dist = df["diagnosis"].value_counts().sort_index()
    class_dist_df = class_dist.reset_index()
    class_dist_df.columns = ["diagnosis", "count"]
    class_dist_df["label"] = class_dist_df["diagnosis"].map({
        0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"
    })
    class_dist_df.to_csv("outputs/reports/class_distribution.csv", index=False)
    print("Saved class_distribution.csv")
    print(class_dist_df.to_string(index=False))

    # ── Duplicate detection ────────────────────────────────────────────────────
    print("\nRunning duplicate detection (SHA-256 + perceptual hash)...")
    image_paths = []
    for img_id in df["id_code"]:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                image_paths.append(p)
                break

    detector = DuplicateDetector(image_paths)
    dup_report = detector.find_duplicates(report_path="outputs/reports/duplicate_report.json")
    print(f"Exact duplicates: {len(dup_report['exact_duplicates'])}")
    print(f"Near duplicates: {len(dup_report['near_duplicates'])}")

    # Remove exact duplicates from df
    if len(dup_report["exact_duplicates"]) > 0:
        dup_ids = set()
        for d in dup_report["exact_duplicates"]:
            dup_id = os.path.basename(d["duplicate"]).split(".")[0]
            dup_ids.add(dup_id)
        original_len = len(df)
        df = df[~df["id_code"].isin(dup_ids)]
        print(f"Removed {original_len - len(df)} exact duplicates from dataset.")

    # ── Dataset summary ────────────────────────────────────────────────────────
    summary = {
        "dataset": dataset_name,
        "total_images": len(df),
        "class_distribution": class_dist.to_dict(),
        "exact_duplicates_removed": len(dup_report["exact_duplicates"]),
        "near_duplicates_flagged": len(dup_report["near_duplicates"]),
    }

    # ── Stratified split ───────────────────────────────────────────────────────
    print("\nSplitting dataset (70/15/15 stratified)...")
    train_df, val_df, test_df = create_stratified_splits(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    summary["split"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

    # ── Leakage check ──────────────────────────────────────────────────────────
    print("\nChecking for data leakage...")
    leak_detector = LeakageDetector(df)
    leak_report = leak_detector.check_splits(train_df, val_df, test_df,
                                              report_path="outputs/reports/leakage_report.json")
    if leak_report["image_leakage"]:
        print(f"WARNING: Image leakage detected: {len(leak_report['image_leakage'])} images")
        summary["leakage_detected"] = True
    else:
        print("No data leakage detected.")
        summary["leakage_detected"] = False

    # ── Dataset report ─────────────────────────────────────────────────────────
    with open("outputs/reports/dataset_report.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved dataset_report.json")

    # ── Summary CSV ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame([{
        "dataset": summary["dataset"],
        "total_images": summary["total_images"],
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "exact_dups_removed": summary["exact_duplicates_removed"],
        "near_dups_flagged": summary["near_duplicates_flagged"],
        "leakage_detected": summary["leakage_detected"]
    }])
    summary_df.to_csv("outputs/reports/dataset_summary.csv", index=False)
    print("Saved dataset_summary.csv")
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
