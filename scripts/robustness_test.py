"""
scripts/robustness_test.py — Evaluate model robustness under image degradation
============================================================
Usage:
    python scripts/robustness_test.py --config configs/config.yaml --image data/sample.jpg
    python scripts/robustness_test.py --config configs/config.yaml --image-dir data/raw/aptos/train_images --n 50

Generates:
    outputs/reports/robustness_report.csv
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.quality.assessment import QualityAssessor
from src.preprocessing.transforms import get_transforms
from src.classification.model import get_model

MODEL_PATH = os.environ.get("MODEL_CHECKPOINT", "models/classification/best_model.pth")


# ── Degradation functions ────────────────────────────────────────────────────

def apply_blur(img, sigma=3):
    """Simulate out-of-focus camera."""
    k = int(sigma * 3) | 1  # ensure odd
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_gaussian_noise(img, std=25):
    """Simulate sensor noise."""
    noise = np.random.randn(*img.shape) * std
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_low_illumination(img, factor=0.4):
    """Simulate poor lighting conditions."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_high_illumination(img, factor=1.8):
    """Simulate overexposure."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_reduced_resolution(img, scale=0.25):
    """Simulate low-resolution portable camera."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * scale), 32), max(int(h * scale), 32)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(img, quality=20):
    """Simulate heavy JPEG compression artifact."""
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_contrast_change(img, alpha=0.5, beta=30):
    """Simulate poor contrast."""
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "original": lambda img: img,
    "blur_mild": lambda img: apply_blur(img, sigma=2),
    "blur_heavy": lambda img: apply_blur(img, sigma=5),
    "noise_mild": lambda img: apply_gaussian_noise(img, std=15),
    "noise_heavy": lambda img: apply_gaussian_noise(img, std=40),
    "low_illumination": lambda img: apply_low_illumination(img, factor=0.35),
    "high_illumination": lambda img: apply_high_illumination(img, factor=1.9),
    "low_resolution": lambda img: apply_reduced_resolution(img, scale=0.25),
    "jpeg_compression": lambda img: apply_jpeg_compression(img, quality=15),
    "low_contrast": lambda img: apply_contrast_change(img, alpha=0.5, beta=20),
}


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = get_model(num_classes=5, model_name="efficientnet_b0")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[WARNING] Model load failed: {e}")
        return None


def predict(model, img_rgb, transform, device):
    t0 = time.time()
    transformed = transform(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    latency = (time.time() - t0) * 1000
    return int(pred.item()), float(conf.item()), latency


def get_image_paths(args):
    if args.image:
        return [args.image]
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png"}
        paths = [str(p) for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts]
        random.shuffle(paths)
        return paths[:args.n]
    return []


def main():
    parser = argparse.ArgumentParser(description="Robustness testing for DR classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image", type=str, default=None, help="Single image to test")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--n", type=int, default=50, help="Max images (when using --image-dir)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if model is None:
        print("[ERROR] Model not found. Train the model before running robustness tests.")
        print(f"  Expected at: {MODEL_PATH}")
        return

    transform = get_transforms(image_size=512, mode="val")
    assessor = QualityAssessor()

    image_paths = get_image_paths(args)
    if not image_paths:
        print("[ERROR] No images provided. Use --image or --image-dir.")
        return

    print(f"Testing robustness on {len(image_paths)} image(s) with {len(DEGRADATIONS)} conditions...")
    results = []

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_name = os.path.basename(img_path)

        # Get original prediction
        orig_grade, orig_conf, _ = predict(model, img_rgb, transform, device)

        for deg_name, deg_fn in DEGRADATIONS.items():
            try:
                degraded = deg_fn(img_rgb.copy())
                quality = assessor.assess(degraded)
                grade, conf, latency = predict(model, degraded, transform, device)

                results.append({
                    "image": img_name,
                    "degradation": deg_name,
                    "original_grade": orig_grade,
                    "predicted_grade": grade,
                    "grade_changed": grade != orig_grade,
                    "confidence": round(conf, 4),
                    "quality_score": round(quality["quality_score"], 4),
                    "quality_status": quality["status"],
                    "latency_ms": round(latency, 1),
                })
            except Exception as e:
                print(f"  [WARN] {deg_name} failed for {img_name}: {e}")

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    os.makedirs("outputs/reports", exist_ok=True)
    df.to_csv("outputs/reports/robustness_report.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════")
    print("        ROBUSTNESS SUMMARY")
    print("═══════════════════════════════════════════════")
    summary = df.groupby("degradation").agg(
        grade_change_rate=("grade_changed", "mean"),
        avg_confidence=("confidence", "mean"),
        avg_quality=("quality_score", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
    ).round(4)
    print(summary.to_string())
    print("═══════════════════════════════════════════════")
    print("Saved: outputs/reports/robustness_report.csv")


if __name__ == "__main__":
    main()
