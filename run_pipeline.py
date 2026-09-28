"""
run_pipeline.py — Master entry point for DR-Screening-XAI
==========================================
Usage:
    python run_pipeline.py --image path/to/fundus_image.jpg
    python run_pipeline.py --image path/to/fundus_image.jpg --output-dir outputs/pipeline_run

Executes the complete end-to-end pipeline:
  1. Validate image
  2. Quality assessment
  3. Enhancement
  4. Lesion analysis (OpenCV heuristic)
  5. DR classification (EfficientNet-B0)
  6. Confidence calibration
  7. Grad-CAM explainability
  8. PDF report generation
  9. Print JSON summary
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.quality.assessment import QualityAssessor
from src.preprocessing.enhancement import ImageEnhancer
from src.preprocessing.transforms import get_transforms
from src.segmentation.lesion_analysis import LesionAnalyzer
from src.classification.model import get_model
from src.explainability.gradcam import XAI_Visualizer
from src.reporting.pdf_generator import ReportGenerator

DR_LABELS = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

MODEL_PATH = os.environ.get("MODEL_CHECKPOINT", "models/classification/best_model.pth")


def validate_image(image_path: str):
    """Check file exists, is a valid image, and within size limits."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = Path(image_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(f"Unsupported format: {ext}. Use JPG or PNG.")

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > 50:
        raise ValueError(f"Image too large: {file_size_mb:.1f} MB (max 50 MB)")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image (corrupted or unsupported): {image_path}")

    return img


def load_model(device):
    """Load classification model if checkpoint exists, else return None (demo mode)."""
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}. Running in DEMO MODE.")
        return None

    try:
        model = get_model(num_classes=5, model_name="efficientnet_b0")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[WARNING] Failed to load model: {e}. Running in DEMO MODE.")
        return None


def run_pipeline(image_path: str, output_dir: str = "outputs/pipeline_run") -> dict:
    """
    Execute the complete DR screening pipeline on a single image.
    Returns a JSON-serialisable result dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    pipeline_start = time.time()

    timings = {}

    # ── Step 1: Validate ─────────────────────────────────────────────────────
    print("\n[1/9] Validating image...")
    t0 = time.time()
    img_cv = validate_image(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    timings["validation_ms"] = round((time.time() - t0) * 1000, 1)
    print(f"      Image shape: {img_rgb.shape} | File: {os.path.basename(image_path)}")

    # ── Step 2: Quality Assessment ───────────────────────────────────────────
    print("[2/9] Running image quality assessment...")
    t0 = time.time()
    assessor = QualityAssessor()
    quality = assessor.assess(img_rgb)
    timings["quality_ms"] = round((time.time() - t0) * 1000, 1)
    print(f"      Quality: {quality['status']} (score={quality['quality_score']:.3f})")

    if quality["recapture_required"]:
        result = {
            "status": "rejected",
            "reason": "Image quality insufficient. Please recapture the fundus image.",
            "quality": quality,
        }
        print(f"\n[REJECTED] {result['reason']}")
        print(json.dumps(result, indent=2))
        return result

    # ── Step 3: Enhancement ──────────────────────────────────────────────────
    print("[3/9] Enhancing image (CLAHE + illumination normalisation)...")
    t0 = time.time()
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(img_rgb)
    timings["enhancement_ms"] = round((time.time() - t0) * 1000, 1)

    # ── Step 4: Lesion Analysis ──────────────────────────────────────────────
    print("[4/9] Running lesion analysis (OpenCV heuristic pipeline)...")
    t0 = time.time()
    analyzer = LesionAnalyzer()
    lesions = analyzer.analyze(enhanced)
    lesion_overlay_path = os.path.join(output_dir, f"{image_name}_lesion_overlay.png")
    analyzer.generate_lesion_overlay(enhanced, lesions, output_path=lesion_overlay_path)
    timings["lesion_ms"] = round((time.time() - t0) * 1000, 1)
    ma = lesions["microaneurysms"]["count"]
    hem = lesions["hemorrhages"]["count"]
    exu = lesions["exudates"]["count"]
    print(f"      Microaneurysms: {ma} | Hemorrhages: {hem} | Exudates: {exu}")

    # ── Step 5: Classification ────────────────────────────────────────────────
    print("[5/9] Running DR classification...")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    demo_mode = model is None

    if model is not None:
        val_transform = get_transforms(image_size=512, mode="val")
        tensor_img = val_transform(image=enhanced)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_grade = int(pred.item())
            confidence = float(conf.item())
            all_probs = probs[0].tolist()
    else:
        pred_grade, confidence = 2, 0.75
        all_probs = [0.05, 0.10, 0.75, 0.07, 0.03]

    timings["classification_ms"] = round((time.time() - t0) * 1000, 1)
    conf_tier = "HIGH" if confidence >= 0.8 else ("MODERATE" if confidence >= 0.6 else "LOW")
    print(f"      Grade: {pred_grade} ({DR_LABELS[pred_grade]}) | Confidence: {confidence:.3f} ({conf_tier})")

    # ── Step 6: Confidence Calibration ───────────────────────────────────────
    print("[6/9] Confidence calibration (temperature scaling — T=1.0, not yet tuned)...")
    # Temperature scaling requires a validation set to tune.
    # Default T=1.0 (no change). See src/explainability/calibration.py for full impl.

    # ── Step 7: GradCAM ──────────────────────────────────────────────────────
    print("[7/9] Generating Grad-CAM explainability map...")
    t0 = time.time()
    gradcam_path = None
    if model is not None:
        try:
            target_layer = model.model.features[-1]
            xai = XAI_Visualizer(model=model, target_layer=target_layer, device=str(device))
            img_resized = cv2.resize(enhanced, (512, 512))
            val_transform = get_transforms(image_size=512, mode="val")
            tensor_cam = val_transform(image=enhanced)["image"].unsqueeze(0).to(device)
            gradcam_out = os.path.join(output_dir, f"{image_name}_gradcam.png")
            _, viz = xai.generate_heatmap(tensor_cam, img_resized, output_path=gradcam_out)
            if viz is not None:
                gradcam_path = gradcam_out
                print(f"      Saved: {gradcam_out}")
        except Exception as e:
            print(f"      [WARNING] GradCAM failed: {e}")
    else:
        print("      Skipped (demo mode — no model loaded)")
    timings["gradcam_ms"] = round((time.time() - t0) * 1000, 1)

    # ── Step 8: PDF Report ───────────────────────────────────────────────────
    print("[8/9] Generating PDF report...")
    t0 = time.time()
    image_id = f"{image_name}_{int(time.time())}"
    total_ms = (time.time() - pipeline_start) * 1000

    result_json = {
        "image_id": image_id,
        "quality": quality,
        "prediction": {
            "grade": pred_grade,
            "label": DR_LABELS[pred_grade],
            "confidence": confidence,
            "confidence_tier": conf_tier,
            "referable": pred_grade >= 2,
            "demo_mode": demo_mode,
        },
        "lesions": lesions,
        "explainability": {
            "gradcam_url": gradcam_path,
            "confidence_tier": conf_tier,
            "class_probabilities": {DR_LABELS[i]: round(p, 4) for i, p in enumerate(all_probs)},
        },
        "processing_time_ms": round(total_ms, 1),
    }

    reporter = ReportGenerator(output_dir=output_dir)
    pdf_path = reporter.generate(
        result_json=result_json,
        image_path=image_path,
        gradcam_path=gradcam_path,
        overlay_path=lesion_overlay_path if os.path.exists(lesion_overlay_path) else None,
    )
    timings["report_ms"] = round((time.time() - t0) * 1000, 1)
    print(f"      Saved: {pdf_path}")

    # ── Step 9: Save outputs & summary ──────────────────────────────────────
    print("[9/9] Saving outputs and summary...")
    summary = {
        "status": "success",
        "image": os.path.basename(image_path),
        "quality": quality["status"],
        "quality_score": round(quality["quality_score"], 4),
        "dr_grade": pred_grade,
        "dr_label": DR_LABELS[pred_grade],
        "confidence": round(confidence, 4),
        "confidence_tier": conf_tier,
        "referable": pred_grade >= 2,
        "demo_mode": demo_mode,
        "lesions": {
            "microaneurysms": ma,
            "hemorrhages": hem,
            "exudates": exu,
        },
        "outputs": {
            "report": pdf_path,
            "gradcam": gradcam_path,
            "lesion_overlay": lesion_overlay_path,
        },
        "timings_ms": timings,
        "total_pipeline_ms": round(total_ms, 1),
    }

    summary_path = os.path.join(output_dir, f"{image_name}_result.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="DR-Screening-XAI Master Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image (JPG/PNG)")
    parser.add_argument("--output-dir", type=str, default="outputs/pipeline_run",
                        help="Directory to save all outputs")
    args = parser.parse_args()

    print("══════════════════════════════════════════════")
    print("  DR-Screening-XAI — End-to-End Pipeline")
    print("══════════════════════════════════════════════")

    result = run_pipeline(args.image, args.output_dir)

    print("\n══════════════════════════════════════════════")
    print("  PIPELINE RESULT")
    print("══════════════════════════════════════════════")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
