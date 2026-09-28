import time
import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from api.database import SessionLocal, Analysis
from src.quality.assessment import QualityAssessor
from src.preprocessing.enhancement import ImageEnhancer
from src.segmentation.lesion_analysis import LesionAnalyzer
from src.classification.model import get_model
from src.preprocessing.transforms import get_transforms
from src.explainability.gradcam import XAI_Visualizer
from src.reporting.pdf_generator import ReportGenerator

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_CHECKPOINT", "models/classification/best_model.pth")
NUM_CLASSES = 5

DR_LABELS = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

# ── Global model instances (loaded once at startup) ──────────────────────────
_classifier = None
_xai_visualizer = None
_device = None

# ── Ensure output directories exist ─────────────────────────────────────────
for d in ["outputs/gradcam", "outputs/annotated", "outputs/reports"]:
    os.makedirs(d, exist_ok=True)


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_classifier():
    global _classifier
    if _classifier is None:
        device = get_device()
        try:
            model = get_model(num_classes=NUM_CLASSES, model_name="efficientnet_b0")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()
            _classifier = model
            print(f"[INFO] Model loaded from {MODEL_PATH} on {device}")
        except FileNotFoundError:
            print(f"[WARNING] Model not found at {MODEL_PATH}. Running in demo mode.")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}. Running in demo mode.")
    return _classifier


def get_xai_visualizer():
    global _xai_visualizer
    model = get_classifier()
    if _xai_visualizer is None and model is not None:
        try:
            target_layer = model.model.features[-1]
            device = get_device()
            _xai_visualizer = XAI_Visualizer(model=model, target_layer=target_layer, device=str(device))
            print("[INFO] GradCAM visualizer initialized.")
        except Exception as e:
            print(f"[WARNING] GradCAM init failed: {e}")
    return _xai_visualizer


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="DR-Screening-XAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ── Database dependency ──────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic models ──────────────────────────────────────────────────────────
class ReviewUpdate(BaseModel):
    analysis_id: str
    decision: str
    comment: str


# ── Startup: pre-load model ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[INFO] Starting DR-Screening-XAI API...")
    get_classifier()   # Pre-warm model
    get_xai_visualizer()


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH) and _classifier is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(get_device()),
        "demo_mode": not model_loaded
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Validate & store image, return image_id for subsequent /api/analyze call."""
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    image_id = f"img_{int(time.time() * 1000)}"
    upload_dir = "outputs/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f"{image_id}.png")
    cv2.imwrite(save_path, img)

    return {"image_id": image_id, "upload_path": save_path, "status": "uploaded"}


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), db=Depends(get_db)):
    """Full end-to-end analysis: quality → enhance → lesions → classify → GradCAM → PDF."""
    start_time = time.time()

    # ── Read & validate image ────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image."})

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_id = f"img_{int(time.time() * 1000)}"

    # Save original for report
    original_path = f"outputs/originals/{image_id}.png"
    os.makedirs("outputs/originals", exist_ok=True)
    cv2.imwrite(original_path, img_cv)

    # ── Image Quality Assessment ─────────────────────────────────────────────
    assessor = QualityAssessor()
    quality = assessor.assess(img_rgb)

    if quality["recapture_required"]:
        return {
            "image_id": image_id,
            "error": "Image quality insufficient for reliable screening. Please recapture the fundus image.",
            "quality": quality
        }

    # ── Enhancement (always applied for consistency) ─────────────────────────
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(img_rgb)

    # ── Lesion Analysis (OpenCV heuristic) ───────────────────────────────────
    analyzer = LesionAnalyzer()
    lesions = analyzer.analyze(enhanced)

    # Generate lesion overlay image
    lesion_overlay_path = f"outputs/annotated/{image_id}_lesions.png"
    analyzer.generate_lesion_overlay(enhanced, lesions, output_path=lesion_overlay_path)
    lesion_overlay_url = f"/outputs/annotated/{image_id}_lesions.png"

    # ── DR Classification ────────────────────────────────────────────────────
    model = get_classifier()
    device = get_device()

    if model is not None:
        val_transform = get_transforms(image_size=512, mode="val")
        transformed = val_transform(image=enhanced)
        tensor_img = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_grade = int(pred.item())
            confidence = float(conf.item())
            all_probs = probs[0].tolist()
    else:
        # Demo mode — clearly labelled as such
        pred_grade = 2
        confidence = 0.75
        all_probs = [0.05, 0.10, 0.75, 0.07, 0.03]
        print("[INFO] Demo mode: returning placeholder prediction.")

    conf_tier = "HIGH" if confidence >= 0.8 else ("MODERATE" if confidence >= 0.6 else "LOW")

    # ── GradCAM Explainability ───────────────────────────────────────────────
    gradcam_url = None
    overlay_url = None

    if model is not None:
        xai = get_xai_visualizer()
        if xai is not None:
            try:
                img_resized = cv2.resize(enhanced, (512, 512))
                val_transform = get_transforms(image_size=512, mode="val")
                tensor_for_cam = val_transform(image=enhanced)["image"].unsqueeze(0).to(device)

                gradcam_path = f"outputs/gradcam/{image_id}.png"
                _, visualization = xai.generate_heatmap(
                    input_tensor=tensor_for_cam,
                    original_image=img_resized,
                    target_category=None,
                    output_path=gradcam_path
                )
                if visualization is not None:
                    gradcam_url = f"/outputs/gradcam/{image_id}.png"
                    overlay_url = gradcam_url
                    print(f"[INFO] GradCAM saved: {gradcam_path}")
            except Exception as e:
                print(f"[WARNING] GradCAM failed: {e}")

    explainability = {
        "gradcam_url": gradcam_url,
        "overlay_url": overlay_url,
        "lesion_overlay_url": lesion_overlay_url,
        "confidence_tier": conf_tier,
        "class_probabilities": {DR_LABELS[i]: round(p, 4) for i, p in enumerate(all_probs)},
        "demo_mode": model is None,
    }

    processing_time = (time.time() - start_time) * 1000

    # ── Persist to DB ────────────────────────────────────────────────────────
    try:
        record = Analysis(
            image_id=image_id,
            quality_score=quality["quality_score"],
            quality_status=quality["status"],
            dr_grade=pred_grade,
            dr_label=DR_LABELS[pred_grade],
            confidence=confidence,
            referable=(pred_grade >= 2),
            processing_time_ms=processing_time
        )
        db.add(record)
        db.commit()
    except Exception as e:
        print(f"[WARNING] DB write failed: {e}")

    result = {
        "image_id": image_id,
        "quality": quality,
        "prediction": {
            "grade": pred_grade,
            "label": DR_LABELS[pred_grade],
            "confidence": confidence,
            "confidence_tier": conf_tier,
            "referable": (pred_grade >= 2),
            "demo_mode": model is None,
        },
        "lesions": lesions,
        "explainability": explainability,
        "processing_time_ms": round(processing_time, 1),
    }

    # ── Generate PDF Report ──────────────────────────────────────────────────
    try:
        reporter = ReportGenerator()
        gradcam_file = gradcam_path if gradcam_url else None
        pdf_path = reporter.generate(
            result_json=result,
            image_path=original_path,
            gradcam_path=gradcam_file,
            overlay_path=lesion_overlay_path if os.path.exists(lesion_overlay_path) else None
        )
        result["report_url"] = f"/outputs/reports/{os.path.basename(pdf_path)}"
    except Exception as e:
        print(f"[WARNING] PDF generation failed: {e}")
        result["report_url"] = None

    return result


@app.get("/api/result/{image_id}")
def get_result(image_id: str, db=Depends(get_db)):
    """Retrieve a previously stored analysis result from the database."""
    record = db.query(Analysis).filter(Analysis.image_id == image_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail=f"No result found for image_id: {image_id}")

    return {
        "image_id": record.image_id,
        "quality_score": record.quality_score,
        "quality_status": record.quality_status,
        "prediction": {
            "grade": record.dr_grade,
            "label": record.dr_label,
            "confidence": record.confidence,
            "referable": record.referable,
        },
        "processing_time_ms": record.processing_time_ms,
        "report_url": f"/outputs/reports/report_{image_id}.pdf"
        if os.path.exists(f"outputs/reports/report_{image_id}.pdf") else None,
    }


@app.get("/api/report/{image_id}")
def get_report(image_id: str):
    """Serve the PDF report for a given image_id."""
    pdf_path = f"outputs/reports/report_{image_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found. Run analysis first.")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"DR_Report_{image_id}.pdf"
    )


@app.post("/api/quality-check")
async def quality_check(file: UploadFile = File(...)):
    """Standalone quality assessment without full analysis."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    assessor = QualityAssessor()
    quality = assessor.assess(img_rgb)
    return quality


@app.post("/api/review")
def submit_review(review: ReviewUpdate, db=Depends(get_db)):
    """Submit a clinician review decision for a given analysis."""
    from api.database import Review
    new_review = Review(
        analysis_id=review.analysis_id,
        decision=review.decision,
        comment=review.comment
    )
    db.add(new_review)
    db.commit()
    return {"status": "Review saved", "analysis_id": review.analysis_id}
