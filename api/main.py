import time
import os
import io
import warnings
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

import api.database as _db_module
from api.database import verify_connection
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
    """Yield db, attempting a lazy reconnect if the initial connection failed."""
    # Use the live module-level reference (may be updated by verify_connection)
    current_db = _db_module.db
    if current_db is None:
        # Attempt a reconnect once
        verify_connection()
        current_db = _db_module.db
    if current_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Check MongoDB Atlas IP whitelist (add 0.0.0.0/0) and credentials."
        )
    yield current_db


# ── Pydantic models ──────────────────────────────────────────────────────────
class ReviewUpdate(BaseModel):
    analysis_id: str
    decision: str
    comment: str

class PatientCreate(BaseModel):
    name: str
    phone: str

# ── Startup: pre-load model ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("  DR-Screening-XAI — Starting up...")
    print("=" * 60)

    # MongoDB connection
    print("[DB]    Connecting to MongoDB Atlas...")
    if verify_connection():
        print("[DB]    ✅ MongoDB Atlas connected successfully.")
    else:
        print("[DB]    ⚠️  MongoDB connection failed. Retries may occur on first request.")

    # Model loading
    print("[MODEL] Loading EfficientNet-B0 classifier...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress torchvision pretrained deprecation
        get_classifier()
        get_xai_visualizer()

    model_loaded = _classifier is not None
    if model_loaded:
        print(f"[MODEL] ✅ EfficientNet-B0 loaded on {get_device()}.")
    else:
        print(f"[MODEL] ⚠️  Model not found at '{MODEL_PATH}'. Running in DEMO mode.")

    print("[API]   ✅ All endpoints ready.")
    print("=" * 60 + "\n")


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


@app.post("/api/register")
def register_patient(patient: PatientCreate):
    """Register a patient profile. Works with or without DB — returns a local ID if DB is down."""
    patient_id = f"pat_{int(time.time() * 1000)}"
    current_db = _db_module.db
    if current_db is None:
        verify_connection()
        current_db = _db_module.db
    if current_db is not None:
        try:
            record = {
                "patient_id": patient_id,
                "name": patient.name,
                "phone": patient.phone,
                "created_at": time.time()
            }
            current_db.patients.insert_one(record)
        except Exception as e:
            print(f"[WARNING] DB write failed for register: {e}")
            # Return success anyway — patient_id is still usable for the session
    return {"patient_id": patient_id, "name": patient.name, "phone": patient.phone, "db_saved": current_db is not None}

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patientId: str = Form("")):
    """Full end-to-end analysis: quality → enhance → lesions → classify → GradCAM → PDF."""
    # Resolve db reference lazily (survive startup DB failure)
    current_db = _db_module.db
    if current_db is None:
        verify_connection()
        current_db = _db_module.db
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
            "error": quality.get("error", "Image quality insufficient for reliable screening. Please recapture the fundus image."),
            "quality": quality
        }

    # ── Enhancement (always applied for consistency) ─────────────────────────
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(img_rgb)

    # ── Lesion Analysis (OpenCV heuristic) ───────────────────────────────────
    analyzer = LesionAnalyzer()
    lesions = analyzer.analyze(enhanced)  # detect on enhanced (better contrast)

    # Generate lesion overlay on the ORIGINAL image (not enhanced) so it's not dark
    lesion_overlay_path = f"outputs/annotated/{image_id}_lesions.png"
    analyzer.generate_lesion_overlay(img_rgb, lesions, output_path=lesion_overlay_path)
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
    gradcam_path = None  # initialize so PDF generator always has this variable

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

    # ── Fetch Patient Details (if provided) ──────────────────────────────────
    pat_name = ""
    pat_phone = ""
    if patientId and current_db is not None:
        pat_record = current_db.patients.find_one({"patient_id": patientId})
        if pat_record:
            pat_name = pat_record.get("name", "")
            pat_phone = pat_record.get("phone", "")

    # ── Persist to DB ────────────────────────────────────────────────────────
    if current_db is not None:
        try:
            record = {
                "image_id": image_id,
                "patient_id": patientId,
                "patient_name": pat_name,
                "phone_number": pat_phone,
                "quality_score": quality["quality_score"],
                "quality_status": quality["status"],
                "dr_grade": pred_grade,
                "dr_label": DR_LABELS[pred_grade],
                "confidence": confidence,
                "referable": (pred_grade >= 2),
                "processing_time_ms": processing_time
            }
            current_db.analyses.insert_one(record)
        except Exception as e:
            print(f"[WARNING] DB write failed: {e}")
    else:
        print("[INFO] DB unavailable — skipping analysis persistence.")

    result = {
        "image_id": image_id,
        "patient_name": pat_name,
        "phone_number": pat_phone,
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
        pdf_path = reporter.generate(
            result_json=result,
            image_path=original_path,
            gradcam_path=gradcam_path,
            overlay_path=lesion_overlay_path if os.path.exists(lesion_overlay_path) else None,
            patient_name=pat_name,
            phone_number=pat_phone
        )
        result["report_url"] = f"/outputs/reports/{os.path.basename(pdf_path)}"
    except Exception as e:
        print(f"[WARNING] PDF generation failed: {e}")
        result["report_url"] = None

    return result


@app.get("/api/result/{image_id}")
def get_result(image_id: str, db=Depends(get_db)):
    """Retrieve a previously stored analysis result from the database."""
    record = db.analyses.find_one({"image_id": image_id})
    if record is None:
        raise HTTPException(status_code=404, detail=f"No result found for image_id: {image_id}")

    return {
        "image_id": record["image_id"],
        "patient_name": record.get("patient_name", ""),
        "phone_number": record.get("phone_number", ""),
        "quality_score": record["quality_score"],
        "quality_status": record["quality_status"],
        "prediction": {
            "grade": record["dr_grade"],
            "label": record["dr_label"],
            "confidence": record["confidence"],
            "referable": record["referable"],
        },
        "processing_time_ms": record["processing_time_ms"],
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
    new_review = {
        "analysis_id": review.analysis_id,
        "decision": review.decision,
        "comment": review.comment
    }
    db.reviews.insert_one(new_review)
    return {"status": "Review saved", "analysis_id": review.analysis_id}
