# DR-Screening-XAI

**Explainable and Clinically Validated AI-Based Diabetic Retinopathy Screening System for Rural Telemedicine**

---

## Project Overview

An end-to-end retinal image analysis system for diabetic retinopathy (DR) screening in resource-limited telemedicine settings. The system accepts a fundus photograph and outputs an explainable DR severity grading, lesion evidence map, Grad-CAM heatmap, and a PDF clinical report.

> ⚠️ This system is a **screening research prototype**. It is NOT an approved medical device and does NOT replace ophthalmologist assessment.

---

## System Architecture

```
Fundus Image
    ↓
Image Quality Assessment (OpenCV)
    ↓
Enhancement (CLAHE + Illumination Normalization)
    ↓
Lesion Analysis (OpenCV morphological heuristics)
    ↓
DR Classification (EfficientNet-B0 / PyTorch)
    ↓
Grad-CAM Explainability
    ↓
Confidence Calibration (Temperature Scaling)
    ↓
PDF Report + Lesion Overlay
    ↓
FastAPI Backend → React Frontend
    ↓
Clinician Review Interface
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Classification | PyTorch, EfficientNet-B0 |
| Image Processing | OpenCV, NumPy, Pillow |
| Transforms | Albumentations |
| Explainability | pytorch-grad-cam (Grad-CAM) |
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | React, Vite |
| Reports | ReportLab |
| MATLAB | Image Processing, Computer Vision Toolbox |
| Simulation | MATLAB M/M/c queuing model (Erlang-C) |
| Dataset | APTOS 2019 Blindness Detection |

---

## DR Severity Scale

| Grade | Label | Referable |
|---|---|---|
| 0 | No apparent DR | No |
| 1 | Mild NPDR | No |
| 2 | Moderate NPDR | **Yes** |
| 3 | Severe NPDR | **Yes** |
| 4 | Proliferative DR | **Yes** |

Referable threshold: Grade ≥ 2.

---

## Installation

```powershell
cd d:\e-commerce\dr_screening_xai
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python albumentations --no-deps
pip install numpy pandas Pillow scipy scikit-learn torch torchvision matplotlib seaborn tqdm PyYAML fastapi uvicorn pydantic SQLAlchemy reportlab grad-cam captum mlflow python-dotenv pytest
```

---

## Running the Project

### 1. Start the Backend API

```powershell
.venv\Scripts\activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 2. Start the Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

### 3. Run Single Image Through Pipeline

```powershell
python run_pipeline.py --image data/sample.jpg
```

### 4. Prepare Dataset (requires APTOS data)

```powershell
python scripts/prepare_dataset.py --config configs/config.yaml
```

### 5. Train the Model

```powershell
python scripts/train_classifier.py --config configs/config.yaml
```

### 6. Evaluate the Model

```powershell
python scripts/evaluate.py --config configs/config.yaml
```

### 7. Validate Dataset

```powershell
python scripts/validate_dataset.py --config configs/config.yaml
```

### 8. Robustness Testing

```powershell
python scripts/robustness_test.py --config configs/config.yaml --image data/sample.jpg
```

### 9. Run Tests

```powershell
python -m pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System health + model loaded status |
| POST | `/api/upload` | Upload image, get image_id |
| POST | `/api/analyze` | Full end-to-end analysis |
| POST | `/api/quality-check` | Quality check only |
| GET | `/api/result/{image_id}` | Retrieve stored analysis |
| GET | `/api/report/{image_id}` | Download PDF report |
| POST | `/api/review` | Submit clinician review |

---

## Dataset

- **APTOS 2019 Blindness Detection** (Kaggle)
- Place at: `data/raw/aptos/train.csv` + `data/raw/aptos/train_images/`
- Splits: 70% Train / 15% Validation / 15% Test (stratified by grade)

---

## MATLAB Pipeline

```matlab
results = main_pipeline('path/to/fundus.jpg', 'outputs/matlab_baseline');
```

Stages: quality assessment, illumination normalization, CLAHE, vessel segmentation, microaneurysm candidate detection, optic disc localization, visual figures.

---

## Telemedicine Simulation

```matlab
dr_telemedicine_sim()        % all 3 scenarios
dr_telemedicine_sim('rural') % rural clinic only
```

Implements M/M/c Erlang-C queuing model for 3 scenarios (rural / district / urban). Outputs throughput, queue length, waiting time, server utilization.

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Referable Sensitivity | > 90% |
| Referable Specificity | > 85% |

> Targets are NOT claimed as achieved—they are the experimental validation goals. Actual metrics depend on the trained model.

---

## Known Limitations

1. **Lesion detection** uses OpenCV heuristics, not a trained DL model. Counts are image-derived estimates, not ground-truth segmentations.
2. **Confidence calibration** temperature is not yet tuned (T=1.0). Requires validation set to optimize.
3. **Simulink `.slx` binary** cannot be auto-generated; equivalent MATLAB M/M/c simulation is provided.
4. **Demo mode**: When `best_model.pth` is absent, placeholder predictions are clearly flagged but should not be used clinically.
5. **Not a medical device**: No regulatory approval. Ophthalmologist review is mandatory for all screening decisions.

---

## Project Structure

```
dr_screening_xai/
├── api/                   FastAPI backend
├── frontend/              React UI
├── src/
│   ├── classification/    EfficientNet-B0 model
│   ├── data/              Dataset loader, splits, leakage
│   ├── explainability/    GradCAM, calibration
│   ├── preprocessing/     Enhancement, transforms, deduplication
│   ├── quality/           Quality assessment
│   ├── reporting/         PDF generator
│   └── segmentation/      U-Net, lesion analysis
├── scripts/               Training, evaluation, robustness
├── matlab/                MATLAB image processing pipeline
├── simulink/              Telemedicine queuing simulation
├── configs/               config.yaml, experiment_config.yaml
├── data/                  Dataset splits
├── outputs/               Results, reports, GradCAM images
├── tests/                 API test suite
├── run_pipeline.py        Master entry point
└── requirements.txt
```
