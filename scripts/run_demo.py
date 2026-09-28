import cv2
import json
import os
import time
import argparse
from src.quality.assessment import QualityAssessor
from src.preprocessing.enhancement import ImageEnhancer
from src.segmentation.lesion_analysis import LesionAnalyzer
from src.reporting.pdf_generator import ReportGenerator
import shutil

def run_demo(image_path):
    print(f"Starting E2E Demo for {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image!")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_id = "demo_" + str(int(time.time()))
    
    # Quality
    assessor = QualityAssessor()
    quality = assessor.assess(image_rgb)
    print("Quality Assessment:", quality["status"])
    
    # Enhance
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(image_rgb)
    
    cv2.imwrite(f"outputs/annotated/{image_id}_enhanced.png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    
    # Lesions
    analyzer = LesionAnalyzer()
    lesions = analyzer.analyze(enhanced)
    
    # Mock Classification
    print("Demo model — not clinically validated")
    grade = 2
    label = "Moderate NPDR"
    confidence = 0.91
    
    # Dummy Grad-CAM for demo
    gradcam_path = f"outputs/gradcam/{image_id}_gradcam.png"
    os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
    shutil.copy(image_path, gradcam_path) # placeholder
    
    result = {
        "image_id": image_id,
        "quality": quality,
        "prediction": {
            "grade": grade,
            "label": label,
            "confidence": confidence,
            "referable": True
        },
        "lesions": lesions
    }
    
    with open(f"outputs/reports/{image_id}_result.json", "w") as f:
        json.dump(result, f, indent=4)
        
    print("Generating PDF Report...")
    pdf_gen = ReportGenerator()
    pdf_path = pdf_gen.generate(result, image_path, gradcam_path=gradcam_path)
    
    print(f"Demo complete! Outputs stored in outputs/")
    print(f"Result JSON: outputs/reports/{image_id}_result.json")
    print(f"Report PDF: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image")
    args = parser.parse_args()
    os.makedirs("outputs/annotated", exist_ok=True)
    os.makedirs("outputs/gradcam", exist_ok=True)
    run_demo(args.image)
