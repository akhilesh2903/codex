import cv2
import os
import argparse
import yaml
from src.quality.assessment import QualityAssessor
from src.preprocessing.enhancement import ImageEnhancer
import matplotlib.pyplot.subplots as subplots
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, "r") as f:
         return yaml.safe_load(f)

def run_quality_pipeline(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    assessor = QualityAssessor()
    enhancer = ImageEnhancer()
    
    quality_result = assessor.assess(image)
    print("Initial Quality:", quality_result)
    
    enhanced = enhancer.enhance(image)
    
    # Save combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Original\nScore: {quality_result['quality_score']:.2f} ({quality_result['status']})")
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced (CLAHE+Norm)")
    axes[1].axis('off')
    
    os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.basename(image_path).replace('.', '_quality.')
    plt.savefig(os.path.join(output_dir, out_filename))
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Quality")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/reports/quality")
    args = parser.parse_args()
    
    run_quality_pipeline(args.image, args.out)
