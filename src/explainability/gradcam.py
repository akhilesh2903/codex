import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import matplotlib.pyplot as plt

class XAI_Visualizer:
    def __init__(self, model, target_layer, device="cpu"):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.model.eval()
        # Initialize GradCAM
        try:
            self.cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception as e:
            print("GradCAM initialization failed:", e)
            self.cam = None

    def generate_heatmap(self, input_tensor, original_image, target_category=None, output_path=None):
        if self.cam is None:
            return None, None
            
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        # Using GradCAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=target_category)
        
        # In this batch, we expect one image
        grayscale_cam = grayscale_cam[0, :]
        
        # Original image must be float normalized [0,1]
        img_normalized = np.float32(original_image) / 255.0
        
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        if output_path is not None:
             os.makedirs(os.path.dirname(output_path), exist_ok=True)
             plt.imsave(output_path, visualization)
             
        return grayscale_cam, visualization
