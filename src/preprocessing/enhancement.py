import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def apply_clahe(self, image):
        # Convert to LAB space for CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        l_clahe = self.clahe.apply(l)
        
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return enhanced
        
    def normalize_illumination(self, image):
        # Background subtraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        background = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
        normalized_v = cv2.subtract(v, background)
        normalized_v = cv2.normalize(normalized_v, None, 0, 255, cv2.NORM_MINMAX)
        hsv[:, :, 2] = normalized_v
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    def denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def enhance(self, image):
        # Pipeline: Denoise -> Illumination Normalization -> CLAHE
        denoised = self.denoise(image)
        norm = self.normalize_illumination(denoised)
        enhanced = self.apply_clahe(norm)
        return enhanced
