import cv2
import numpy as np

class QualityAssessor:
    def __init__(self, blur_threshold=100.0, contrast_threshold=0.2):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold

    def calculate_blur(self, image):
        # Variance of Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_brightness_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        return mean_brightness, std_contrast

    def calculate_retinal_area(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        area = np.sum(thresh == 255)
        total_area = image.shape[0] * image.shape[1]
        return area / total_area

    def assess(self, image):
        blur_score = self.calculate_blur(image)
        brightness, contrast = self.calculate_brightness_contrast(image)
        retinal_area = self.calculate_retinal_area(image)
        
        # --- OOD (Out-Of-Distribution) Detection ---
        # < 0.08 means nearly all black (invalid/empty image)
        # > 0.99 means image has no dark circular mask at all (not a fundus photo)
        if retinal_area < 0.08 or retinal_area > 0.99:
            return {
                "quality_score": 0.0,
                "status": "OOD_REJECTED",
                "blur_score": float(blur_score),
                "brightness": float(brightness),
                "contrast_score": float(contrast),
                "field_of_view_score": float(retinal_area),
                "recapture_required": True,
                "ood_detected": True,
                "error": "The image does not appear to be a valid retinal fundus image. Fundus images have a distinctive circular field. Please upload a correct retinal scan."
            }
            
        # Normalize scores to 0-1 broadly based on empirical values
        normalized_blur = np.clip(blur_score / 1000.0, 0, 1)
        normalized_brightness = 1.0 - abs(brightness - 127.5) / 127.5
        normalized_contrast = np.clip(contrast / 100.0, 0, 1)
        
        quality_score = (normalized_blur * 0.4 + 
                         normalized_brightness * 0.2 + 
                         normalized_contrast * 0.2 + 
                         retinal_area * 0.2)
        
        if quality_score >= 0.7:
            status = "GOOD"
            recapture = False
        elif quality_score >= 0.4:
            status = "BORDERLINE"
            recapture = False
        else:
            status = "UNGRADABLE"
            recapture = True
            
        return {
            "quality_score": float(quality_score),
            "status": status,
            "blur_score": float(blur_score),
            "brightness": float(brightness),
            "contrast_score": float(contrast),
            "field_of_view_score": float(retinal_area),
            "recapture_required": recapture
        }
