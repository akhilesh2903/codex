import hashlib
import os
from PIL import Image
import imagehash
import json

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def calculate_phash(filepath):
    try:
        img = Image.open(filepath)
        return str(imagehash.phash(img))
    except Exception:
        return None

class DuplicateDetector:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def find_duplicates(self, report_path="outputs/reports/duplicate_report.json"):
        sha_hashes = {}
        phash_hashes = {}
        exact_duplicates = []
        near_duplicates = []
        
        for p in self.image_paths:
            if not os.path.exists(p):
                continue
            s_hash = calculate_sha256(p)
            p_hash = calculate_phash(p)
            
            if s_hash in sha_hashes:
                exact_duplicates.append({"original": sha_hashes[s_hash], "duplicate": p, "sha256": s_hash})
            else:
                sha_hashes[s_hash] = p
                
            if p_hash in phash_hashes:
                near_duplicates.append({"original": phash_hashes[p_hash], "duplicate": p, "phash": p_hash})
            else:
                phash_hashes[p_hash] = p
                
        report = {
            "exact_duplicates": exact_duplicates,
            "near_duplicates": near_duplicates
        }
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        return report
