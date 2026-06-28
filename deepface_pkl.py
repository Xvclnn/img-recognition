import os
import glob
import pickle
import time
import numpy as np
from deepface import DeepFace
import config

OUTPUT_PKL = "./embeddings/deepface_arcface.pkl"

image_paths = sorted(glob.glob(os.path.join(config.IMG_PATH, "*.jpeg")) + 
                     glob.glob(os.path.join(config.IMG_PATH, "*.jpg")))

total_images = len(image_paths)

representations = []
start_time = time.time()

for idx, path in enumerate(image_paths):
    try:
        face_objs = DeepFace.represent(
            img_path=path,
            model_name=config.MODEL_NAME,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=True
        )
        for face_idx, face_obj in enumerate(face_objs):
            embedding = face_obj["embedding"]
            box = face_obj["facial_area"]            
            representations.append({
                "path": path,
                "face_index": face_idx,
                "bounding_box": box,
                "embedding": list(embedding)
            })
            
    except Exception as e:
        print(f"Skipping error frame {path}: {e}")
        
    if (idx + 1) % 100 == 0:
        print(f"{idx + 1}/{total_images} file уншигдлаа...")

print(f"\nWriting master face lookup table ({len(representations)} faces found) to {OUTPUT_PKL}...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(representations, f)

elapsed = time.time() - start_time
print(f"Зарцуулсан хугацаа: {elapsed:.1f}с")