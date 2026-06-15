import pickle
import numpy as np
from deepface import DeepFace
import time
import os
import config

HAIH_ZURAG = "./converted_assets/Aaron_Peirsol_0001.jpeg" 
if not os.path.exists(config.PKL_PATH):
    print(f"Error: {config.PKL_PATH} not found! .PKL file-аа үүсгэнэ үү.")
    exit()

print("reserve fail-г уншиж байна...")
with open(config.PKL_PATH, "rb") as f:
    database = pickle.load(f)

paths = []
boxes = []
embeddings_list = []

for item in database: 
    paths.append(item["path"])
    boxes.append(item["bounding_box"])
    embeddings_list.append(item["embedding"])

embeddings = np.array(embeddings_list)

print(f"Дараах нэртэй зургаар хайж байна: {os.path.basename(HAIH_ZURAG)}...")
try:
    target_faces = DeepFace.represent(
        img_path=HAIH_ZURAG, 
        model_name=config.MODEL_NAME, 
        detector_backend=config.DETECTOR_BACKEND,
        enforce_detection=True
    )
except Exception as e:
    print(f"Оруулсан зурганд царай олдсонгүй. Error: {e}")
    exit()

print(f"Эх зурганд {len(target_faces)} царай олдлоо...\n")

for face_no, target_obj in enumerate(target_faces):
    target_embedding = np.array(target_obj["embedding"])
    target_box = target_obj["facial_area"]
    
    print(f"--- АДИЛ ЦАРАЙ #{face_no + 1} (Face Box: {target_box}) ---")
    
    start_time = time.time()
    
    dot_product = np.dot(embeddings, target_embedding)
    norms_db = np.linalg.norm(embeddings, axis=1)
    norm_target = np.linalg.norm(target_embedding)
    
    norms_db[norms_db == 0] = 1e-10
    cosine_distances = 1 - (dot_product / (norms_db * norm_target))
    
    sorted_indices = np.argsort(cosine_distances)
    search_time = time.time() - start_time
    
    print(f"Matrix хайлт {search_time:.6f} секундэд дуусав.")
    match_count = 0
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        zai = cosine_distances[idx]
        if zai < config.THRESHOLD:
            match_count += 1
            print(f"  {match_count}. {os.path.basename(paths[idx])} (Зай: {zai:.4f}, Box: {boxes[idx]})")
        else:
            break