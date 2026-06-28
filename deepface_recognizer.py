import pickle
import numpy as np
import os
import time
from collections import Counter
import config

# ----------------------------
# Helper
# ----------------------------
def get_identity(path):
    filename = os.path.basename(path)
    return filename.rsplit("_", 1)[0]


# ----------------------------
# Load database
# ----------------------------
print("Loading database...")

with open(config.PKL_PATH, "rb") as f:
    database = pickle.load(f)

paths = []
embeddings = []

for item in database:
    paths.append(item["path"])
    embeddings.append(item["embedding"])

embeddings = np.asarray(embeddings, dtype=np.float32)

print(f"Faces loaded     : {len(paths)}")
print(f"Embedding size   : {embeddings.shape[1]}")

# ----------------------------
# Normalize embeddings
# ----------------------------
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-10)

# ----------------------------
# Count images per identity
# ----------------------------
identity_counts = Counter(get_identity(p) for p in paths)

valid_queries = [
    i for i, p in enumerate(paths)
    if identity_counts[get_identity(p)] >= 2
]

print(f"People           : {len(identity_counts)}")
print(f"Valid queries    : {len(valid_queries)}")

# ----------------------------
# Evaluation
# ----------------------------
top1_correct = 0
top5_correct = 0
total_search_time = 0

shown_failures = 0
MAX_FAILURES = 10

for query_idx in valid_queries:

    query_embedding = embeddings[query_idx]
    true_identity = get_identity(paths[query_idx])

    start = time.time()

    similarity = embeddings @ query_embedding
    distance = 1 - similarity

    sorted_idx = np.argsort(distance)

    total_search_time += time.time() - start

    predictions = []

    for idx in sorted_idx:

        # Skip the exact same image
        if idx == query_idx:
            continue

        predictions.append(get_identity(paths[idx]))

        if len(predictions) == 5:
            break

    # ----------------------------
    # Top-1
    # ----------------------------
    if predictions[0] == true_identity:
        top1_correct += 1

    # ----------------------------
    # Top-5
    # ----------------------------
    if true_identity in predictions:
        top5_correct += 1

    # ----------------------------
    # Print a few failures
    # ----------------------------
    elif shown_failures < MAX_FAILURES:
        shown_failures += 1

        print("\n------------------------------------")
        print("Query :", os.path.basename(paths[query_idx]))
        print("Truth :", true_identity)
        print("Top 5 :")

        for rank, idx in enumerate(sorted_idx[1:6], start=1):
            print(
                f"{rank}. "
                f"{os.path.basename(paths[idx])} "
                f"(Identity={get_identity(paths[idx])}, "
                f"Distance={distance[idx]:.4f})"
            )

# ----------------------------
# Results
# ----------------------------
n = len(valid_queries)

print("\n========== RESULTS ==========")
print(f"Valid Queries    : {n}")
print(f"Top-1 Accuracy   : {top1_correct / n:.4f}")
print(f"Top-5 Accuracy   : {top5_correct / n:.4f}")
print(f"Average Search   : {(total_search_time / n) * 1000:.3f} ms")