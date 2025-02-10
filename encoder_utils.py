import torch
from tqdm import tqdm
import faiss
import re
import numpy as np
from torch.utils.data import DataLoader


def build_faiss_index(dataloader, preprocess_fn, device="cuda"):
    """
    Build a FAISS index with preprocessing applied directly.

    Args:
        dataloader (DataLoader): DataLoader for labeled data.
        preprocess_fn (function): Preprocessing function for embeddings.
        device (str): Device to run computations ("cuda" or "cpu").

    Returns:
        faiss_labels (np.ndarray): Labels corresponding to indexed embeddings.
        faiss_index (faiss.Index): Built FAISS index.
    """
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Building FAISS Index"):
            imgs = imgs.to(device)

            embeddings = preprocess_fn(imgs)
            features.append(embeddings.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    features = np.vstack(features)
    faiss_labels = np.array(labels)

    dim = features.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(features)

    print(f"FAISS index built with {len(labels)} entries.")
    return faiss_labels, faiss_index


def predict_with_faiss(dataloader, preprocess_fn, faiss_index, faiss_labels,
                              device="cuda", top_k=5, distractor_classes=None):
    """
    Predict top-k classes using FAISS with duplicate and distractor handling.

    Args:
        dataloader (DataLoader): DataLoader for test data.
        preprocess_fn (function): Preprocessing function for embeddings.
        faiss_index (faiss.Index): Prebuilt FAISS index.
        faiss_labels (np.ndarray): Labels corresponding to FAISS index.
        device (str): Device to run computations ("cuda" or "cpu").
        top_k (int): Number of predictions to return.
        distractor_classes (set): Classes treated as distractors.

    Returns:
        results (list): List of tuples (file_id, [top_k_predictions]).
    """
    assert faiss_index is not None, "FAISS index is not built. Call build_faiss_index_global() first."

    if distractor_classes is None:
        distractor_classes = {}

    results = []

    with torch.no_grad():
        for imgs, file_ids in tqdm(dataloader, desc="Predicting with FAISS"):
            imgs = imgs.to(device)
            embeddings = preprocess_fn(imgs)

            features = embeddings.cpu().numpy()
            distances, indices = faiss_index.search(features, top_k * 2)

            for i in range(len(features)):
                top_classes = faiss_labels[indices[i]].tolist()

                seen = set()
                filtered_classes = []
                for cls in top_classes:
                    if cls not in seen:
                        filtered_classes.append(cls)
                        seen.add(cls)
                    if len(filtered_classes) == top_k:
                        break

                predictions = []
                if filtered_classes[0] in distractor_classes:
                    predictions.append(filtered_classes[0])
                    predictions.append(-1)
                    predictions.extend(filtered_classes[1:])
                else:
                    predictions = filtered_classes

                predictions = predictions[:top_k]

                if len(predictions) < top_k:
                    predictions += [-1] * (top_k - len(predictions))
                    
                results.append((file_ids[i], predictions))

    return results

