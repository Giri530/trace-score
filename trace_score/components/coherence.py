import numpy as np
from typing import List, Tuple, Dict
def compute_C(
    conversation : List[Tuple[str, str]],
    sbert_model  = None,
    gamma : float = 0.80
) -> Dict:
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts = [text for _, text in conversation]
    N = len(all_texts)
    if N <= 1:
        return {
            "score": 1.0,
            "per_turn_sims": [],
            "decay_weights": [],
            "mean_drift":0.0,
            "explanation":f"Only {N} turn(s). C = 1.0."
        }
    embeddings = sbert_model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    )
    sim_vector = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    sim_vector = np.clip(sim_vector, 0.0, 1.0)
    per_turn_sims = [round(float(s), 4) for s in sim_vector]
    M = len(per_turn_sims) 
    decay_weights = [gamma ** (M - 1 - t) for t in range(M)]
    Z = sum(decay_weights)
    C_score = sum(w * s for w, s in zip(decay_weights, per_turn_sims)) / Z
    mean_drift = 1.0 - float(np.mean(sim_vector))
    return {
        "score": round(C_score, 4),
        "per_turn_sims": per_turn_sims,
        "decay_weights": [round(w, 4) for w in decay_weights],
        "mean_drift": round(mean_drift, 4),
        "explanation": (
            f"Encoded {N} turns via SBERT. "
            f"Adjacent-pair sims: {per_turn_sims}. "
            f"C = {C_score:.4f}, mean drift = {mean_drift:.4f}."
        )
    }
