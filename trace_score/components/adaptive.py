import re
import numpy as np
from typing import List, Tuple, Dict, Optional
CORRECTION_MARKERS = [
    r"\bactually\b",       r"\bno[,\s]",
    r"\bnot exactly\b",    r"\bi said\b",
    r"\bi mentioned\b",    r"\bi told you\b",
    r"\bplease avoid\b",   r"\bplease don'?t\b",
    r"\bdon'?t forget\b",  r"\bi already said\b",
    r"\bcorrection\b",     r"\bwait[,\s]",
    r"\bi meant\b",        r"\bthat'?s wrong\b",
    r"\bnot quite\b",      r"\byou forgot\b",
    r"\byou missed\b",     r"\bremember[,\s]",
]
def has_correction_marker(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in CORRECTION_MARKERS)
def is_correction_turn(
    user_text        : str,
    prev_asst_text   : Optional[str],
    nli_model        = None
) -> bool:
    if not has_correction_marker(user_text):
        return False
    if prev_asst_text is None or nli_model is None:
        return True
    scores = nli_model.predict(
        [[prev_asst_text, user_text]],
        apply_softmax=True
    )
    return float(scores[0][0]) >= 0.35 or has_correction_marker(user_text)
def compute_A(
    conversation:List[Tuple[str, str]],
    sbert_model=None,
    nli_model=None,
    retention_threshold: float = 0.55,
    gamma:float = 0.80
) -> Dict:
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    if nli_model is None:
        from sentence_transformers import CrossEncoder
        nli_model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            max_length=512
        )
    corrections    = []
    prev_assistant = None
    for i, (role, text) in enumerate(conversation):
        if role == "assistant":
            prev_assistant = text
        elif role == "user":
            if is_correction_turn(text, prev_assistant, nli_model):
                corrections.append({"turn_index": i, "text": text})
    K = len(corrections)
    if K == 0:
        return {
            "score": 1.0,
            "per_turn_scores": [],
            "decay_weights": [],
            "corrections_found": 0,
            "retained_count": 0,
            "details": [],
            "explanation": "No correction turns detected. A = 1.0 (vacuously true)."
        }
    per_turn_scores = []
    details = []
    for k, correction in enumerate(corrections):
        turn_idx = correction["turn_index"]
        c_text   = correction["text"]
        subsequent = [
            text for i, (role, text) in enumerate(conversation)
            if role == "assistant" and i > turn_idx
        ]
        if not subsequent:
            A_k  = 1.0
            max_sim = 1.0
        else:
            c_emb   = sbert_model.encode(
                [c_text], convert_to_numpy=True, normalize_embeddings=True
            )
            s_embs  = sbert_model.encode(
                subsequent, convert_to_numpy=True, normalize_embeddings=True
            )
            sims    = np.dot(c_emb, s_embs.T)[0]
            max_sim = float(sims.max())
            A_k     = 1.0 if max_sim >= retention_threshold else 0.0
        per_turn_scores.append(A_k)
        details.append({
            "turn_index":   turn_idx,
            "text":         c_text[:100],
            "retained":     bool(A_k),
            "max_sim":      round(max_sim, 4),
        })
    decay_weights = [gamma ** (K - 1 - k) for k in range(K)]
    Z = sum(decay_weights)
    A_score = sum(w * s for w, s in zip(decay_weights, per_turn_scores)) / Z
    return {
        "score": round(A_score, 4),
        "per_turn_scores": per_turn_scores,
        "decay_weights": [round(w, 4) for w in decay_weights],
        "corrections_found": K,
        "retained_count": int(sum(per_turn_scores)),
        "details": details,
        "explanation": (
            f"Found {K} correction(s). "
            f"Per-correction scores: {per_turn_scores}. "
            f"A = {A_score:.4f}."
        )
    }
