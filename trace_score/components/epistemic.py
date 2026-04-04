import numpy as np
from typing import List, Tuple, Dict
CONFIDENCE_LEXICON = {
    "definitely": 1.00,"certainly": 1.00,
    "absolutely": 1.00,"undoubtedly": 0.98,
    "without a doubt": 0.97,"for sure": 0.95,
    "clearly": 0.94,"obviously": 0.93,
    "i am certain": 0.95,"i am confident": 0.92,
    "i know": 0.90,"i believe": 0.80,
    "i think": 0.75,"likely": 0.77,
    "probably": 0.74,"it seems": 0.72,
    "generally": 0.76,"typically": 0.75,
    "usually": 0.74,"might": 0.52,
    "could": 0.51,"may": 0.56,
    "possibly": 0.46,"i suppose": 0.51,
    "i guess": 0.46,"sometimes": 0.56,
    "perhaps": 0.31,"maybe": 0.31,
    "i am not sure": 0.20,"i'm not sure": 0.20,
    "uncertain": 0.26,"unclear": 0.26,
    "it depends": 0.31,"hard to say": 0.20,
    "i doubt": 0.22,"i cannot say": 0.15,
}
DEFAULT_CONFIDENCE = 0.68
HIGH_CONF_ANCHOR   = "I am absolutely certain and confident about this."
LOW_CONF_ANCHOR    = "I am not sure and quite uncertain about this perhaps."
def lexicon_confidence(text: str) -> float:
    text_lower   = text.lower()
    found_scores = [
        score for marker, score in sorted(
            CONFIDENCE_LEXICON.items(), key=lambda x: len(x[0]), reverse=True
        )
        if marker in text_lower
    ]
    return float(np.mean(found_scores)) if found_scores else DEFAULT_CONFIDENCE
def sbert_anchor_confidence(text, sbert_model, high_emb, low_emb) -> float:
    text_emb = sbert_model.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    sim_high = float(np.dot(text_emb, high_emb))
    sim_low  = float(np.dot(text_emb, low_emb))
    total    = sim_high + sim_low
    return float(sim_high / total) if total > 0 else DEFAULT_CONFIDENCE
def compute_E(
    conversation    : List[Tuple[str, str]],
    sbert_model     = None,
    gamma           : float = 0.80,
    lexicon_weight  : float = 0.60,
    sbert_weight    : float = 0.40
) -> Dict:
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    anchors  = sbert_model.encode(
        [HIGH_CONF_ANCHOR, LOW_CONF_ANCHOR],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    high_emb = anchors[0]
    low_emb  = anchors[1]
    assistant_texts = [text for role, text in conversation if role == "assistant"]
    N = len(assistant_texts)
    if N == 0:
        return {
            "score": 1.0,
            "variance_penalty": 0.0,
            "turn_confidences": [],
            "decay_weights": [],
            "mean_confidence":  DEFAULT_CONFIDENCE,
            "raw_variance": 0.0,
            "explanation": "No assistant turns. E = 1.0, V = 0.0."
        }
    if N == 1:
        conf = (
            lexicon_weight * lexicon_confidence(assistant_texts[0]) +
            sbert_weight   * sbert_anchor_confidence(assistant_texts[0], sbert_model, high_emb, low_emb)
        )
        return {
            "score": round(conf, 4),
            "variance_penalty": 0.0,
            "turn_confidences": [round(conf, 4)],
            "decay_weights": [1.0],
            "mean_confidence": round(conf, 4),
            "raw_variance": 0.0,
            "explanation": f"Single assistant turn. E = {conf:.4f}, V = 0.0."
        }
    turn_confidences = [
        lexicon_weight * lexicon_confidence(t) +
        sbert_weight   * sbert_anchor_confidence(t, sbert_model, high_emb, low_emb)
        for t in assistant_texts
    ]
    conf_array = np.array(turn_confidences)
    decay_weights = [gamma ** (N - 1 - t) for t in range(N)]
    Z = sum(decay_weights)
    E_score = sum(w * c for w, c in zip(decay_weights, turn_confidences)) / Z
    mu = float(conf_array.mean())
    raw_variance = float(np.mean((conf_array - mu) ** 2))
    V_norm = min(raw_variance / 0.25, 1.0)
    return {
        "score": round(E_score, 4),
        "variance_penalty": round(V_norm, 4),
        "turn_confidences": [round(c, 4) for c in turn_confidences],
        "decay_weights": [round(w, 4) for w in decay_weights],
        "mean_confidence": round(mu, 4),
        "raw_variance": round(raw_variance, 6),
        "explanation": (
            f"Analyzed {N} assistant turns. "
            f"Confidences: {[round(c,3) for c in turn_confidences]}. "
            f"E = {E_score:.4f}, V_norm = {V_norm:.4f}."
        )
    }
