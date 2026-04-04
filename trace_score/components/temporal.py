import re
import numpy as np
from typing import List, Tuple, Dict
FACT_PATTERNS = [
    (r"i(?:'m| am) ([\w\s]+?)(?:\s+and\s+|\s*,\s*|\.|\?|$)",
     "user is {}"),
    (r"i have ([\w\s]+?)(?:\s+and\s+|\s*,\s*|\.|\?|$)",
     "user has {}"),
    (r"i (hate|love|prefer|avoid|dislike|like|eat|drink) ([\w\s]+?)(?:\s+and\s+|\s*,\s*|\.|\?|$)",
     "user {}s {}"),
    (r"allergic to ([\w\s]+?)(?:\s+and\s+|\s*,\s*|\.|\?|$)",
     "user is allergic to {}"),
    (r"my ([\w\s]+?) is ([\w\s\d]+?)(?:\s+and\s+|\s*,\s*|\.|\?|$)",
     "user's {} is {}"),
]
def extract_atomic_facts(text: str) -> List[str]:
    facts      = []
    text_lower = text.lower().strip()
    for pattern, template in FACT_PATTERNS:
        for match in re.findall(pattern, text_lower):
            if isinstance(match, tuple):
                fact = template.format(*[m.strip() for m in match])
            else:
                fact = template.format(match.strip())
            if len(fact.split()) >= 3:
                facts.append(fact)
    return list(set(facts))
def compute_T(
    conversation : List[Tuple[str, str]],
    sbert_model  = None,
    threshold    : float = 0.60,
    gamma        : float = 0.80
) -> Dict:
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    cumulative_facts = []
    assistant_turns  = [] 
    for role, text in conversation:
        if role == "user":
            new_facts = extract_atomic_facts(text)
            for f in new_facts:
                if f not in cumulative_facts:
                    cumulative_facts.append(f)
        elif role == "assistant":
            assistant_turns.append({
                "text":   text,
                "facts":  list(cumulative_facts)  
            })
    N = len(assistant_turns)
    if N == 0:
        return {
            "score": 1.0,
            "per_turn_scores": [],
            "decay_weights":   [],
            "user_facts":      cumulative_facts,
            "explanation":     "No assistant turns found. T = 1.0."
        }
    per_turn_scores = []
    for turn_data in assistant_turns:
        facts = turn_data["facts"]
        text  = turn_data["text"]
        if not facts:
            per_turn_scores.append(1.0)
            continue
        fact_embs = sbert_model.encode(
            facts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        turn_emb = sbert_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        sims     = np.dot(fact_embs, turn_emb.T).flatten()
        recalled = int(np.sum(sims >= threshold))
        T_t      = recalled / len(facts)
        per_turn_scores.append(round(T_t, 4))
    decay_weights = [gamma ** (N - 1 - t) for t in range(N)]
    Z = sum(decay_weights)
    T_score = sum(w * s for w, s in zip(decay_weights, per_turn_scores)) / Z
    return {
        "score":           round(T_score, 4),
        "per_turn_scores": per_turn_scores,
        "decay_weights":   [round(w, 4) for w in decay_weights],
        "user_facts":      cumulative_facts,
        "explanation":     (
            f"Computed T across {N} assistant turns with gamma={gamma}. "
            f"Per-turn scores: {per_turn_scores}. "
            f"Decay weights: {[round(w,3) for w in decay_weights]}. "
            f"T = {T_score:.4f}."
        )
    }
