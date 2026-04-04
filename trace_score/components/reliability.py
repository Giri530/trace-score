from itertools import combinations
from typing import List, Tuple, Dict
import numpy as np
def compute_R(
    conversation  : List[Tuple[str, str]],
    nli_model     = None,
    threshold     : float = 0.75,
    gamma         : float = 0.80,
    bidirectional : bool  = True
) -> Dict:
    if nli_model is None:
        from sentence_transformers import CrossEncoder
        nli_model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            max_length=512
        )
    assistant_turns = [
        (i, text)
        for i, (role, text) in enumerate(conversation)
        if role == "assistant"
    ]
    N = len(assistant_turns)
    if N < 2:
        return {
            "score":               1.0,
            "penalty":             0.0,
            "per_turn_scores":     [1.0] * N,
            "decay_weights":       [1.0] * N,
            "total_pairs":         0,
            "contradiction_count": 0,
            "contradictions":      [],
            "explanation":         f"Only {N} assistant turn(s). R = 1.0, P = 0.0."
        }
    all_pairs = list(combinations(range(N), 2))
    forward_inputs = [
        [assistant_turns[i][1], assistant_turns[j][1]]
        for i, j in all_pairs
    ]
    forward_scores = nli_model.predict(forward_inputs, apply_softmax=True)
    forward_probs  = forward_scores[:, 0]
    if bidirectional:
        backward_inputs = [
            [assistant_turns[j][1], assistant_turns[i][1]]
            for i, j in all_pairs
        ]
        backward_scores = nli_model.predict(backward_inputs, apply_softmax=True)
        backward_probs  = backward_scores[:, 0]
        contradiction_probs = np.maximum(forward_probs, backward_probs)
    else:
        contradiction_probs = forward_probs
    contradictions   = []
    contradicted_set = set()   
    for k, (i, j) in enumerate(all_pairs):
        prob = float(contradiction_probs[k])
        if prob >= threshold:
            contradictions.append({
                "turn_i":                    assistant_turns[i][0],
                "turn_j":                    assistant_turns[j][0],
                "contradiction_probability": round(prob, 4),
                "text_i":                    assistant_turns[i][1][:100],
                "text_j":                    assistant_turns[j][1][:100],
            })
            contradicted_set.add(i)
            contradicted_set.add(j)
    per_turn_scores = [
        0.0 if t in contradicted_set else 1.0
        for t in range(N)
    ]
    decay_weights = [gamma ** (N - 1 - t) for t in range(N)]
    Z = sum(decay_weights)
    R_score = sum(w * s for w, s in zip(decay_weights, per_turn_scores)) / Z
    total_pairs = len(all_pairs)
    P = len(contradictions) / total_pairs if total_pairs > 0 else 0.0
    return {
        "score":               round(R_score, 4),
        "penalty":             round(P, 4),
        "per_turn_scores":     per_turn_scores,
        "decay_weights":       [round(w, 4) for w in decay_weights],
        "total_pairs":         total_pairs,
        "contradiction_count": len(contradictions),
        "contradictions":      contradictions,
        "explanation":         (
            f"Checked {total_pairs} pairs across {N} assistant turns. "
            f"Contradictions: {len(contradictions)}. "
            f"R = {R_score:.4f}, P (penalty) = {P:.4f}."
        )
    }
