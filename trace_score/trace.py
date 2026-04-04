from typing import List, Tuple, Dict, Optional
from .components.temporal    import compute_T
from .components.reliability import compute_R
from .components.adaptive    import compute_A
from .components.coherence   import compute_C
from .components.epistemic   import compute_E
WEIGHT_PRESETS = {
    "equal":            {"w_T": 0.20, "w_R": 0.20, "w_A": 0.20, "w_C": 0.20, "w_E": 0.20},
    "customer_service": {"w_T": 0.30, "w_R": 0.20, "w_A": 0.30, "w_C": 0.10, "w_E": 0.10},
    "technical_qa":     {"w_T": 0.20, "w_R": 0.30, "w_A": 0.10, "w_C": 0.10, "w_E": 0.30},
    "medical_chatbot":  {"w_T": 0.30, "w_R": 0.30, "w_A": 0.20, "w_C": 0.10, "w_E": 0.10},
    "education_tutor":  {"w_T": 0.20, "w_R": 0.10, "w_A": 0.30, "w_C": 0.30, "w_E": 0.10},
}
INTERPRETATION_THRESHOLDS = [
    (0.85, "Excellent — conversation is highly consistent"),
    (0.70, "Good — minor consistency issues"),
    (0.55, "Moderate — notable consistency failures"),
    (0.40, "Poor — significant consistency problems"),
    (0.00, "Very poor — conversation is highly inconsistent"),
]
class TRACEEvaluator:
    def __init__(self):
        self._sbert = None
        self._nli   = None
    def _load_models(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
        if self._nli is None:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small",
                max_length=512
            )
    def evaluate(
        self,
        conversation : List[Tuple[str, str]],
        weights      : Optional[Dict[str, float]] = None,
        preset       : str   = "equal",
        gamma        : float = 0.80,
        lambda_      : float = 0.15,
        delta        : float = 0.10,
        alpha        : float = 0.05,
        beta         : float = 0.05,
        verbose      : bool  = False
    ) -> Dict:
        self._load_models()
        if not conversation:
            return self._empty_result(weights or WEIGHT_PRESETS[preset])
        for role, text in conversation:
            if role not in ("user", "assistant"):
                raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")
            if not text or not text.strip():
                raise ValueError("Empty turn text found in conversation.")
        w = self._resolve_weights(weights, preset)
        t_result = compute_T(conversation, sbert_model=self._sbert, gamma=gamma)
        r_result = compute_R(conversation, nli_model=self._nli,     gamma=gamma)
        a_result = compute_A(conversation, sbert_model=self._sbert,
                             nli_model=self._nli, gamma=gamma)
        c_result = compute_C(conversation, sbert_model=self._sbert, gamma=gamma)
        e_result = compute_E(conversation, sbert_model=self._sbert, gamma=gamma)
        T = t_result["score"]
        R = r_result["score"]
        A = a_result["score"]
        C = c_result["score"]
        E = e_result["score"]
        P = r_result["penalty"]           
        V = e_result["variance_penalty"] 
        base_score = (
            w["w_T"] * T +
            w["w_R"] * R +
            w["w_A"] * A +
            w["w_C"] * C +
            w["w_E"] * E
        )
        penalty_term = lambda_ * P + delta * V
        interaction_term = alpha * (T * C) + beta * (A * R)
        raw_trace = base_score - penalty_term + interaction_term
        trace_score = round(max(0.0, min(1.0, raw_trace)), 4)
        formula_breakdown = (
            f"TRACE = [{w['w_T']}×{T:.3f} + {w['w_R']}×{R:.3f} + "
            f"{w['w_A']}×{A:.3f} + {w['w_C']}×{C:.3f} + {w['w_E']}×{E:.3f}]"
            f" - [{lambda_}×{P:.3f} + {delta}×{V:.3f}]"
            f" + [{alpha}×({T:.3f}×{C:.3f}) + {beta}×({A:.3f}×{R:.3f})]"
            f" = {base_score:.4f} - {penalty_term:.4f} + {interaction_term:.4f}"
            f" = {raw_trace:.4f} → clamped → {trace_score:.4f}"
        )
        result = {
            "trace_score":       trace_score,
            "base_score":        round(base_score, 4),
            "penalty_term":      round(penalty_term, 4),
            "interaction_term":  round(interaction_term, 4),
            "T": T, "R": R, "A": A, "C": C, "E": E,
            "P": P, "V": V,
            "weights":           w,
            "preset":            preset if weights is None else "custom",
            "gamma":             gamma,
            "lambda":            lambda_,
            "delta":             delta,
            "alpha":             alpha,
            "beta":              beta,
            "interpretation":    self._interpret(trace_score),
            "formula_breakdown": formula_breakdown,
            "num_turns":         len(conversation),
        }
        if verbose:
            result["details"] = {
                "T": t_result,
                "R": r_result,
                "A": a_result,
                "C": c_result,
                "E": e_result,
            }
        return result
    def _resolve_weights(self, weights, preset):
        if weights is not None:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total:.4f}.")
            return weights
        if preset not in WEIGHT_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(WEIGHT_PRESETS.keys())}"
            )
        return WEIGHT_PRESETS[preset]
    def _interpret(self, score: float) -> str:
        for threshold, label in INTERPRETATION_THRESHOLDS:
            if score >= threshold:
                return label
        return INTERPRETATION_THRESHOLDS[-1][1]
    def _empty_result(self, w):
        return {
            "trace_score": 1.0,
            "base_score":  1.0,
            "penalty_term": 0.0,
            "interaction_term": 0.0,
            "T": 1.0, "R": 1.0, "A": 1.0, "C": 1.0, "E": 1.0,
            "P": 0.0, "V": 0.0,
            "weights": w,
            "interpretation": "Empty conversation. TRACE = 1.0.",
            "num_turns": 0,
        }
def compute_TRACE(
    conversation : List[Tuple[str, str]],
    weights      : Optional[Dict[str, float]] = None,
    preset       : str   = "equal",
    gamma        : float = 0.80,
    lambda_      : float = 0.15,
    delta        : float = 0.10,
    alpha        : float = 0.05,
    beta         : float = 0.05,
    verbose      : bool  = False
) -> Dict:
    evaluator = TRACEEvaluator()
    return evaluator.evaluate(
        conversation,
        weights=weights,
        preset=preset,
        gamma=gamma,
        lambda_=lambda_,
        delta=delta,
        alpha=alpha,
        beta=beta,
        verbose=verbose
    )
