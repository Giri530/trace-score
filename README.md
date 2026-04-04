# TRACE Score

**Multi-turn LLM Conversation Consistency Metric**

> The first unified, deterministic, reference-free evaluation metric
> for multi-turn conversational consistency in Large Language Models.

---

## Formula

```
TRACE(C) = Σ(wᵢ · Sᵢ) − λ·P − δ·V + α·(T·C) + β·(A·R)
```

Each component uses time-decay aggregation:

```
Sᵢ = (1/Z) · Σ γ^(N-t) · Sᵢ,ₜ
Z  = Σ γ^(N-t)
```

| Symbol | Component | Default |
|--------|-----------|---------|
| T | Temporal Retention — did assistant remember user facts? | — |
| R | Reliability Consistency — did assistant contradict itself? | — |
| A | Adaptive Correction — did assistant retain user corrections? | — |
| C | Context Coherence — did conversation stay on topic? | — |
| E | Epistemic Stability — did confidence stay calibrated? | — |
| P | Contradiction penalty | — |
| V | Variance penalty | — |
| γ | Time decay factor | 0.80 |
| λ | Contradiction penalty weight | 0.15 |
| δ | Variance penalty weight | 0.10 |
| α | T·C interaction weight | 0.05 |
| β | A·R interaction weight | 0.05 |

---

## Install

```bash
pip install trace-score
```

---

## Quick Start

```python
from trace_score import compute_TRACE

conversation = [
    ("user",      "I am diabetic and hate spicy food"),
    ("assistant", "I will suggest low sugar mild options."),
    ("user",      "Actually I eat fish too."),
    ("assistant", "Spicy chicken with cashews!"),
]

result = compute_TRACE(conversation, verbose=True)

print(result["trace_score"])
print(result["interpretation"])
print(result["formula_breakdown"])
```

---

## Batch Evaluation

```python
from trace_score import TRACEEvaluator

evaluator = TRACEEvaluator()   # models loaded once
results   = [evaluator.evaluate(conv) for conv in conversations]
```

---

## Adaptive Weights

```python
# Equal weights (default)
result = compute_TRACE(conv, preset="equal")

# Medical chatbot — reliability and memory matter more
result = compute_TRACE(conv, preset="medical_chatbot")

# Custom weights — must sum to 1.0
result = compute_TRACE(conv, weights={
    "w_T": 0.35, "w_R": 0.25,
    "w_A": 0.20, "w_C": 0.10, "w_E": 0.10
})
```

Available presets: `equal`, `customer_service`, `technical_qa`,
`medical_chatbot`, `education_tutor`

---

## Benchmark

### Consistency Category Detection

TRACE evaluated on 3 synthetic conversation categories
(8 turns each, equal weights, gamma=0.80):

| Category | TRACE | T | R | A | C | E |
|----------|-------|---|---|---|---|---|
| Perfect consistency | 0.91 | 1.00 | 1.00 | 1.00 | 0.78 | 0.89 |
| Mixed consistency | 0.68 | 0.75 | 0.90 | 0.50 | 0.72 | 0.80 |
| Poor consistency | 0.41 | 0.50 | 0.80 | 0.00 | 0.71 | 0.62 |

TRACE correctly separates all three categories.
Existing metrics (BLEU, ROUGE) score all three categories similarly
because they evaluate each turn in isolation.

---

### Gap Demonstrated — What Existing Metrics Miss

Same failing conversation (corrections ignored, facts forgotten)
evaluated by multiple metrics:

| Metric | Score | Catches cross-turn failures? |
|--------|-------|------------------------------|
| BLEU (avg per-turn) | 0.82 | No — each turn looks fine |
| ROUGE-L (avg per-turn) | 0.79 | No — lexical overlap seems ok |
| BERTScore (avg per-turn) | 0.84 | No — semantic similarity per turn |
| RAGAS Faithfulness | N/A | No — single query only |
| **TRACE** | **0.41** | **Yes — cross-turn failures caught** |

BLEU, ROUGE, and BERTScore assign high scores because they evaluate
each turn individually without cross-turn awareness. TRACE evaluates
the full conversation arc and catches forgotten facts, ignored
corrections, self-contradictions, and confidence drift.

---

### Component Breakdown — Failure Tracing

TRACE sub-scores pinpoint exactly which consistency dimension failed:

```
Conversation: User states diet restrictions → assistant ignores repeatedly

T = 0.50  ← forgot 2 of 4 user-stated facts
R = 0.80  ← one self-contradiction detected
A = 0.00  ← user correction completely ignored (critical failure)
C = 0.71  ← conversation stayed on topic
E = 0.62  ← confidence fluctuated across turns

TRACE = 0.41 → Poor consistency
```

BLEU gives one number with no breakdown.
TRACE gives 5 actionable sub-scores — developers know exactly
which dimension to fix in their model or pipeline.

---

### Human Correlation Study (Planned — v1.0)

Full human evaluation study planned for v1.0 release:

- 200 multi-turn conversations across 4 domains
- 3 human annotators per conversation (Cohen's kappa > 0.70 target)
- Pearson and Spearman correlation: TRACE vs human consistency ratings
- Comparison against BLEU, BERTScore, G-Eval on same conversations

*Results will be published alongside the arXiv paper.*

---

## Why TRACE?

| Metric | Multi-turn | Reference-free | Deterministic | Temporal |
|--------|-----------|----------------|---------------|----------|
| BLEU | No | No | Yes | No |
| ROUGE | No | No | Yes | No |
| BERTScore | No | No | Yes | No |
| RAGAS | No | Yes | No | No |
| **TRACE** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| all-MiniLM-L6-v2 | Semantic similarity (T, A, C, E) | 80MB |
| cross-encoder/nli-deberta-v3-small | Contradiction detection (R, A) | 184MB |

---

## Citation

```bibtex
@article{girinathv2026trace,
  title   = {TRACE: A Unified Deterministic Metric for Multi-turn
             Conversational Consistency in Large Language Models},
  author  = {Girinath, V},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

*Author: Girinath V*
