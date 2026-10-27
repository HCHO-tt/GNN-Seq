"""Explicit Seq MOEA/D train/test problem matrix and evaluation summaries."""
from collections import Counter

import numpy as np


SEEN = ("DTLZ2", "WFG4", "WFG6")
UNSEEN = ("DTLZ4", "WFG5", "WFG7", "WFG8", "WFG9")


def training_objectives(key):
    """Single-scale registered training keys; mixed-scale needs an explicit m."""
    if key in {f"M_2_46_{m}" for m in (3, 5, 7)}:
        return int(key.rsplit("_", 1)[1])
    if key == "M_2_46_357":
        return None
    try:
        value = int(key.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return value


def evaluation_tasks(suite, train_key, objectives=None):
    if suite == "single":
        return (train_key,)
    if suite not in {"seen", "unseen", "all"}:
        raise ValueError("Unknown evaluation suite")
    if train_key not in {"M_2_46_3", "M_2_46_5", "M_2_46_7", "M_2_46_357"}:
        raise ValueError("Seq seen/unseen suite requires a registered mixed-function training key")
    inferred = training_objectives(train_key)
    if inferred is not None and objectives is not None and objectives != inferred:
        raise ValueError("Evaluation objectives must match the fixed training scale")
    m = objectives if objectives is not None else inferred
    if m is None:
        raise ValueError("Mixed-objective training key requires --objectives")
    if m not in (3, 5, 7):
        raise ValueError("Seq evaluation suite supports 3, 5, or 7 objectives")
    functions = (SEEN if suite in ("seen", "all") else ()) + (
        UNSEEN if suite in ("unseen", "all") else ())
    return tuple(f"{function}_{m}" for function in functions)


def task_counts(rollouts):
    return dict(sorted(Counter(f"{item['function']}_{item['objectives']}"
                               for item in rollouts).items()))


def summarize_evaluations(rows):
    """Keep final IGD primary, and preserve best IGD as a distinct diagnostic."""
    return {
        "runs": len(rows),
        "mean_final_igd": float(np.mean([r["last_igd"] for r in rows])),
        "std_final_igd": float(np.std([r["last_igd"] for r in rows], ddof=1)) if len(rows) > 1 else 0.,
        "mean_best_igd": float(np.mean([r["best_igd"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
    }
