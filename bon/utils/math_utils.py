from typing import Sequence

import numpy as np
import scipy.special
import sklearn.metrics


def logprobs_to_logodds(logprobs: np.ndarray, base: float | None = None) -> np.ndarray:
    """
    Converts logprobs to logodds.
    Assumes logprobs are in log base e.
    If base is None, the natural logarithm is used.
    """
    logodds = logprobs - np.log(-scipy.special.expm1(logprobs))
    if base is not None:
        logodds /= np.log(base)
    return logodds


def probs_to_logodds(probs: np.ndarray, base: float | None = None) -> np.ndarray:
    """
    Converts logprobs to logodds.
    If base is None, the natural logarithm is used.
    """
    logprobs = np.log(probs)
    return logprobs_to_logodds(logprobs, base=base)


def logodds_to_probs(logodds: np.ndarray, base: float | None = None) -> np.ndarray:
    """
    Converts logodds to logprobs.
    If base is None, the natural logarithm is used.
    """
    if base is None:
        return scipy.special.expit(logodds)

    return scipy.special.expit(logodds * np.log(base))


def logsumexp(logprobs: Sequence[float]) -> float:
    if len(logprobs) == 0:
        return -np.inf
    return scipy.special.logsumexp(logprobs)


def roc_curve_with_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        y_true=y_true,
        y_score=y_score,
    )
    auc = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, auc


def two_set_roc(benign_scores: np.ndarray, adv_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_score = np.concatenate([benign_scores, adv_scores])
    y_true = np.concatenate(
        [
            np.zeros_like(benign_scores, dtype=bool),
            np.ones_like(adv_scores, dtype=bool),
        ]
    )

    return roc_curve_with_auc(y_true, y_score)
