import numpy as np
from scipy.stats import binom


def sample_at_AUROC(
    N: int, target_AUROC: float, pos_prev: float, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """This function samples probabilities and binary labels at the given AUROC.

    Args:
        N: The number of samples to return
        target_AUROC: The desired AUROC the resulting probabilities and labels
            should display.
        pos_prev: The prevalence of the positive label.
        seed: The random seed that should be used.
    """

    if N <= 0:
        raise ValueError(f"N must be a positive int; got {N}")
    if target_AUROC < 0 or target_AUROC > 1:
        raise ValueError(f"target_AUROC must be in [0, 1]; got {target_AUROC}")
    if pos_prev <= 0 or pos_prev >= 1:
        raise ValueError(f"pos_prev must be in (0, 1); got {pos_prev}")

    rng = np.random.default_rng(seed)

    N_true = int(round(pos_prev * N))
    N_false = N - N_true

    if N_true == 0 or N_false == 0:
        raise ValueError(f"pos_prev is too extreme; got {N_true} and {N_false}")

    pos_scores = list(sorted(rng.uniform(low=0, high=1, size=N_true)))

    bounds = [0] + pos_scores
    neg_score_window_probs = []
    for n_pos_lt, lb in enumerate(bounds):
        # probability that a negative sample is > n_pos_lt is (1-AUC)**n_pos_lt
        neg_score_window_probs.append(binom.pmf(n_pos_lt, N_true, 1 - target_AUROC))

    neg_scores = []
    for _ in range(N_false):
        window = rng.choice(len(bounds), p=neg_score_window_probs, size=1).item()
        window = int(window)
        lb = bounds[window]
        ub = (bounds + [1])[window + 1]
        neg_scores.append(rng.uniform(low=lb, high=ub, size=1).item())

    scores = np.concatenate((pos_scores, neg_scores))
    Ys = np.array([True] * N_true + [False] * N_false)

    return scores, Ys
