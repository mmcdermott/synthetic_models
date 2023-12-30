import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Callable, Sequence

def get_pairwise_probs(
    subgroup_sizes, subgroup_AUROCs, overall_AUROC, subgroup_prevalences, subgroup_pos_distributions,
    subgroup_discrimination_AUROC,
):
    """This function computes the pairwise ordering probabilities between subgroups X labels.
    """

    n_subgroups = len(subgroup_sizes)
    n_labels = 2

    if n_subgroups != 2:
        raise ValueError(f"Only two subgroups are supported; got {n_subgroups}")

    # out will be indexed such that out[i, yi, j, yj] is the probability that a random sample
    # from subgroup i with label yi will have a higher score than a random sample from subgroup
    # j with label yj.

    out = np.zeros((n_subgroups, n_labels, n_subgroups, n_labels))
    for i in range(n_subgroups):
        for y in range(n_labels):
            # On-diagonal entries are 0.5, since this is comparing a group to itself.
            out[i, y, i, y] = 0.5

        # Within each subgroup, we can also compare labels against one another via subgroup AUROCs.
        out[i, 1, i, 0] = subgroup_AUROCs[i]
        out[i, 0, i, 1] = 1-subgroup_AUROCs[i]

    # Postive - positive comparisons can be computed empirically, as we have the subgroup_pos_distributions.
    for i in range(n_subgroups):
        for j in range(n_subgroups):
            if j == i:
                continue
            out[i, 1, j, 1] = roc_auc_score(
                np.concatenate((np.ones(subgroup_sizes[i]), np.zeros(subgroup_sizes[j]))),
                np.concatenate((subgroup_pos_distributions[i], subgroup_pos_distributions[j]))
            )

    # Cross subgroup-label entries require us to leverage additional information. The overall AUROC
    # will give us part of the picture and the subgroup_discrimination_AUROC will give us the rest.

    raise NotImplementedError


def _validate_subgroup_inputs(
    N: int | None,
    subgroup_sizes: Sequence[int | float],
    subgroup_AUROCs: Sequence[float],
    subgroup_prevalences: Sequence[float],
    subgroup_pos_distributions: Sequence[str | Callable[[int], np.ndarray]],
    subgroup_calibrations: Sequence[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:

    # Check lengths match
    L = len(subgroup_sizes)
    if len(subgroup_AUROCs) != L:
        raise ValueError(f"subgroup_AUROCs must have length {L}; got {len(subgroup_AUROCs)}")
    if len(subgroup_prevalences) != L:
        raise ValueError(f"subgroup_prevalences must have length {L}; got {len(subgroup_prevalences)}")
    if len(subgroup_pos_distributions) != L:
        raise ValueError(
            f"subgroup_pos_distributions must have length {L}; got {len(subgroup_pos_distributions)}"
        )
    if len(subgroup_calibrations) != L:
        raise ValueError(f"subgroup_calibrations must have length {L}; got {len(subgroup_calibrations)}")

    # Check subgroup sizes
    if N is not None:
        if type(N) is not int:
            raise TypeError(f"N must be an int or None; got {type(N)}")
        elif N <= 0:
            raise ValueError(f"N must be positive; got {N}")

    subgroup_sizes_out = []
    for subgroup_size in subgroup_sizes:
        match subgroup_size:
            case int() if subgroup_size <= 0:
                raise ValueError(f"subgroup_sizes must be positive; got {subgroup_size}")
            case int() if N is not None and subgroup_size >= N:
                raise ValueError(f"subgroup_sizes must be no more than N if specified; got {subgroup_size}")
            case int():
                subgroup_sizes_out.append(subgroup_size)
            case float() as subgroup_prev if subgroup_prev <= 0 or subgroup_prev >= 1:
                raise ValueError(f"subgroup_sizes must be in (0, 1) if float; got {subgroup_prev}")
            case float() as subgroup_prev if N is None:
                raise ValueError(
                    f"subgroup_sizes must be all ints if N is None; got {subgroup_prev}"
                )
            case float() as subgroup_prev:
                subgroup_sizes_out.append(int(round(subgroup_prev * N)))
            case _:
                raise TypeError(
                    f"subgroup_sizes must be int or float; got {type(subgroup_size)}({subgroup_size})"
                )
    if N is None:
        N = sum(subgroup_sizes_out)
    elif sum(subgroup_sizes_out) != N:
        raise ValueError(f"subgroup_sizes must sum to N={N}; got {sum(subgroup_sizes_out)}")

    subgroup_sizes_out = np.array(subgroup_sizes_out)

    raise NotImplementedError

def sample_at_AUROC_subgroups(
    N: int | None,
    subgroup_sizes: Sequence[int | float],
    subgroup_AUROCs: Sequence[float],
    subgroup_prevalences: Sequence[float],
    subgroup_pos_distributions: Sequence[str | Callable[[int], np.ndarray]],
    subgroup_calibrations: Sequence[float],
    overall_AUROC: float,
    overall_calibration: float,
    subgroup_discrimination_AUROC: float,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function samples probabilities and binary labels at the given constraints.
    """

    if len(subgroup_sizes) != 2: raise ValueError("Only two subgroups are supported")

    rng = np.random.default_rng(seed)

    (
        subgroup_sizes, subgroup_AUROCs, subgroup_prevalences, subgroup_pos_distributions,
        subgroup_calibrations
    ) = _validate_subgroup_inputs(
        N, subgroup_sizes, subgroup_AUROCs, subgroup_prevalences, subgroup_pos_distributions,
        subgroup_calibrations, rng
    )

    pairwise_probs = get_pairwise_probs(
        subgroup_AUROCs, overall_AUROC, subgroup_prevalences, subgroup_pos_distributions,
        subgroup_discrimination_AUROC
    )

    # pairwise_probs will be indexed such that pairwise_probs[i, yi, j, yj] is the probability that a random
    # sample from subgroup i with label yi will have a higher score than a random sample from subgroup j with
    # label yj.


    # Calibration is
    # P(y = 1 | p = p)
    # = P(p_+ = p) * P(y=1)/(P(p_+ = p)*P(y=1) + P(p_- = p)*(1-P(y=1)))
    # Of these terms, I know P(p_+ = p), P(y=1).
    #
    # So, if P(y=1 | p = p) = p + \varepsilon (\varepsilon \in \N(0, \sigma)), then
    # P(p_- = p) = (P(p_+ = p) * P(y=1))/(p + \varepsilon) - P(p_+ = p) * P(y=1)
    #            = (P(p_+ = p) * P(y=1)) * (1/(p + \varepsilon) - 1)

    raise NotImplementedError
