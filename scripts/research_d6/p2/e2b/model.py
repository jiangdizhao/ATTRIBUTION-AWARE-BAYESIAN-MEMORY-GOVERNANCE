"""Feature-only linear and patch-MIL learners for LP-D6-P2-E2-B."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    nonnegative = x >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    ex = np.exp(x[~nonnegative])
    out[~nonnegative] = ex / (1.0 + ex)
    return out


def binary_labels(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("binary learner accepts only one-dimensional labels in {0,1}")
    if len(values) == 0 or len(np.unique(values)) != 2:
        raise ValueError("training history must contain both binary classes")
    return values.astype(np.float64, copy=False)


def parameter_digest(theta: np.ndarray) -> str:
    value = np.ascontiguousarray(theta, dtype=np.float64)
    h = hashlib.sha256()
    h.update(repr(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def global_objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                     regularization: float) -> tuple[float, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    labels = binary_labels(y)
    params = np.asarray(theta, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(labels) or params.shape != (features.shape[1] + 1,):
        raise ValueError("global objective shape mismatch")
    weights, bias = params[:-1], params[-1]
    logits = features @ weights + bias
    loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits)
                 + 0.5 * regularization * (weights @ weights))
    residual = sigmoid(logits) - labels
    grad = np.empty_like(params)
    grad[:-1] = (features.T @ residual) / len(labels) + regularization * weights
    grad[-1] = float(np.mean(residual))
    return loss, grad


def build_global_readout(readout: str, cls: np.ndarray, patches: np.ndarray,
                         valid_mask: np.ndarray) -> np.ndarray:
    cls_vec = np.asarray(cls, dtype=np.float64).reshape(-1)
    patch = np.asarray(patches, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    if patch.ndim != 2 or patch.shape[1] != len(cls_vec) or len(mask) != len(patch):
        raise ValueError("readout feature/mask shape mismatch")
    if not mask.any():
        raise ValueError("image bag has no valid patch")
    selected = patch[mask]
    mean = np.mean(selected, axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise FloatingPointError("valid patch mean has zero or nonfinite norm")
    mean = mean / norm
    if readout == "G-CLS":
        result = cls_vec
    elif readout == "G-Mean":
        result = mean
    elif readout == "G-Concat":
        result = np.concatenate([cls_vec, mean])
        norm = float(np.linalg.norm(result))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise FloatingPointError("concatenated global feature has zero or nonfinite norm")
        result = result / norm
    else:
        raise ValueError(f"unknown global readout: {readout}")
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite global readout")
    return np.asarray(result, dtype=np.float64)


def mil_initial_parameters(n_features: int, seed: int) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    rng = np.random.default_rng(int(seed))
    weights = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=n_features)
    return np.concatenate([weights.astype(np.float64), np.asarray([0.0], dtype=np.float64)])


def valid_bag(patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(patches)
    valid = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.ndim != 2 or len(valid) != len(values) or not valid.any():
        raise ValueError("bag must have a nonempty valid patch mask matching its patch matrix")
    chosen = np.asarray(values[valid], dtype=np.float64)
    if not np.isfinite(chosen).all():
        raise FloatingPointError("nonfinite valid patch feature")
    return chosen


def bag_scores_and_means(bags: Sequence[np.ndarray], theta: np.ndarray,
                         tau: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if tau <= 0 or not bags:
        raise ValueError("MIL requires positive temperature and at least one bag")
    params = np.asarray(theta, dtype=np.float64)
    dimension = len(params) - 1
    weights, bias = params[:-1], params[-1]
    scores = np.empty(len(bags), dtype=np.float64)
    means = np.empty((len(bags), dimension), dtype=np.float64)
    for i, raw_bag in enumerate(bags):
        z = np.asarray(raw_bag, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != dimension or len(z) == 0:
            raise ValueError("MIL bag shape mismatch or empty valid bag")
        logits = z @ weights + bias
        scaled = logits / tau
        maximum = float(np.max(scaled))
        exp_scaled = np.exp(scaled - maximum)
        denominator = float(np.sum(exp_scaled))
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("invalid MIL softmax denominator")
        probs = exp_scaled / denominator
        scores[i] = tau * (maximum + np.log(denominator) - np.log(len(z)))
        means[i] = probs @ z
    if not np.isfinite(scores).all() or not np.isfinite(means).all():
        raise FloatingPointError("nonfinite MIL bag score or gradient feature")
    return scores, means


def pair_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = binary_labels(y)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    pos = np.repeat(positives, len(negatives))
    neg = np.tile(negatives, len(positives))
    return pos, neg


@dataclass(frozen=True)
class MILLossParts:
    total: float
    bce: float
    l2: float
    rank: float
    pair_count: int
    gradient: np.ndarray


def mil_objective_parts(theta: np.ndarray, bags: Sequence[np.ndarray], y: np.ndarray,
                        regularization: float, *, rho: float = 0.0,
                        tau: float = 1.0, margin: float = 1.0,
                        kappa: float = 0.1) -> MILLossParts:
    labels = binary_labels(y)
    if len(bags) != len(labels):
        raise ValueError("MIL bag/label count mismatch")
    if rho < 0 or regularization < 0 or kappa <= 0 or tau <= 0:
        raise ValueError("invalid MIL loss scalar")
    params = np.asarray(theta, dtype=np.float64)
    weights = params[:-1]
    scores, score_means = bag_scores_and_means(bags, params, tau=tau)
    residual = sigmoid(scores) - labels
    bce = float(np.mean(np.logaddexp(0.0, scores) - labels * scores))
    l2 = float(0.5 * regularization * (weights @ weights))
    gradient = np.empty_like(params)
    gradient[:-1] = (residual @ score_means) / len(labels) + regularization * weights
    gradient[-1] = float(np.mean(residual))

    rank = 0.0
    pair_count = 0
    if rho > 0.0:
        pos_idx, neg_idx = pair_indices(labels)
        pair_count = len(pos_idx)
        violations = margin - scores[pos_idx] + scores[neg_idx]
        pair_losses = kappa * np.logaddexp(0.0, violations / kappa)
        pair_derivatives = sigmoid(violations / kappa)
        rank = float(np.mean(pair_losses))
        rank_gradient_w = np.mean(
            pair_derivatives[:, None] * (-score_means[pos_idx] + score_means[neg_idx]), axis=0
        )
        gradient[:-1] += rho * rank_gradient_w
        # The common bias cancels exactly in every positive-negative score difference.
    total = bce + l2 + rho * rank
    if not np.isfinite(total) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite MIL objective or gradient")
    return MILLossParts(total, bce, l2, rank, pair_count, gradient)


def mil_objective(theta: np.ndarray, bags: Sequence[np.ndarray], y: np.ndarray,
                  regularization: float, *, rho: float = 0.0,
                  tau: float = 1.0, margin: float = 1.0,
                  kappa: float = 0.1) -> tuple[float, np.ndarray]:
    parts = mil_objective_parts(theta, bags, y, regularization, rho=rho,
                                tau=tau, margin=margin, kappa=kappa)
    return parts.total, parts.gradient


def optimize_lbfgsb(objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
                    initial: np.ndarray, *, options: dict[str, float | int]) -> dict:
    start = time.perf_counter()
    result = minimize(objective, np.asarray(initial, dtype=np.float64), method="L-BFGS-B",
                      jac=True, options={k: v for k, v in options.items()
                                         if k in {"ftol", "gtol", "maxiter", "maxls", "maxcor"}})
    elapsed = time.perf_counter() - start
    theta = np.asarray(result.x, dtype=np.float64)
    value, gradient = objective(theta)
    grad_inf = float(np.max(np.abs(gradient)))
    finite = bool(np.isfinite(theta).all() and np.isfinite(value) and np.isfinite(gradient).all())
    gradient_ok = finite and grad_inf <= float(options["accept_gradient_inf"])
    accepted = bool(result.success and gradient_ok)
    message = str(result.message)
    if accepted:
        status = "PASS"
    elif finite and result.success and not gradient_ok:
        status = "FTOL_ONLY_NOT_CONVERGED"
    elif not finite:
        status = "NONFINITE_FAILURE"
    else:
        status = "SOLVER_FAILURE"
    return {
        "theta": theta,
        "accepted": accepted,
        "status": status,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "message": message,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(value),
        "gradient_inf_norm": grad_inf,
        "finite": finite,
        "elapsed_seconds": elapsed,
    }


def fit_global(x: np.ndarray, y: np.ndarray, initial: np.ndarray | None,
               regularization: float, options: dict) -> dict:
    features = np.asarray(x, dtype=np.float64)
    labels = binary_labels(y)
    if len(features) != len(labels):
        raise ValueError("global feature/label count mismatch")
    start = np.zeros(features.shape[1] + 1, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
    return optimize_lbfgsb(lambda t: global_objective(t, features, labels, regularization), start, options=options)


def fit_mil(bags: Sequence[np.ndarray], y: np.ndarray, initial: np.ndarray | None,
            regularization: float, *, rho: float, tau: float, margin: float,
            kappa: float, options: dict) -> dict:
    labels = binary_labels(y)
    if len(bags) != len(labels):
        raise ValueError("MIL bag/label count mismatch")
    dimension = int(np.asarray(bags[0]).shape[1])
    start = np.zeros(dimension + 1, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
    return optimize_lbfgsb(
        lambda t: mil_objective(t, bags, labels, regularization,
                                rho=rho, tau=tau, margin=margin, kappa=kappa),
        start, options=options,
    )


def predict_global(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    params = np.asarray(theta, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) @ params[:-1] + params[-1]


def predict_mil(theta: np.ndarray, bags: Sequence[np.ndarray], tau: float) -> np.ndarray:
    return bag_scores_and_means(bags, theta, tau=tau)[0]

