#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable normal-state models for ABMG E3.

All models operate on one image-level evidence vector at a time.  The caller is
responsible for using the same frozen representation and update stream.

Implemented:
  deterministic_diag  running plug-in diagonal Gaussian
  nig                 diagonal Normal-Inverse-Gamma / Student-t predictive
  hierarchical_nig    same online NIG update with source-derived hierarchical prior
  robust_nig          NIG with bounded predictive-residual influence on updates

The robust model is a robust generalized-Bayes update, not an exact second
conjugate probabilistic model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm as scipy_norm
from scipy.stats import t as scipy_t


MODEL_NAMES = (
    "deterministic_diag",
    "nig",
    "hierarchical_nig",
    "robust_nig",
)


def _as_row(x: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if int(a.size) != int(dim):
        raise ValueError(f"expected dim={dim}, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError("state update contains non-finite values")
    return a


def _as_matrix(x: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or int(a.shape[1]) != int(dim):
        raise ValueError(f"expected [N,{dim}], got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("predictive input contains non-finite values")
    return a


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        n = max(float(np.linalg.norm(a)), float(eps))
        return a / n
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, float(eps))


def _safe_variance(x: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    ddof = 1 if a.shape[axis] > 1 else 0
    v = np.var(a, axis=axis, ddof=ddof)
    positive = v[v > 0]
    ref = float(np.median(positive)) if positive.size else 1.0
    floor = max(1e-10, ref * 1e-4)
    return np.maximum(v, floor)


@dataclass(frozen=True)
class NIGPrior:
    mu0: np.ndarray
    kappa0: np.ndarray
    alpha0: np.ndarray
    beta0: np.ndarray
    kind: str
    metadata: Dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.mu0.size)

    def validate(self) -> None:
        d = self.dim
        for name, arr in (
            ("kappa0", self.kappa0),
            ("alpha0", self.alpha0),
            ("beta0", self.beta0),
        ):
            if np.asarray(arr).shape != (d,):
                raise ValueError(f"{name} shape mismatch")
        if np.any(self.kappa0 <= 0):
            raise ValueError("kappa0 must be > 0")
        if np.any(self.alpha0 <= 1.0):
            raise ValueError("alpha0 must be > 1 for finite mean variance")
        if np.any(self.beta0 <= 0):
            raise ValueError("beta0 must be > 0")


def fit_weak_nig_prior(
    category_to_normal: Mapping[str, np.ndarray],
    *,
    kappa0: float = 0.01,
    alpha0: float = 2.5,
) -> NIGPrior:
    """Weak empirical-Bayes prior from development normal TRAIN vectors."""
    if float(kappa0) <= 0:
        raise ValueError("kappa0 must be > 0")
    if float(alpha0) <= 1.0:
        raise ValueError("alpha0 must be > 1")
    blocks = [
        np.asarray(v, dtype=np.float64)
        for _, v in sorted(category_to_normal.items())
        if len(v)
    ]
    if not blocks:
        raise ValueError("no development normal evidence for prior")
    x = np.concatenate(blocks, axis=0)
    mu = np.mean(x, axis=0)
    var = _safe_variance(x, axis=0)
    d = int(mu.size)
    kap = np.full(d, float(kappa0), dtype=np.float64)
    alp = np.full(d, float(alpha0), dtype=np.float64)
    beta = (alp - 1.0) * var
    prior = NIGPrior(
        mu0=mu,
        kappa0=kap,
        alpha0=alp,
        beta0=beta,
        kind="weak_global",
        metadata={
            "n_categories": len(blocks),
            "n_vectors": int(len(x)),
            "kappa0": float(kappa0),
            "alpha0": float(alpha0),
            "median_prior_variance": float(np.median(var)),
        },
    )
    prior.validate()
    return prior


def fit_hierarchical_nig_prior(
    category_to_normal: Mapping[str, np.ndarray],
    *,
    alpha0: float = 3.0,
    kappa_min: float = 0.05,
    kappa_max: float = 20.0,
) -> NIGPrior:
    """Category-balanced empirical-Bayes NIG hyperprior.

    The model uses the approximation

        Var(mu_category) ~= sigma_within^2 / kappa0

    after subtracting the finite-sample contribution of each category mean.
    """
    if float(alpha0) <= 1.0:
        raise ValueError("alpha0 must be > 1")
    cats = []
    means = []
    variances = []
    ns = []
    for cat, block in sorted(category_to_normal.items()):
        x = np.asarray(block, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            continue
        cats.append(str(cat))
        means.append(np.mean(x, axis=0))
        variances.append(_safe_variance(x, axis=0))
        ns.append(int(len(x)))
    if len(cats) < 2:
        raise ValueError("hierarchical prior needs >=2 development categories")

    means_a = np.stack(means, axis=0)
    vars_a = np.stack(variances, axis=0)
    ns_a = np.asarray(ns, dtype=np.float64)

    mu0 = means_a.mean(axis=0)
    within = np.mean(vars_a, axis=0)
    raw_between = np.var(means_a, axis=0, ddof=1)
    finite_sample = np.mean(
        vars_a / ns_a[:, None],
        axis=0,
    )
    between = np.maximum(raw_between - finite_sample, 1e-10)

    kappa = within / between
    kappa = np.clip(kappa, float(kappa_min), float(kappa_max))
    alpha = np.full_like(mu0, float(alpha0), dtype=np.float64)
    beta = (alpha - 1.0) * within

    prior = NIGPrior(
        mu0=mu0,
        kappa0=kappa,
        alpha0=alpha,
        beta0=np.maximum(beta, 1e-12),
        kind="hierarchical_empirical_bayes",
        metadata={
            "n_categories": int(len(cats)),
            "n_vectors": int(sum(ns)),
            "alpha0": float(alpha0),
            "kappa_median": float(np.median(kappa)),
            "kappa_q10": float(np.quantile(kappa, 0.10)),
            "kappa_q90": float(np.quantile(kappa, 0.90)),
            "median_within_variance": float(np.median(within)),
            "median_between_variance": float(np.median(between)),
        },
    )
    prior.validate()
    return prior


class NormalState:
    name = "base"

    def __init__(self, dim: int):
        self.dim = int(dim)

    def clone(self) -> "NormalState":
        return copy.deepcopy(self)

    def update(
        self,
        x: np.ndarray,
        weight: float = 1.0,
        *,
        robust: bool = True,
    ) -> float:
        raise NotImplementedError

    def predictive_log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predictive_interval_coverage(
        self,
        x: np.ndarray,
        *,
        level: float = 0.90,
    ) -> float:
        raise NotImplementedError

    def uncertainty(self) -> Dict[str, float]:
        raise NotImplementedError

    def state_summary(self) -> Dict[str, Any]:
        raise NotImplementedError

    def state_drift(self, reference: "NormalState") -> Dict[str, float]:
        raise NotImplementedError


class DeterministicDiagGaussian(NormalState):
    name = "deterministic_diag"

    def __init__(
        self,
        dim: int,
        *,
        fallback_mean: np.ndarray,
        fallback_variance: np.ndarray,
        variance_floor_ratio: float = 0.05,
    ):
        super().__init__(dim)
        self.n = 0.0
        self.mean = np.asarray(fallback_mean, dtype=np.float64).copy()
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.fallback_variance = np.asarray(
            fallback_variance, dtype=np.float64
        ).copy()
        self.floor = np.maximum(
            self.fallback_variance * float(variance_floor_ratio),
            1e-10,
        )
        self.update_count = 0

    def update(self, x, weight=1.0, *, robust=True) -> float:
        del robust
        w = float(weight)
        if w <= 0:
            return 0.0
        row = _as_row(x, self.dim)
        if self.n <= 0:
            self.mean = row.copy()
            self.n = w
            self.update_count += 1
            return w

        new_n = self.n + w
        delta = row - self.mean
        new_mean = self.mean + (w / new_n) * delta
        self.m2 += w * delta * (row - new_mean)
        self.mean = new_mean
        self.n = new_n
        self.update_count += 1
        return w

    def variance(self) -> np.ndarray:
        if self.n < 2.0:
            return self.fallback_variance.copy()
        v = self.m2 / max(self.n, 1.0)
        return np.maximum(v, self.floor)

    def predictive_log_prob(self, x) -> np.ndarray:
        a = _as_matrix(x, self.dim)
        var = self.variance()
        lp = -0.5 * (
            np.log(2.0 * np.pi * var)[None, :]
            + (a - self.mean[None, :]) ** 2 / var[None, :]
        )
        return np.mean(lp, axis=1)

    def predictive_interval_coverage(self, x, *, level=0.90) -> float:
        a = _as_matrix(x, self.dim)
        z = float(scipy_norm.ppf((1.0 + float(level)) / 2.0))
        sd = np.sqrt(self.variance())
        lo = self.mean - z * sd
        hi = self.mean + z * sd
        return float(np.mean((a >= lo[None, :]) & (a <= hi[None, :])))

    def uncertainty(self) -> Dict[str, float]:
        return {
            "epistemic_mean_variance": 0.0,
            "predictive_variance_mean": float(np.mean(self.variance())),
            "effective_n": float(self.n),
        }

    def state_summary(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "effective_n": float(self.n),
            "mean_norm": float(np.linalg.norm(self.mean)),
            "median_variance": float(np.median(self.variance())),
            "update_count": int(self.update_count),
        }

    def state_drift(self, reference: NormalState) -> Dict[str, float]:
        if not isinstance(reference, DeterministicDiagGaussian):
            raise TypeError("reference type mismatch")
        v0 = reference.variance()
        v1 = self.variance()
        return {
            "mean_l2": float(np.linalg.norm(self.mean - reference.mean)),
            "mean_per_coordinate_abs_shift": float(
                np.mean(np.abs(self.mean - reference.mean))
            ),
            "mean_abs_log_variance_ratio": float(
                np.mean(
                    np.abs(
                        np.log(np.maximum(v1, 1e-12))
                        - np.log(np.maximum(v0, 1e-12))
                    )
                )
            ),
        }


class DiagonalNIG(NormalState):
    name = "nig"

    def __init__(self, prior: NIGPrior):
        prior.validate()
        super().__init__(prior.dim)
        self.prior = prior
        self.mu = prior.mu0.astype(np.float64, copy=True)
        self.kappa = prior.kappa0.astype(np.float64, copy=True)
        self.alpha = prior.alpha0.astype(np.float64, copy=True)
        self.beta = prior.beta0.astype(np.float64, copy=True)
        self.effective_n = 0.0
        self.update_count = 0
        self.applied_weights: list[float] = []

    def _influence_weight(self, x: np.ndarray) -> float:
        del x
        return 1.0

    def update(self, x, weight=1.0, *, robust=True) -> float:
        row = _as_row(x, self.dim)
        external = float(weight)
        if external <= 0:
            return 0.0
        influence = self._influence_weight(row) if robust else 1.0
        w = external * float(influence)
        if w <= 0:
            return 0.0

        k0 = self.kappa.copy()
        mu0 = self.mu.copy()
        kn = k0 + w
        mun = (k0 * mu0 + w * row) / kn
        self.beta = self.beta + 0.5 * (k0 * w / kn) * (row - mu0) ** 2
        self.alpha = self.alpha + 0.5 * w
        self.kappa = kn
        self.mu = mun
        self.effective_n += w
        self.update_count += 1
        self.applied_weights.append(float(influence))
        return float(w)

    def predictive_scale2(self) -> np.ndarray:
        return (
            self.beta
            * (self.kappa + 1.0)
            / np.maximum(self.alpha * self.kappa, 1e-18)
        )

    def predictive_variance(self) -> np.ndarray:
        # Student-t variance for df=2*alpha.
        return (
            self.beta
            * (self.kappa + 1.0)
            / np.maximum(self.kappa * (self.alpha - 1.0), 1e-18)
        )

    def mean_parameter_variance(self) -> np.ndarray:
        return self.beta / np.maximum(
            self.kappa * (self.alpha - 1.0),
            1e-18,
        )

    def predictive_log_prob(self, x) -> np.ndarray:
        a = _as_matrix(x, self.dim)
        nu = 2.0 * self.alpha
        scale2 = np.maximum(self.predictive_scale2(), 1e-18)
        d = a - self.mu[None, :]
        lp = (
            gammaln((nu + 1.0) / 2.0)[None, :]
            - gammaln(nu / 2.0)[None, :]
            - 0.5 * np.log(nu * np.pi * scale2)[None, :]
            - ((nu + 1.0) / 2.0)[None, :]
            * np.log1p((d * d) / (nu * scale2)[None, :])
        )
        return np.mean(lp, axis=1)

    def predictive_interval_coverage(self, x, *, level=0.90) -> float:
        a = _as_matrix(x, self.dim)
        nu = 2.0 * self.alpha
        q = scipy_t.ppf((1.0 + float(level)) / 2.0, df=nu)
        half = q * np.sqrt(self.predictive_scale2())
        lo = self.mu - half
        hi = self.mu + half
        return float(np.mean((a >= lo[None, :]) & (a <= hi[None, :])))

    def uncertainty(self) -> Dict[str, float]:
        return {
            "epistemic_mean_variance": float(
                np.mean(self.mean_parameter_variance())
            ),
            "predictive_variance_mean": float(
                np.mean(self.predictive_variance())
            ),
            "median_df": float(np.median(2.0 * self.alpha)),
            "effective_n": float(self.effective_n),
        }

    def state_summary(self) -> Dict[str, Any]:
        out = {
            "model": self.name,
            "prior_kind": self.prior.kind,
            "effective_n": float(self.effective_n),
            "mean_norm": float(np.linalg.norm(self.mu)),
            "median_expected_variance": float(
                np.median(self.beta / np.maximum(self.alpha - 1.0, 1e-18))
            ),
            "median_kappa": float(np.median(self.kappa)),
            "median_alpha": float(np.median(self.alpha)),
            "update_count": int(self.update_count),
        }
        if self.applied_weights:
            out["mean_influence_weight"] = float(
                np.mean(self.applied_weights)
            )
            out["min_influence_weight"] = float(
                np.min(self.applied_weights)
            )
        return out

    def state_drift(self, reference: NormalState) -> Dict[str, float]:
        if not isinstance(reference, DiagonalNIG):
            raise TypeError("reference type mismatch")
        v0 = reference.beta / np.maximum(reference.alpha - 1.0, 1e-18)
        v1 = self.beta / np.maximum(self.alpha - 1.0, 1e-18)
        return {
            "mean_l2": float(np.linalg.norm(self.mu - reference.mu)),
            "mean_per_coordinate_abs_shift": float(
                np.mean(np.abs(self.mu - reference.mu))
            ),
            "mean_abs_log_variance_ratio": float(
                np.mean(
                    np.abs(
                        np.log(np.maximum(v1, 1e-18))
                        - np.log(np.maximum(v0, 1e-18))
                    )
                )
            ),
            "mean_abs_log_kappa_ratio": float(
                np.mean(
                    np.abs(
                        np.log(np.maximum(self.kappa, 1e-18))
                        - np.log(np.maximum(reference.kappa, 1e-18))
                    )
                )
            ),
        }


class HierarchicalDiagonalNIG(DiagonalNIG):
    name = "hierarchical_nig"


class RobustDiagonalNIG(DiagonalNIG):
    name = "robust_nig"

    def __init__(
        self,
        prior: NIGPrior,
        *,
        robust_df: float = 4.0,
        min_influence: float = 0.05,
    ):
        super().__init__(prior)
        if float(robust_df) <= 0:
            raise ValueError("robust_df must be > 0")
        if not (0 < float(min_influence) <= 1.0):
            raise ValueError("min_influence must be in (0,1]")
        self.robust_df = float(robust_df)
        self.min_influence = float(min_influence)
        self.clean_update_weights: list[float] = []
        self.contaminated_update_weights: list[float] = []

    def _influence_weight(self, x: np.ndarray) -> float:
        pv = np.maximum(self.predictive_variance(), 1e-18)
        delta2 = float(np.mean((x - self.mu) ** 2 / pv))
        w = (self.robust_df + 1.0) / (self.robust_df + delta2)
        return float(np.clip(w, self.min_influence, 1.0))

    def record_last_weight(self, *, contaminated: bool) -> None:
        if not self.applied_weights:
            return
        w = float(self.applied_weights[-1])
        if contaminated:
            self.contaminated_update_weights.append(w)
        else:
            self.clean_update_weights.append(w)

    def state_summary(self) -> Dict[str, Any]:
        out = super().state_summary()
        out["model"] = self.name
        out["robust_df"] = float(self.robust_df)
        out["min_influence"] = float(self.min_influence)
        out["mean_clean_influence"] = (
            float(np.mean(self.clean_update_weights))
            if self.clean_update_weights
            else float("nan")
        )
        out["mean_contaminated_influence"] = (
            float(np.mean(self.contaminated_update_weights))
            if self.contaminated_update_weights
            else float("nan")
        )
        return out


def expected_prior_variance(prior: NIGPrior) -> np.ndarray:
    return prior.beta0 / np.maximum(prior.alpha0 - 1.0, 1e-18)


def build_normal_state(
    name: str,
    *,
    weak_prior: NIGPrior,
    hierarchical_prior: NIGPrior,
    deterministic_floor_ratio: float = 0.05,
    robust_df: float = 4.0,
    robust_min_influence: float = 0.05,
) -> NormalState:
    name = str(name)
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown normal-state model={name!r}")
    if weak_prior.dim != hierarchical_prior.dim:
        raise ValueError("prior dimension mismatch")

    if name == "deterministic_diag":
        return DeterministicDiagGaussian(
            weak_prior.dim,
            fallback_mean=weak_prior.mu0,
            fallback_variance=expected_prior_variance(weak_prior),
            variance_floor_ratio=float(deterministic_floor_ratio),
        )
    if name == "nig":
        return DiagonalNIG(weak_prior)
    if name == "hierarchical_nig":
        return HierarchicalDiagonalNIG(hierarchical_prior)
    if name == "robust_nig":
        return RobustDiagonalNIG(
            weak_prior,
            robust_df=float(robust_df),
            min_influence=float(robust_min_influence),
        )
    raise AssertionError("unreachable")
