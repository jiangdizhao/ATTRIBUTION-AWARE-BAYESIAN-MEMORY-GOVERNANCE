"""Deterministic float64 full-history proximal logistic learner for LP-D6-P2-E1."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


LAMBDA = 1e-4
GAMMAS = (1e-4, 1e-3)
LBFGSB_OPTIONS = {
    "ftol": 1e-12,
    "gtol": 1e-8,
    "maxiter": 10000,
    "maxls": 50,
    "maxcor": 10,
}
GRAD_INF_TOL = 1e-6


class SolverFailure(RuntimeError):
    """A registered optimization configuration did not meet its fixed contract."""

    def __init__(self, status: str, message: str, *, result: Any | None = None):
        super().__init__(message)
        self.status = status
        self.result = result


def _unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("theta must contain at least one weight and an intercept")
    return values[:-1], float(values[-1])


def objective_and_gradient(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    *,
    lam: float = LAMBDA,
    gamma: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Mean binary log loss + L2(w) + proximal L2(w-w0); intercept is free."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    anchor = np.asarray(w0, dtype=np.float64).reshape(-1)
    w, b = _unpack(theta)
    if x64.ndim != 2 or x64.shape != (y64.size, w.size) or anchor.shape != w.shape:
        raise ValueError("incompatible x, y, theta, or anchor dimensions")
    if y64.size == 0 or not np.isin(y64, (0.0, 1.0)).all():
        raise ValueError("objective requires nonempty binary labels")
    if lam < 0.0 or gamma < 0.0:
        raise ValueError("regularization strengths must be nonnegative")
    scores = x64 @ w + b
    residual = expit(scores) - y64
    delta = w - anchor
    value = float(
        np.mean(np.logaddexp(0.0, scores) - y64 * scores)
        + 0.5 * lam * np.dot(w, w)
        + 0.5 * gamma * np.dot(delta, delta)
    )
    grad_w = (x64.T @ residual) / y64.size + lam * w + gamma * delta
    grad_b = float(np.mean(residual))
    gradient = np.concatenate([grad_w, np.asarray([grad_b], dtype=np.float64)])
    return value, gradient


def hessian_vector_product(
    theta: np.ndarray,
    vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = LAMBDA,
    gamma: float = 0.0,
) -> np.ndarray:
    """Analytical Hessian-vector product for independent Newton-CG verification."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    w, b = _unpack(theta)
    vw, vb = _unpack(vector)
    if x64.shape != (y64.size, w.size) or vw.shape != w.shape:
        raise ValueError("incompatible Hessian-vector product dimensions")
    p = expit(x64 @ w + b)
    weighted = p * (1.0 - p) * (x64 @ vw + vb) / y64.size
    hw = x64.T @ weighted + (lam + gamma) * vw
    hb = float(np.sum(weighted))
    return np.concatenate([hw, np.asarray([hb], dtype=np.float64)])


@dataclass(frozen=True)
class FitDiagnostics:
    elapsed_seconds: float
    objective: float
    gradient_inf_norm: float
    iterations: int
    function_evaluations: int
    solver_message: str
    status: str


def fit_lbfgsb(
    x: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    start_theta: np.ndarray,
    *,
    lam: float = LAMBDA,
    gamma: float,
    minimize_fn: Callable[..., Any] = minimize,
) -> tuple[np.ndarray, FitDiagnostics]:
    """Run the frozen L-BFGS-B contract; never relax a failed condition."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    anchor = np.asarray(w0, dtype=np.float64).reshape(-1)
    theta0 = np.asarray(start_theta, dtype=np.float64).reshape(-1)
    start = time.perf_counter()
    result = minimize_fn(
        lambda z, xx, yy, ww: objective_and_gradient(
            z, xx, yy, ww, lam=lam, gamma=gamma
        ),
        theta0,
        args=(x64, y64, anchor),
        method="L-BFGS-B",
        jac=True,
        options=dict(LBFGSB_OPTIONS),
    )
    elapsed = time.perf_counter() - start
    theta = np.asarray(result.x, dtype=np.float64).reshape(-1)
    value, gradient = objective_and_gradient(theta, x64, y64, anchor, lam=lam, gamma=gamma)
    grad_inf = float(np.max(np.abs(gradient)))
    message = str(result.message)
    ftol_only = grad_inf > GRAD_INF_TOL and (
        "RELATIVE REDUCTION" in message.upper() or bool(result.success)
    )
    if ftol_only:
        raise SolverFailure(
            "FTOL_ONLY_NOT_CONVERGED",
            f"L-BFGS-B met function tolerance but gradient infinity norm {grad_inf:.6g} exceeds {GRAD_INF_TOL:g}",
            result=result,
        )
    if not bool(result.success):
        raise SolverFailure("SOLVER_FAILED", f"L-BFGS-B failed: {message}", result=result)
    if not np.isfinite(theta).all() or not np.isfinite(value) or not np.isfinite(gradient).all():
        raise SolverFailure("NONFINITE_SOLUTION", "L-BFGS-B returned nonfinite parameters/objective/gradient", result=result)
    if grad_inf > GRAD_INF_TOL:
        raise SolverFailure(
            "GRADIENT_TOLERANCE_FAILED",
            f"gradient infinity norm {grad_inf:.6g} exceeds {GRAD_INF_TOL:g}",
            result=result,
        )
    diag = FitDiagnostics(
        elapsed_seconds=elapsed,
        objective=value,
        gradient_inf_norm=grad_inf,
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        solver_message=message,
        status="PASS",
    )
    return theta.copy(), diag


def _array_digest(*arrays: np.ndarray, extra: bytes = b"") -> str:
    """Match P1-B's dtype/shape/bytes digest convention."""
    h = hashlib.sha256(extra)
    for arr in arrays:
        value = np.ascontiguousarray(arr)
        h.update(str(value.dtype).encode("ascii"))
        h.update(repr(value.shape).encode("ascii"))
        h.update(value.tobytes())
    return h.hexdigest()


class ProximalLogistic:
    """Full-history binary Logistic with immutable source anchor and proximal term."""

    def __init__(
        self,
        base_x: np.ndarray,
        base_y: np.ndarray,
        w0: np.ndarray,
        b0: float,
        *,
        gamma: float,
    ) -> None:
        if gamma not in (0.0, *GAMMAS):
            raise ValueError(f"unregistered gamma: {gamma}")
        self.base_x = np.asarray(base_x, dtype=np.float32).copy()
        self.base_y = np.asarray(base_y, dtype=np.int8).reshape(-1).copy()
        if self.base_x.ndim != 2 or len(self.base_y) != len(self.base_x):
            raise ValueError("base feature/label shape mismatch")
        if not np.isin(self.base_y, (0, 1)).all() or len(np.unique(self.base_y)) != 2:
            raise ValueError("base history must contain both binary classes")
        anchor = np.asarray(w0, dtype=np.float64).reshape(-1).copy()
        if anchor.size != self.base_x.shape[1]:
            raise ValueError("anchor dimension differs from the feature dimension")
        anchor.setflags(write=False)
        self.anchor_w = anchor
        self.anchor_b = float(b0)
        self.gamma = float(gamma)
        self.w = anchor.copy()
        self.b = float(b0)
        self.feedback_x: list[np.ndarray] = []
        self.feedback_y: list[int] = []
        self.event_keys: list[str] = []
        self.optimization_count = 0
        self.last_fit: FitDiagnostics | None = None

    def scores(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.w.size:
            raise ValueError("prediction feature dimension mismatch")
        return values @ self.w + self.b

    def training_history(self) -> tuple[np.ndarray, np.ndarray]:
        if self.feedback_x:
            x = np.vstack([self.base_x.astype(np.float64), np.stack(self.feedback_x).astype(np.float64)])
            y = np.concatenate([self.base_y.astype(np.int8), np.asarray(self.feedback_y, dtype=np.int8)])
        else:
            x = self.base_x.astype(np.float64)
            y = self.base_y.astype(np.int8)
        return x, y

    def objective_gradient(self) -> tuple[float, float]:
        x, y = self.training_history()
        theta = np.concatenate([self.w, np.asarray([self.b], dtype=np.float64)])
        value, gradient = objective_and_gradient(
            theta, x, y, self.anchor_w, lam=LAMBDA, gamma=self.gamma
        )
        return value, float(np.max(np.abs(gradient)))

    def observe(self, x: np.ndarray, y: int, event_key: str) -> FitDiagnostics:
        if y not in (0, 1):
            raise ValueError("only a revealed binary label may update the learner")
        if event_key in self.event_keys:
            raise ValueError("a feedback event may be used only once")
        candidate_x = np.asarray(x, dtype=np.float32).reshape(-1)
        if candidate_x.size != self.w.size or not np.isfinite(candidate_x).all():
            raise ValueError("invalid feedback feature")
        next_feedback_x = self.feedback_x + [candidate_x.copy()]
        next_feedback_y = self.feedback_y + [int(y)]
        next_X = np.vstack([self.base_x.astype(np.float64), np.stack(next_feedback_x).astype(np.float64)])
        next_y = np.concatenate([self.base_y.astype(np.int8), np.asarray(next_feedback_y, dtype=np.int8)])
        start_theta = np.concatenate([self.w, np.asarray([self.b], dtype=np.float64)])
        fitted, diagnostics = fit_lbfgsb(
            next_X,
            next_y,
            self.anchor_w,
            start_theta,
            lam=LAMBDA,
            gamma=self.gamma,
        )
        # Commit the new history/state only after a fully successful solve.
        self.w = fitted[:-1].copy()
        self.b = float(fitted[-1])
        self.feedback_x = next_feedback_x
        self.feedback_y = next_feedback_y
        self.event_keys.append(str(event_key))
        self.optimization_count += 1
        self.last_fit = diagnostics
        return diagnostics

    def state_digest(self) -> str:
        coef = self.w.reshape(1, -1).astype(np.float64, copy=False)
        intercept = np.asarray([self.b], dtype=np.float64)
        arrays: list[np.ndarray] = [coef, intercept]
        extra = b""
        if self.feedback_x:
            arrays.extend([
                np.stack(self.feedback_x).astype(np.float32),
                np.asarray(self.feedback_y, dtype=np.int8),
            ])
            extra = b"\0".join(k.encode("utf-8") for k in self.event_keys)
        return _array_digest(*arrays, extra=extra)

    def state_payload_bytes(self) -> int:
        return int(
            self.w.reshape(1, -1).nbytes
            + np.asarray([self.b], dtype=np.float64).nbytes
            + self.base_x.nbytes
            + self.base_y.nbytes
            + sum(x.nbytes + 1 + len(k.encode("utf-8"))
                  for x, k in zip(self.feedback_x, self.event_keys))
        )

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.w.reshape(1, -1).astype(np.float64, copy=True),
            "intercept": np.asarray([self.b], dtype=np.float64),
            "anchor_coef": self.anchor_w.copy(),
            "anchor_intercept": np.asarray([self.anchor_b], dtype=np.float64),
            "feedback_x": (np.stack(self.feedback_x).astype(np.float32)
                           if self.feedback_x else np.empty((0, self.w.size), dtype=np.float32)),
            "feedback_y": np.asarray(self.feedback_y, dtype=np.int8),
            "feedback_event_keys_utf8_hex": np.asarray(
                [k.encode("utf-8").hex() for k in self.event_keys], dtype="U"
            ),
        }

