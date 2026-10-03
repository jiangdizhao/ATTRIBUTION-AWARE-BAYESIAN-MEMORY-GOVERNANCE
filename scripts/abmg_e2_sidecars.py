#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Address-sidecar models for ABMG E2.

The frozen DINO feature remains the detector representation.  These sidecars
exist only to produce an address code for factor routing / later memory credit.

Candidates:
  raw              identity baseline
  pca              linear PCA projection
  ica              linear ICA projection
  beta_tcvae       VAE + explicit minibatch total-correlation penalty
  factor_vae       VAE + density-ratio total-correlation discriminator
  corrvae          grouped variational address code:
                   * within-group correlation is allowed;
                   * cross-group covariance is penalized;
                   * group-L2 sparsity encourages a few active groups.
  address_residual deterministic address + residual autoencoder
  sparse_ae        deterministic sparse autoencoder

The CorrVAE-inspired candidate intentionally implements the project principle
"strong structure within a factor, weak coupling between factors".  It does
NOT force scalar coordinates inside a group to be independent.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FastICA, PCA
from torch.utils.data import DataLoader, TensorDataset


CANDIDATES = (
    "raw",
    "pca",
    "ica",
    "beta_tcvae",
    "factor_vae",
    "corrvae",
    "address_residual",
    "sparse_ae",
)


@dataclass(frozen=True)
class SidecarTrainConfig:
    address_dim: int = 32
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    residual_dim: int = 64
    corr_groups: int = 8
    batch_size: int = 512
    epochs: int = 30
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    vae_kl_weight: float = 1e-3
    beta_tc_weight: float = 5e-3
    factor_tc_weight: float = 5e-3
    factor_disc_lr: float = 1e-3
    corr_cross_weight: float = 5e-3
    corr_group_sparse_weight: float = 1e-3
    address_aux_weight: float = 0.25
    address_l1_weight: float = 1e-4
    sparse_l1_weight: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 0

    def validate(self) -> None:
        if self.address_dim <= 0:
            raise ValueError("address_dim must be positive")
        if self.corr_groups <= 0 or self.address_dim % self.corr_groups != 0:
            raise ValueError(
                "address_dim must be divisible by corr_groups "
                f"(got {self.address_dim}, {self.corr_groups})"
            )
        if self.batch_size <= 1:
            raise ValueError("batch_size must be > 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


class FeatureNormalizer:
    """Train-only coordinate standardization for neural sidecars."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.detach().float().cpu()
        self.std = std.detach().float().cpu()

    @classmethod
    def fit(cls, x: torch.Tensor) -> "FeatureNormalizer":
        xf = x.detach().float().cpu()
        mean = xf.mean(dim=0)
        std = xf.std(dim=0, unbiased=False)
        positive = std[std > 0]
        ref = float(positive.median().item()) if positive.numel() else 1.0
        floor = max(1e-6, ref * 1e-3)
        return cls(mean, std.clamp_min(floor))

    def to(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean.to(device), self.std.to(device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.to(x.device)
        return (x.float() - mean) / std

    def metadata(self) -> Dict[str, Any]:
        return {
            "n_features": int(self.mean.numel()),
            "mean_abs_mean": float(self.mean.abs().mean().item()),
            "median_std": float(self.std.median().item()),
        }


class AddressSidecar:
    name: str = "base"

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def fit(
        self,
        train_x: torch.Tensor,
        val_x: torch.Tensor,
        cfg: SidecarTrainConfig,
        device: torch.device,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {}


class RawSidecar(AddressSidecar):
    name = "raw"

    def __init__(self, input_dim: int):
        self.input_dim = int(input_dim)

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def fit(self, train_x, val_x, cfg, device):
        return {"status": "identity", "n_train": int(train_x.shape[0])}

    @torch.no_grad()
    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        return x.detach().float().to(device)


class LinearProjectionSidecar(AddressSidecar):
    def __init__(self, name: str, input_dim: int, output_dim: int, seed: int):
        if name not in {"pca", "ica"}:
            raise ValueError(name)
        self.name = name
        self.input_dim = int(input_dim)
        self._output_dim = int(output_dim)
        self.seed = int(seed)
        self.mean_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self._diag: Dict[str, Any] = {}

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def fit(self, train_x, val_x, cfg, device):
        x = train_x.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if self.name == "pca":
            model = PCA(
                n_components=self.output_dim,
                svd_solver="randomized",
                random_state=self.seed,
            )
            model.fit(x)
            self.mean_ = torch.from_numpy(
                np.asarray(model.mean_, dtype=np.float32)
            )
            self.components_ = torch.from_numpy(
                np.asarray(model.components_, dtype=np.float32)
            )
            self._diag = {
                "explained_variance_ratio_sum": float(
                    np.sum(model.explained_variance_ratio_)
                )
            }
        else:
            model = FastICA(
                n_components=self.output_dim,
                whiten="unit-variance",
                random_state=self.seed,
                max_iter=800,
                tol=1e-4,
            )
            model.fit(x)
            self.mean_ = torch.from_numpy(
                np.asarray(model.mean_, dtype=np.float32)
            )
            self.components_ = torch.from_numpy(
                np.asarray(model.components_, dtype=np.float32)
            )
            self._diag = {
                "n_iter": int(model.n_iter_),
                "converged": bool(model.n_iter_ < int(model.max_iter)),
            }

        val_z = self.transform(val_x, torch.device("cpu"))
        return {
            "status": "fit",
            "n_train": int(train_x.shape[0]),
            "n_val": int(val_x.shape[0]),
            "output_dim": int(self.output_dim),
            "val_code_std_mean": float(
                val_z.float().std(dim=0, unbiased=False).mean().item()
            ),
            **self._diag,
        }

    @torch.no_grad()
    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError(f"{self.name} sidecar has not been fitted")
        mean = self.mean_.to(x.device)
        comp = self.components_.to(x.device)
        z = (x.float() - mean) @ comp.T
        return z.to(device)

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diag)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        address_dim: int,
        hidden: int,
        bottleneck: int,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden, bottleneck)
        self.mu = nn.Linear(bottleneck, address_dim)
        self.logvar = nn.Linear(bottleneck, address_dim)
        self.decoder = nn.Sequential(
            nn.Linear(address_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-10.0, 8.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        recon = self.decoder(z)
        return recon, mu, logvar, z


class FactorDiscriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        width = max(64, 4 * int(latent_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class AddressResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        address_dim: int,
        residual_dim: int,
        hidden: int,
        bottleneck: int,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden, bottleneck)
        self.address = nn.Linear(bottleneck, address_dim)
        self.residual = nn.Linear(bottleneck, residual_dim)
        self.decoder_full = nn.Sequential(
            nn.Linear(address_dim + residual_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )
        self.decoder_address = nn.Sequential(
            nn.Linear(address_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        a = self.address(h)
        r = self.residual(h)
        full = self.decoder_full(torch.cat([a, r], dim=-1))
        coarse = self.decoder_address(a)
        return full, coarse, a, r


class SparseAENet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        address_dim: int,
        hidden: int,
        bottleneck: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, address_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(address_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        torch.sum(torch.exp(logvar) + mu.square() - 1.0 - logvar, dim=1)
    )


def _log_normal_diag_matrix(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Return log q(z_i | x_j) per coordinate with shape [B,B,D]."""
    z_i = z[:, None, :]
    mu_j = mu[None, :, :]
    lv_j = logvar[None, :, :]
    return -0.5 * (
        math.log(2.0 * math.pi)
        + lv_j
        + (z_i - mu_j).square() * torch.exp(-lv_j)
    )


def minibatch_total_correlation(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Simple minibatch estimate of KL(q(z) || prod_j q(z_j))."""
    b = int(z.shape[0])
    if b <= 1:
        return z.new_tensor(0.0)
    log_qzi_xj = _log_normal_diag_matrix(z, mu, logvar)
    log_b = math.log(float(b))
    log_qz = torch.logsumexp(log_qzi_xj.sum(dim=2), dim=1) - log_b
    log_prod_qzi = (
        torch.logsumexp(log_qzi_xj, dim=1) - log_b
    ).sum(dim=1)
    return torch.mean(log_qz - log_prod_qzi)


def permute_dims(z: torch.Tensor) -> torch.Tensor:
    out = []
    for j in range(z.shape[1]):
        idx = torch.randperm(z.shape[0], device=z.device)
        out.append(z[idx, j])
    return torch.stack(out, dim=1)


def cross_group_covariance_penalty(
    z: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Penalize ONLY cross-group covariance; within-group covariance is free."""
    if z.ndim != 2:
        raise ValueError("z must be [batch, dim]")
    if z.shape[1] % int(groups) != 0:
        raise ValueError("latent dim must be divisible by groups")
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    zg = z.reshape(z.shape[0], int(groups), z.shape[1] // int(groups))
    zg = zg - zg.mean(dim=0, keepdim=True)
    terms = []
    denom = float(max(1, z.shape[0] - 1))
    for g in range(int(groups)):
        for h in range(g + 1, int(groups)):
            cov = zg[:, g, :].T @ zg[:, h, :] / denom
            terms.append(cov.square().mean())
    return torch.stack(terms).mean() if terms else z.new_tensor(0.0)


def group_l2_sparsity(z: torch.Tensor, groups: int) -> torch.Tensor:
    if z.shape[1] % int(groups) != 0:
        raise ValueError("latent dim must be divisible by groups")
    zg = z.reshape(z.shape[0], int(groups), z.shape[1] // int(groups))
    energy = torch.sqrt(zg.square().mean(dim=2) + 1e-8)
    return energy.sum(dim=1).mean()


class NeuralSidecar(AddressSidecar):
    def __init__(
        self,
        name: str,
        input_dim: int,
        cfg: SidecarTrainConfig,
    ):
        if name not in {
            "beta_tcvae",
            "factor_vae",
            "corrvae",
            "address_residual",
            "sparse_ae",
        }:
            raise ValueError(name)
        self.name = name
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.normalizer: Optional[FeatureNormalizer] = None
        self.model: Optional[nn.Module] = None
        self.discriminator: Optional[FactorDiscriminator] = None
        self._diag: Dict[str, Any] = {}

    @property
    def output_dim(self) -> int:
        return int(self.cfg.address_dim)

    def _build(self, device: torch.device) -> None:
        c = self.cfg
        if self.name in {"beta_tcvae", "factor_vae", "corrvae"}:
            self.model = GaussianAE(
                self.input_dim,
                c.address_dim,
                c.hidden_dim,
                c.bottleneck_dim,
            )
            if self.name == "factor_vae":
                self.discriminator = FactorDiscriminator(c.address_dim)
        elif self.name == "address_residual":
            self.model = AddressResidualNet(
                self.input_dim,
                c.address_dim,
                c.residual_dim,
                c.hidden_dim,
                c.bottleneck_dim,
            )
        elif self.name == "sparse_ae":
            self.model = SparseAENet(
                self.input_dim,
                c.address_dim,
                c.hidden_dim,
                c.bottleneck_dim,
            )
        else:
            raise ValueError(self.name)
        self.model.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)

    def _main_loss(
        self,
        xb: torch.Tensor,
        *,
        sample: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        assert self.model is not None
        c = self.cfg

        if self.name in {"beta_tcvae", "factor_vae", "corrvae"}:
            model = self.model
            assert isinstance(model, GaussianAE)
            recon, mu, logvar, z = model(xb, sample=sample)
            recon_loss = F.mse_loss(recon, xb)
            kl = _kl_standard_normal(mu, logvar)
            loss = recon_loss + float(c.vae_kl_weight) * kl
            comp: Dict[str, float] = {
                "recon": float(recon_loss.detach().item()),
                "kl": float(kl.detach().item()),
            }

            if self.name == "beta_tcvae":
                tc = minibatch_total_correlation(z, mu, logvar)
                loss = loss + float(c.beta_tc_weight) * tc
                comp["tc"] = float(tc.detach().item())

            elif self.name == "factor_vae":
                if self.discriminator is None:
                    raise RuntimeError("FactorVAE discriminator missing")
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                tc_proxy = self.discriminator(z).mean()
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                loss = loss + float(c.factor_tc_weight) * tc_proxy
                comp["tc_proxy"] = float(tc_proxy.detach().item())

            elif self.name == "corrvae":
                # Project-specific structured factorization:
                # - coordinates inside each group may correlate;
                # - coupling between groups is penalized;
                # - group-L2 sparsity makes a small set of groups address-active.
                cross = cross_group_covariance_penalty(mu, c.corr_groups)
                group_sparse = group_l2_sparsity(mu, c.corr_groups)
                loss = (
                    loss
                    + float(c.corr_cross_weight) * cross
                    + float(c.corr_group_sparse_weight) * group_sparse
                )
                comp["cross_group_cov"] = float(cross.detach().item())
                comp["group_l2_sparsity"] = float(group_sparse.detach().item())

            return loss, comp, mu

        if self.name == "address_residual":
            model = self.model
            assert isinstance(model, AddressResidualNet)
            full, coarse, a, _r = model(xb)
            full_loss = F.mse_loss(full, xb)
            coarse_loss = F.mse_loss(coarse, xb)
            l1 = a.abs().mean()
            loss = (
                full_loss
                + float(c.address_aux_weight) * coarse_loss
                + float(c.address_l1_weight) * l1
            )
            return loss, {
                "recon_full": float(full_loss.detach().item()),
                "recon_address_only": float(coarse_loss.detach().item()),
                "address_l1": float(l1.detach().item()),
            }, a

        if self.name == "sparse_ae":
            model = self.model
            assert isinstance(model, SparseAENet)
            recon, z = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            sparse = z.abs().mean()
            loss = recon_loss + float(c.sparse_l1_weight) * sparse
            return loss, {
                "recon": float(recon_loss.detach().item()),
                "code_l1": float(sparse.detach().item()),
            }, z

        raise ValueError(self.name)

    def _factor_disc_step(
        self,
        xb: torch.Tensor,
        disc_opt: torch.optim.Optimizer,
    ) -> float:
        if self.name != "factor_vae":
            return 0.0
        assert isinstance(self.model, GaussianAE)
        assert self.discriminator is not None
        with torch.no_grad():
            _, _, _, z = self.model(xb, sample=True)
        z_joint = z.detach()
        z_perm = permute_dims(z_joint)
        logits_joint = self.discriminator(z_joint)
        logits_perm = self.discriminator(z_perm)
        ones = torch.ones_like(logits_joint)
        zeros = torch.zeros_like(logits_perm)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(logits_joint, ones)
            + F.binary_cross_entropy_with_logits(logits_perm, zeros)
        )
        disc_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), self.cfg.grad_clip
        )
        disc_opt.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def _eval_loss(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        assert self.model is not None
        self.model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        total = 0.0
        n = 0
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device, non_blocking=True).float()
            assert self.normalizer is not None
            xb = self.normalizer.transform(xb)
            loss, _, _ = self._main_loss(xb, sample=False)
            bs = int(xb.shape[0])
            total += float(loss.item()) * bs
            n += bs
        return total / max(n, 1)

    def fit(
        self,
        train_x: torch.Tensor,
        val_x: torch.Tensor,
        cfg: SidecarTrainConfig,
        device: torch.device,
    ) -> Dict[str, Any]:
        cfg.validate()
        if cfg != self.cfg:
            raise ValueError("sidecar config mismatch")
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg.seed))

        self.normalizer = FeatureNormalizer.fit(train_x)
        self._build(device)
        assert self.model is not None

        train_ds = TensorDataset(train_x.detach().cpu())
        val_ds = TensorDataset(val_x.detach().cpu())
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(cfg.seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            generator=gen,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        disc_opt = None
        if self.discriminator is not None:
            disc_opt = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=float(cfg.factor_disc_lr),
            )

        best_state = copy.deepcopy(self.model.state_dict())
        best_disc = (
            copy.deepcopy(self.discriminator.state_dict())
            if self.discriminator is not None
            else None
        )
        best_val = float("inf")
        best_epoch = 0
        bad_epochs = 0
        history = []

        for epoch in range(1, int(cfg.epochs) + 1):
            self.model.train()
            if self.discriminator is not None:
                self.discriminator.train()
            sum_loss = 0.0
            sum_n = 0
            disc_losses = []

            for (xb_cpu,) in train_loader:
                xb = xb_cpu.to(device, non_blocking=True).float()
                xb = self.normalizer.transform(xb)

                opt.zero_grad(set_to_none=True)
                loss, _, _ = self._main_loss(xb, sample=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(cfg.grad_clip)
                )
                opt.step()

                if disc_opt is not None:
                    disc_losses.append(self._factor_disc_step(xb, disc_opt))

                bs = int(xb.shape[0])
                sum_loss += float(loss.detach().item()) * bs
                sum_n += bs

            val_loss = self._eval_loss(val_loader, device)
            train_loss = sum_loss / max(sum_n, 1)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "disc_loss": (
                        float(np.mean(disc_losses)) if disc_losses else None
                    ),
                }
            )

            if val_loss < best_val - 1e-6:
                best_val = float(val_loss)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(self.model.state_dict())
                best_disc = (
                    copy.deepcopy(self.discriminator.state_dict())
                    if self.discriminator is not None
                    else None
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(cfg.patience):
                    break

        self.model.load_state_dict(best_state)
        if self.discriminator is not None and best_disc is not None:
            self.discriminator.load_state_dict(best_disc)
        self.model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()

        with torch.no_grad():
            val_code = self.transform(val_x[: min(len(val_x), 4096)], device)
            coord_std = val_code.float().std(dim=0, unbiased=False)
            active = int((coord_std > 1e-3).sum().item())
            self._diag = {
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "epochs_ran": int(len(history)),
                "active_coordinates_std_gt_1e-3": int(active),
                "mean_coordinate_std": float(coord_std.mean().item()),
                "history_tail": history[-5:],
                "normalizer": self.normalizer.metadata(),
            }
            if self.name == "corrvae":
                self._diag["corr_groups"] = int(cfg.corr_groups)
                self._diag["group_dim"] = int(
                    cfg.address_dim // cfg.corr_groups
                )
                self._diag["design"] = (
                    "within-group covariance allowed; cross-group covariance "
                    "penalized; group-L2 sparsity encourages few active groups"
                )

        return {
            "status": "fit",
            "n_train": int(train_x.shape[0]),
            "n_val": int(val_x.shape[0]),
            "output_dim": int(self.output_dim),
            **self._diag,
        }

    @torch.no_grad()
    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.model is None or self.normalizer is None:
            raise RuntimeError(f"{self.name} sidecar has not been fitted")
        self.model.eval()
        xb = x.detach().float().to(device)
        xb = self.normalizer.transform(xb)

        if self.name in {"beta_tcvae", "factor_vae", "corrvae"}:
            assert isinstance(self.model, GaussianAE)
            mu, _ = self.model.encode(xb)
            return mu

        if self.name == "address_residual":
            assert isinstance(self.model, AddressResidualNet)
            h = self.model.encoder(xb)
            return self.model.address(h)

        if self.name == "sparse_ae":
            assert isinstance(self.model, SparseAENet)
            return self.model.encoder(xb)

        raise ValueError(self.name)

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diag)


def build_sidecar(
    name: str,
    input_dim: int,
    cfg: SidecarTrainConfig,
) -> AddressSidecar:
    name = str(name).strip().lower()
    if name not in CANDIDATES:
        raise ValueError(f"unknown sidecar={name!r}; allowed={CANDIDATES}")
    if name == "raw":
        return RawSidecar(input_dim)
    if name in {"pca", "ica"}:
        return LinearProjectionSidecar(
            name,
            input_dim,
            cfg.address_dim,
            cfg.seed,
        )
    return NeuralSidecar(name, input_dim, cfg)


def coordinate_participation_ratio(z: torch.Tensor) -> float:
    """Average effective number of active coordinates based on |z|."""
    x = z.detach().float().abs()
    num = x.sum(dim=1).square()
    den = x.square().sum(dim=1).clamp_min(1e-12)
    return float((num / den).mean().item())


def group_participation_ratio(
    z: torch.Tensor,
    groups: int,
) -> float:
    if z.shape[1] % int(groups) != 0:
        return float("nan")
    x = z.detach().float().reshape(
        z.shape[0],
        int(groups),
        z.shape[1] // int(groups),
    )
    e = torch.sqrt(x.square().mean(dim=2) + 1e-12)
    num = e.sum(dim=1).square()
    den = e.square().sum(dim=1).clamp_min(1e-12)
    return float((num / den).mean().item())


def sidecar_config_dict(cfg: SidecarTrainConfig) -> Dict[str, Any]:
    return asdict(cfg)
