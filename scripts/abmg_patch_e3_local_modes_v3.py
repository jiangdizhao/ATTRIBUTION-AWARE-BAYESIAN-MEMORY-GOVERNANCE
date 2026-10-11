#!/usr/bin/env python3
"""Patch-local multimodal E3 (source-CV validation, no target access).

Every cached patch is scored before top-k image aggregation. Frozen support-only
cosine keys route patches to independent local Gaussian/NIG beliefs. This is
FEATURE-LOCAL, not spatial-grid-local: the E4 cache is score-sorted.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score

MODELS = ("gaussian", "nig")
CHECKPOINTS = (0, 4, 8, 16, 32)


def load_evidence(path: str, fold: int) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if data.get("schema") != "abmg.e4.evidence.v1" or data.get("outer_fold") != fold:
        raise ValueError("Expected E4 full-patch evidence for the selected outer fold")
    x = data.get("candidate_features")
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 784:
        raise ValueError("E3 v3 requires all 784 patches per image; use --candidates 784 evidence")
    if len(data["items"]) != x.shape[0] or x.shape[2] < 1:
        raise ValueError("Malformed evidence metadata/dimensions")
    if len(data.get("source_cv_groups", [])) != 4:
        raise ValueError("Expected four source-CV groups")
    return data


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def patches_at(data: dict, i: int) -> np.ndarray:
    return np.asarray(data["candidate_features"][i].numpy(), dtype=np.float32)


def choose_keys(support_patches: np.ndarray, slots: int, seed: int) -> np.ndarray:
    """Deterministic cosine farthest-first anchors, from support normal patches only."""
    x = normalize(support_patches)
    if not 1 <= slots <= len(x):
        raise ValueError("Invalid slots / insufficient initial support")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(x)))
    indices = [first]
    distance = 1.0 - x @ x[first]
    for _ in range(1, slots):
        j = int(np.argmax(distance))
        indices.append(j)
        distance = np.minimum(distance, 1.0 - x @ x[j])
    return x[indices].copy()


class LocalPatchMemory:
    """One diagonal predictive state per visual appearance mode.

    One verified image contributes at most 1 total unit, distributed over its
    selected correlated patches. NIG is a composite-likelihood approximation;
    fixed support-derived mode keys ensure a stable comparison across branches.
    """
    def __init__(self, keys, prior_mean, prior_var, model="nig"):
        if model not in MODELS:
            raise ValueError("Unknown normal-state model")
        self.model = model
        self.keys = normalize(keys)
        self.k, self.dim = self.keys.shape
        mu = np.asarray(prior_mean, dtype=np.float64).reshape(-1)
        v = np.maximum(np.asarray(prior_var, dtype=np.float64).reshape(-1), 1e-7)
        if mu.size != self.dim or v.size != self.dim:
            raise ValueError("Prior dimensions do not match frozen keys")
        self.mu = np.tile(mu, (self.k, 1))
        self.var0 = np.tile(v, (self.k, 1))
        self.kappa = np.full((self.k, self.dim), 0.05)
        self.alpha = np.full((self.k, self.dim), 2.5)
        self.beta = 1.5 * self.var0.copy()
        self.m2 = np.zeros((self.k, self.dim), dtype=np.float64)
        self.mass = np.zeros(self.k, dtype=np.float64)
        self.images_seen = 0

    def assign(self, patches: np.ndarray) -> np.ndarray:
        p = np.asarray(patches, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != self.dim:
            raise ValueError("Patch shape incompatible with local normal memory")
        return np.argmax(normalize(p) @ self.keys.T, axis=1)

    def surprise(self, patches: np.ndarray, assigned=None) -> np.ndarray:
        p = np.asarray(patches, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.dim:
            raise ValueError("Invalid patch evidence")
        index = self.assign(patches) if assigned is None else np.asarray(assigned, dtype=int)
        if len(index) != len(p) or np.any(index < 0) or np.any(index >= self.k):
            raise ValueError("Invalid appearance-mode assignment")
        d = p - self.mu[index]
        if self.model == "nig":
            nu = 2.0 * self.alpha[index]
            scale2 = np.maximum(self.beta[index] * (self.kappa[index] + 1.0)
                                / (self.alpha[index] * self.kappa[index]), 1e-9)
            logp = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                    - .5 * np.log(nu * np.pi * scale2)
                    - .5 * (nu + 1.0) * np.log1p(d * d / (nu * scale2)))
        else:
            variance = np.maximum((self.var0[index] + self.m2[index]) /
                                  (1.0 + self.mass[index, None]), .05*self.var0[index])
            logp = -.5 * (np.log(2.0 * np.pi * variance) + d*d/variance)
        return -np.mean(logp, axis=1)

    def image_score(self, patches: np.ndarray, topk: int = 8) -> float:
        # Score all 784 patches; pool the anomaly-score tail afterward.
        s = self.surprise(patches)
        n = min(max(int(topk), 1), len(s))
        return float(np.partition(s, -n)[-n:].mean())

    def update(self, patches: np.ndarray, weights=None) -> dict:
        p = np.asarray(patches, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.dim or len(p) == 0:
            raise ValueError("Invalid verified-normal patch batch")
        if weights is None:
            weights = np.full(len(p), 1.0 / len(p))
        w_all = np.asarray(weights, dtype=np.float64)
        if w_all.shape != (len(p),) or np.any(w_all < 0) or not np.all(np.isfinite(w_all)):
            raise ValueError("Invalid patch evidence weights")
        if w_all.sum() > 1.0 + 1e-8:
            raise ValueError("One verified image cannot exceed one effective unit")
        assignments = self.assign(p)
        touched = []
        for j in range(self.k):
            mask = assignments == j
            if not np.any(mask):
                continue
            weights_j = w_all[mask]
            w = float(weights_j.sum())
            if w <= 0: continue
            x = p[mask]
            xb = np.sum(x * weights_j[:, None], axis=0)/w
            residual = np.sum(weights_j[:, None]*(x-xb)**2, axis=0)
            k0 = self.kappa[j].copy()
            m0 = self.mu[j].copy()
            kn = k0+w
            cross = (k0*w/kn)*(xb-m0)**2
            self.mu[j] = (k0*m0+w*xb)/kn
            self.kappa[j] = kn
            self.alpha[j] += .5*w
            self.beta[j] += .5*(residual+cross)
            self.m2[j] += residual+cross
            self.mass[j] += w
            touched.append(j)
        self.images_seen += 1
        return {"affected_modes": touched, "effective_mass_added": float(w_all.sum())}


def empirical_prior(data: dict, heldout: set, images_per_cat=3, patches_per_img=16):
    """Source-CV development TRAIN normals only; no heldout/sentinel priors."""
    used = {}
    chunks = []
    for i, item in enumerate(data["items"]):
        c = item["category"]
        if c in heldout or item["role"] != "train_normal" or used.get(c, 0) >= images_per_cat:
            continue
        used[c] = used.get(c, 0) + 1
        p = patches_at(data, i)
        idx = np.linspace(0, len(p)-1, min(patches_per_img,len(p)), dtype=int)
        chunks.append(p[idx].astype(np.float64))
    if not chunks:
        raise ValueError("No development TRAIN normals for empirical prior")
    x = np.concatenate(chunks)
    return x.mean(axis=0), np.maximum(x.var(axis=0), 1e-5)


def metrics(model, normal, defect, initial_normal=None, initial_defect=None, topk=8):
    ns = np.asarray([model.image_score(p,topk) for p in normal], dtype=np.float64)
    ds = np.asarray([model.image_score(p,topk) for p in defect], dtype=np.float64)
    auc = float(roc_auc_score(np.r_[np.zeros(len(ns)),np.ones(len(ds))],np.r_[ns,ds]))
    out = {"auroc":auc,"normal_surprise":float(ns.mean()),"defect_surprise":float(ds.mean()),
           "separation":float(ds.mean()-ns.mean()),"n_normal":len(ns),"n_defect":len(ds),
           "mode_mass":model.mass.tolist(),"n_modes":model.k}
    if initial_normal is not None:
        out["normal_score_reduced_frac"] = float(np.mean(ns < initial_normal - 1e-9))
        out["defect_score_reduced_frac"] = float(np.mean(ds < initial_defect - 1e-9))
        out["normal_score_delta"] = float((ns-initial_normal).mean())
        out["defect_score_delta"] = float((ds-initial_defect).mean())
    return out,ns,ds


def run(data: dict, cv_fold: int, budget=16, support=2, seed=0,
        modes=8, patch_write=32, score_topk=8):
    if not 0 <= cv_fold < len(data["source_cv_groups"]):
        raise ValueError("Invalid source-CV fold")
    if min(budget, support, modes, patch_write, score_topk) < 1:
        raise ValueError("Parameters must be positive")
    heldout = set(data["source_cv_groups"][cv_fold])
    prior_mean, prior_var = empirical_prior(data, heldout)
    bycat = {c:{"train_normal":[], "test_normal":[], "defect":[]} for c in heldout}
    for i,item in enumerate(data["items"]):
        c = item["category"]
        if c in bycat and item["role"] in bycat[c]:
            bycat[c][item["role"]].append(i)
    rng = np.random.default_rng(seed + 1009 * cv_fold)
    output = {"schema":"abmg.patch_e3.local_modes.v3", "outer_fold":data["outer_fold"],
              "source_cv_fold":cv_fold, "categories":sorted(heldout),
              "protocol":{"budget":budget,"support":support,"seed":seed,"modes":modes,
                          "write_patches":patch_write,"score_topk":score_topk,
                          "score_patches":784,
                          "feedback_selection":"offline known-normal high-surprise, shared cold-start local NIG",
                          "evaluation":"held-out source-CV test-normal and defect sentinels, no writes",
                          "spatial_coordinates":"unavailable in E4 cache; appearance-local only",
                          "image_effective_mass_cap":1.0,
                          "model_branches":["global_gaussian","global_nig","local_gaussian","local_nig"]},
              "results":[],"skipped":[]}
    for cat in sorted(heldout):
        ids = bycat[cat]
        train = rng.permutation(ids["train_normal"]).tolist()
        test = rng.permutation(ids["test_normal"]).tolist()
        n_sentinel = max(1, len(test)//2)
        sent_normal = test[:n_sentinel]
        queried_test = test[n_sentinel:]
        sent_defect = list(ids["defect"])
        if len(train) < support + 1 or not sent_normal or not sent_defect:
            output["skipped"].append(cat)
            continue
        support_ids = train[:support]
        pool = train[support:] + queried_test
        initial = np.concatenate([patches_at(data, i) for i in support_ids])
        keys_local = choose_keys(initial, modes, seed + 37 + len(output["results"]))
        keys_global = choose_keys(initial, 1, seed)
        branches = {}
        for mode, keys in (("global",keys_global),("local",keys_local)):
            for model in MODELS:
                name = f"{mode}_{model}"
                st = LocalPatchMemory(keys,prior_mean,prior_var,model)
                for i in support_ids:
                    st.update(patches_at(data,i))
                branches[name] = st
        # Same offline verified-normal query manifest AND same selected patches
        # for all models, determined once from frozen cold-start NIG.
        selector = copy.deepcopy(branches["local_nig"])
        pool_scores = [(selector.image_score(patches_at(data,i),score_topk),i) for i in pool]
        pool_scores.sort(key=lambda t:(-t[0],t[1]))
        queried_ids = [i for _,i in pool_scores[:budget]]
        write_indices = {}
        for i in queried_ids:
            p = patches_at(data,i)
            s = selector.surprise(p)
            kk = min(patch_write,len(p))
            write_indices[i] = np.argsort(-s,kind="stable")[:kk].copy()
        normals = [patches_at(data,i) for i in sent_normal]
        defects = [patches_at(data,i) for i in sent_defect]
        initial_scores = {}
        for name,st in branches.items():
            _,ns,ds = metrics(st,normals,defects,topk=score_topk)
            initial_scores[name] = (ns,ds)
        checkpoints=[]
        for t in range(len(queried_ids)+1):
            if t in CHECKPOINTS or t == len(queried_ids):
                row = {}
                for name,st in branches.items():
                    row[name] = metrics(st,normals,defects,*initial_scores[name],topk=score_topk)[0]
                checkpoints.append({"queries":t,"branches":row})
            if t == len(queried_ids): break
            i=queried_ids[t]
            patch_batch=patches_at(data,i)[write_indices[i]]
            for st in branches.values():
                st.update(patch_batch)
        names = [data["items"][i]["image_id"] for i in queried_ids]
        sentinel_ids = [data["items"][i]["image_id"] for i in sent_normal+sent_defect]
        if set(names) & set(sentinel_ids):
            raise AssertionError("Feedback/sentinel overlap")
        output["results"].append({"category":cat,"n_support":len(support_ids),
                                  "n_queries":len(queried_ids),
                                  "n_sentinel_normal":len(normals),"n_sentinel_defect":len(defects),
                                  "query_image_ids":names,"sentinel_image_ids":sentinel_ids,
                                  "checkpoints":checkpoints})
    if not output["results"]:
        raise RuntimeError("No evaluable product categories")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold",type=int,choices=range(5),required=True)
    p.add_argument("--evidence",required=True)
    p.add_argument("--source-cv-fold",type=int,choices=range(4),default=0)
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--support",type=int,default=2)
    p.add_argument("--modes",type=int,default=8)
    p.add_argument("--patch-write",type=int,default=32)
    p.add_argument("--score-topk",type=int,default=8)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out-dir",default="outputs/patch_e3_v3")
    a=p.parse_args()
    artifact=load_evidence(a.evidence,a.fold)
    result=run(artifact,a.source_cv_fold,a.budget,a.support,a.seed,a.modes,a.patch_write,a.score_topk)
    out=Path(a.out_dir)/f"fold_{a.fold}"/f"cv_{a.source_cv_fold}"
    out.mkdir(parents=True,exist_ok=True)
    file=out/"patch_e3_local_modes_audit.json"
    file.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {file}; evaluated {len(result['results'])} categories")
    for cp in (0,min(a.budget,16)):
        vals=[]
        for row in result["results"]:
            matches=[r for r in row["checkpoints"] if r["queries"]==cp]
            if matches:
                vals.append(matches[0]["branches"])
        if vals:
            print(f"checkpoint={cp}"," ".join(f"{n}={np.mean([v[n]['auroc'] for v in vals]):.4f}" for n in vals[0]))


if __name__=="__main__":
    main()
