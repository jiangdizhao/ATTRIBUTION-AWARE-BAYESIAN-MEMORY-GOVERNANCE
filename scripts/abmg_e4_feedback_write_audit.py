#!/usr/bin/env python3
"""E4 source-side fixed-query experiment: asymmetric verified-feedback writes.

E4A (N0/N1): ordinary NIG update on *raw* frozen evidence in both branches;
N1 additionally writes verified-normal false-positive top-8 patches to a bounded,
category-local negative correction bank. Scoring may rerank a cached top-M
candidate patch pool. NIG histories must remain exactly equal between N0/N1.

E4B (D0/D1/D2/D3): freeze the selected E4A normal policy, vary ONLY credit for
FN-to-factor memory updates: current responsibility, past source centroid,
shuffled source centroid, or offline oracle one-hot. No defect can update NIG.

E4C: factorial C00/C10/C01/C11 after independent A/B gates. No controller,
no revealed masks, no online DINO updates, and no outer-target access. The E4
cache must come from abmg_e4_extract_evidence.py (not the pooled-only E3 file).
All source labels are offline-only until their own fixed feedback event occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

try:
    from abmg_e3_normal_state_models import DiagonalNIG, fit_weak_nig_prior
    from abmg_sequential_local_update_audit import (init_memory, uniform_query_positions)
    from abmg_four_branch_structural_credit_audit import (make_derangement,
                                                           update_memory_weighted)
    from abmg_prototype_addressability_audit import CORE4, category_diverse_support_indices
    from abmg_memory_addressability_audit import fit_shared_diag_prior
except ImportError:
    from scripts.abmg_e3_normal_state_models import DiagonalNIG, fit_weak_nig_prior
    from scripts.abmg_sequential_local_update_audit import (init_memory, uniform_query_positions)
    from scripts.abmg_four_branch_structural_credit_audit import (make_derangement,
                                                                   update_memory_weighted)
    from scripts.abmg_prototype_addressability_audit import CORE4, category_diverse_support_indices
    from scripts.abmg_memory_addressability_audit import fit_shared_diag_prior

K = 8
T = 20.0
CORE_SET = frozenset(CORE4)
CHECKPOINTS = (0, 4, 8, 16, 32)


def stable_hash(seed: int, key: str) -> str:
    return hashlib.sha1(f'{int(seed)}|{key}'.encode('utf-8')).hexdigest()


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(np.clip(x, -700, 0))
    return e / e.sum()


def routing_responsibility(logits: np.ndarray) -> np.ndarray:
    """Uncalibrated routing responsibility proxy, NOT a Bayesian posterior."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if x.size != len(CORE4):
        raise ValueError('routing logits must contain exactly Core-4 factors')
    return stable_softmax((x - x.mean()) / max(float(x.std()), 1e-8))


def score_routing_fractional(
    x: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float = 1.0,
) -> np.ndarray:
    """Faithful diag_shrunk scorer with FLOAT effective sample counts.

    The Stage-1 score_memory casts n to int, which would silently discard
    fractional E4 source credit. This local scorer deliberately retains n.
    """
    arr=norm(np.asarray(x,dtype=np.float64).reshape(1,-1))[0]
    mu0=np.asarray(shared_prior['mu0'],dtype=np.float64)
    var=np.asarray(shared_prior['var'],dtype=np.float64)
    if mu0.shape!=arr.shape or var.shape!=arr.shape or np.any(var<=0):
        raise ValueError('invalid routing prior')
    scores=[]
    for g in CORE4:
        n=float(memory[g]['n'])
        total=np.asarray(memory[g]['sum'],dtype=np.float64)
        if n<=0 or total.shape!=arr.shape:
            raise ValueError('invalid factor sufficient-statistic state')
        mu=(float(kappa0)*mu0+total)/(float(kappa0)+n)
        d=arr-mu
        scores.append(-.5*np.sum(d*d/var + np.log(var)))
    return np.asarray(scores,dtype=np.float64)


@dataclass
class PrototypeBank:
    """E1B-style bounded cosine prototypes (CPU NumPy for cached E4 evidence)."""
    capacity: int = 8

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError('capacity must be positive')
        self.vectors: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.observations = 0

    @property
    def size(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def add(self, x: np.ndarray) -> None:
        arr = norm(np.asarray(x, dtype=np.float64).reshape(1, -1))[0]
        if self.vectors is None:
            self.vectors = arr[None, :]
            self.counts = np.ones(1, dtype=np.float64)
        elif self.size < self.capacity:
            self.vectors = np.vstack([self.vectors, arr])
            self.counts = np.append(self.counts, 1.0)
        else:
            j = int(np.argmax(self.vectors @ arr))
            c = float(self.counts[j])
            self.vectors[j] = norm((self.vectors[j] * c + arr).reshape(1, -1))[0]
            self.counts[j] = c + 1.0
        self.observations += 1

    def max_sim(self, patch_features: np.ndarray) -> np.ndarray:
        x = norm(np.asarray(patch_features, dtype=np.float64))
        if self.vectors is None:
            return np.zeros(len(x), dtype=np.float64)
        return np.clip(np.max(x @ self.vectors.T, axis=1), 0, 1)


def corrected_evidence(
    candidates: np.ndarray,
    scores: np.ndarray,
    full_score_std: float,
    bank: Optional[PrototypeBank],
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-8 from cached top-M; preserve E3 score-softmax pooling.

    A missing bank reproduces raw top-8 (after cache fp16 rounding). This is
    tested to numerical tolerance; raw E3 pooled vectors are used when empty.
    """
    x = np.asarray(candidates, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or s.shape != (len(x),) or len(x) < K:
        raise ValueError('candidate patch shapes invalid')
    c = s if bank is None or bank.size == 0 else s - float(full_score_std) * bank.max_sim(x)
    # deterministic tie resolution by original rank (already ordered by frozen score)
    idx = np.argsort(-c, kind='stable')[:K]
    w = stable_softmax(T * c[idx])
    vec = norm(np.sum(w[:, None] * x[idx], axis=0).reshape(1, -1))[0]
    return vec, idx


@dataclass
class SourceCentroids:
    """Past-only responsibilities by human-revealed source (not latent causes)."""
    count: Dict[str, int]
    total: Dict[str, np.ndarray]

    @classmethod
    def empty(cls) -> 'SourceCentroids':
        return cls({}, {})

    def get(self, source: str) -> Optional[np.ndarray]:
        n = self.count.get(str(source), 0)
        return self.total[str(source)] / n if n else None

    def add(self, source: str, responsibility: np.ndarray) -> None:
        if source not in CORE_SET:
            raise ValueError('only Core-4 verified sources are supported')
        r = np.asarray(responsibility, dtype=np.float64)
        if r.shape != (len(CORE4),) or np.any(r < 0) or not np.isclose(r.sum(), 1):
            raise ValueError('invalid responsibility for source memory')
        self.total[source] = self.total.get(source, np.zeros_like(r)) + r
        self.count[source] = self.count.get(source, 0) + 1


def defect_credit(
    branch: str,
    responsibility: np.ndarray,
    source: str,
    history: SourceCentroids,
    permutation: Mapping[str, str],
    eta: float,
) -> np.ndarray:
    """Credit computed with past-only source memory, BEFORE history.add()."""
    r = np.asarray(responsibility, dtype=np.float64)
    if source not in CORE_SET or r.shape != (len(CORE4),):
        raise ValueError('invalid Core-4 source or responsibility')
    if not 0 <= eta <= 1:
        raise ValueError('eta must be in [0,1]')
    if branch == 'binary':
        out = r.copy()
    elif branch == 'correct':
        q = history.get(source)
        out = r.copy() if q is None else eta * r + (1-eta) * q
    elif branch == 'shuffled':
        q = history.get(permutation[source])
        out = r.copy() if q is None else eta * r + (1-eta) * q
    elif branch == 'oracle':
        out = np.eye(len(CORE4))[list(CORE4).index(source)]
    else:
        raise ValueError(f'unknown defect credit branch {branch}')
    if np.any(out < 0) or not np.isclose(out.sum(), 1.0, atol=1e-10):
        raise AssertionError('credit must be nonnegative unit-mass')
    return out


def split_stream_sentinel(items: Sequence[Mapping[str, Any]], seed: int,
                          sentinel_fraction: float = .25) -> tuple[list[int], list[int]]:
    """Stratified OFFLINE split only: labels never used to select query positions."""
    if not 0 < sentinel_fraction < 1:
        raise ValueError('sentinel_fraction outside (0,1)')
    cells: Dict[tuple[str,str], list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        source = item.get('defect_source_offline_only') if item['role'] == 'defect' else 'normal'
        cells[(str(item['category']), str(source))].append(i)
    stream, sentinel = [], []
    for key, candidates in sorted(cells.items()):
        sorted_ids = sorted(candidates, key=lambda i: stable_hash(seed, str(items[i]['image_id'])))
        if len(sorted_ids) < 2:
            stream.extend(sorted_ids)
            continue
        n = min(len(sorted_ids)-1, max(1, round(len(sorted_ids)*sentinel_fraction)))
        sentinel.extend(sorted_ids[:n]); stream.extend(sorted_ids[n:])
    if set(stream) & set(sentinel) or len(stream)+len(sentinel) != len(items):
        raise AssertionError('stream/sentinel leakage or incompleteness')
    stream.sort(key=lambda i: stable_hash(seed+1, str(items[i]['image_id'])))
    sentinel.sort(key=lambda i: str(items[i]['image_id']))
    return stream, sentinel


def branches_for(mode: str, normal_policy: str) -> dict[str, tuple[bool,str]]:
    if mode == 'e4a':
        return {'N0':(False,'binary'), 'N1':(True,'binary')}
    if mode == 'e4b':
        return {f'D{i}':(normal_policy == 'n1',kind) for i,kind in enumerate(
            ('binary','correct','shuffled','oracle'))}
    if mode == 'e4c':
        return {'C00':(False,'binary'), 'C10':(True,'binary'),
                'C01':(False,'correct'), 'C11':(True,'correct')}
    raise ValueError('mode must be e4a, e4b or e4c')


@dataclass
class BranchState:
    hard_normal: bool
    defect_kind: str
    normal_by_category: dict[str, Any]
    router_memory: dict[str, Any]
    neg_banks: dict[str, PrototypeBank]
    normal_updates: Counter
    defect_writes: Counter

    def fingerprint_nig(self, category: str) -> tuple[bytes,...]:
        s = self.normal_by_category[category]
        return tuple(np.asarray(getattr(s,k)).tobytes() for k in ('mu','kappa','alpha','beta'))


def apply_verified_feedback(
    state: BranchState, *, is_defect: bool, is_false_alarm: bool,
    category: str, vector_raw: np.ndarray, top8_patches: np.ndarray,
    source: Optional[str], responsibility: Optional[np.ndarray],
    history: SourceCentroids, permutation: Mapping[str,str], eta: float,
) -> dict[str, Any]:
    """Exactly one feedback event: output an auditable write receipt."""
    before_nig = state.fingerprint_nig(category)
    receipt: dict[str,Any] = {'normal_write':False,'negative_writes':0,
                              'factor_credit':None,'factor_write':False}
    if not is_defect:
        state.normal_by_category[category].update(vector_raw, 1.0, robust=False)
        state.normal_updates[category] += 1
        receipt['normal_write'] = True
        if state.hard_normal and is_false_alarm:
            bank = state.neg_banks.setdefault(category, PrototypeBank())
            for x in top8_patches:
                bank.add(x)
                receipt['negative_writes'] += 1
    else:
        if source in CORE_SET and responsibility is not None:
            # is_false_alarm here signifies a wrong operational prediction:
            # for a defect this means a false negative.
            if is_false_alarm:
                weights = defect_credit(state.defect_kind, responsibility, str(source),
                                        history, permutation, eta)
                update_memory_weighted(state.router_memory,
                                       {g:float(weights[i]) for i,g in enumerate(CORE4)}, vector_raw)
                state.defect_writes[str(source)] += 1
                receipt['factor_credit'] = weights.tolist()
                receipt['factor_write'] = True
        if state.fingerprint_nig(category) != before_nig:
            raise AssertionError('DEFECT WRITE CONTAMINATED NORMAL NIG STATE')
    return receipt


def _collect_artifact(artifact: Mapping[str,Any]) -> tuple[list[dict[str,Any]],np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if artifact.get('schema') != 'abmg.e4.evidence.v1':
        raise ValueError('E4 requires patch candidate cache, NOT pooled-only E3 cache')
    items = list(artifact['items'])
    x=norm(artifact['evidence'].float().cpu().numpy())
    pf=artifact['candidate_features'].float().cpu().numpy().astype(np.float32)
    ps=artifact['candidate_scores'].float().cpu().numpy().astype(np.float32)
    sd=artifact['full_patch_score_std'].float().cpu().numpy().astype(np.float64)
    if x.shape[0] != len(items) or pf.shape[:2] != ps.shape or len(sd) != len(items):
        raise ValueError('E4 cache dimension mismatch')
    if pf.shape[-1] != x.shape[-1] or pf.shape[1]<K:
        raise ValueError('E4 feature dimension/candidate pool mismatch')
    if set(artifact.get('source_categories',[])) & set(artifact.get('target_categories_untouched',[])):
        raise ValueError('source/target leakage in E4 evidence manifest')
    if any(i['category'] not in set(artifact['source_categories']) for i in items):
        raise ValueError('E4 evidence contains a sealed target category')
    return items,x,pf,ps,sd


def _initial_nig(prior: Any, train_normal: np.ndarray, image_ids: Sequence[str],
                 shots:int,seed:int)->Any:
    if len(train_normal)<shots:
        raise ValueError('insufficient normal support')
    order=sorted(range(len(image_ids)),key=lambda j:stable_hash(seed,image_ids[j]))[:shots]
    state=DiagonalNIG(prior)
    for j in order:
        state.update(train_normal[j],1.,robust=False)
    return state


def _source_prior_and_router(artifact:Mapping[str,Any],items:Sequence[Mapping[str,Any]],x:np.ndarray,
                             dev_cats:set[str],router_seed:int)->tuple[Any,Any,Any]:
    train_normals={c:x[[i for i,v in enumerate(items) if v['category']==c and v['role']=='train_normal']]
                   for c in sorted(dev_cats)}
    if not all(len(v)>=2 for v in train_normals.values()):
        raise ValueError('development training normals missing')
    prior=fit_weak_nig_prior(train_normals,kappa0=.01,alpha0=2.5)
    dev_idx=[i for i,v in enumerate(items) if v['category'] in dev_cats]
    dev_x=x[dev_idx]
    train_factor=[i for i in dev_idx if items[i]['role']=='defect' and
                  items[i]['defect_source_offline_only'] in CORE_SET]
    y=np.asarray([items[i]['defect_source_offline_only'] for i in train_factor],dtype=object)
    c=np.asarray([items[i]['category'] for i in train_factor],dtype=object)
    if set(y.tolist())!=CORE_SET:
        raise ValueError('source-CV development products lack Core-4 factor coverage')
    support=category_diverse_support_indices(y,c,CORE4,8,router_seed)
    factor_mem=init_memory(x[train_factor],support)
    shared_prior=fit_shared_diag_prior(dev_x)
    return prior, factor_mem, shared_prior


def _development_threshold(prior:Any,items:Sequence[Mapping[str,Any]],x:np.ndarray,
                           dev_cats:set[str],support_seed:int,shots:int)->float:
    """Fixed 95th percentile of held-in development-product TEST normal surprise."""
    scores=[]
    for cat in sorted(dev_cats):
        train=[i for i,v in enumerate(items) if v['category']==cat and v['role']=='train_normal']
        test=[i for i,v in enumerate(items) if v['category']==cat and v['role']=='test_normal']
        if len(train)<shots or not test:continue
        state=_initial_nig(prior,x[train],[items[i]['image_id'] for i in train],shots,support_seed)
        scores.extend((-state.predictive_log_prob(x[test])).tolist())
    if not scores: raise ValueError('no development normal scores for calibration')
    return float(np.quantile(scores,0.95))


def _score_item(state:BranchState,cat:str, idx:int,x:np.ndarray,pf:np.ndarray,
                ps:np.ndarray,sd:np.ndarray,prior_router:Any,
                kappa:float,initial_raw_score:bool=False)->tuple[float,np.ndarray,np.ndarray,np.ndarray]:
    raw=x[idx]
    if state.hard_normal and state.neg_banks.get(cat) and state.neg_banks[cat].size:
        vec,top = corrected_evidence(pf[idx],ps[idx],float(sd[idx]),state.neg_banks[cat])
    else:
        # Exact original pooled vector, rather than fp16 roundtrip through shortlist.
        vec=raw
        top=np.arange(K,dtype=np.int64)
    surprise=-float(state.normal_by_category[cat].predictive_log_prob(vec.reshape(1,-1))[0])
    scores=score_routing_fractional(vec,state.router_memory,prior_router,kappa)
    return surprise, scores, vec, top


def _sentinel_metrics(state:BranchState, sentinels:Sequence[int],items:Sequence[Mapping[str,Any]],
                      x:np.ndarray,pf:np.ndarray,ps:np.ndarray,sd:np.ndarray,
                      router_prior:Any,threshold:float)->dict[str,Any]:
    truth=[]; surprise=[]; core_y=[]; core_pred=[]; core_scores=[]; reranked=0
    for idx in sentinels:
        itm=items[idx];cat=str(itm['category']); y=int(itm['role']=='defect')
        s, logits, _, top=_score_item(state,cat,idx,x,pf,ps,sd,router_prior,1.0)
        truth.append(y);surprise.append(s)
        reranked+=int(not np.array_equal(top,np.arange(K)))
        source=itm.get('defect_source_offline_only')
        if y and source in CORE_SET:
            core_y.append(str(source));core_pred.append(str(CORE4[int(np.argmax(logits))]));core_scores.append(logits)
    y=np.asarray(truth,dtype=np.int64);score=np.asarray(surprise,dtype=np.float64)
    pred=(score>threshold).astype(np.int64)
    mask_n=(y==0);mask_d=(y==1)
    routing_margin = np.asarray(core_scores, dtype=np.float64)
    if core_y:
        correct_idx = np.asarray([list(CORE4).index(g) for g in core_y])
        own = routing_margin[np.arange(len(correct_idx)), correct_idx]
        other = routing_margin.copy()
        other[np.arange(len(correct_idx)), correct_idx] = -np.inf
        margins = own - other.max(axis=1)
    else:
        margins = np.empty(0,dtype=np.float64)
    result={
        'routing_true_margin_mean':float(margins.mean()) if len(margins) else float('nan'),
        'routing_true_margin_by_source':{
            g:(float(margins[np.asarray(core_y)==g].mean()) if g in core_y else None)
            for g in CORE4},
        'n_sentinel':len(y),'n_normal':int(mask_n.sum()),'n_defect':int(mask_d.sum()),
        'threshold_dev_only':float(threshold),
        'normal_fpr':float(pred[mask_n].mean()) if mask_n.any() else float('nan'),
        'defect_recall':float(pred[mask_d].mean()) if mask_d.any() else float('nan'),
        'auroc':float(roc_auc_score(y,score)) if mask_n.any() and mask_d.any() else float('nan'),
        'ap':float(average_precision_score(y,score)) if mask_n.any() and mask_d.any() else float('nan'),
        'routing_macro_f1':float(f1_score(core_y,core_pred,labels=list(CORE4),average='macro',zero_division=0)) if core_y else float('nan'),
        'n_core4':len(core_y),'spatial_rerank_fraction':float(reranked/len(y)) if len(y) else float('nan'),
        'pred_core4':core_pred, 'true_core4':core_y,
        'core4_logits':np.asarray(core_scores,dtype=np.float64).tolist(),
        'core4_indices':[int(i) for i in sentinels if items[i]['role']=='defect' and items[i].get('defect_source_offline_only') in CORE_SET],
        'n_hard_normal_obs':sum(b.observations for b in state.neg_banks.values()),
        'n_positive_factor_writes':sum(state.defect_writes.values()),
        'n_normal_updates':sum(state.normal_updates.values()),
    }
    return result


def _compact_metrics(m:Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in m.items() if k not in ('pred_core4','true_core4','core4_logits','core4_indices')}


def _harmful_flips(a:Mapping[str,Any],b:Mapping[str,Any])->float:
    if list(a['core4_indices'])!=list(b['core4_indices']):raise AssertionError('unpaired sentinel evaluations')
    ya=np.asarray(a['true_core4'],dtype=object)
    pa=np.asarray(a['pred_core4'],dtype=object)
    pb=np.asarray(b['pred_core4'],dtype=object)
    return float(np.mean((pa==ya)&(pb!=ya))) if len(ya) else float('nan')


def run_one_fold(artifact:Mapping[str,Any], *, cv_fold:int, support_seed:int,
                 mode:str,normal_policy:str,query_budget:int,stream_seed:int,
                 eta:float,shots:int,checkpoints:Sequence[int])->dict[str,Any]:
    items,x,pf,ps,sd=_collect_artifact(artifact)
    test_cats=set(str(c) for c in artifact['source_cv_groups'][cv_fold])
    dev_cats=set(str(c) for c in artifact['source_categories'])-test_cats
    if test_cats & set(artifact['target_categories_untouched']):
        raise AssertionError('outer target category accessed')
    prior, factor_initial, routing_prior=_source_prior_and_router(artifact,items,x,dev_cats,
                                                                  router_seed=support_seed+100000*cv_fold)
    threshold=_development_threshold(prior,items,x,dev_cats,support_seed,shots)
    # Supports and held-out evaluation use only held-out TRAIN normals and TEST images respectively.
    ng={}
    for cat in sorted(test_cats):
        train=[i for i,v in enumerate(items) if v['category']==cat and v['role']=='train_normal']
        ng[cat]=_initial_nig(prior,x[train],[items[i]['image_id'] for i in train],shots,support_seed)
    test_indices=[i for i,v in enumerate(items) if v['category'] in test_cats and v['role'] in ('test_normal','defect')]
    test_items=[items[i] for i in test_indices]
    local_stream,local_sentinel=split_stream_sentinel(test_items,stream_seed+cv_fold)
    stream=[test_indices[i] for i in local_stream]
    sentinel=[test_indices[i] for i in local_sentinel]
    query_pos=uniform_query_positions(len(stream),query_budget,stream_seed+971*cv_fold)
    if len(query_pos)!=query_budget:
        raise ValueError('insufficient held-out stream for query budget')
    schedule=set(int(p) for p in query_pos.tolist())
    if not sentinel: raise ValueError('empty sentinel set')
    specs=branches_for(mode,normal_policy)
    states={name:BranchState(hard,kind,{c:copy.deepcopy(st) for c,st in ng.items()},
                             copy.deepcopy(factor_initial),{},Counter(),Counter())
            for name,(hard,kind) in specs.items()}
    history=SourceCentroids.empty()
    perm=make_derangement(CORE4,seed=stream_seed+cv_fold)
    rows=[];events=[]; n_query=0

    def evaluate(checkpoint:int)->None:
        branch_metrics={name:_sentinel_metrics(st,sentinel,items,x,pf,ps,sd,routing_prior,threshold)
                        for name,st in states.items()}
        keys=list(states)
        refs={name:_harmful_flips(branch_metrics[keys[0]],m)
              for name,m in branch_metrics.items()}
        paired_margin={}
        for left,right in (('N1','N0'),('D1','D0'),('D1','D2'),
                           ('C10','C00'),('C01','C00'),('C11','C00')):
            if left in branch_metrics and right in branch_metrics:
                a,b=branch_metrics[left],branch_metrics[right]
                diff={g:(float(a['routing_true_margin_by_source'][g]-b['routing_true_margin_by_source'][g])
                         if a['routing_true_margin_by_source'][g] is not None
                         and b['routing_true_margin_by_source'][g] is not None else None)
                      for g in CORE4}
                paired_margin[f'{left}-minus-{right}']= {
                    'true_margin_by_source':diff,
                    'harmful_flips_from_right':_harmful_flips(b,a),
                    'helpful_flips_from_right':_harmful_flips(a,b),
                }
        rows.append({'checkpoint':checkpoint,'branches':{n:_compact_metrics(m) for n,m in branch_metrics.items()},
                     'harmful_routing_flips_vs_'+keys[0]:refs,
                     'paired_routing_effects':paired_margin})

    if 0 in checkpoints:evaluate(0)
    for pos,item_idx in enumerate(stream):
        itm=items[item_idx];cat=str(itm['category'])
        # Predictions must be recorded BEFORE feedback, on the SAME item, in EVERY branch.
        scores_before={name:_score_item(st,cat,item_idx,x,pf,ps,sd,routing_prior,1.0)[0]
                       for name,st in states.items()}
        if pos not in schedule:continue
        n_query+=1
        is_defect=(itm['role']=='defect')
        source=itm.get('defect_source_offline_only') if is_defect else None
        # Fixed responsibility oracle: initial source-side routing memory only.
        # Current source label is NOT used to compute r_t.
        raw_logits=score_routing_fractional(x[item_idx],factor_initial,routing_prior,1.0)
        r=routing_responsibility(raw_logits) if is_defect and source in CORE_SET else None
        receipts={}
        for name,st in states.items():
            pre_defect=bool(scores_before[name]>threshold)
            wrong=(pre_defect != is_defect)
            receipts[name]=apply_verified_feedback(
                st,is_defect=is_defect,is_false_alarm=wrong,category=cat,
                vector_raw=x[item_idx],top8_patches=pf[item_idx,:K],
                source=str(source) if source else None,responsibility=r,
                history=history,permutation=perm,eta=eta,
            )
        # NIG states must agree across branches regardless of their correction/routing state.
        names=list(states)
        for other in names[1:]:
            if states[other].fingerprint_nig(cat)!=states[names[0]].fingerprint_nig(cat):
                raise AssertionError('NIG trajectory changed across branches')
        # NEVER reveal current source to source-memory until all writes have completed.
        if is_defect and source in CORE_SET:
            history.add(str(source),r)
        events.append({'position':int(pos),'item_id':str(itm['image_id']),
                       'feedback':'defect' if is_defect else 'normal',
                       'verified_source':str(source) if is_defect else None,
                       'source_history_before':history.count.get(str(source),0)-int(is_defect and source in CORE_SET)
                           if source in CORE_SET else None,
                       'receipts':receipts})
        if n_query in checkpoints:evaluate(n_query)
    if n_query!=query_budget:raise AssertionError('query-count mismatch')
    return {'source_cv_fold':cv_fold,'support_seed':support_seed,'mode':mode,
            'normal_policy':normal_policy,'test_categories':sorted(test_cats),
            'dev_categories':sorted(dev_cats),'query_positions':query_pos.tolist(),
            'query_budget':n_query,'n_stream':len(stream),'n_sentinel':len(sentinel),
            'threshold_dev_only':threshold,'permutation':perm,
            'checkpoints':rows,'query_events':events,
            'n_queried_normals':sum(e['feedback']=='normal' for e in events),
            'n_queried_defects':sum(e['feedback']=='defect' for e in events),
            'n_core4_queries':sum(e['verified_source'] in CORE_SET for e in events)}


def summarize(runs:Sequence[Mapping[str,Any]],checkpoint:int)->dict[str,Any]:
    table:Dict[str,Dict[str,list[float]]]=defaultdict(lambda:defaultdict(list))
    contrast:Dict[str,Dict[str,list[float]]]=defaultdict(lambda:defaultdict(list))
    for run in runs:
        cps=[cp for cp in run['checkpoints'] if cp['checkpoint']==checkpoint]
        if len(cps)!=1:raise ValueError('missing final checkpoint')
        m=cps[0]['branches']
        for branch,metrics in m.items():
            for key in ('normal_fpr','defect_recall','auroc','ap','routing_macro_f1','routing_true_margin_mean',
                        'spatial_rerank_fraction','n_normal_updates','n_hard_normal_obs','n_positive_factor_writes'):
                table[branch][key].append(float(metrics[key]))
        for a,b in (('N1','N0'),('D1','D0'),('D1','D2'),('D3','D1'),
                    ('C10','C00'),('C01','C00'),('C11','C00')):
            if a in m and b in m:
                for key in ('normal_fpr','defect_recall','auroc','routing_macro_f1'):
                    contrast[a+'-minus-'+b][key].append(float(m[a][key])-float(m[b][key]))
    def summarize_values(vs:Sequence[float])->dict[str,Any]:
        a=np.asarray(vs,dtype=np.float64)
        a=a[np.isfinite(a)]
        return {'mean':float(np.mean(a)) if len(a) else None,
                'std':float(np.std(a)) if len(a) else None,
                'n':int(len(a)),'positive_count':int((a>0).sum())}
    return {'final_checkpoint':checkpoint,
            'branches':{b:{k:summarize_values(v) for k,v in d.items()} for b,d in table.items()},
            'paired_differences':{b:{k:summarize_values(v) for k,v in d.items()} for b,d in contrast.items()},
            'run_count':len(runs)}


def run(args:argparse.Namespace)->int:
    if args.shots!=4 or args.query_budget!=32 or args.eta_source!=.5:
        raise ValueError('Primary E4 fixed at four state supports, 32 feedbacks, eta_source=0.5')
    checkpoints=tuple(int(s) for s in args.checkpoints.split(','))
    if checkpoints!=CHECKPOINTS:raise ValueError(f'primary checkpoints must be {CHECKPOINTS}')
    try:
        art=torch.load(args.evidence,map_location='cpu',weights_only=False)
    except TypeError:
        art=torch.load(args.evidence,map_location='cpu')
    if art.get('outer_fold')!=args.fold:raise ValueError('evidence outer-fold mismatch')
    _collect_artifact(art)
    ncv=len(art['source_cv_groups'])
    folds=list(range(ncv)) if args.source_cv_folds=='all' else [int(v) for v in args.source_cv_folds.split(',')]
    if not folds or any(v<0 or v>=ncv for v in folds): raise ValueError('source CV fold index invalid')
    seeds=[int(s) for s in args.support_seeds.split(',')]
    if not seeds:raise ValueError('support seeds missing')
    results=[]
    for cv in folds:
        for seed in seeds:
            print(f'E4 {args.mode} source-CV {cv}/{ncv} support seed {seed}',flush=True)
            r=run_one_fold(art,cv_fold=cv,support_seed=seed,mode=args.mode,
                           normal_policy=args.normal_policy,query_budget=args.query_budget,
                           stream_seed=args.stream_seed,eta=args.eta_source,
                           shots=args.shots,checkpoints=checkpoints)
            results.append(r)
            print(f"  queries: normal={r['n_queried_normals']} defect={r['n_queried_defects']} Core4={r['n_core4_queries']}",flush=True)
    output=Path(args.out_dir)/f'fold_{args.fold}'
    output.mkdir(parents=True,exist_ok=True)
    payload={'schema':'abmg.e4.audit.v1','outer_fold':args.fold,
             'target_categories_untouched':art['target_categories_untouched'],
             'source_cv_folds':folds,'mode':args.mode,'normal_policy':args.normal_policy,
             'config':{'shots':args.shots,'eta_source':args.eta_source,
                       'query_budget':args.query_budget,'support_seeds':seeds,
                       'stream_seed':args.stream_seed,'checkpoints':list(checkpoints),
                       'candidate_limit':art['candidate_limit'],
                       'sensor_fingerprint':art['sensor_fingerprint']},
             'results':results,'summary':summarize(results,checkpoints[-1]),
             'limitation':'source-side CV only; cached top-M patch reranking; uncalibrated factor responsibility; no target confirmation, no query controller'}
    path=output/f'e4_{args.mode}_{args.normal_policy}_audit.json'
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(payload['summary'],indent=2))
    print(f'Wrote {path}')
    return 0


def make_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--evidence',required=True)
    p.add_argument('--out-dir',default='outputs/e4_asymmetric_feedback')
    p.add_argument('--mode',choices=('e4a','e4b','e4c'),default='e4a')
    p.add_argument('--normal-policy',choices=('n0','n1'),default='n0',
                   help='Only E4B; set based on independently evaluated E4A gate')
    p.add_argument('--source-cv-folds',default='all')
    p.add_argument('--support-seeds',default='0,1,2')
    p.add_argument('--shots',type=int,default=4)
    p.add_argument('--query-budget',type=int,default=32)
    p.add_argument('--checkpoints',default='0,4,8,16,32')
    p.add_argument('--eta-source',type=float,default=.5)
    p.add_argument('--stream-seed',type=int,default=20261008)
    return p


def main(argv:Optional[Sequence[str]]=None)->int:
    return run(make_parser().parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
