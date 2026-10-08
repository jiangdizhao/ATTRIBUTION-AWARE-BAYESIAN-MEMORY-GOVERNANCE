#!/usr/bin/env python3
"""Extract source-only E4 evidence including bounded raw-DINO patch candidates.

This is intentionally separate from the E3 extractor. E3 pooled-only artifacts
cannot reproduce post-feedback spatial rescoring. We cache the highest-scoring
64 frozen-Sensor patches per image (or another predeclared candidate count) and
the standard deviation of ALL frozen patch scores. Corrections cannot recover
patches outside the cached candidate set; that limitation is reported.

No ground-truth defect mask is loaded. Offline source codes are stored for the
simulated feedback/evaluator, never passed into Sensor feature extraction.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

try:
    from abmg_e3_extract_evidence import (
        _records_for_category, pool_topk_raw, PRIMARY_SENSOR_K,
        PRIMARY_SCORE_TEMPERATURE,
    )
    from abmg_stage0_foundation import (
        PaperAlignedFrozenSensor, Stage0Config, config_fingerprint,
        load_realiad_records, parse_layers, select_supports, set_seed,
    )
    from abmg_factorizability_audit import (
        DEFAULT_FOLDS, deterministic_stratified_defect_sample,
        load_fold_protocol, nearest_cosine, source_defect_records,
    )
except ImportError:
    from scripts.abmg_e3_extract_evidence import (
        _records_for_category, pool_topk_raw, PRIMARY_SENSOR_K,
        PRIMARY_SCORE_TEMPERATURE,
    )
    from scripts.abmg_stage0_foundation import (
        PaperAlignedFrozenSensor, Stage0Config, config_fingerprint,
        load_realiad_records, parse_layers, select_supports, set_seed,
    )
    from scripts.abmg_factorizability_audit import (
        DEFAULT_FOLDS, deterministic_stratified_defect_sample,
        load_fold_protocol, nearest_cosine, source_defect_records,
    )


def run(args: argparse.Namespace) -> int:
    if args.sensor_k != PRIMARY_SENSOR_K or args.score_temperature != PRIMARY_SCORE_TEMPERATURE:
        raise ValueError('E4 freezes top-8 and score temperature 20, matching E3')
    if args.detector_shots != 4:
        raise ValueError('E4 freezes detector support at four shots')
    if args.candidates < args.sensor_k:
        raise ValueError('candidate shortlist must include the frozen top-8')
    protocol = load_fold_protocol(args.fold, args.folds)
    records = load_realiad_records(args.root, args.json_dir, set(protocol.source_categories))
    defects = deterministic_stratified_defect_sample(
        source_defect_records(records, protocol.source_categories),
        max_per_type_per_category=args.max_per_type_per_category,
        seed=args.seed,
    )
    defect_ids = {str(r.image_id) for r in defects}
    cfg = Stage0Config(
        model_name=args.model_name, layers=parse_layers(args.layers),
        resize_size=args.resize_size, crop_size=args.crop_size, patch_size=args.patch_size,
        layer_fusion='mean', support_augmentation='paper_geometric', query_view='identity',
        top_fraction=0.01, knn=1, shots=4, support_seed=args.detector_support_seed,
        cache_dtype='float16' if not args.no_fp16 else 'float32',
    )
    set_seed(args.seed)
    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    vectors: list[torch.Tensor] = []
    patch_features: list[torch.Tensor] = []
    patch_scores_out: list[torch.Tensor] = []
    patch_std: list[float] = []
    items: list[dict[str, Any]] = []
    supports: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    try:
        for ci, category in enumerate(sorted(protocol.source_categories), 1):
            support = select_supports(records, category, 4, args.detector_support_seed)
            support_ids = {str(r.image_id) for r in support}
            bank = sensor.build_category_bank(support)
            rows = _records_for_category(
                records, category, defect_ids, support_ids,
                max_train_normal=args.max_train_normal_per_category,
                max_test_normal=args.max_test_normal_per_category,
                normal_sample_seed=args.normal_sample_seed,
            )
            supports[category] = sorted(support_ids)
            print(f'[{ci}/{len(protocol.source_categories)}] {category}: {len(rows)} items', flush=True)
            for start in range(0, len(rows), args.query_batch_size):
                block = rows[start:start + args.query_batch_size]
                q_batch, _ = sensor.encode_path_batch([r for _, r in block])
                b, p, dim = q_batch.shape
                best, _ = nearest_cosine(q_batch.reshape(b*p, dim), bank, chunk=args.nn_chunk)
                score_batch = (1.0 - best).reshape(b, p)
                for i, (role, r) in enumerate(block):
                    q = q_batch[i].float()
                    scores = score_batch[i].float()
                    order = torch.argsort(scores, descending=True, stable=True)[:args.candidates]
                    # The first eight are always the E3 frozen Sensor evidence.
                    z = pool_topk_raw(q, scores, k=8, temperature=20.0)
                    vectors.append(z.cpu().to(torch.float16))
                    patch_features.append(q[order].cpu().to(torch.float16))
                    patch_scores_out.append(scores[order].cpu().to(torch.float16))
                    patch_std.append(float(scores.std(unbiased=False).item()))
                    items.append({
                        'item_index': len(items), 'image_id': str(r.image_id),
                        'category': str(r.category), 'role': str(role),
                        'split': str(r.split), 'is_good': bool(r.is_good),
                        'defect_source_offline_only': str(r.defect_source) if role == 'defect' else None,
                        'relative_path': str(r.relative_path),
                    })
                    counts[role] += 1
            del bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()
    if not items:
        raise RuntimeError('empty E4 evidence')
    out = Path(args.out_dir) / f'fold_{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        'schema': 'abmg.e4.evidence.v1',
        'outer_fold': int(args.fold),
        'source_categories': list(protocol.source_categories),
        'source_cv_groups': [list(x) for x in protocol.source_cv_groups],
        'target_categories_untouched': list(protocol.target_categories),
        'sensor_config': asdict(cfg), 'sensor_fingerprint': config_fingerprint(cfg),
        'evidence_definition': 'raw DINO Sensor top-8 score-softmax pooled and L2-normalized',
        'candidate_limit': int(args.candidates),
        'candidate_limit_caveat': 'corrections only rerank cached top-M patches; outside top-M inaccessible',
        'items': items, 'evidence': torch.stack(vectors),
        'candidate_features': torch.stack(patch_features),
        'candidate_scores': torch.stack(patch_scores_out),
        'full_patch_score_std': torch.tensor(patch_std, dtype=torch.float32),
        'detector_support_image_ids': supports,
    }
    torch.save(artifact, out / 'e4_evidence.pt')
    manifest = {
        'schema':'abmg.e4.evidence_manifest.v1', 'outer_fold':args.fold,
        'target_categories_untouched':list(protocol.target_categories),
        'source_categories':list(protocol.source_categories),
        'source_cv_groups':[list(x) for x in protocol.source_cv_groups],
        'n_items':len(items), 'role_counts':dict(counts),
        'candidates':args.candidates, 'feature_dim':int(artifact['evidence'].shape[1]),
        'sensor_fingerprint':artifact['sensor_fingerprint'],
        'evidence_path':str(out/'e4_evidence.pt'),
        'limitation':artifact['candidate_limit_caveat'],
    }
    (out/'e4_evidence_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(manifest, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--json-dir',required=True)
    p.add_argument('--folds',default=str(DEFAULT_FOLDS))
    p.add_argument('--out-dir',default='outputs/e4_evidence')
    p.add_argument('--model-name',default='dinov2_vitl14_reg')
    p.add_argument('--layers',default='4-18')
    p.add_argument('--resize-size',type=int,default=448)
    p.add_argument('--crop-size',type=int,default=392)
    p.add_argument('--patch-size',type=int,default=14)
    p.add_argument('--detector-shots',type=int,default=4)
    p.add_argument('--detector-support-seed',type=int,default=0)
    p.add_argument('--sensor-k',type=int,default=8)
    p.add_argument('--score-temperature',type=float,default=20.0)
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--max-per-type-per-category',type=int,default=16)
    p.add_argument('--max-train-normal-per-category',type=int,default=32)
    p.add_argument('--max-test-normal-per-category',type=int,default=64)
    p.add_argument('--normal-sample-seed',type=int,default=271828)
    p.add_argument('--query-batch-size',type=int,default=4)
    p.add_argument('--nn-chunk',type=int,default=2048)
    p.add_argument('--device',default='cuda')
    p.add_argument('--no-fp16',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    return p


def main(argv:Optional[Sequence[str]]=None)->int:
    return run(make_parser().parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
