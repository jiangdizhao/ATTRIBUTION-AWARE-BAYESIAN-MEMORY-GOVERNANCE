#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract one frozen raw-DINO evidence cache for ABMG E3.

The detector is fixed to the established 4-shot normal-NN Sensor.  Every image
is represented by the score-softmax weighted mean of its frozen Sensor top-8
DINO patch descriptors.  The output vector is L2-normalized.

State-support normals are deliberately separated from detector-support images.
Only outer-fold source categories are opened.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from abmg_stage0_foundation import (
    PaperAlignedFrozenSensor,
    Stage0Config,
    Stage0Record,
    config_fingerprint,
    load_realiad_records,
    parse_layers,
    select_supports,
    set_seed,
)
from abmg_factorizability_audit import (
    DEFAULT_FOLDS,
    deterministic_stratified_defect_sample,
    load_fold_protocol,
    nearest_cosine,
    source_defect_records,
)


PRIMARY_SENSOR_K = 8
PRIMARY_SCORE_TEMPERATURE = 20.0


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


@torch.no_grad()
def pool_topk_raw(
    qf: torch.Tensor,
    patch_scores: torch.Tensor,
    *,
    k: int,
    temperature: float,
) -> torch.Tensor:
    kk = min(max(1, int(k)), int(patch_scores.numel()))
    idx = torch.topk(patch_scores, k=kk, largest=True).indices
    w = torch.softmax(float(temperature) * patch_scores[idx].float(), dim=0)
    z = torch.sum(w[:, None] * qf[idx].float(), dim=0)
    return torch.nn.functional.normalize(z, dim=0)


def _records_for_category(
    records: Sequence[Stage0Record],
    category: str,
    defect_keep_ids: set[str],
    detector_support_ids: set[str],
) -> List[Tuple[str, Stage0Record]]:
    out: List[Tuple[str, Stage0Record]] = []
    for r in records:
        if r.category != category:
            continue
        if r.split == "train" and r.is_good:
            if r.image_id not in detector_support_ids:
                out.append(("train_normal", r))
        elif r.split == "test" and r.is_good:
            out.append(("test_normal", r))
        elif r.split == "test" and (not r.is_good):
            if r.image_id in defect_keep_ids:
                out.append(("defect", r))
    return sorted(
        out,
        key=lambda x: (x[0], x[1].defect_source, x[1].relative_path),
    )


def run(args: argparse.Namespace) -> int:
    if int(args.sensor_k) != PRIMARY_SENSOR_K:
        raise ValueError("E3 extraction fixes Sensor top-k at 8")
    if float(args.score_temperature) != PRIMARY_SCORE_TEMPERATURE:
        raise ValueError("E3 extraction fixes score-softmax temperature at 20")
    if int(args.detector_shots) != 4:
        raise ValueError("E3 detector support is fixed at 4 shots")

    protocol = load_fold_protocol(args.fold, args.folds)
    # HARD DATA BOUNDARY: outer target category JSONs are not loaded.
    records = load_realiad_records(
        args.root,
        args.json_dir,
        set(protocol.source_categories),
    )
    defects = deterministic_stratified_defect_sample(
        source_defect_records(records, protocol.source_categories),
        max_per_type_per_category=int(args.max_per_type_per_category),
        seed=int(args.seed),
    )
    defect_keep_ids = {str(r.image_id) for r in defects}

    cfg = Stage0Config(
        model_name=args.model_name,
        layers=parse_layers(args.layers),
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        patch_size=int(args.patch_size),
        layer_fusion="mean",
        support_augmentation="paper_geometric",
        query_view="identity",
        top_fraction=0.01,
        knn=1,
        shots=int(args.detector_shots),
        support_seed=int(args.detector_support_seed),
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors: List[torch.Tensor] = []
    items: List[Dict[str, Any]] = []
    support_meta: Dict[str, Any] = {}
    role_counts: Counter[str] = Counter()
    category_role_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    sensor = PaperAlignedFrozenSensor(
        cfg,
        device=args.device,
        use_fp16=not args.no_fp16,
    )
    try:
        categories = sorted(protocol.source_categories)
        for ci, category in enumerate(categories, 1):
            detector_support = select_supports(
                records,
                category,
                int(args.detector_shots),
                int(args.detector_support_seed),
            )
            detector_support_ids = {str(r.image_id) for r in detector_support}
            bank = sensor.build_category_bank(detector_support)
            rows = _records_for_category(
                records,
                category,
                defect_keep_ids,
                detector_support_ids,
            )
            support_meta[category] = {
                "detector_support_image_ids": [
                    str(r.image_id) for r in detector_support
                ],
                "n_detector_bank_patches": int(bank.shape[0]),
            }
            print(
                f"[{ci}/{len(categories)}] {category}: "
                f"detector_support={len(detector_support)}, evidence_items={len(rows)}"
            )

            for start in range(0, len(rows), int(args.query_batch_size)):
                block = rows[start : start + int(args.query_batch_size)]
                recs = [r for _, r in block]
                q_batch, _ = sensor.encode_path_batch(recs)

                for bi, (role, rec) in enumerate(block):
                    qf = q_batch[bi]
                    best_sim, _ = nearest_cosine(
                        qf,
                        bank,
                        chunk=int(args.nn_chunk),
                    )
                    patch_scores = 1.0 - best_sim
                    vec = pool_topk_raw(
                        qf,
                        patch_scores,
                        k=int(args.sensor_k),
                        temperature=float(args.score_temperature),
                    )
                    vectors.append(vec.detach().cpu().to(torch.float16))
                    items.append(
                        {
                            "item_index": len(items),
                            "image_id": str(rec.image_id),
                            "category": str(rec.category),
                            "role": str(role),
                            "split": str(rec.split),
                            "is_good": bool(rec.is_good),
                            "defect_source_offline_only": (
                                str(rec.defect_source)
                                if role == "defect"
                                else None
                            ),
                            "relative_path": str(rec.relative_path),
                        }
                    )
                    role_counts[role] += 1
                    category_role_counts[category][role] += 1

                done = min(start + len(block), len(rows))
                if int(args.progress_every) > 0 and (
                    done % int(args.progress_every) == 0
                    or done == len(rows)
                ):
                    print(f"  {done}/{len(rows)}")

            del bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    if not vectors:
        raise RuntimeError("no E3 evidence vectors extracted")
    evidence = torch.stack(vectors, dim=0).contiguous()

    artifact = {
        "schema": "abmg.e3.evidence.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
        "sensor_config": asdict(cfg),
        "sensor_fingerprint": config_fingerprint(cfg),
        "evidence_definition": (
            "L2-normalized score-softmax pooled frozen-DINO Sensor top-8"
        ),
        "items": items,
        "evidence": evidence,
        "detector_support": support_meta,
    }
    torch.save(artifact, out_dir / "e3_evidence.pt")

    manifest = {
        "schema": "abmg.e3.evidence_manifest.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "n_items": int(len(items)),
        "evidence_dim": int(evidence.shape[1]),
        "role_counts": dict(sorted(role_counts.items())),
        "category_role_counts": {
            c: dict(sorted(v.items()))
            for c, v in sorted(category_role_counts.items())
        },
        "detector_support": support_meta,
        "defect_sampling": {
            "max_per_type_per_category": int(
                args.max_per_type_per_category
            ),
            "n_defect_items": int(role_counts.get("defect", 0)),
        },
    }
    _json_dump(manifest, out_dir / "e3_evidence_manifest.json")
    _json_dump(
        {
            "schema": "abmg.e3.extract_config.v1",
            "outer_fold": int(args.fold),
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "sensor_k": int(args.sensor_k),
            "score_temperature": float(args.score_temperature),
            "max_per_type_per_category": int(
                args.max_per_type_per_category
            ),
            "seed": int(args.seed),
        },
        out_dir / "resolved_config.json",
    )

    print(json.dumps(manifest, indent=2, allow_nan=True))
    print(f"Wrote E3 evidence to: {out_dir}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--json-dir", type=str, required=True)
    p.add_argument("--folds", type=str, default=str(DEFAULT_FOLDS))
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/e3_evidence",
    )

    p.add_argument("--model-name", type=str, default="dinov2_vitl14_reg")
    p.add_argument("--layers", type=str, default="4-18")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--detector-shots", type=int, default=4)
    p.add_argument("--detector-support-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--query-batch-size", type=int, default=2)
    p.add_argument("--nn-chunk", type=int, default=256)
    p.add_argument("--sensor-k", type=int, default=PRIMARY_SENSOR_K)
    p.add_argument(
        "--score-temperature",
        type=float,
        default=PRIMARY_SCORE_TEMPERATURE,
    )

    p.add_argument(
        "--max-per-type-per-category",
        type=int,
        default=64,
        help="Screening default=64; use 0 later only for full confirmation.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=64)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
