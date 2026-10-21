#!/usr/bin/env python3
"""Read-only Real-IAD JSON label inventory; no images, DINO, or target-fold training.

Counting semantics match scripts/abmg_stage0_foundation.py:
- each <json-dir>/*.json is a source dataset metadata file;
- product is rec.category, falling back to the JSON filename;
- a record is normal if anomaly_class (fallback defect_type, then normal_class)
  equals the file's meta.normal_class (fallback 'OK');
- train/test are kept separate; no samples are chosen for experimentation.

Important: missing labels default to normal in the predecessor loader. This tool
counts them the same way *but flags them explicitly* as implicit-normal records.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "abmg.realiad_inventory.v1"
SPLITS = ("train", "test")
COUNT_FIELDS = (
    "train_normal", "train_defect", "test_normal", "test_defect",
    "train_defect_with_mask", "train_defect_without_mask",
    "test_defect_with_mask", "test_defect_without_mask",
)
ISSUE_EXAMPLE_LIMIT = 25


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _new_counts() -> Counter[str]:
    return Counter({field: 0 for field in COUNT_FIELDS})


def _report_counts(c: Counter[str]) -> dict[str, int | float]:
    result: dict[str, int | float] = {field: int(c[field]) for field in COUNT_FIELDS}
    result["normal"] = int(c["train_normal"] + c["test_normal"])
    result["defect"] = int(c["train_defect"] + c["test_defect"])
    result["train"] = int(c["train_normal"] + c["train_defect"])
    result["test"] = int(c["test_normal"] + c["test_defect"])
    result["total"] = int(result["normal"] + result["defect"])
    result["normal_fraction"] = round(float(result["normal"]) / result["total"], 6) if result["total"] else 0.0
    result["defect_fraction"] = round(float(result["defect"]) / result["total"], 6) if result["total"] else 0.0
    return result


def _present_path(root: Path, prefix: str, rel: str) -> bool:
    """Match the path candidates in abmg_stage0_foundation._resolve_existing."""
    if not rel:
        return False
    relative = rel.replace("\\", "/").lstrip("/")
    prefix = prefix.replace("\\", "/").strip("/")
    return any(p.is_file() for p in (
        root / "images" / prefix / relative,
        root / "realiad_1024" / prefix / relative,
        root / prefix / relative,
        root / relative,
    ))


def inventory(root: Path, json_dir: Path, verify_files: bool = False) -> dict[str, Any]:
    if not json_dir.is_dir():
        raise ValueError(f"JSON directory does not exist: {json_dir}")
    if verify_files and not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No *.json files found in {json_dir}")

    total = _new_counts()
    product = defaultdict(_new_counts)
    defects_by_type = defaultdict(_new_counts)
    defects_product_type = defaultdict(_new_counts)
    product_sources: dict[str, set[str]] = defaultdict(set)
    normal_names: dict[str, set[str]] = defaultdict(set)
    issues: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    n_implicit_normal = 0
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    known_split: dict[tuple[str, str], str] = {}

    def issue(code: str, file: str, split: str, row: int, details: str) -> None:
        issues[code] += 1
        if len(examples) < ISSUE_EXAMPLE_LIMIT:
            examples.append({"code": code, "file": file, "split": split,
                             "row": row, "details": details})

    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed reading {file}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {file}")
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"Expected meta object in {file}")
        normal_label = _clean(meta.get("normal_class")) or "OK"
        prefix = _clean(meta.get("prefix"))
        if any(key not in data for key in SPLITS):
            issue("missing_split_key", file.name, "", 0, "Expected train and test keys")
        for split in SPLITS:
            entries = data.get(split, [])
            if entries is None:
                entries = []
            if not isinstance(entries, list):
                raise ValueError(f"Expected '{split}' array in {file}")
            for i, rec in enumerate(entries):
                if not isinstance(rec, dict):
                    raise ValueError(f"Expected record object in {file} {split}[{i}]")
                product_name = _clean(rec.get("category")) or file.stem
                product_sources[product_name].add(file.name)
                normal_names[product_name].add(normal_label)
                if product_name != file.stem:
                    issue("category_filename_mismatch", file.name, split, i,
                          f"record.category={product_name!r} vs JSON file stem={file.stem!r}")

                label = _clean(rec.get("anomaly_class"))
                fallback = _clean(rec.get("defect_type"))
                if not label:
                    label = fallback
                if not label:
                    # Intentional parity with Stage-0 loader; explicit audit warning.
                    label = normal_label
                    n_implicit_normal += 1
                    issue("implicit_normal_label", file.name, split, i,
                          "Neither anomaly_class nor defect_type present; Stage-0 treats as normal")
                elif fallback and _clean(rec.get("anomaly_class")) and fallback != label:
                    issue("anomaly_class_defect_type_disagree", file.name, split, i,
                          f"anomaly_class={label!r}, defect_type={fallback!r}")
                is_normal = label == normal_label

                img_rel = _clean(rec.get("image_path"))
                mask_rel = _clean(rec.get("mask_path"))
                if not img_rel:
                    issue("missing_image_path", file.name, split, i, "Empty image_path")
                else:
                    # Duplicate metadata rows remain counted; flag them explicitly.
                    key = (product_name, img_rel.replace("\\", "/"))
                    if key[1] in seen[(product_name, split)]:
                        issue("duplicate_image_in_split", file.name, split, i, img_rel)
                    seen[(product_name, split)].add(key[1])
                    if key in known_split and known_split[key] != split:
                        issue("image_in_both_splits", file.name, split, i, img_rel)
                    known_split[key] = split

                field = f"{split}_{'normal' if is_normal else 'defect'}"
                total[field] += 1
                product[product_name][field] += 1
                if not is_normal:
                    defects_by_type[label][field] += 1
                    defects_product_type[(product_name, label)][field] += 1
                    mask_field = f"{split}_defect_{'with_mask' if mask_rel else 'without_mask'}"
                    for group in (total, product[product_name], defects_by_type[label],
                                  defects_product_type[(product_name, label)]):
                        group[mask_field] += 1
                elif mask_rel:
                    issue("normal_has_mask_path", file.name, split, i, mask_rel)

                if verify_files:
                    if not _present_path(root, prefix, img_rel):
                        issue("image_file_missing", file.name, split, i, img_rel)
                    if mask_rel and not _present_path(root, prefix, mask_rel):
                        issue("mask_file_missing", file.name, split, i, mask_rel)

    product_rows = [dict(product=p, source_json_files=sorted(product_sources[p]),
                         normal_label_values=sorted(normal_names[p]), **_report_counts(product[p]))
                    for p in sorted(product)]
    type_rows = [dict(defect_type=t, **_report_counts(defects_by_type[t]))
                 for t in sorted(defects_by_type)]
    product_type_rows = [dict(product=p, defect_type=t, **_report_counts(defects_product_type[(p,t)]))
                         for p,t in sorted(defects_product_type)]

    return {
        "schema": SCHEMA,
        "dataset_root": str(root),
        "json_dir": str(json_dir),
        "source_json_files": [f.name for f in files],
        "definitions": {
            "record_unit": "one entry in train[] or test[]; repeated paths counted and reported",
            "normal": "anomaly_class (fallback defect_type, then meta.normal_class) == meta.normal_class",
            "defect_type": "non-normal anomaly_class (fallback defect_type); exact original spelling retained",
            "mask": "mask_path is non-empty; availability of physical files only if --verify-files",
            "implicit_normal": "neither label present; counted normal to match Stage-0 loader, flagged",
        },
        "verify_files": verify_files,
        "overall": _report_counts(total),
        "n_json_files": len(files),
        "n_products": len(product_rows),
        "n_distinct_defect_types": len(type_rows),
        "n_implicit_normal_labels": n_implicit_normal,
        "per_product": product_rows,
        "per_defect_type": type_rows,
        "per_product_defect_type": product_type_rows,
        "issues": {"counts": dict(sorted(issues.items())), "examples": examples,
                   "examples_capped_at": ISSUE_EXAMPLE_LIMIT},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], leading: list[str]) -> None:
    cols = leading + [field for field in COUNT_FIELDS] + [
        "train", "test", "normal", "defect", "total", "normal_fraction", "defect_fraction"]
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ";".join(row[k]) if isinstance(row.get(k), list) else row.get(k, "")
                             for k in cols})


def write_outputs(result: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "realiad_inventory.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    product_csv = out_dir / "per_product.csv"
    type_csv = out_dir / "per_defect_type.csv"
    product_type_csv = out_dir / "per_product_defect_type.csv"
    _write_csv(product_csv, result["per_product"], ["product", "source_json_files", "normal_label_values"])
    _write_csv(type_csv, result["per_defect_type"], ["defect_type"])
    _write_csv(product_type_csv, result["per_product_defect_type"], ["product", "defect_type"])
    return [json_path, product_csv, type_csv, product_type_csv]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Real-IAD dataset root")
    parser.add_argument("--json-dir", type=Path, required=True, help="Directory of per-product Real-IAD JSONs")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/realiad_inventory"))
    parser.add_argument("--verify-files", action="store_true", help="Also check image/mask path existence (slower)")
    parser.add_argument("--fail-on-issues", action="store_true", help="Exit 2 if any inventory warnings are found")
    args = parser.parse_args(argv)
    try:
        result = inventory(args.root, args.json_dir, args.verify_files)
        paths = write_outputs(result, args.out_dir)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    o = result["overall"]
    print(f"Real-IAD inventory: {result['n_products']} products, {result['n_distinct_defect_types']} defect types")
    print(f"Train: {o['train_normal']} normal / {o['train_defect']} defect")
    print(f"Test:  {o['test_normal']} normal / {o['test_defect']} defect")
    print(f"Total: {o['normal']} normal / {o['defect']} defect ({o['defect_fraction']:.2%} defect)")
    print(f"Defect mask paths declared: {o['train_defect_with_mask'] + o['test_defect_with_mask']}")
    print(f"Implicit-normal labels: {result['n_implicit_normal_labels']}")
    print("Inventory warnings: " + (str(result['issues']['counts']) if result['issues']['counts'] else "none"))
    for p in paths:
        print("Saved: " + str(p))
    return 2 if args.fail_on_issues and result["issues"]["counts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
