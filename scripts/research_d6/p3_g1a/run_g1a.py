"""One-shot LP-D6-P3-G1A preparation runner; never trains or scores labels."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import g1a_common as c

PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
REPORT_DIR = ROOT / "docs/research_d6/p3_g1a"


def write_csv_once(path: Path, rows: list[dict], fields=None):
    return c.write_csv_once(path, rows, fields)


def save_public_reports(public: dict, product_rows: list[dict], bin_rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = ["| 产品 | SHA 成功 | 特征成功 | Representative 组 | Stress bin | 最终组 |", "|---|---:|---:|---:|---|---:|"]
    table.extend(f"| {r['product']} | {r['sha_success']} | {r['feature_success']} | {r['representative_groups']} | {r['stress_bins']} | {r['development_groups']} |" for r in product_rows)
    lines = [
        "# P3-G1A 冻结特征与无标签分层报告", "",
        f"状态：`{public['status']}`", "",
        f"Git：`{public['branch']}` / `{public['head']}`；授权记录：`{c.AUTHORIZATION}`。", "",
        "## 执行摘要", "",
        f"白名单 {public['authorized_views']} 张；原图 SHA 成功 {public['sha_success']} 张、失败 {public['sha_failure']} 张；DINOv2 特征成功 {public['feature_success']} 张。",
        f"H12 重建差异最大值 {public.get('h12_max_abs_score_delta', '未运行')}（容差 1e-8）；历史纠正锚点核验 {public.get('anchor_count', 0)}/5。",
        f"最终开发角色 {public.get('development_groups', 0)} 组 / {public.get('development_views', 0)} 视角；D-cal {public.get('d_cal_groups', 0)} 组，D-eval {public.get('d_eval_groups', 0)} 组。", "",
        "## 产品级聚合", "", *table, "",
        "## 预注册 Stress bin 覆盖", "",
        "| 产品 | 低置信低距 | 低置信高距 | 高置信低距 | 高置信高距 | 状态 |", "|---|---:|---:|---:|---:|---|"]
    for r in bin_rows:
        table_bins = r.get("bin_counts", {})
        lines.append(f"| {r['product']} | {table_bins.get('LOW_LOW', 0)} | {table_bins.get('LOW_HIGH', 0)} | {table_bins.get('HIGH_LOW', 0)} | {table_bins.get('HIGH_HIGH', 0)} | {r['status']} |")
    lines.extend(["", "## 权限与解释边界", "",
                  "本任务读取的候选只登记为新论文 D-dev 开发数据；旧 protected 集合的逐文件交集仍为 UNKNOWN。它们及可识别近重复不得进入未来 D-conf。未读取新人工属性标签、GT 缺陷 mask 或旧 protected 存储；未训练或更新模型。Stress 分层仅使用冻结特征、H12 分数、旧的五个合法纠正锚点和固定身份排序。",
                  "", "本报告不评价属性准确率、修复/损害或模型泛化；Stress 组不代表自然发生率。",
                  "", f"私有执行目录：`{public['private_output_relpath']}`（已由 Git 忽略并限制 Windows ACL）。",
                  "", "## 运行及完整性", "",
                  f"特征输出 {public.get('output_bytes', 0):,} bytes；GPU 峰值已分配 {public.get('gpu_peak_allocated_bytes', 0):,} bytes；用时 {public.get('elapsed_seconds', 0):.1f}s。",
                  "输入 manifest、模型、代码与逐样本 SHA 收据均保存在私有输出中。", ""])
    feature_path = REPORT_DIR / "P3_G1A_FEATURE_AND_SELECTION_REPORT.md"
    review_path = REPORT_DIR / "P3_G1A_REVIEW_AND_NEXT_ACTION.md"
    for path, content in ((feature_path, "\n".join(lines)),
                          (review_path, "\n".join([
                              "# P3-G1A 审核与下一步", "", f"当前状态：`{public['status']}`。", "",
                              f"已验证 {public['authorized_views']} 个授权候选输入；SHA 成功 {public['sha_success']}，特征成功 {public['feature_success']}。H12/锚点状态：{public.get('history_status', '未完成')}。",
                              f"最终选择 {public.get('development_groups', 0)} 组、{public.get('development_views', 0)} 视角；Stress bin 完整产品数 {public.get('products_with_all_stress_bins', 0)}/6。", "",
                              "旧保护集合交集仍是 UNKNOWN；本次仅作开发曝光，不进入 D-conf。未读取新标签或 GT mask，未训练/更新模型。", "",
                              "研究负责人下一步唯一决定：是否批准在上述冻结开发角色上开展有限人工属性标注。当前报告本身不授予标注或 G1 评价权限。", ""] ))):
        if path.exists():
            raise FileExistsError(f"PUBLIC_REPORT_EXISTS:{path.name}")
        path.write_text(content, encoding="utf-8", newline="\n")


def run() -> dict:
    start = time.monotonic()
    git = c.current_git()
    if not PYTHON.is_file():
        raise RuntimeError("PINNED_PYTHON_MISSING")
    relative_out = str(c.OUT.relative_to(ROOT)).replace("\\", "/")
    chk = subprocess.run(["git", "check-ignore", "-q", relative_out], cwd=ROOT)
    if chk.returncode != 0:
        raise RuntimeError("OUTPUT_NOT_GIT_IGNORED")
    if c.OUT.exists():
        raise FileExistsError("G1A_OUTPUT_ALREADY_EXISTS; use a new version directory, do not overwrite")
    if any((REPORT_DIR / name).exists() for name in ("P3_G1A_FEATURE_AND_SELECTION_REPORT.md",
                                                       "P3_G1A_REVIEW_AND_NEXT_ACTION.md")):
        raise FileExistsError("PUBLIC_G1A_REPORT_EXISTS; preserve and use an explicitly versioned report")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("NVIDIA_SMI_UNAVAILABLE")

    bundle = c.validate_r2c()
    allowed = bundle["allowed"]
    if len(allowed) > 216:
        raise RuntimeError("ALLOWLIST_LIMIT_EXCEEDED")
    # Direct, item-scoped metadata preflight only. No recursive discovery or file-byte read here.
    missing_paths = [r["opaque_image_id"] for r in allowed if not Path(r["source_image_path"]).is_file()]
    products = c.read_csv(c.R2C / "conditional_products.csv")
    selected_products = [r for r in products if r["conditional_role"] in {"TENTATIVE_D_CAL", "TENTATIVE_D_EVAL"}]
    if len(selected_products) != 6 or sum(r["conditional_role"] == "TENTATIVE_D_CAL" for r in selected_products) != 2:
        raise RuntimeError("CONDITIONAL_PRODUCT_ROLES_INVALID")
    product_slot = {r["product"]: str(r["product_rank"]) for r in selected_products}
    role_map = {r["product"]: "P3_G1A_D_CAL" if r["conditional_role"] == "TENTATIVE_D_CAL" else "P3_G1A_D_EVAL"
                for r in selected_products}
    if {r["product"] for r in allowed} != set(product_slot):
        raise RuntimeError("ALLOWLIST_PRODUCT_MAP_MISMATCH")

    # Estimate run environment without loading model weights.
    probe_code = ("import json,platform,torch,torchvision,numpy,PIL,shutil; p=torch.cuda.get_device_properties(0); "
                  "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,'cuda':torch.cuda.is_available(),'gpu':p.name,'gpu_total':p.total_memory,'gpu_free':torch.cuda.mem_get_info(0)[0],'disk_free':shutil.disk_usage(" + repr(str(ROOT)) + ").free}))")
    env_probe = subprocess.run([str(PYTHON), "-c", probe_code],
        cwd=ROOT, text=True, capture_output=True, timeout=45)
    if env_probe.returncode:
        raise RuntimeError("PINNED_RUNTIME_PREFLIGHT_FAILED")
    runtime = json.loads(env_probe.stdout.strip().splitlines()[-1])
    if (runtime["python"], runtime["torch"], runtime["torchvision"], runtime["numpy"], runtime["pillow"], runtime["cuda"]) != (
            "3.11.14", "2.0.1+cu117", "0.15.2+cu117", "1.24.1", "9.3.0", True):
        raise RuntimeError("PINNED_RUNTIME_MISMATCH")
    if runtime["gpu"] != "NVIDIA GeForce RTX 3090 Ti" or runtime["gpu_free"] < 10 * 1024**3:
        raise RuntimeError("GPU_RESOURCE_GATE")
    if runtime["disk_free"] < 3 * 1024**3:
        raise RuntimeError("OUTPUT_DISK_RESOURCE_GATE")

    c.OUT.mkdir(parents=True, exist_ok=False)
    c.restrict_acl(c.OUT)
    try:
        from scripts.research_d6.p3_g0_r2.partition_eligibility import order_key, SEP
        rows_auth = []
        stress_rows = []
        for row in allowed:
            slot = product_slot[row["product"]]
            role = role_map[row["product"]]
            rows_auth.append({"opaque_image_id": row["opaque_image_id"], "product": row["product"],
                              "opaque_group_id": row["opaque_group_id"], "view": row["view"],
                              "conditional_role": row["conditional_role"], "candidate_kind": row["candidate_kind"],
                              "source_image_path": row["source_image_path"],
                              "protected_exclusion_status": row.get("protected_exclusion_status", "UNKNOWN")})
            digest, _ = order_key("STRESS_TIE", row["product"] + SEP + row["opaque_group_id"])
            stress_rows.append({"opaque_image_id": row["opaque_image_id"], "opaque_group_id": row["opaque_group_id"],
                                "product_slot": slot, "study_role": role, "view": row["view"],
                                "candidate_kind": row["candidate_kind"], "stress_tie_key": digest.hex()})
        auth_sha = write_csv_once(c.OUT / "authorized_images.csv", rows_auth)
        stress_sha = write_csv_once(c.OUT / "stress_input_manifest.csv", stress_rows)
        identity = {"schema": "lp.d6.p3.g1a.run-identity.v1", "authorization": c.AUTHORIZATION,
                    "branch": git["branch"], "head": git["head"], "remote_head": git["remote_head"],
                    "whitelist_sha256": c.WHITELIST_SHA, "selection_receipt_sha256": c.SELECTION_SHA,
                    "r2c_validation": bundle["bundle"]["counts"], "authorized_views": len(allowed),
                    "known_protected_conflict_excluded": len(bundle["known_protected_excluded"]),
                    "protection_status_remaining": "UNKNOWN", "historical_protected_storage_opened": False,
                    "authorized_images_manifest_sha256": auth_sha, "stress_input_manifest_sha256": stress_sha,
                    "runtime_preflight": runtime, "missing_path_metadata_count": len(missing_paths),
                    "code_sha256": {p.name: c.sha_file(p) for p in HERE.glob("*.py")},
                    "new_attribute_labels_read": False, "gt_masks_read": False, "training_or_update": False}
        c.write_json_once(c.OUT / "run_identity.json", identity)
        if missing_paths:
            # Keep the run scoped to the exact original list; the extractor records per-item failures.
            c.write_json_once(c.OUT / "path_preflight.json", {"missing_count": len(missing_paths),
                                  "missing_opaque_ids": missing_paths, "no_reselection": True})

        commands = [
            [str(PYTHON), str(HERE / "extract_features.py")],
            [str(PYTHON), str(HERE / "validate_history.py"), "--output", str(c.OUT / "history")],
            [str(PYTHON), str(HERE / "select_stress.py"), "--output", str(c.OUT)],
        ]
        logs = []
        for index, command in enumerate(commands):
            child_start = time.monotonic()
            proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                  timeout=max(1, c.MAX_RUNTIME_SECONDS - int(time.monotonic() - start)))
            log_path = c.OUT / f"child_{index + 1:02d}.log"
            if log_path.exists():
                raise FileExistsError("CHILD_LOG_EXISTS")
            log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
            logs.append({"command_name": Path(command[1]).name, "returncode": proc.returncode,
                         "elapsed_seconds": time.monotonic() - child_start,
                         "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                         "stderr_sha256": hashlib.sha256(proc.stderr.encode()).hexdigest()})
            if proc.returncode != 0:
                raise RuntimeError(f"CHILD_FAILED:{Path(command[1]).name}:{proc.returncode}")
            if c.check_output_budget(c.OUT) > c.MAX_OUTPUT_BYTES:
                raise RuntimeError("OUTPUT_BUDGET_EXCEEDED")
        extraction = json.loads((c.OUT / "extraction_summary.json").read_text(encoding="utf-8"))
        history = json.loads((c.OUT / "history" / "history_validation.json").read_text(encoding="utf-8"))
        selection = json.loads((c.OUT / "selection_summary.json").read_text(encoding="utf-8"))
        final_roles = c.read_csv(c.OUT / "development_roles.csv")
        role_groups = {}
        for row in final_roles:
            role_groups.setdefault((row["product_slot"], row["opaque_group_id"], row["study_role"]), 0)
            role_groups[(row["product_slot"], row["opaque_group_id"], row["study_role"])] += 1
        if any(n != 2 for n in role_groups.values()):
            raise RuntimeError("FINAL_ROLE_GROUP_SPLIT")
        full = (extraction["canonical_success"] == len(allowed)
                and extraction["feature_success"] == len(allowed)
                and history["status"] == "PASS"
                and selection["status"] == "PASS"
                and len(role_groups) == 60 and len(final_roles) == 120
                and len(selection["per_product_bin_diagnostics"]) == 6
                and all(x["status"] == "PASS" for x in selection["per_product_bin_diagnostics"]))
        status = "P3_G1A_READY_FOR_ANNOTATION_APPROVAL" if full else "P3_G1A_PARTIAL"
        product_names = {v: k for k, v in product_slot.items()}
        stress_diag = {r["product_slot"]: r for r in selection["per_product_bin_diagnostics"]}
        auth_by_product = {name: {"sha_success": 0, "feature_success": 0} for name in product_slot}
        for row in c.read_csv(c.OUT / "exposure_ledger.csv"):
            if row["read_success"] == "YES":
                pname = next(x["product"] for x in allowed if x["opaque_image_id"] == row["opaque_image_id"])
                auth_by_product[pname]["sha_success"] += 1
        for row in c.read_csv(c.OUT / "feature_manifest.csv"):
            pname = product_names[row["product_slot"]]
            auth_by_product[pname]["feature_success"] += 1
        selected_product_counts = {}
        representative_ids = {}
        for row in final_roles:
            slot = row["product_slot"]
            pname = product_names[slot]
            selected_product_counts[pname] = selected_product_counts.get(pname, 0) + 1
            if row["candidate_kind"] == "REPRESENTATIVE_CANDIDATE":
                representative_ids.setdefault(pname, set()).add(row["opaque_group_id"])
        product_rows, bin_rows = [], []
        for name in product_slot:
            d = stress_diag.get(product_slot[name], {})
            nrep = len(representative_ids.get(name, set()))
            product_rows.append({"product": name, "sha_success": auth_by_product[name]["sha_success"],
                                 "feature_success": auth_by_product[name]["feature_success"],
                                 "representative_groups": nrep, "stress_bins": d.get("status", "NOT_RUN"),
                                 "development_groups": selected_product_counts.get(name, 0)})
            bin_rows.append({"product": name, "status": d.get("status", "NOT_RUN"),
                             "bin_counts": d.get("bin_counts", {})})
        result = {"status": status, "branch": git["branch"], "head": git["head"],
                  "authorized_views": len(allowed), "sha_success": extraction["sha_success"],
                  "sha_failure": extraction["sha_failure"],
                  "feature_success": extraction["feature_success"], "history_status": history["status"],
                  "h12_max_abs_score_delta": history["max_abs_score_delta"], "anchor_count": history["historical_anchor_count"],
                  "development_groups": len(role_groups), "development_views": len(final_roles),
                  "d_cal_groups": sum(1 for _, _, role in role_groups if role == "P3_G1A_D_CAL"),
                  "d_eval_groups": sum(1 for _, _, role in role_groups if role == "P3_G1A_D_EVAL"),
                  "products_with_all_stress_bins": sum(x["status"] == "PASS" for x in selection["per_product_bin_diagnostics"]),
                  "private_output_relpath": relative_out, "output_bytes": extraction["resources"]["output_bytes"],
                  "gpu_peak_allocated_bytes": extraction["resources"]["peak_allocated_bytes"],
                  "elapsed_seconds": time.monotonic() - start, "new_attribute_labels_read": False,
                  "gt_masks_read": False, "protected_storage_accessed": False, "training_or_update": False}
        c.write_json_once(c.OUT / "run_summary.json", {**result, "children": logs,
                          "runtime": runtime, "r2c_unknown_protected_intersection_preserved": True,
                          "selection_output_sha256": c.sha_file(c.OUT / "development_roles.csv")})
        save_public_reports(result, product_rows, bin_rows)
        return result
    except Exception as exc:
        failure = {"status": "P3_G1A_BLOCKED_RUNTIME" if "RUNTIME" in str(exc) or "GPU" in str(exc) else "P3_G1A_PARTIAL",
                   "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0],
                   "elapsed_seconds": time.monotonic() - start,
                   "authorized_views": len(allowed), "new_attribute_labels_read": False,
                   "gt_masks_read": False, "protected_storage_accessed": False, "training_or_update": False}
        if not (c.OUT / "run_failure.json").exists():
            c.write_json_once(c.OUT / "run_failure.json", failure)
        raise


if __name__ == "__main__":
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0]}, ensure_ascii=False))
        raise
