"""Tkinter annotator for the frozen LP-D6-P3-G1B queue.

Real input validation and image reads are gated by a separate approval receipt.
`--self-test` uses synthetic pixels and a temporary output tree only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageDraw, ImageOps, ImageTk

from scripts.research_d6.p3_g1b import g1b_core as core

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / core.RUN_REL
ATTRIBUTE_NAME = "surface_abrasion_visible"
STATE_LABELS = {
    "present": "Present",
    "absent_assessable": "Absent, assessable",
    "unobservable_or_uncertain": "Unobservable or uncertain",
    "not_applicable": "Not applicable",
}
CONFIDENCE_VALUES = ["", "high", "medium", "low"]
STATE_ORDER = ["present", "absent_assessable", "unobservable_or_uncertain", "not_applicable"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_sha(actual: str, expected: str, code: str) -> None:
    if actual.casefold() != expected.casefold():
        raise core.IntegrityError(code)


def _secure_read_image(path_text: str, expected_sha256: str) -> bytes:
    from scripts.research_d6.p3_g1a.secure_io import read_and_hash_whitelisted
    data, metadata = read_and_hash_whitelisted(path_text)
    core.verify_image_hash(metadata["sha256"], expected_sha256)
    return data


def production_preflight(run_dir: Path, authorization_path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict]:
    """Validate the approval, frozen hashes, role identity and all selected source bytes.

    This function reads real source-image bytes and must only be reached after a
    valid, separate annotation authorization file has been issued.
    """
    run_dir = run_dir.resolve(strict=True)
    if run_dir != DEFAULT_RUN.resolve(strict=True):
        raise core.IntegrityError("G1B_RUN_DIRECTORY_NOT_ALLOWLISTED")
    expected_authorization = run_dir / "annotation_authorization.json"
    if authorization_path.resolve(strict=False) != expected_authorization.resolve(strict=False):
        raise PermissionError("ANNOTATION_AUTHORIZATION_PATH_NOT_ALLOWLISTED")
    receipt_path = run_dir / "input_freeze_receipt.json"
    receipt = read_json(receipt_path)
    authorization = read_json(authorization_path)
    receipt_hash = core.sha256_file(receipt_path)
    if (authorization.get("schema") != "lp.d6.p3.g1b.annotation-authorization.v1"
            or authorization.get("status") != "APPROVED"
            or not authorization.get("approved_by")
            or not authorization.get("approved_at_utc")
            or not authorization.get("authorization_reference")
            or authorization.get("input_freeze_receipt_sha256") != receipt_hash):
        raise PermissionError("REAL_ANNOTATION_AUTHORIZATION_INVALID")
    if receipt.get("annotation_authorized") is not False or receipt.get("status") != "PREP_READY_NO_REAL_ANNOTATION_AUTHORIZATION":
        raise core.IntegrityError("G1B_RECEIPT_STATUS_INVALID")

    for relative, expected in receipt["code_sha256"].items():
        path = ROOT / Path(relative)
        if not path.is_file():
            raise core.IntegrityError("G1B_CODE_FILE_MISSING")
        _assert_sha(core.sha256_file(path), expected, "G1B_CODE_SHA_MISMATCH")
    g1a_root = ROOT / core.G1A_ROOT_REL
    for name, expected in receipt["g1a_input_sha256"].items():
        _assert_sha(core.sha256_file(g1a_root / name), expected, "G1A_INPUT_SHA_MISMATCH")
    for filename, key in [
        ("image_mapping.csv", "image_mapping_sha256"),
        ("annotation_queue.csv", "annotation_queue_sha256"),
        ("blind_repeat_mapping.csv", "blind_repeat_mapping_sha256"),
        ("blind_repeat_groups.csv", "blind_repeat_groups_sha256"),
        ("exposure_ledger.csv", "exposure_ledger_sha256"),
    ]:
        _assert_sha(core.sha256_file(run_dir / filename), receipt["queue"][key], "G1B_FROZEN_OUTPUT_SHA_MISMATCH")
    _assert_sha(
        core.sha256_file(run_dir / "attribute_definition_card_v1_1.md"),
        receipt["attribute_definition_snapshot_sha256"],
        "G1B_ATTRIBUTE_DEFINITION_SHA_MISMATCH",
    )

    roles, current_mapping, _, _ = core.load_g1a_metadata(g1a_root)
    frozen_mapping_rows = core.read_csv(run_dir / "image_mapping.csv")
    frozen_mapping = {row["opaque_image_id"]: row for row in frozen_mapping_rows}
    if set(frozen_mapping) != set(current_mapping) or len(frozen_mapping) != 120:
        raise core.IntegrityError("G1B_FROZEN_IMAGE_SET_MISMATCH")
    for image_id, row in current_mapping.items():
        for field in ("opaque_group_id", "product", "product_slot", "study_role", "view",
                      "candidate_kind", "source_image_path", "source_sha256"):
            if row[field] != frozen_mapping[image_id][field]:
                raise core.IntegrityError("G1B_FROZEN_IDENTITY_MAPPING_MISMATCH")
    queue = core.read_csv(run_dir / "annotation_queue.csv")
    core.validate_queue(queue, current_mapping)

    # Exact selected source bytes are re-opened only after the approval receipt
    # passes. No replacement search is performed.
    for row in frozen_mapping_rows:
        _secure_read_image(row["source_image_path"], row["source_sha256"])
    return queue, frozen_mapping, receipt


def oriented_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def decode_image_bytes(image_bytes: bytes, expected_sha256: str) -> Image.Image:
    core.verify_image_hash(hashlib.sha256(image_bytes).hexdigest(), expected_sha256)
    return oriented_image(image_bytes)


def visible_widget_text(widget: tk.Misc) -> list[str]:
    texts = []
    try:
        value = widget.cget("text")
        if value:
            texts.append(str(value))
    except tk.TclError:
        pass
    try:
        for child in widget.winfo_children():
            texts.extend(visible_widget_text(child))
    except tk.TclError:
        pass
    return texts


def _read_annotation_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return core.read_csv(path)


def _write_jsonl_append(path: Path, record: dict[str, Any]) -> None:
    encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class AnnotationApp:
    """Single-window queue annotator. Queue metadata never enters visible text."""

    def __init__(self, run_dir: Path, queue: list[dict[str, str]], mapping: dict[str, dict[str, str]], *,
                 synthetic_images: dict[str, Image.Image] | None = None, root: tk.Tk | None = None):
        self.run_dir = run_dir
        self.queue = sorted(queue, key=lambda row: int(row["presentation_order"]))
        self.mapping = mapping
        self.synthetic_images = synthetic_images
        self.synthetic = synthetic_images is not None
        self.root = root if root is not None else tk.Tk()
        self.root.title("人工属性标注")
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)
        self.cursor = 0
        self.current_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fit_mode = True
        self.drag_origin: tuple[int, int] | None = None
        self.point_mode = False
        self.point: tuple[float, float] | None = None
        self.annotations: dict[str, dict[str, str]] = {}
        self.session_timing: dict[str, dict[str, Any]] = {"presentations": {}}
        self.timer_started_monotonic = time.monotonic()
        self.first_displayed_utc = core.utc_now()
        self.dirty = False
        self._loading = False
        self._test_rendered = False

        for role in ("P3_G1A_D_CAL", "P3_G1A_D_EVAL"):
            self.annotations[role] = {
                row["presentation_id"]: row
                for row in _read_annotation_file(self._annotation_path(role))
            }
        timing_path = self.run_dir / "session_timing.json"
        if timing_path.exists():
            self.session_timing = read_json(timing_path)

        self._make_widgets()
        self._load_task(0)

    def _make_widgets(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.root, background="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        panel = ttk.Frame(self.root, padding=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        panel.columnconfigure(0, weight=1)

        self.progress_var = tk.StringVar(master=self.root, value="")
        ttk.Label(panel, textvariable=self.progress_var, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(panel, text="surface_abrasion_visible", font=("Segoe UI", 11, "bold")).grid(row=1, column=0, sticky="w")
        ttk.Label(panel, text="请只判断当前显示视角中可见的表面擦伤。反光、阴影、曝光变化、单独孔洞/裂缝/接缝/规则纹理，或缺少预期纹理，本身都不足以证明擦伤。证据不足请选择“不确定”。").grid(row=2, column=0, sticky="w", pady=(5, 10))

        self.state_var = tk.StringVar(master=self.root, value="")
        for index, state in enumerate(STATE_ORDER):
            ttk.Radiobutton(panel, text=STATE_LABELS[state], value=state, variable=self.state_var,
                            command=self._mark_dirty).grid(row=3 + index, column=0, sticky="w", pady=2)
        ttk.Label(panel, text="置信度").grid(row=7, column=0, sticky="w", pady=(12, 2))
        self.confidence_var = tk.StringVar(master=self.root, value="")
        ttk.Combobox(panel, textvariable=self.confidence_var, values=CONFIDENCE_VALUES[1:], state="readonly", width=16).grid(row=8, column=0, sticky="w")
        self.confidence_var.trace_add("write", lambda *_: self._mark_dirty())

        point_bar = ttk.Frame(panel)
        point_bar.grid(row=9, column=0, sticky="ew", pady=(10, 2))
        self.point_button = ttk.Button(point_bar, text="标记位置", command=self._toggle_point_mode)
        self.point_button.pack(side="left")
        ttk.Button(point_bar, text="清除位置", command=self._clear_point).pack(side="left", padx=6)
        self.point_label_var = tk.StringVar(master=self.root, value="未标记")
        ttk.Label(panel, textvariable=self.point_label_var).grid(row=10, column=0, sticky="w")

        ttk.Label(panel, text="备注（可选）").grid(row=11, column=0, sticky="w", pady=(12, 2))
        self.notes = tk.Text(panel, height=6, wrap="word")
        self.notes.grid(row=12, column=0, sticky="ew")
        self.notes.bind("<<Modified>>", self._notes_modified)

        actions = ttk.Frame(panel)
        actions.grid(row=13, column=0, sticky="ew", pady=(14, 4))
        ttk.Button(actions, text="上一张", command=lambda: self._navigate(-1)).pack(side="left")
        ttk.Button(actions, text="保存", command=self.save_current).pack(side="left", padx=5)
        ttk.Button(actions, text="下一张", command=lambda: self._navigate(1)).pack(side="left")
        ttk.Button(actions, text="关闭", command=self.close).pack(side="right")
        view_actions = ttk.Frame(panel)
        view_actions.grid(row=14, column=0, sticky="ew", pady=(12, 2))
        ttk.Button(view_actions, text="适合窗口", command=self.fit_image).pack(side="left")
        ttk.Button(view_actions, text="100%", command=self.original_size).pack(side="left", padx=6)
        ttk.Label(panel, text="滚轮缩放；拖动平移。视图操作不会改变标签或当前呈现项。").grid(row=15, column=0, sticky="w", pady=(4, 0))
        ttk.Button(panel, text="查看完整属性定义 v1.1", command=self._show_definition).grid(row=16, column=0, sticky="w", pady=(8, 0))

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._wheel_zoom_linux(event, 1.1))
        self.canvas.bind("<Button-5>", lambda event: self._wheel_zoom_linux(event, 0.9))
        self.canvas.bind("<ButtonPress-1>", self._pointer_down)
        self.canvas.bind("<B1-Motion>", self._pointer_move)
        self.canvas.bind("<ButtonRelease-1>", self._pointer_up)

    def _annotation_path(self, role: str) -> Path:
        if role == "P3_G1A_D_CAL":
            return self.run_dir / "D-cal" / "annotations.csv"
        if role == "P3_G1A_D_EVAL":
            return self.run_dir / "D-eval" / "annotations.csv"
        raise core.IntegrityError("ANNOTATION_ROLE_INVALID")

    def _mapping_row(self, event: dict[str, str]) -> dict[str, str]:
        row = self.mapping.get(event["opaque_image_id"])
        if row is None:
            raise core.IntegrityError("QUEUE_IMAGE_MAPPING_MISSING")
        return row

    def _load_task(self, index: int) -> None:
        if not 0 <= index < len(self.queue):
            return
        self._flush_timer()
        self.cursor = index
        event = self.queue[index]
        mapping = self._mapping_row(event)
        presentation_id = event["presentation_id"]
        self._loading = True
        self.state_var.set("")
        self.confidence_var.set("")
        self.notes.delete("1.0", "end")
        self.point = None
        saved = self.annotations.get(event["study_role"], {}).get(presentation_id)
        if saved:
            if saved.get("attribute_state") in core.VALID_STATES:
                self.state_var.set(saved["attribute_state"])
            if saved.get("confidence") in CONFIDENCE_VALUES:
                self.confidence_var.set(saved.get("confidence", ""))
            self.notes.insert("1.0", saved.get("notes", ""))
            x, y = saved.get("point_x_normalized", ""), saved.get("point_y_normalized", "")
            if x and y:
                self.point = (float(x), float(y))
        timing = self.session_timing.setdefault("presentations", {}).setdefault(presentation_id, {})
        self.first_displayed_utc = timing.get("first_displayed_utc") or core.utc_now()
        timing["first_displayed_utc"] = self.first_displayed_utc
        timing.setdefault("elapsed_seconds", 0.0)
        self.timer_started_monotonic = time.monotonic()
        self._persist_timing()
        self.dirty = False
        self._loading = False
        self.progress_var.set(f"呈现 {index + 1} / {len(self.queue)}")
        self.point_label_var.set("已标记位置" if self.point else "未标记")
        self.fit_mode = True
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        try:
            if self.synthetic:
                image = self.synthetic_images[event["opaque_image_id"]]
                self.current_image = image.copy()
            else:
                payload = _secure_read_image(mapping["source_image_path"], mapping["source_sha256"])
                self.current_image = oriented_image(payload)
            self.fit_image()
            self._test_rendered = self.current_image is not None
        except Exception as exc:
            if self.synthetic:
                raise
            messagebox.showerror("图像校验失败", "输入校验未通过；当前图像已停止显示。请联系数据管理员。", parent=self.root)
            raise core.IntegrityError("ANNOTATION_IMAGE_LOAD_BLOCKED") from exc

    def _flush_timer(self) -> None:
        if not hasattr(self, "queue") or not self.queue:
            return
        event = self.queue[self.cursor]
        presentation_id = event["presentation_id"]
        elapsed = max(0.0, time.monotonic() - self.timer_started_monotonic)
        timing = self.session_timing.setdefault("presentations", {}).setdefault(presentation_id, {})
        timing["first_displayed_utc"] = timing.get("first_displayed_utc") or self.first_displayed_utc
        timing["elapsed_seconds"] = float(timing.get("elapsed_seconds", 0.0)) + elapsed
        self.timer_started_monotonic = time.monotonic()
        if not self.synthetic:
            self._persist_timing()

    def _persist_timing(self) -> None:
        core.atomic_write_bytes(
            self.run_dir / "session_timing.json",
            (json.dumps(self.session_timing, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )

    def _mark_dirty(self) -> None:
        if not self._loading:
            self.dirty = True

    def _notes_modified(self, _event=None) -> None:
        if self.notes.edit_modified():
            if not self._loading:
                self.dirty = True
            self.notes.edit_modified(False)

    def _navigate(self, delta: int) -> None:
        target = min(max(self.cursor + delta, 0), len(self.queue) - 1)
        if target == self.cursor:
            return
        if self.dirty:
            answer = messagebox.askyesnocancel("未保存的更改", "是否先保存当前呈现？", parent=self.root)
            if answer is None:
                return
            if answer and not self.save_current():
                return
        self._load_task(target)

    def _current_values(self) -> dict[str, str]:
        state = self.state_var.get()
        if state and state not in core.VALID_STATES:
            raise core.IntegrityError("ANNOTATION_STATE_INVALID")
        if self.confidence_var.get() and self.confidence_var.get() not in CONFIDENCE_VALUES:
            raise core.IntegrityError("CONFIDENCE_INVALID")
        x = "" if self.point is None else f"{self.point[0]:.8f}"
        y = "" if self.point is None else f"{self.point[1]:.8f}"
        return {
            "attribute_state": state,
            "confidence": self.confidence_var.get(),
            "location_kind": "point" if self.point is not None else "",
            "point_x_normalized": x,
            "point_y_normalized": y,
            "notes": self.notes.get("1.0", "end-1c").strip(),
        }

    def save_current(self, *, correction_reason: str | None = None) -> bool:
        event = self.queue[self.cursor]
        role = event["study_role"]
        presentation_id = event["presentation_id"]
        mapping = self._mapping_row(event)
        try:
            values = self._current_values()
        except core.IntegrityError:
            messagebox.showerror("无法保存", "标注状态无效。", parent=self.root)
            return False
        old = self.annotations.get(role, {}).get(presentation_id)
        self._flush_timer()
        timer = self.session_timing["presentations"][presentation_id]
        now = core.utc_now()
        revision = int(old.get("annotation_revision", "0")) if old else 0
        fixed = {
            "presentation_id": presentation_id,
            "opaque_image_id": event["opaque_image_id"],
            "opaque_group_id": event["opaque_group_id"],
            "product": event["product"],
            "study_role": role,
            "view": event["view"],
            "attribute_name": ATTRIBUTE_NAME,
            **values,
            "first_displayed_utc": timer["first_displayed_utc"],
            "last_saved_utc": now,
            "cumulative_active_seconds": f"{float(timer['elapsed_seconds']):.3f}",
            "annotator_id": os.environ.get("LP_G1B_ANNOTATOR_ID", ""),
            "annotation_revision": str(revision + (1 if old else 0)),
        }
        if old:
            changed = any(old.get(key, "") != fixed.get(key, "") for key in values)
            if changed:
                reason = correction_reason
                if reason is None:
                    reason = simpledialog.askstring("修订记录", "请简要记录修订原因：", parent=self.root)
                if not reason or not reason.strip():
                    return False
                audit = {
                    "recorded_at_utc": now,
                    "presentation_id": presentation_id,
                    "revision_from": revision,
                    "revision_to": revision + 1,
                    "reason": reason.strip(),
                    "previous_record": old,
                    "new_record": fixed,
                }
                _write_jsonl_append(self._correction_path(role), audit)
        rows = list(self.annotations.get(role, {}).values())
        previous = self.annotations.get(role, {}).get(presentation_id)
        if previous:
            rows = [row for row in rows if row["presentation_id"] != presentation_id]
        rows.append(fixed)
        rows.sort(key=lambda row: row["presentation_id"])
        try:
            target = self._annotation_path(role)
            core.assert_output_within_run(target, self.run_dir)
            core.atomic_write_csv(target, rows)
        except Exception:
            messagebox.showerror("保存失败", "已有记录保持不变。请停止操作并联系数据管理员。", parent=self.root)
            return False
        self.annotations.setdefault(role, {})[presentation_id] = fixed
        self.dirty = False
        self.root.title("人工属性标注 - 已保存")
        return True

    def _correction_path(self, role: str) -> Path:
        folder = "D-cal" if role == "P3_G1A_D_CAL" else "D-eval"
        return self.run_dir / folder / "annotation_corrections.jsonl"

    def _show_definition(self) -> None:
        card_path = self.run_dir / "attribute_definition_card_v1_1.md"
        try:
            text = card_path.read_text(encoding="utf-8")
        except Exception:
            messagebox.showerror("属性定义不可用", "冻结的属性定义文件无法读取。请停止并联系数据管理员。", parent=self.root)
            return
        cutoff = text.find("## Revision basis")
        if cutoff >= 0:
            text = text[:cutoff]
        window = tk.Toplevel(self.root)
        window.title("Surface Abrasion — Definition v1.1")
        window.geometry("760x680")
        view = tk.Text(window, wrap="word", padx=12, pady=12)
        scroll = ttk.Scrollbar(window, orient="vertical", command=view.yview)
        view.configure(yscrollcommand=scroll.set)
        view.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        view.insert("1.0", text)
        view.configure(state="disabled")

    def close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("未保存的更改", "是否先保存当前呈现？", parent=self.root)
            if answer is None:
                return
            if answer and not self.save_current():
                return
        self._flush_timer()
        self.root.destroy()

    def _navigate_by_close(self) -> None:
        self.close()

    def fit_image(self) -> None:
        if self.current_image is None:
            return
        self.root.update_idletasks()
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / self.current_image.width, height / self.current_image.height, 1.0)
        self.scale = max(scale, 0.05)
        self.fit_mode = True
        self.pan_x = (width - self.current_image.width * self.scale) / 2
        self.pan_y = (height - self.current_image.height * self.scale) / 2
        self._render()

    def original_size(self) -> None:
        if self.current_image is None:
            return
        self.root.update_idletasks()
        self.scale = 1.0
        self.fit_mode = False
        self.pan_x = (self.canvas.winfo_width() - self.current_image.width) / 2
        self.pan_y = (self.canvas.winfo_height() - self.current_image.height) / 2
        self._render()

    def _render(self) -> None:
        if self.current_image is None:
            return
        width = max(1, int(round(self.current_image.width * self.scale)))
        height = max(1, int(round(self.current_image.height * self.scale)))
        resized = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, image=self.photo, anchor="nw", tags="source_image")
        if self.point is not None:
            x = self.pan_x + self.point[0] * width
            y = self.pan_y + self.point[1] * height
            radius = 7
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline="#ffef00", width=2, tags="point_marker")

    def _on_canvas_resize(self, _event=None) -> None:
        if self.fit_mode:
            self.fit_image()

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.current_image is None:
            return
        old = self.scale
        new = min(8.0, max(0.05, old * factor))
        if new == old:
            return
        self.pan_x = x - (x - self.pan_x) * new / old
        self.pan_y = y - (y - self.pan_y) * new / old
        self.scale = new
        self.fit_mode = False
        self._render()

    def _wheel_zoom(self, event) -> None:
        self._zoom_at(event.x, event.y, 1.1 if event.delta > 0 else 0.9)

    def _wheel_zoom_linux(self, event, factor: float) -> None:
        self._zoom_at(event.x, event.y, factor)

    def _toggle_point_mode(self) -> None:
        self.point_mode = not self.point_mode
        self.point_button.configure(text="点击图像标记" if self.point_mode else "标记位置")

    def _clear_point(self) -> None:
        self.point = None
        self.point_label_var.set("未标记")
        self.dirty = True
        self._render()

    def _pointer_down(self, event) -> None:
        self.drag_origin = (event.x, event.y)

    def _pointer_move(self, event) -> None:
        if self.drag_origin is None or self.point_mode:
            return
        dx, dy = event.x - self.drag_origin[0], event.y - self.drag_origin[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_origin = (event.x, event.y)
        self._render()

    def _pointer_up(self, event) -> None:
        if self.point_mode and self.current_image is not None:
            width = self.current_image.width * self.scale
            height = self.current_image.height * self.scale
            x = (event.x - self.pan_x) / width
            y = (event.y - self.pan_y) / height
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.point = (float(x), float(y))
                self.point_label_var.set("已标记位置")
                self.dirty = True
                self.point_mode = False
                self.point_button.configure(text="标记位置")
                self._render()
        self.drag_origin = None


def _synthetic_session(run_dir: Path | None = None) -> dict[str, Any]:
    """Create a temporary, two-item synthetic GUI session and test persistence."""
    import tempfile
    from collections import Counter
    temporary = tempfile.TemporaryDirectory(prefix="lp_g1b_synthetic_") if run_dir is None else None
    root_dir = Path(temporary.name) if temporary else run_dir
    (root_dir / "D-cal").mkdir(parents=True, exist_ok=True)
    (root_dir / "D-eval").mkdir(parents=True, exist_ok=True)
    for role in ("P3_G1A_D_CAL", "P3_G1A_D_EVAL"):
        core.atomic_write_csv(root_dir / ("D-cal" if role.endswith("D_CAL") else "D-eval") / "annotations.csv", [])
    core.atomic_write_bytes(root_dir / "session_timing.json", b'{"schema":"test","presentations":{}}')
    images = {
        "synthetic-cal": Image.new("RGB", (320, 240), (160, 170, 180)),
        "synthetic-eval": Image.new("RGB", (320, 240), (190, 140, 120)),
    }
    draw = ImageDraw.Draw(images["synthetic-cal"])
    draw.line((50, 50, 260, 170), fill=(25, 25, 25), width=5)
    draw = ImageDraw.Draw(images["synthetic-eval"])
    draw.ellipse((90, 70, 210, 190), outline=(15, 15, 15), width=5)
    queue = [
        {"presentation_order": "1", "presentation_id": "SYN-P-1", "opaque_image_id": "synthetic-cal", "opaque_group_id": "syn-g1", "product": "hidden-A", "product_slot": "1", "study_role": "P3_G1A_D_CAL", "view": "C1", "candidate_kind": "SYNTHETIC", "presentation_kind": "FIRST", "first_presentation_id": "SYN-P-1", "queue_nonce": "N1"},
        {"presentation_order": "2", "presentation_id": "SYN-P-2", "opaque_image_id": "synthetic-eval", "opaque_group_id": "syn-g2", "product": "hidden-B", "product_slot": "2", "study_role": "P3_G1A_D_EVAL", "view": "C2", "candidate_kind": "SYNTHETIC", "presentation_kind": "FIRST", "first_presentation_id": "SYN-P-2", "queue_nonce": "N2"},
        {"presentation_order": "3", "presentation_id": "SYN-P-3", "opaque_image_id": "synthetic-cal", "opaque_group_id": "syn-g1", "product": "hidden-A", "product_slot": "1", "study_role": "P3_G1A_D_CAL", "view": "C1", "candidate_kind": "SYNTHETIC", "presentation_kind": "REPEAT", "first_presentation_id": "SYN-P-1", "queue_nonce": "N3"},
    ]
    mapping = {
        "synthetic-cal": {"opaque_image_id": "synthetic-cal", "opaque_group_id": "syn-g1", "product": "hidden-A", "product_slot": "1", "study_role": "P3_G1A_D_CAL", "view": "C1", "candidate_kind": "SYNTHETIC", "source_image_path": "synthetic-only", "source_sha256": ""},
        "synthetic-eval": {"opaque_image_id": "synthetic-eval", "opaque_group_id": "syn-g2", "product": "hidden-B", "product_slot": "2", "study_role": "P3_G1A_D_EVAL", "view": "C2", "candidate_kind": "SYNTHETIC", "source_image_path": "synthetic-only", "source_sha256": ""},
    }
    window = tk.Tk()
    app = AnnotationApp(root_dir, queue, mapping, synthetic_images=images, root=window)
    window.update_idletasks()
    window.update()
    if not app._test_rendered or not app.canvas.find_withtag("source_image"):
        raise RuntimeError("SYNTHETIC_GUI_RENDER_FAILED")
    before_identity = app.queue[app.cursor]["presentation_id"]
    app.state_var.set("present")
    app.confidence_var.set("high")
    app.dirty = True
    if not app.save_current():
        raise RuntimeError("SYNTHETIC_GUI_SAVE_FAILED")
    app.original_size()
    app._zoom_at(100, 100, 1.1)
    app.pan_x += 3
    app._render()
    if app.queue[app.cursor]["presentation_id"] != before_identity or app.state_var.get() != "present":
        raise RuntimeError("VIEW_TRANSFORM_CHANGED_ANNOTATION_OR_IDENTITY")
    app._load_task(2)
    if app.state_var.get() != "":
        raise RuntimeError("BLIND_REPEAT_STATE_WAS_COPIED")
    app.state_var.set("unobservable_or_uncertain")
    app.dirty = True
    if not app.save_current():
        raise RuntimeError("SYNTHETIC_REPEAT_SAVE_FAILED")
    loaded = core.read_csv(root_dir / "D-cal" / "annotations.csv")
    if len(loaded) != 2 or {row["attribute_state"] for row in loaded} != {"present", "unobservable_or_uncertain"}:
        raise RuntimeError("SYNTHETIC_SAVE_RESTORE_FAILED")
    app.close()
    restored_window = tk.Tk()
    restored = AnnotationApp(root_dir, queue, mapping, synthetic_images=images, root=restored_window)
    restored_window.update_idletasks()
    restored_window.update()
    if restored.state_var.get() != "present":
        raise RuntimeError("SYNTHETIC_RESTORE_FAILED")
    restored._load_task(2)
    if restored.state_var.get() != "unobservable_or_uncertain":
        raise RuntimeError("SYNTHETIC_REPEAT_RESTORE_FAILED")
    visible = " ".join(visible_widget_text(restored_window)).casefold()
    for forbidden in ("hidden-a", "hidden-b", "syn-g", "synthetic-cal", "synthetic-eval", "synthetic-only"):
        if forbidden in visible:
            raise RuntimeError("SYNTHETIC_GUI_METADATA_LEAK")
    restored.close()
    if temporary:
        temporary.cleanup()
    return {
        "synthetic_images": 2,
        "synthetic_presentations": 3,
        "rendered_canvas_items": True,
        "save_restore": True,
        "blind_repeat_independent": True,
        "zoom_pan_preserved_identity_and_label": True,
        "hidden_metadata_not_visible": True,
    }


def launch(run_dir: Path = DEFAULT_RUN, authorization_path: Path | None = None) -> None:
    auth_path = authorization_path or (run_dir / "annotation_authorization.json")
    lock = core.SingleWriterLock(run_dir / "annotation_session.lock")
    try:
        queue, mapping, _receipt = production_preflight(run_dir, auth_path)
        lock.acquire()
        app = AnnotationApp(run_dir, queue, mapping)
        app.root.protocol("WM_DELETE_WINDOW", app.close)
        app.root.mainloop()
    except PermissionError:
        print("ANNOTATION_NOT_AUTHORIZED_OR_D_EVAL_ACCESS_BLOCKED", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"ANNOTATION_STARTUP_BLOCKED:{type(exc).__name__}", file=sys.stderr)
        raise
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LP-D6-P3-G1B human attribute annotation")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="Use only synthetic pixels and temporary outputs")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(_synthetic_session(), sort_keys=True))
        return 0
    launch(args.run_dir, args.authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
