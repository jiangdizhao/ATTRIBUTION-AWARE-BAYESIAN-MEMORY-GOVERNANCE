"""Windows handle-based, fail-closed reading for the exact G1A whitelist."""
from __future__ import annotations
import ctypes
import hashlib
import os
from ctypes import wintypes
from pathlib import Path, PureWindowsPath

ROOT = PureWindowsPath(r"G:\dataset\realiad")
MAX_BLOCK = 1024 * 1024
GENERIC_READ = 0x80000000
FILE_READ_ATTRIBUTES = 0x80
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_ATTRIBUTE_DIRECTORY = 0x10
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [("dwFileAttributes", wintypes.DWORD), ("ftCreationTime", FILETIME),
                ("ftLastAccessTime", FILETIME), ("ftLastWriteTime", FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD), ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD), ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD), ("nFileIndexLow", wintypes.DWORD)]


class UnsafePath(RuntimeError):
    pass


class SourceChanged(RuntimeError):
    pass


def _configure_api():
    if os.name != "nt":
        raise UnsafePath("WINDOWS_HANDLE_API_REQUIRED")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
    k32.GetFileInformationByHandle.restype = wintypes.BOOL
    k32.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    k32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    k32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    k32.GetFileAttributesW.restype = wintypes.DWORD
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    k32.ReadFile.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


def _signature(info: BY_HANDLE_FILE_INFORMATION) -> tuple:
    return (info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow,
            info.nFileSizeHigh, info.nFileSizeLow,
            info.ftLastWriteTime.dwHighDateTime, info.ftLastWriteTime.dwLowDateTime,
            info.dwFileAttributes)


def _info(k32, handle) -> BY_HANDLE_FILE_INFORMATION:
    result = BY_HANDLE_FILE_INFORMATION()
    if not k32.GetFileInformationByHandle(handle, ctypes.byref(result)):
        raise UnsafePath("HANDLE_METADATA_FAILED")
    return result


def _final_path(k32, handle) -> str:
    buf = ctypes.create_unicode_buffer(32768)
    size = k32.GetFinalPathNameByHandleW(handle, buf, len(buf), 0)
    if not size or size >= len(buf):
        raise UnsafePath("HANDLE_FINAL_PATH_FAILED")
    path = buf.value.replace("/", "\\").rstrip("\\")
    if path.casefold().startswith("\\\\?\\unc\\"):
        raise UnsafePath("FINAL_PATH_UNC")
    return path


def _open(k32, path: str, *, directory: bool):
    access = FILE_READ_ATTRIBUTES if directory else GENERIC_READ
    flags = FILE_FLAG_OPEN_REPARSE_POINT | (FILE_FLAG_BACKUP_SEMANTICS if directory else FILE_FLAG_SEQUENTIAL_SCAN)
    handle = k32.CreateFileW(path, access, FILE_SHARE_READ, None, OPEN_EXISTING, flags, None)
    if handle == INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        raise UnsafePath(f"CREATE_FILE_FAILED_{error}")
    return handle


def _check_components(k32, candidate: PureWindowsPath) -> None:
    try:
        relative = candidate.relative_to(ROOT)
    except ValueError as exc:
        raise UnsafePath("PATH_OUTSIDE_APPROVED_ROOT") from exc
    if not relative.parts or any(part in (".", "..") or ":" in part for part in relative.parts):
        raise UnsafePath("PATH_TRAVERSAL_OR_ALT_STREAM")
    parts = [ROOT, *[ROOT.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]]
    for index, part in enumerate(parts):
        attrs = k32.GetFileAttributesW(str(part))
        if attrs == INVALID_FILE_ATTRIBUTES:
            raise UnsafePath("PATH_COMPONENT_MISSING")
        if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
            raise UnsafePath("PATH_REPARSE_POINT")
        is_last = index == len(parts) - 1
        if not is_last and not (attrs & FILE_ATTRIBUTE_DIRECTORY):
            raise UnsafePath("NON_DIRECTORY_PARENT")
        if is_last and attrs & FILE_ATTRIBUTE_DIRECTORY:
            raise UnsafePath("TARGET_NOT_REGULAR_FILE")


def read_and_hash_whitelisted(path_text: str) -> tuple[bytes, dict]:
    """Open one exact listed image, validate final handle path and stream/hash bytes.

    The returned bytes are from the same read used to compute SHA-256 and are
    passed directly to Pillow via BytesIO; the source path is never reopened.
    """
    k32 = _configure_api()
    candidate = PureWindowsPath(path_text)
    if (not candidate.is_absolute() or candidate.drive.casefold() != ROOT.drive.casefold()
            or candidate.anchor.startswith("\\\\")):
        raise UnsafePath("ABSOLUTE_PATH_INVALID")
    _check_components(k32, candidate)
    root_handle = _open(k32, str(ROOT), directory=True)
    try:
        root_info = _info(k32, root_handle)
        if root_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT or not root_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY:
            raise UnsafePath("ROOT_NOT_TRUSTED_DIRECTORY")
        root_final = _final_path(k32, root_handle)
    finally:
        k32.CloseHandle(root_handle)

    handle = _open(k32, str(candidate), directory=False)
    try:
        before = _info(k32, handle)
        if before.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT):
            raise UnsafePath("TARGET_NOT_REGULAR_NONREPARSE_FILE")
        actual_final = _final_path(k32, handle)
        relative = candidate.relative_to(ROOT)
        expected_final = root_final + "\\" + "\\".join(relative.parts)
        if actual_final.casefold() != expected_final.casefold():
            raise UnsafePath("FINAL_PATH_IDENTITY_MISMATCH")
        digest = hashlib.sha256()
        data = bytearray()
        while True:
            buffer = ctypes.create_string_buffer(MAX_BLOCK)
            count = wintypes.DWORD()
            if not k32.ReadFile(handle, buffer, MAX_BLOCK, ctypes.byref(count), None):
                raise OSError("SOURCE_READ_FAILED")
            if not count.value:
                break
            block = buffer.raw[:count.value]
            digest.update(block)
            data.extend(block)
        after = _info(k32, handle)
        if _signature(before) != _signature(after):
            raise SourceChanged("SOURCE_CHANGED_DURING_READ")
        size = (int(before.nFileSizeHigh) << 32) | int(before.nFileSizeLow)
        if len(data) != size:
            raise SourceChanged("READ_SIZE_MISMATCH")
        return bytes(data), {"sha256": digest.hexdigest(), "size_bytes": size,
                             "file_id": f"{before.dwVolumeSerialNumber:08x}:{before.nFileIndexHigh:08x}{before.nFileIndexLow:08x}",
                             "last_write_filetime": f"{before.ftLastWriteTime.dwHighDateTime:08x}{before.ftLastWriteTime.dwLowDateTime:08x}",
                             "final_path_verified": True, "reparse_check": "PASS"}
    finally:
        k32.CloseHandle(handle)
