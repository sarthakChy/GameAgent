from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread, current_thread
from typing import Optional

if platform.system() != "Windows":  # pragma: no cover - Windows-specific tool
    raise SystemExit("This input recorder is Windows-only.")

from ctypes import wintypes


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

if not hasattr(wintypes, "ULONG_PTR"):
    wintypes.ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong  # type: ignore[attr-defined]

if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = ctypes.c_ssize_t if hasattr(ctypes, "c_ssize_t") else (ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long)  # type: ignore[attr-defined]

for _handle_name in ("HINSTANCE", "HICON", "HCURSOR", "HBRUSH"):
    if not hasattr(wintypes, _handle_name):
        setattr(wintypes, _handle_name, wintypes.HANDLE)  # type: ignore[attr-defined]


def _configure_winapi_prototypes() -> None:
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = wintypes.LRESULT

    user32.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = wintypes.LRESULT

    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HANDLE

    user32.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL

    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.RegisterClassW.restype = wintypes.ATOM

    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND

    user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, ctypes.c_uint]
    user32.RegisterRawInputDevices.restype = wintypes.BOOL

    user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, ctypes.POINTER(wintypes.UINT), ctypes.c_uint]
    user32.GetRawInputData.restype = wintypes.UINT

    user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
    user32.PeekMessageW.restype = wintypes.BOOL

    user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int

    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL

    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype = wintypes.LRESULT

    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL

    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.PostQuitMessage.restype = None

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE

    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    user32.GetClipCursor.argtypes = [ctypes.POINTER(RECT)]
    user32.GetClipCursor.restype = wintypes.BOOL

    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int

    get_precise = getattr(kernel32, "GetSystemTimePreciseAsFileTime", None)
    if get_precise is not None:
        get_precise.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
        get_precise.restype = None

    get_filetime = getattr(kernel32, "GetSystemTimeAsFileTime", None)
    if get_filetime is not None:
        get_filetime.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
        get_filetime.restype = None

HC_ACTION = 0
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14

WM_NULL = 0x0000
WM_QUIT = 0x0012
WM_DESTROY = 0x0002
WM_INPUT = 0x00FF
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
WM_MOUSEHWHEEL = 0x020E
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0

LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_ALTDOWN = 0x20
LLKHF_UP = 0x80

LLMHF_INJECTED = 0x00000001
LLMHF_LOWER_IL_INJECTED = 0x00000002

FILETIME_EPOCH_OFFSET_100NS = 116444736000000000

MOUSE_MOVE_ABSOLUTE = 0x01  # lLastX/lLastY are absolute coords, not relative


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR),
    ]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR),
    ]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWMOUSE_BUTTONS(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class RAWMOUSE_BUTTONS_UNION(ctypes.Union):
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("buttons", RAWMOUSE_BUTTONS),
    ]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("buttons_union",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("buttons_union", RAWMOUSE_BUTTONS_UNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWHID(ctypes.Structure):
    _fields_ = [
        ("dwSizeHid", wintypes.DWORD),
        ("dwCount", wintypes.DWORD),
        ("bRawData", wintypes.BYTE * 1),
    ]


class RAWINPUTDATA(ctypes.Union):
    _fields_ = [
        ("mouse", RAWMOUSE),
        ("keyboard", RAWKEYBOARD),
        ("hid", RAWHID),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", RAWINPUTDATA),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


WNDPROC = ctypes.WINFUNCTYPE(wintypes.LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
HOOKPROC = ctypes.WINFUNCTYPE(wintypes.LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


_configure_winapi_prototypes()


def _is_cursor_confined() -> bool:
    """Returns True when the game has clipped the cursor to a sub-screen rect (world/3D mode).
    Returns False when the cursor moves freely (GUI/menu mode).
    Works for the majority of games: Source, Unreal, Unity, Genshin, GTA, etc."""
    clip = RECT()
    if not user32.GetClipCursor(ctypes.byref(clip)):
        return False
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    clip_w = clip.right - clip.left
    clip_h = clip.bottom - clip.top
    # Allow a small margin — some games clip to window size which may be slightly
    # smaller than full screen even in fullscreen mode
    return clip_w < (screen_w - 2) or clip_h < (screen_h - 2)


LOW_LEVEL_MOUSE_MESSAGES = {
    WM_MOUSEMOVE: "move",
    WM_LBUTTONDOWN: "left_down",
    WM_LBUTTONUP: "left_up",
    WM_RBUTTONDOWN: "right_down",
    WM_RBUTTONUP: "right_up",
    WM_MBUTTONDOWN: "middle_down",
    WM_MBUTTONUP: "middle_up",
    WM_MOUSEWHEEL: "wheel",
    WM_XBUTTONDOWN: "x_down",
    WM_XBUTTONUP: "x_up",
    WM_MOUSEHWHEEL: "hwheel",
}


VK_NAME_MAP = {
    0x08: "backspace",
    0x09: "tab",
    0x0D: "enter",
    0x10: "shift",
    0x11: "ctrl",
    0x12: "alt",
    0x13: "pause",
    0x14: "caps_lock",
    0x1B: "esc",
    0x20: "space",
    0x21: "page_up",
    0x22: "page_down",
    0x23: "end",
    0x24: "home",
    0x25: "left",
    0x26: "up",
    0x27: "right",
    0x28: "down",
    0x2C: "print_screen",
    0x2D: "insert",
    0x2E: "delete",
    0x5B: "lwin",
    0x5C: "rwin",
    0x5D: "apps",
    0x60: "numpad_0",
    0x61: "numpad_1",
    0x62: "numpad_2",
    0x63: "numpad_3",
    0x64: "numpad_4",
    0x65: "numpad_5",
    0x66: "numpad_6",
    0x67: "numpad_7",
    0x68: "numpad_8",
    0x69: "numpad_9",
    0x6A: "multiply",
    0x6B: "add",
    0x6C: "separator",
    0x6D: "subtract",
    0x6E: "decimal",
    0x6F: "divide",
    0x70: "f1",
    0x71: "f2",
    0x72: "f3",
    0x73: "f4",
    0x74: "f5",
    0x75: "f6",
    0x76: "f7",
    0x77: "f8",
    0x78: "f9",
    0x79: "f10",
    0x7A: "f11",
    0x7B: "f12",
    0x7C: "f13",
    0x7D: "f14",
    0x7E: "f15",
    0x7F: "f16",
    0x80: "f17",
    0x81: "f18",
    0x82: "f19",
    0x83: "f20",
    0x84: "f21",
    0x85: "f22",
    0x86: "f23",
    0x87: "f24",
    0x90: "num_lock",
    0x91: "scroll_lock",
    0xA0: "lshift",
    0xA1: "rshift",
    0xA2: "lctrl",
    0xA3: "rctrl",
    0xA4: "lalt",
    0xA5: "ralt",
}


def _precise_timestamp_ns() -> int:
    filetime = wintypes.FILETIME()
    get_precise = getattr(kernel32, "GetSystemTimePreciseAsFileTime", None)

    if get_precise is None:
        kernel32.GetSystemTimeAsFileTime(ctypes.byref(filetime))
    else:
        get_precise(ctypes.byref(filetime))

    ticks = (int(filetime.dwHighDateTime) << 32) | int(filetime.dwLowDateTime)
    return (ticks - FILETIME_EPOCH_OFFSET_100NS) * 100


def _timestamp_payload() -> dict:
    timestamp_ns = _precise_timestamp_ns()
    wall_seconds, wall_remainder = divmod(timestamp_ns, 1_000_000_000)
    wall_time = datetime.fromtimestamp(wall_seconds, tz=timezone.utc).replace(microsecond=wall_remainder // 1000)
    return {
        "timestamp_ns": timestamp_ns,
        "wall_time_utc": wall_time.isoformat(),
    }


def _decode_wheel_delta(mouse_data: int) -> int:
    value = ctypes.c_short((mouse_data >> 16) & 0xFFFF).value
    return int(value)


def _map_virtual_key_name(vk_code: int) -> str:
    if 0x30 <= vk_code <= 0x39 or 0x41 <= vk_code <= 0x5A:
        return chr(vk_code).lower()

    return VK_NAME_MAP.get(vk_code, f"vk_{vk_code}")


class InputRecorder:
    def __init__(
        self,
        output_path: Path,
        stop_key: str = "f10",
        poll_interval_ms: int = 5,
        start_wall_time: Optional[datetime] = None,
        start_perf_ns: Optional[int] = None,
        suppress_keys: Optional[set] = None,
    ):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("w", encoding="utf-8", buffering=1)

        self.lock = RLock()
        self.stop_event = Event()
        self.start_perf_ns = start_perf_ns if start_perf_ns is not None else time.perf_counter_ns()
        self.start_wall_time = start_wall_time if start_wall_time is not None else datetime.now(timezone.utc)
        self.stop_key_name = stop_key.lower()
        self.poll_interval_s = max(poll_interval_ms, 1) / 1000.0
        # Keys to fully silence (both keydown and keyup) — orchestrator hotkeys
        self.suppress_keys: set = {k.lower() for k in suppress_keys} if suppress_keys else set()

        self.sequence_id = 0
        self.closed = False

        self.pressed_keys: set[str] = set()
        self.pressed_buttons: set[str] = set()
        self.absolute_mouse_position: Optional[tuple[int, int]] = None
        self.absolute_mouse_delta: tuple[int, int] = (0, 0)
        self.relative_mouse_delta_since_snapshot: tuple[int, int] = (0, 0)

        self.record_queue: Queue[Optional[dict]] = Queue()
        self.keyboard_hook: Optional[int] = None
        self.mouse_hook: Optional[int] = None
        self.snapshot_thread: Optional[Thread] = None
        self.consumer_thread: Optional[Thread] = None
        self._thread_id: Optional[int] = None
        self._window_handle: Optional[int] = None
        self._class_name: Optional[str] = None

        self._wndproc_ref = WNDPROC(self._window_proc)
        self._keyboard_hook_ref = HOOKPROC(self._keyboard_hook_proc)
        self._mouse_hook_ref = HOOKPROC(self._mouse_hook_proc)

        atexit.register(self.close)

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter_ns() - self.start_perf_ns) / 1_000_000.0, 3)

    def _build_record_locked(self, payload: dict, include_relative_delta: bool = True) -> dict:
        record = dict(payload)
        record.update(self._snapshot_state(include_relative_delta=include_relative_delta))
        timestamp_payload = _timestamp_payload()
        record["wall_time_utc"] = timestamp_payload["wall_time_utc"]
        record["timestamp_ns"] = timestamp_payload["timestamp_ns"]
        record["elapsed_ms"] = self._elapsed_ms()
        record["sequence_id"] = self.sequence_id
        self.sequence_id += 1
        return record

    def _snapshot_state(self, include_relative_delta: bool = True) -> dict:
        if self.absolute_mouse_position is None:
            mouse_position = None
        else:
            mouse_position = [self.absolute_mouse_position[0], self.absolute_mouse_position[1]]

        if include_relative_delta:
            relative_delta = [self.relative_mouse_delta_since_snapshot[0], self.relative_mouse_delta_since_snapshot[1]]
        else:
            relative_delta = [0, 0]

        return {
            "held_keys": sorted(self.pressed_keys),
            "held_buttons": sorted(self.pressed_buttons),
            "mouse_position": mouse_position,
            "mouse_delta": relative_delta,
            "cursor_confined": _is_cursor_confined(),  # True=world mode, False=GUI mode
        }

    def _enqueue_record(self, payload: dict, include_relative_delta: bool = True) -> None:
        with self.lock:
            self.record_queue.put(self._build_record_locked(payload, include_relative_delta=include_relative_delta))

    def _record_mouse_relative_delta(self, dx: int, dy: int, payload: dict) -> None:
        with self.lock:
            self.relative_mouse_delta_since_snapshot = (
                self.relative_mouse_delta_since_snapshot[0] + dx,
                self.relative_mouse_delta_since_snapshot[1] + dy,
            )
            self.record_queue.put(self._build_record_locked(payload))

    def _emit_snapshot(self) -> None:
        with self.lock:
            relative_delta = self.relative_mouse_delta_since_snapshot
            self.relative_mouse_delta_since_snapshot = (0, 0)
            payload = {
                "event_type": "state_snapshot",
                "snapshot_interval_ms": int(self.poll_interval_s * 1000),
                "relative_mouse_delta": [relative_delta[0], relative_delta[1]],
            }
            self.record_queue.put(self._build_record_locked(payload, include_relative_delta=False))

    def _snapshot_thread_loop(self) -> None:
        while not self.stop_event.wait(self.poll_interval_s):
            self._emit_snapshot()

    def _consumer_thread_loop(self) -> None:
        while True:
            try:
                record = self.record_queue.get(timeout=0.1)
            except Empty:
                continue

            if record is None:
                break

            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

        while True:
            try:
                record = self.record_queue.get_nowait()
            except Empty:
                break

            if record is not None:
                self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.file.flush()

    def _keyboard_hook_proc(self, n_code, w_param, l_param):
        if n_code != HC_ACTION:
            return user32.CallNextHookEx(self.keyboard_hook, n_code, w_param, l_param)

        info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk_name = _map_virtual_key_name(int(info.vkCode))
        is_key_down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
        is_key_up = w_param in (WM_KEYUP, WM_SYSKEYUP)
        is_injected = bool(info.flags & LLKHF_INJECTED)

        # Drop injected events (e.g. from playback via SendInput) — not real gameplay.
        # Also fully silence orchestrator hotkeys on both keydown and keyup.
        if is_injected or vk_name in self.suppress_keys:
            with self.lock:
                self.pressed_keys.discard(vk_name)
            return user32.CallNextHookEx(self.keyboard_hook, n_code, w_param, l_param)

        if is_key_down:
            with self.lock:
                if vk_name not in self.pressed_keys:
                    self.pressed_keys.add(vk_name)
                    self.record_queue.put(self._build_record_locked({
                        "event_type": "keyboard",
                        "action": "down",
                        "key": vk_name,
                        "vk_code": int(info.vkCode),
                        "scan_code": int(info.scanCode),
                        "injected": False,
                        "extended": bool(info.flags & LLKHF_EXTENDED),
                        "alt_down": bool(info.flags & LLKHF_ALTDOWN),
                    }))

            if vk_name == self.stop_key_name:
                self.stop_event.set()
                self._post_quit_message()
                return 1  # suppress — do not pass stop key to the game

        elif is_key_up:
            with self.lock:
                self.pressed_keys.discard(vk_name)
                self.record_queue.put(self._build_record_locked({
                    "event_type": "keyboard",
                    "action": "up",
                    "key": vk_name,
                    "vk_code": int(info.vkCode),
                    "scan_code": int(info.scanCode),
                    "injected": False,
                    "extended": bool(info.flags & LLKHF_EXTENDED),
                    "alt_down": bool(info.flags & LLKHF_ALTDOWN),
                }))

        return user32.CallNextHookEx(self.keyboard_hook, n_code, w_param, l_param)

    def _mouse_hook_proc(self, n_code, w_param, l_param):
        if n_code != HC_ACTION:
            return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

        info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
        x_pos = int(info.pt.x)
        y_pos = int(info.pt.y)
        message_name = LOW_LEVEL_MOUSE_MESSAGES.get(int(w_param), f"msg_{int(w_param)}")

        with self.lock:
            if self.absolute_mouse_position is None:
                absolute_delta = (0, 0)
            else:
                absolute_delta = (
                    x_pos - self.absolute_mouse_position[0],
                    y_pos - self.absolute_mouse_position[1],
                )

            self.absolute_mouse_position = (x_pos, y_pos)
            self.absolute_mouse_delta = absolute_delta

            if w_param == WM_MOUSEMOVE:
                self.record_queue.put(self._build_record_locked({
                    "event_type": "mouse_absolute",
                    "action": "move",
                    "x": x_pos,
                    "y": y_pos,
                    "dx": absolute_delta[0],
                    "dy": absolute_delta[1],
                    "injected": bool(info.flags & LLMHF_INJECTED),
                    "lower_il_injected": bool(info.flags & LLMHF_LOWER_IL_INJECTED),
                }))
                return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

            if w_param in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN, WM_XBUTTONDOWN):
                button_name = self._mouse_button_name_from_message(int(w_param), int(info.mouseData))
                if button_name not in self.pressed_buttons:
                    self.pressed_buttons.add(button_name)

                self.record_queue.put(self._build_record_locked({
                    "event_type": "mouse_button",
                    "action": "down",
                    "button": button_name,
                    "x": x_pos,
                    "y": y_pos,
                    "message": message_name,
                    "injected": bool(info.flags & LLMHF_INJECTED),
                }))
                return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

            if w_param in (WM_LBUTTONUP, WM_RBUTTONUP, WM_MBUTTONUP, WM_XBUTTONUP):
                button_name = self._mouse_button_name_from_message(int(w_param), int(info.mouseData))
                self.pressed_buttons.discard(button_name)

                self.record_queue.put(self._build_record_locked({
                    "event_type": "mouse_button",
                    "action": "up",
                    "button": button_name,
                    "x": x_pos,
                    "y": y_pos,
                    "message": message_name,
                    "injected": bool(info.flags & LLMHF_INJECTED),
                }))
                return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

            if w_param == WM_MOUSEWHEEL:
                wheel_delta = _decode_wheel_delta(int(info.mouseData))
                self.record_queue.put(self._build_record_locked({
                    "event_type": "mouse_scroll",
                    "action": "vertical",
                    "delta": wheel_delta,
                    "x": x_pos,
                    "y": y_pos,
                    "message": message_name,
                    "injected": bool(info.flags & LLMHF_INJECTED),
                }))
                return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

            if w_param == WM_MOUSEHWHEEL:
                wheel_delta = _decode_wheel_delta(int(info.mouseData))
                self.record_queue.put(self._build_record_locked({
                    "event_type": "mouse_scroll",
                    "action": "horizontal",
                    "delta": wheel_delta,
                    "x": x_pos,
                    "y": y_pos,
                    "message": message_name,
                    "injected": bool(info.flags & LLMHF_INJECTED),
                }))
                return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

        return user32.CallNextHookEx(self.mouse_hook, n_code, w_param, l_param)

    def _mouse_button_name_from_message(self, message: int, mouse_data: int) -> str:
        # Prefixed with 'lbutton'/'rbutton'/'mbutton' to avoid collision with
        # arrow key names ('left', 'right') in action strings fed to the LLM.
        if message in (WM_LBUTTONDOWN, WM_LBUTTONUP):
            return "lbutton"
        if message in (WM_RBUTTONDOWN, WM_RBUTTONUP):
            return "rbutton"
        if message in (WM_MBUTTONDOWN, WM_MBUTTONUP):
            return "mbutton"
        if message in (WM_XBUTTONDOWN, WM_XBUTTONUP):
            button = (mouse_data >> 16) & 0xFFFF
            return "xbutton1" if button == 0x0001 else "xbutton2"
        return f"button_{message}"

    def _window_proc(self, hwnd, msg, w_param, l_param):
        if msg == WM_INPUT:
            self._handle_raw_input(l_param)
            return 0

        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0

        return user32.DefWindowProcW(hwnd, msg, w_param, l_param)

    def _handle_raw_input(self, raw_input_handle):
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(RAWINPUTHEADER)

        result = user32.GetRawInputData(raw_input_handle, RID_INPUT, None, ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF:
            return

        buffer = ctypes.create_string_buffer(size.value)
        result = user32.GetRawInputData(raw_input_handle, RID_INPUT, buffer, ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF:
            return

        raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
        if int(raw.header.dwType) != RIM_TYPEMOUSE:
            return

        mouse_data = raw.data.mouse
        # If MOUSE_MOVE_ABSOLUTE is set, lLastX/lLastY are absolute coords not deltas.
        # This happens in GUI scenes in many games. Skip here — absolute position is
        # already captured by the LL mouse hook via WM_MOUSEMOVE.
        if int(mouse_data.usFlags) & MOUSE_MOVE_ABSOLUTE:
            return

        dx = int(mouse_data.lLastX)
        dy = int(mouse_data.lLastY)
        if dx == 0 and dy == 0:
            return

        self._record_mouse_relative_delta(dx, dy, {
            "event_type": "mouse_relative",
            "dx": dx,
            "dy": dy,
            "raw_flags": int(mouse_data.usFlags),
            "raw_buttons": int(mouse_data.ulRawButtons),
            "raw_extra_information": int(mouse_data.ulExtraInformation),
        })

    def _create_message_window(self) -> None:
        self._class_name = f"InputRecorderWindow_{id(self):x}"
        hinstance = kernel32.GetModuleHandleW(None)

        wnd_class = WNDCLASSW()
        wnd_class.style = 0
        wnd_class.lpfnWndProc = self._wndproc_ref
        wnd_class.cbClsExtra = 0
        wnd_class.cbWndExtra = 0
        wnd_class.hInstance = hinstance
        wnd_class.hIcon = None
        wnd_class.hCursor = None
        wnd_class.hbrBackground = None
        wnd_class.lpszMenuName = None
        wnd_class.lpszClassName = self._class_name

        atom = user32.RegisterClassW(ctypes.byref(wnd_class))
        if atom == 0:
            error = ctypes.get_last_error()
            if error != 1410:  # ERROR_CLASS_ALREADY_EXISTS
                raise ctypes.WinError(error)

        self._window_handle = user32.CreateWindowExW(
            0,
            self._class_name,
            self._class_name,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            hinstance,
            None,
        )
        if not self._window_handle:
            raise ctypes.WinError(ctypes.get_last_error())

        raw_devices = (RAWINPUTDEVICE * 1)()
        raw_devices[0].usUsagePage = 0x01
        raw_devices[0].usUsage = 0x02
        raw_devices[0].dwFlags = RIDEV_INPUTSINK
        raw_devices[0].hwndTarget = self._window_handle

        if not user32.RegisterRawInputDevices(raw_devices, 1, ctypes.sizeof(RAWINPUTDEVICE)):
            raise ctypes.WinError(ctypes.get_last_error())

    def _install_hooks(self) -> None:
        hinstance = kernel32.GetModuleHandleW(None)
        self.keyboard_hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._keyboard_hook_ref, hinstance, 0)
        if not self.keyboard_hook:
            raise ctypes.WinError(ctypes.get_last_error())

        self.mouse_hook = user32.SetWindowsHookExW(WH_MOUSE_LL, self._mouse_hook_ref, hinstance, 0)
        if not self.mouse_hook:
            user32.UnhookWindowsHookEx(self.keyboard_hook)
            self.keyboard_hook = None
            raise ctypes.WinError(ctypes.get_last_error())

    def _post_quit_message(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    def start(self) -> None:
        with self.lock:
            self.record_queue.put(self._build_record_locked({
                "event_type": "session_start",
                "logger": "windows_hooks_raw_input",
                "stop_key": self.stop_key_name,
                "snapshot_interval_ms": int(self.poll_interval_s * 1000),
                "timestamp_source": "GetSystemTimePreciseAsFileTime",
                "session_start_wall_time_utc": self.start_wall_time.isoformat(),
            }, include_relative_delta=False))

        self._create_message_window()
        self._thread_id = kernel32.GetCurrentThreadId()

        warmup_message = MSG()
        user32.PeekMessageW(ctypes.byref(warmup_message), None, 0, 0, 0)

        self._install_hooks()

        self.consumer_thread = Thread(target=self._consumer_thread_loop, name="Input Record Writer", daemon=True)
        self.consumer_thread.start()

        self.snapshot_thread = Thread(target=self._snapshot_thread_loop, name="Input State Snapshot", daemon=True)
        self.snapshot_thread.start()

        if self.stop_key_name != "__never__":
            print(f"Recording input events to {self.output_path}")
            print(f"Press {self.stop_key_name.upper()} to stop.")

        msg = MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0:
                break
            if result == -1:
                raise ctypes.WinError(ctypes.get_last_error())

            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self.close()

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True

        self.stop_event.set()
        self._post_quit_message()

        if self.snapshot_thread is not None and self.snapshot_thread.is_alive() and self.snapshot_thread is not current_thread():
            self.snapshot_thread.join(timeout=1)

        if self.mouse_hook:
            try:
                user32.UnhookWindowsHookEx(self.mouse_hook)
            except Exception:
                pass
            self.mouse_hook = None

        if self.keyboard_hook:
            try:
                user32.UnhookWindowsHookEx(self.keyboard_hook)
            except Exception:
                pass
            self.keyboard_hook = None

        if self._window_handle:
            try:
                user32.DestroyWindow(self._window_handle)
            except Exception:
                pass
            self._window_handle = None

        with self.lock:
            self.record_queue.put(self._build_record_locked({
                "event_type": "session_end",
            }, include_relative_delta=False))

        self.record_queue.put(None)

        if self.consumer_thread is not None and self.consumer_thread.is_alive() and self.consumer_thread is not current_thread():
            self.consumer_thread.join(timeout=2)

        with self.lock:
            if not self.file.closed:
                self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record keyboard and mouse input with Windows hooks and Raw Input. Works with most PC games.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSONL file that will store the input events.",
    )
    parser.add_argument(
        "--stop-key",
        default="f10",
        help="Key used to stop the recording session. Default: f10",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=5,
        help="Interval between state snapshots in milliseconds. Default: 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recorder = InputRecorder(Path(args.output), stop_key=args.stop_key, poll_interval_ms=args.poll_interval_ms)
    recorder.start()


if __name__ == "__main__":
    main()