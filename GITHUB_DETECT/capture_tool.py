#!/usr/bin/env python3
"""
capture_tool.py — YOLO Dataset Capture Tool (Phase 0 + Phase 1)

Phase 0 — Class Manager:
    Type a class name → hit Enter → appears in numbered list.
    Reorder (Up/Down), rename (double-click or button), remove.
    Tag classes with value reader type (bar_fill, digit_ocr, text_ocr).
    Auto-writes training/classes.txt + training/data.yaml on every change.

Phase 1 — Screenshot Capture:
    F6 = Single screenshot → datasets/raw/
    F7 = Toggle slow auto-capture (every 3s)
    F8 = Toggle fast auto-capture (every 0.5s)
    Esc = Stop auto-capture
    dHash deduplication (hamming distance < 5)

Screen capture starts on-demand (first F6/F7/F8), not at launch.

Requirements:
    pip install PyQt5 opencv-python numpy pynput mss
"""

import numpy as np
import json
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import argparse
import platform

# ---------------------------------------------------------------------------
# Windows DPI — MUST be before any Qt import
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# PyQt5
# ---------------------------------------------------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QListWidget,
    QComboBox, QInputDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV required.  pip install opencv-python")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    print("WARNING: mss not found. pip install mss  (falling back to PIL)")

if not HAS_MSS:
    from PIL import ImageGrab

# ---------------------------------------------------------------------------
# pynput
# ---------------------------------------------------------------------------
try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput required.  pip install pynput")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CAPTURE_FPS = 30
DHASH_THRESHOLD = 5
DHASH_WINDOW_SECONDS = 30
SLOW_CAPTURE_INTERVAL = 3.0
FAST_CAPTURE_INTERVAL = 0.5
OVERLAY_REDRAW_MS = 100

# Dark theme
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
DARK_INPUT = "#0f3460"
DARK_TEXT = "#e0e0e0"
DARK_DIM = "#888888"
DARK_ACCENT = "#533483"

VALUE_READER_OPTIONS = ["", "bar_fill", "digit_ocr", "text_ocr"]
VALUE_READER_LABELS = {
    "": "None",
    "bar_fill": "Bar Fill",
    "digit_ocr": "Digit OCR",
    "text_ocr": "Text OCR",
}


# =============================================================================
# CLASS MANAGER — data model + file I/O
# =============================================================================

class ClassEntry:
    """Single class in the YOLO taxonomy."""

    def __init__(self, name: str, value_reader: str = ""):
        self.name = name
        self.value_reader = value_reader

    def to_dict(self):
        d = {"name": self.name}
        if self.value_reader:
            d["value_reader"] = self.value_reader
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d.get("value_reader", ""))


class ClassManager:
    """
    Manages YOLO class list.
    Source of truth: training/class_config.json
    Auto-generates: training/classes.txt, training/data.yaml
    """

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.training_dir = project_dir / "training"
        self.config_path = self.training_dir / "class_config.json"
        self.classes_txt_path = self.training_dir / "classes.txt"
        self.data_yaml_path = self.training_dir / "data.yaml"
        self.classes: List[ClassEntry] = []

        self.training_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load from class_config.json, or fall back to classes.txt."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.classes = [ClassEntry.from_dict(d) for d in data.get("classes", [])]
        elif self.classes_txt_path.exists():
            with open(self.classes_txt_path, "r") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.classes.append(ClassEntry(name))
            self._save()

    def _save(self):
        """Write class_config.json, classes.txt, and data.yaml."""
        # Source of truth
        with open(self.config_path, "w") as f:
            json.dump({
                "classes": [c.to_dict() for c in self.classes],
                "last_modified": datetime.now().isoformat(),
            }, f, indent=2)

        # classes.txt (for LabelImg)
        with open(self.classes_txt_path, "w") as f:
            for c in self.classes:
                f.write(c.name + "\n")

        # data.yaml (for YOLO training)
        with open(self.data_yaml_path, "w") as f:
            f.write("# Auto-generated by capture tool — do not edit manually\n")
            f.write("path: ../datasets/game_objects\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write("names:\n")
            for i, c in enumerate(self.classes):
                f.write(f"  {i}: {c.name}\n")

    def add(self, name: str, value_reader: str = "") -> bool:
        """Add a class. Returns False if name already exists or is invalid."""
        clean = name.strip().replace(" ", "_").lower()
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        if not clean:
            return False
        if any(c.name == clean for c in self.classes):
            return False
        self.classes.append(ClassEntry(clean, value_reader))
        self._save()
        return True

    def remove(self, index: int) -> Optional[str]:
        """Remove class at index. Returns removed name or None."""
        if 0 <= index < len(self.classes):
            removed = self.classes.pop(index)
            self._save()
            return removed.name
        return None

    def rename(self, index: int, new_name: str) -> bool:
        """Rename class at index. Returns False if invalid or duplicate."""
        clean = new_name.strip().replace(" ", "_").lower()
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        if not clean:
            return False
        if any(i != index and c.name == clean for i, c in enumerate(self.classes)):
            return False
        if 0 <= index < len(self.classes):
            self.classes[index].name = clean
            self._save()
            return True
        return False

    def move_up(self, index: int) -> bool:
        if 0 < index < len(self.classes):
            self.classes[index - 1], self.classes[index] = \
                self.classes[index], self.classes[index - 1]
            self._save()
            return True
        return False

    def move_down(self, index: int) -> bool:
        if 0 <= index < len(self.classes) - 1:
            self.classes[index], self.classes[index + 1] = \
                self.classes[index + 1], self.classes[index]
            self._save()
            return True
        return False

    def set_value_reader(self, index: int, value_reader: str):
        if 0 <= index < len(self.classes):
            self.classes[index].value_reader = value_reader
            self._save()

    def has_annotations(self, class_name: str) -> bool:
        """Check if any YOLO label files reference this class index."""
        labels_dir = self.project_dir / "datasets" / "game_objects" / "labels"
        if not labels_dir.exists():
            return False
        idx = next((i for i, c in enumerate(self.classes) if c.name == class_name), None)
        if idx is None:
            return False
        for txt_file in labels_dir.rglob("*.txt"):
            try:
                with open(txt_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and int(parts[0]) == idx:
                            return True
            except (ValueError, IndexError):
                continue
        return False

    def format_display(self, index: int) -> str:
        """Format class for list display: '0: class_name  [Bar Fill]'"""
        c = self.classes[index]
        vr = f"  [{VALUE_READER_LABELS[c.value_reader]}]" if c.value_reader else ""
        return f"{index:2d}: {c.name}{vr}"


# =============================================================================
# dHash UTILITIES
# =============================================================================

def dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    """Compute difference hash. Returns 64-bit integer."""
    resized = cv2.resize(image_bgr, (hash_size + 1, hash_size))
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum(2**i for i, v in enumerate(diff.flatten()) if v)


def hamming(h1: int, h2: int) -> int:
    """Hamming distance between two hashes."""
    return bin(h1 ^ h2).count('1')


# =============================================================================
# SCREEN CAPTURE — on-demand, lazy start
# =============================================================================

class ScreenCapture:
    """Background screen capture thread. Starts only when start() is called."""

    def __init__(self, target_fps: int = CAPTURE_FPS):
        self.target_fps = target_fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self):
        interval = 1.0 / self.target_fps
        if HAS_MSS:
            sct = mss.mss()
            monitor = sct.monitors[1]
            while self._running:
                t0 = time.perf_counter()
                try:
                    raw = np.array(sct.grab(monitor))
                    with self._lock:
                        self._frame = raw[:, :, :3].copy()
                except Exception:
                    pass
                dt = time.perf_counter() - t0
                if dt < interval:
                    time.sleep(interval - dt)
        else:
            while self._running:
                t0 = time.perf_counter()
                try:
                    img = ImageGrab.grab()
                    with self._lock:
                        self._frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
                dt = time.perf_counter() - t0
                if dt < interval:
                    time.sleep(interval - dt)


# =============================================================================
# CAPTURE CONTROLLER — saves frames with dHash dedup
# =============================================================================

class CaptureController:
    """Manages screenshot saving with dHash deduplication."""

    def __init__(self, output_dir: Path, screen_capture: ScreenCapture):
        self.output_dir = output_dir
        self.screen_capture = screen_capture
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.dup_count = 0
        self._sequence = 0
        self._recent_hashes: List[Tuple[float, int]] = []
        self._last_save_info = "Ready"

    def _ensure_capture(self):
        """Start screen capture if not already running."""
        if not self.screen_capture.is_running:
            self.screen_capture.start()
            # Brief wait for first frame
            for _ in range(10):
                if self.screen_capture.get_frame() is not None:
                    break
                time.sleep(0.05)

    def _prune_hashes(self):
        """Remove hashes older than the dedup window."""
        cutoff = time.time() - DHASH_WINDOW_SECONDS
        self._recent_hashes = [(t, h) for t, h in self._recent_hashes if t > cutoff]

    def _is_duplicate(self, frame_hash: int) -> bool:
        """Check if frame is too similar to recently saved frames."""
        self._prune_hashes()
        for _, h in self._recent_hashes:
            if hamming(frame_hash, h) < DHASH_THRESHOLD:
                return True
        return False

    def save_frame(self) -> Optional[str]:
        """Capture and save current frame. Returns filename if saved, None if skipped."""
        self._ensure_capture()
        frame = self.screen_capture.get_frame()
        if frame is None:
            self._last_save_info = "No frame available"
            return None

        frame_hash = dhash(frame)
        if self._is_duplicate(frame_hash):
            self.dup_count += 1
            self._last_save_info = f"Skipped duplicate (#{self.dup_count})"
            return None

        now = datetime.now()
        self._sequence += 1
        filename = f"frame_{now.strftime('%Y%m%d_%H%M%S')}_{self._sequence:03d}.png"
        filepath = self.output_dir / filename

        cv2.imwrite(str(filepath), frame)

        self._recent_hashes.append((time.time(), frame_hash))
        self.frame_count += 1
        self._last_save_info = f"Saved: {filename}"
        return filename

    @property
    def last_save_info(self) -> str:
        return self._last_save_info


# =============================================================================
# OVERLAY WIDGET — transparent, click-through
# =============================================================================

class OverlayWidget(QWidget):
    """Minimal transparent overlay for capture status."""

    def __init__(self, screen_geo: QRect):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint |
            Qt.WindowTransparentForInput | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(screen_geo)
        self.status_text = ""

    def set_status(self, text: str):
        self.status_text = text
        self.update()

    def clear(self):
        self.status_text = ""
        self.update()

    def paintEvent(self, event):
        if not self.status_text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        sw = self.width()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 160)))
        p.drawRect(0, 6, sw, 36)
        p.setFont(QFont("Consolas", 12, QFont.Bold))
        p.setPen(QPen(QColor(0, 255, 255)))
        p.drawText(QRect(0, 6, sw, 36), Qt.AlignCenter, self.status_text)
        p.end()


# =============================================================================
# HOTKEY BRIDGE — pynput → Qt signals
# =============================================================================

class HotkeyBridge(QObject):
    f6_pressed = pyqtSignal()
    f7_pressed = pyqtSignal()
    f8_pressed = pyqtSignal()
    esc_pressed = pyqtSignal()


# =============================================================================
# CONTROL PANEL — Class Manager + Capture Controls
# =============================================================================

class ControlPanel(QMainWindow):
    """Main window with Class Manager (Phase 0) and Capture Controls (Phase 1)."""

    def __init__(self, class_mgr: ClassManager, capture_ctrl: CaptureController,
                 overlay: OverlayWidget, bridge: HotkeyBridge):
        super().__init__()
        self.class_mgr = class_mgr
        self.capture_ctrl = capture_ctrl
        self.overlay = overlay
        self.bridge = bridge

        self.auto_mode: Optional[str] = None
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_capture_tick)

        self._build_ui()
        self._connect_signals()
        self._refresh_class_list()

    def _build_ui(self):
        self.setWindowTitle("YOLO Capture Tool — Phase 0 + 1")
        self.setFixedSize(480, 750)
        self.move(50, 50)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {DARK_BG}; }}
            QGroupBox {{
                color: {DARK_DIM}; border: 1px solid #333;
                border-radius: 4px; margin-top: 8px; padding-top: 14px;
                background-color: {DARK_PANEL};
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
            QLabel {{ color: {DARK_TEXT}; }}
            QLineEdit {{
                background-color: {DARK_INPUT}; color: {DARK_TEXT};
                border: 1px solid #333; border-radius: 3px; padding: 4px;
                font-family: Consolas; font-size: 11px;
            }}
            QListWidget {{
                background-color: {DARK_INPUT}; color: {DARK_TEXT};
                border: 1px solid #333; border-radius: 3px;
                font-family: Consolas; font-size: 11px;
            }}
            QListWidget::item:selected {{ background-color: {DARK_ACCENT}; }}
            QComboBox {{
                background-color: {DARK_INPUT}; color: {DARK_TEXT};
                border: 1px solid #333; border-radius: 3px; padding: 3px;
                font-family: Consolas; font-size: 11px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {DARK_INPUT}; color: {DARK_TEXT};
                selection-background-color: {DARK_ACCENT};
            }}
            QPushButton {{
                border: none; border-radius: 4px; padding: 6px 12px;
                font-size: 11px; font-weight: bold; color: white;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel("YOLO Capture Tool")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {DARK_TEXT};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ==== CLASS MANAGER (Phase 0) ====
        cm_group = QGroupBox("Class Manager")
        cm_layout = QVBoxLayout(cm_group)

        # Add class row
        add_row = QHBoxLayout()
        self.inp_class = QLineEdit()
        self.inp_class.setPlaceholderText("Type class name, hit Enter")
        self.inp_class.returnPressed.connect(self._add_class)
        add_row.addWidget(self.inp_class, 1)

        btn_add = QPushButton("Add")
        btn_add.setStyleSheet("background-color: #2ca02c;")
        btn_add.clicked.connect(self._add_class)
        add_row.addWidget(btn_add)
        cm_layout.addLayout(add_row)

        # Class list
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        cm_layout.addWidget(self.class_list)

        # Reorder + edit buttons
        btn_row = QHBoxLayout()

        btn_up = QPushButton("\u25b2 Up")
        btn_up.setStyleSheet("background-color: #1f77b4;")
        btn_up.clicked.connect(self._move_up)
        btn_row.addWidget(btn_up)

        btn_down = QPushButton("\u25bc Down")
        btn_down.setStyleSheet("background-color: #1f77b4;")
        btn_down.clicked.connect(self._move_down)
        btn_row.addWidget(btn_down)

        btn_rename = QPushButton("Rename")
        btn_rename.setStyleSheet("background-color: #ff7f0e;")
        btn_rename.clicked.connect(self._rename_class)
        btn_row.addWidget(btn_rename)

        btn_remove = QPushButton("Remove")
        btn_remove.setStyleSheet("background-color: #d62728;")
        btn_remove.clicked.connect(self._remove_class)
        btn_row.addWidget(btn_remove)

        cm_layout.addLayout(btn_row)

        # Value reader assignment
        vr_row = QHBoxLayout()
        vr_row.addWidget(QLabel("Value Reader:"))
        self.cmb_value_reader = QComboBox()
        for key in VALUE_READER_OPTIONS:
            self.cmb_value_reader.addItem(VALUE_READER_LABELS[key], key)
        self.cmb_value_reader.currentIndexChanged.connect(self._on_value_reader_changed)
        vr_row.addWidget(self.cmb_value_reader, 1)
        cm_layout.addLayout(vr_row)

        # Class count
        self.lbl_class_count = QLabel("0 classes")
        self.lbl_class_count.setStyleSheet(
            f"color: {DARK_DIM}; font-family: Consolas; font-size: 10px;")
        cm_layout.addWidget(self.lbl_class_count)

        layout.addWidget(cm_group)

        # ==== CAPTURE CONTROLS (Phase 1) ====
        cap_group = QGroupBox("Screenshot Capture")
        cap_layout = QVBoxLayout(cap_group)

        cap_btns = QHBoxLayout()

        self.btn_single = QPushButton("Single Shot (F6)")
        self.btn_single.setStyleSheet("background-color: #2ca02c;")
        self.btn_single.clicked.connect(self._single_capture)
        cap_btns.addWidget(self.btn_single)

        self.btn_slow = QPushButton("Slow 3s (F7)")
        self.btn_slow.setStyleSheet("background-color: #1f77b4;")
        self.btn_slow.clicked.connect(self._toggle_slow)
        cap_btns.addWidget(self.btn_slow)

        self.btn_fast = QPushButton("Fast 0.5s (F8)")
        self.btn_fast.setStyleSheet("background-color: #ff7f0e;")
        self.btn_fast.clicked.connect(self._toggle_fast)
        cap_btns.addWidget(self.btn_fast)

        cap_layout.addLayout(cap_btns)

        self.lbl_capture_status = QLabel("Idle — press F6 for single shot, F7/F8 for auto.")
        self.lbl_capture_status.setStyleSheet(
            f"color: {DARK_DIM}; font-family: Consolas; font-size: 10px;")
        self.lbl_capture_status.setWordWrap(True)
        cap_layout.addWidget(self.lbl_capture_status)

        self.lbl_frame_count = QLabel("Frames: 0  |  Skipped: 0")
        self.lbl_frame_count.setStyleSheet(
            "color: #00ff00; font-family: Consolas; font-size: 11px;")
        cap_layout.addWidget(self.lbl_frame_count)

        layout.addWidget(cap_group)

        # Hotkey reference
        hk = QLabel("F6=Single  F7=Slow Auto  F8=Fast Auto  Esc=Stop Auto")
        hk.setStyleSheet("color: #555; font-family: Consolas; font-size: 9px;")
        hk.setAlignment(Qt.AlignCenter)
        layout.addWidget(hk)

    def _connect_signals(self):
        self.bridge.f6_pressed.connect(self._single_capture)
        self.bridge.f7_pressed.connect(self._toggle_slow)
        self.bridge.f8_pressed.connect(self._toggle_fast)
        self.bridge.esc_pressed.connect(self._stop_auto)

    # ---- Class Manager actions ----

    def _refresh_class_list(self):
        current = self.class_list.currentRow()
        self.class_list.clear()
        for i in range(len(self.class_mgr.classes)):
            self.class_list.addItem(self.class_mgr.format_display(i))
        if 0 <= current < self.class_list.count():
            self.class_list.setCurrentRow(current)
        n = len(self.class_mgr.classes)
        self.lbl_class_count.setText(
            f"{n} class{'es' if n != 1 else ''}  |  "
            f"training/classes.txt + data.yaml auto-saved")

    def _add_class(self):
        name = self.inp_class.text().strip()
        if not name:
            return
        if self.class_mgr.add(name):
            self.inp_class.clear()
            self._refresh_class_list()
            self.class_list.setCurrentRow(self.class_list.count() - 1)
        else:
            QMessageBox.warning(self, "Cannot Add",
                                f"'{name}' already exists or is invalid.")

    def _remove_class(self):
        idx = self.class_list.currentRow()
        if idx < 0:
            return
        entry = self.class_mgr.classes[idx]

        if self.class_mgr.has_annotations(entry.name):
            reply = QMessageBox.warning(
                self, "Annotations Exist",
                f"'{entry.name}' is referenced in existing label files.\n"
                f"Removing it will shift all higher class indices.\n\n"
                f"Remove anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self, "Remove Class",
                f"Remove '{entry.name}'?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.class_mgr.remove(idx)
        self._refresh_class_list()

    def _rename_class(self):
        idx = self.class_list.currentRow()
        if idx < 0:
            return
        old_name = self.class_mgr.classes[idx].name
        new_name, ok = QInputDialog.getText(
            self, "Rename Class", f"Rename '{old_name}' to:", text=old_name)
        if ok and new_name:
            if not self.class_mgr.rename(idx, new_name):
                QMessageBox.warning(self, "Cannot Rename",
                                    f"'{new_name}' is invalid or already exists.")
            else:
                self._refresh_class_list()

    def _move_up(self):
        idx = self.class_list.currentRow()
        if self.class_mgr.move_up(idx):
            self._refresh_class_list()
            self.class_list.setCurrentRow(idx - 1)

    def _move_down(self):
        idx = self.class_list.currentRow()
        if self.class_mgr.move_down(idx):
            self._refresh_class_list()
            self.class_list.setCurrentRow(idx + 1)

    def _on_class_selected(self, idx):
        if 0 <= idx < len(self.class_mgr.classes):
            vr = self.class_mgr.classes[idx].value_reader
            combo_idx = VALUE_READER_OPTIONS.index(vr) if vr in VALUE_READER_OPTIONS else 0
            self.cmb_value_reader.blockSignals(True)
            self.cmb_value_reader.setCurrentIndex(combo_idx)
            self.cmb_value_reader.blockSignals(False)

    def _on_value_reader_changed(self, combo_idx):
        class_idx = self.class_list.currentRow()
        if class_idx < 0:
            return
        vr = VALUE_READER_OPTIONS[combo_idx] if combo_idx < len(VALUE_READER_OPTIONS) else ""
        self.class_mgr.set_value_reader(class_idx, vr)
        self._refresh_class_list()

    # ---- Capture actions ----

    def _single_capture(self):
        result = self.capture_ctrl.save_frame()
        self._update_capture_ui()
        if result:
            self.overlay.show()
            self.overlay.set_status(
                f"Captured #{self.capture_ctrl.frame_count}: {result}")
            if self.auto_mode is None:
                QTimer.singleShot(2000, self._hide_overlay_if_idle)

    def _hide_overlay_if_idle(self):
        if self.auto_mode is None:
            self.overlay.clear()

    def _toggle_slow(self):
        if self.auto_mode == "slow":
            self._stop_auto()
        else:
            self._start_auto("slow")

    def _toggle_fast(self):
        if self.auto_mode == "fast":
            self._stop_auto()
        else:
            self._start_auto("fast")

    def _start_auto(self, mode: str):
        self._stop_auto()
        self.auto_mode = mode

        interval_ms = int(
            (SLOW_CAPTURE_INTERVAL if mode == "slow" else FAST_CAPTURE_INTERVAL) * 1000)
        self._auto_timer.start(interval_ms)

        # Update button appearance
        if mode == "slow":
            self.btn_slow.setStyleSheet("background-color: #d62728;")
            self.btn_slow.setText("STOP Slow (F7)")
            self.btn_fast.setStyleSheet("background-color: #ff7f0e;")
            self.btn_fast.setText("Fast 0.5s (F8)")
        else:
            self.btn_fast.setStyleSheet("background-color: #d62728;")
            self.btn_fast.setText("STOP Fast (F8)")
            self.btn_slow.setStyleSheet("background-color: #1f77b4;")
            self.btn_slow.setText("Slow 3s (F7)")

        label = "SLOW (3s)" if mode == "slow" else "FAST (0.5s)"
        self.overlay.show()
        self.overlay.set_status(
            f"Auto-capture: {label}  |  {self.capture_ctrl.frame_count} frames  |  Esc=stop")
        self._update_capture_ui()

    def _stop_auto(self):
        self._auto_timer.stop()
        self.auto_mode = None
        self.btn_slow.setStyleSheet("background-color: #1f77b4;")
        self.btn_slow.setText("Slow 3s (F7)")
        self.btn_fast.setStyleSheet("background-color: #ff7f0e;")
        self.btn_fast.setText("Fast 0.5s (F8)")
        self.overlay.clear()
        self._update_capture_ui()

    def _auto_capture_tick(self):
        self.capture_ctrl.save_frame()
        self._update_capture_ui()
        label = "SLOW (3s)" if self.auto_mode == "slow" else "FAST (0.5s)"
        self.overlay.set_status(
            f"Auto: {label}  |  "
            f"{self.capture_ctrl.frame_count} frames  |  "
            f"{self.capture_ctrl.dup_count} dupes skipped  |  "
            f"Esc=stop")

    def _update_capture_ui(self):
        mode_str = {None: "Idle", "slow": "SLOW AUTO (3s)", "fast": "FAST AUTO (0.5s)"}
        self.lbl_capture_status.setText(
            f"Mode: {mode_str[self.auto_mode]}  |  {self.capture_ctrl.last_save_info}")
        self.lbl_frame_count.setText(
            f"Frames: {self.capture_ctrl.frame_count}  |  "
            f"Skipped: {self.capture_ctrl.dup_count}")

    def closeEvent(self, event):
        self._stop_auto()
        self.overlay.close()
        event.accept()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO Capture Tool — Class Manager + Screenshot Capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Class Manager (Phase 0):
  Add/remove/rename/reorder YOLO classes via UI.
  Tag classes with value reader type (bar_fill, digit_ocr, text_ocr).
  Auto-writes training/classes.txt + training/data.yaml on every change.

Screenshot Capture (Phase 1):
  F6 = Single screenshot → datasets/raw/
  F7 = Toggle slow auto-capture (every 3s)
  F8 = Toggle fast auto-capture (every 0.5s)
  Esc = Stop auto-capture

  dHash dedup skips near-duplicate frames (hamming < 5).
  Screen capture starts on-demand, not at launch.
""")
    parser.add_argument("--project_dir", type=str, default=".",
                        help="Project root directory (default: current directory)")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()

    app = QApplication(sys.argv)

    class_mgr = ClassManager(project_dir)

    screen_capture = ScreenCapture(CAPTURE_FPS)
    raw_dir = project_dir / "datasets" / "raw"
    capture_ctrl = CaptureController(raw_dir, screen_capture)

    screen_geo = app.primaryScreen().geometry()
    overlay = OverlayWidget(screen_geo)

    bridge = HotkeyBridge()

    panel = ControlPanel(class_mgr, capture_ctrl, overlay, bridge)

    def on_key(key):
        try:
            mapping = {
                keyboard.Key.f6: bridge.f6_pressed,
                keyboard.Key.f7: bridge.f7_pressed,
                keyboard.Key.f8: bridge.f8_pressed,
                keyboard.Key.esc: bridge.esc_pressed,
            }
            sig = mapping.get(key)
            if sig:
                sig.emit()
        except Exception:
            pass

    kb = keyboard.Listener(on_press=on_key)
    kb.daemon = True
    kb.start()

    panel.show()

    print("\n" + "=" * 60)
    print("YOLO CAPTURE TOOL — Phase 0 + Phase 1")
    print("=" * 60)
    print(f"Project dir:  {project_dir}")
    print(f"Classes:      {len(class_mgr.classes)}")
    print(f"Raw dir:      {raw_dir}")
    print(f"Config:       {class_mgr.config_path}")
    print("Hotkeys:      F6=Single  F7=Slow Auto  F8=Fast Auto  Esc=Stop")
    print("=" * 60)

    def cleanup():
        screen_capture.stop()
        kb.stop()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
