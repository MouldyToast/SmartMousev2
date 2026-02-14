#!/usr/bin/env python3
"""
calibrate_objects.py - Visual Object Calibration Tool v4.0 (PyQt5)

PURPOSE:
    Build a reusable visual object database for game automation.
    Calibrate each game asset ONCE (capturing multiple visual variants),
    then runtime code can find it anywhere on screen regardless of
    camera angle, zoom, or object state.

ARCHITECTURE:
    - Control Panel:   PyQt5 window — metadata entry, object list, status
    - Review Window:   Frozen screenshot — draw rectangle around object (cursor-free)
    - Overlay:         Transparent click-through — shows tracking box during recording
    - Capture Thread:  Continuous screenshots at ~30Hz via mss
    - Variant Tracker: CSRT (discriminative) tracking + automatic variant saving

CALIBRATION FLOW:
    1. Type name, action, category, state label in control panel
    2. Position camera so object is visible, cursor AWAY from object
    3. F2 — freezes current frame, opens Review Window
    4. Draw rectangle around the object on the frozen image (cursor-free capture)
    5. Click Confirm — initial template saved, Review Window closes
    6. F3 — start recording. Slowly rotate camera 360 degrees
       System tracks the object frame-to-frame, automatically saves
       new variants when appearance changes enough
    7. F4 — stop recording → automatically enters TEST MODE
       Searches live screen for ALL variants, draws matches on overlay
    8. F4 again — exit test mode

    Repeat with different state labels for different object states
    (e.g. "full", "half_empty", "depleted")

HOTKEYS:
    F2  = Freeze frame → open Review Window for selection
    F3  = Start recording (after selection confirmed)
    F4  = Stop recording / Exit test mode
    F5  = Test an existing object (select from list first)
    Esc = Cancel current operation

SAVED DATA:
    calibration/
    └── objects/
        └── health_potion/
            ├── config.json             # metadata + variant index
            ├── variants/
            │   ├── v_0000.png          # color
            │   ├── v_0000_gray.png     # grayscale
            │   ├── v_0001.png
            │   └── ...
            └── edges/
                ├── v_0000.png          # Canny edges
                └── ...

REQUIREMENTS:
    pip install PyQt5 opencv-python numpy pynput mss
"""

import numpy as np
import json
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
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
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QListWidget, QMessageBox, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPoint, QRect, QRectF, pyqtSignal, QObject, QSize,
)
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QFont, QBrush, QPolygonF, QImage, QPixmap,
)

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV required.  pip install opencv-python")
    sys.exit(1)

# CSRT tracker availability (API changed across OpenCV versions)
def _create_csrt():
    """Create a CSRT tracker, handling API differences across OpenCV versions."""
    # OpenCV 4.5.1+
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    # OpenCV 4.x with legacy module
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    # Fallback: try contrib
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        pass
    print("ERROR: CSRT tracker not available.")
    print("  Install:  pip install opencv-contrib-python")
    print(f"  Your OpenCV version: {cv2.__version__}")
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
    from pynput import keyboard, mouse
except ImportError:
    print("ERROR: pynput required.  pip install pynput")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CAL_DIR = "calibration"
CAPTURE_FPS = 30
OVERLAY_REDRAW_MS = 33              # ~30 FPS

# Tracking (CSRT)
SAVE_THRESHOLD = 0.85               # Below this vs all saved = new variant worth saving
MIN_FRAMES_BETWEEN_SAVES = 15       # ~0.5s at 30fps, prevents flooding
DRIFT_CHECK_INTERVAL = 30           # Check for drift every N frames
DRIFT_THRESHOLD = 0.30              # If best variant match < this, tracker has drifted
REACQUIRE_THRESHOLD = 0.45          # Confidence needed for multi-variant reacquisition

# Template matching
DEFAULT_MATCH_THRESHOLD = 0.75
TEST_THRESHOLD = 0.65               # Lower for test mode discovery

# Colors
COL_HULL = QColor(0, 255, 0)
COL_HULL_FILL = QColor(0, 255, 0, 30)
COL_BBOX = QColor(255, 255, 0)
COL_CLICK = QColor(255, 0, 0)
COL_MATCH = QColor(0, 255, 0)
COL_MATCH_WEAK = QColor(255, 255, 0)
COL_STATUS_BG = QColor(0, 0, 0, 160)
COL_STATUS_TEXT = QColor(0, 255, 255)
COL_TRACK_BOX = QColor(0, 200, 255)
COL_TRACK_FILL = QColor(0, 200, 255, 25)

# Panel styling
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
DARK_INPUT = "#0f3460"
DARK_TEXT = "#e0e0e0"
DARK_DIM = "#888888"

# Default padding around selection for template crop
DEFAULT_PADDING = 4


# =============================================================================
# DATA: Object Config + Variant Info
# =============================================================================

class VariantInfo:
    """Metadata for a single visual variant of an object."""

    def __init__(self, vid: str, label: str = "", state: str = "default"):
        self.id = vid                       # e.g. "v_0000"
        self.label = label                  # e.g. "initial", "auto_frame_150"
        self.state = state                  # e.g. "full", "half_empty"
        self.template_path = ""             # relative: "variants/v_0000.png"
        self.template_gray_path = ""        # relative: "variants/v_0000_gray.png"
        self.edges_path = ""                # relative: "edges/v_0000.png"
        self.source_frame_idx: int = 0      # frame number during recording
        self.bbox_on_frame: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.saved_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "label": self.label, "state": self.state,
            "template_path": self.template_path,
            "template_gray_path": self.template_gray_path,
            "edges_path": self.edges_path,
            "source_frame_idx": self.source_frame_idx,
            "bbox_on_frame": list(self.bbox_on_frame),
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "VariantInfo":
        v = cls(d["id"], d.get("label", ""), d.get("state", "default"))
        v.template_path = d.get("template_path", "")
        v.template_gray_path = d.get("template_gray_path", "")
        v.edges_path = d.get("edges_path", "")
        v.source_frame_idx = d.get("source_frame_idx", 0)
        v.bbox_on_frame = tuple(d.get("bbox_on_frame", (0, 0, 0, 0)))
        v.saved_at = d.get("saved_at", "")
        return v


class ObjectConfig:
    """Full configuration for a calibrated game object."""

    def __init__(self, name: str, action: str = "", category: str = ""):
        self.name = name
        self.action = action
        self.category = category
        self.tags: List[str] = self._build_tags()
        self.match_threshold: float = DEFAULT_MATCH_THRESHOLD
        self.click_offset_ratio: Tuple[float, float] = (0.5, 0.5)
        self.initial_bbox_size: Tuple[int, int] = (0, 0)
        self.variants: List[VariantInfo] = []
        self.recording_sessions: List[Dict] = []
        self.capture_resolution: Tuple[int, int] = (0, 0)
        self.created_at: str = ""
        self.last_modified: str = ""

    def _build_tags(self) -> List[str]:
        tags = set()
        for field in [self.name, self.action, self.category]:
            if field:
                tags.add(field.lower())
                for part in field.lower().split("_"):
                    if len(part) > 1:
                        tags.add(part)
        return sorted(tags)

    def next_variant_id(self) -> str:
        return f"v_{len(self.variants):04d}"

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "action": self.action, "category": self.category,
            "tags": self.tags, "match_threshold": self.match_threshold,
            "click_offset_ratio": list(self.click_offset_ratio),
            "initial_bbox_size": list(self.initial_bbox_size),
            "variants": [v.to_dict() for v in self.variants],
            "recording_sessions": self.recording_sessions,
            "capture_resolution": list(self.capture_resolution),
            "created_at": self.created_at, "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ObjectConfig":
        obj = cls(d["name"], d.get("action", ""), d.get("category", ""))
        obj.tags = d.get("tags", obj._build_tags())
        obj.match_threshold = d.get("match_threshold", DEFAULT_MATCH_THRESHOLD)
        obj.click_offset_ratio = tuple(d.get("click_offset_ratio", (0.5, 0.5)))
        obj.initial_bbox_size = tuple(d.get("initial_bbox_size", (0, 0)))
        obj.variants = [VariantInfo.from_dict(v) for v in d.get("variants", [])]
        obj.recording_sessions = d.get("recording_sessions", [])
        obj.capture_resolution = tuple(d.get("capture_resolution", (0, 0)))
        obj.created_at = d.get("created_at", "")
        obj.last_modified = d.get("last_modified", "")
        return obj


# =============================================================================
# CALIBRATION STORE
# =============================================================================

class CalibrationStore:
    """
    Manages object calibration data on disk.
    
    Structure:
        calibration/
        ├── manifest.json           # Quick lookup index
        └── objects/
            └── <name>/
                ├── config.json     # Full object config
                ├── variants/       # Template images
                └── edges/          # Canny edge templates
    """

    def __init__(self, cal_dir: str):
        self.cal_dir = Path(cal_dir)
        self.objects_dir = self.cal_dir / "objects"
        self.manifest_path = self.cal_dir / "manifest.json"

        self.cal_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {"objects": {}, "index_by_action": {}, "index_by_category": {},
                "created_at": datetime.now().isoformat()}

    def _save_manifest(self):
        self.manifest["last_modified"] = datetime.now().isoformat()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _rebuild_indexes(self):
        by_action, by_cat = defaultdict(list), defaultdict(list)
        for name, info in self.manifest["objects"].items():
            if info.get("action"):
                by_action[info["action"]].append(name)
            if info.get("category"):
                by_cat[info["category"]].append(name)
        self.manifest["index_by_action"] = dict(by_action)
        self.manifest["index_by_category"] = dict(by_cat)

    def _obj_dir(self, name: str) -> Path:
        d = self.objects_dir / name
        d.mkdir(exist_ok=True)
        (d / "variants").mkdir(exist_ok=True)
        (d / "edges").mkdir(exist_ok=True)
        return d

    def save_config(self, cfg: ObjectConfig):
        """Save object config to disk and update manifest."""
        cfg.last_modified = datetime.now().isoformat()
        if not cfg.created_at:
            cfg.created_at = cfg.last_modified

        obj_dir = self._obj_dir(cfg.name)
        with open(obj_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

        self.manifest["objects"][cfg.name] = {
            "action": cfg.action, "category": cfg.category,
            "tags": cfg.tags, "variant_count": len(cfg.variants),
            "match_threshold": cfg.match_threshold,
            "bbox_size": list(cfg.initial_bbox_size),
            "states": sorted(set(v.state for v in cfg.variants)),
        }
        self._rebuild_indexes()
        self._save_manifest()

    def save_variant_images(self, name: str, vid: str,
                            bgr: np.ndarray, gray: np.ndarray) -> VariantInfo:
        """Save variant template images and return updated VariantInfo."""
        obj_dir = self._obj_dir(name)

        # Color
        tpl_path = f"variants/{vid}.png"
        cv2.imwrite(str(obj_dir / tpl_path), bgr)

        # Grayscale
        gray_path = f"variants/{vid}_gray.png"
        cv2.imwrite(str(obj_dir / gray_path), gray)

        # Canny edges
        edges = cv2.Canny(gray, 50, 200)
        edges_path = f"edges/{vid}.png"
        cv2.imwrite(str(obj_dir / edges_path), edges)

        vi = VariantInfo(vid)
        vi.template_path = tpl_path
        vi.template_gray_path = gray_path
        vi.edges_path = edges_path
        vi.saved_at = datetime.now().isoformat()
        return vi

    def load_config(self, name: str) -> Optional[ObjectConfig]:
        path = self.objects_dir / name / "config.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return ObjectConfig.from_dict(json.load(f))

    def load_all_variant_grays(self, name: str) -> List[np.ndarray]:
        """Load all grayscale variant templates for an object."""
        cfg = self.load_config(name)
        if not cfg:
            return []
        obj_dir = self.objects_dir / name
        grays = []
        for v in cfg.variants:
            path = obj_dir / v.template_gray_path
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    grays.append(img)
        return grays

    def list_objects(self) -> List[str]:
        return sorted(self.manifest.get("objects", {}).keys())

    def delete_object(self, name: str):
        import shutil
        obj_dir = self.objects_dir / name
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        self.manifest["objects"].pop(name, None)
        self._rebuild_indexes()
        self._save_manifest()

    def get_object_summary(self, name: str) -> str:
        info = self.manifest["objects"].get(name, {})
        vc = info.get("variant_count", 0)
        action = info.get("action", "")
        states = info.get("states", [])
        size = info.get("bbox_size", [0, 0])
        parts = [f"{name}: {vc} variants, {size[0]}x{size[1]}px"]
        if action:
            parts.append(f"action={action}")
        if states:
            parts.append(f"states={states}")
        return "  ".join(parts)


# =============================================================================
# SCREEN CAPTURE THREAD
# =============================================================================

class ScreenCapture:
    """Continuous background capture. Thread-safe access to latest frame (BGR)."""

    def __init__(self, target_fps: int = CAPTURE_FPS):
        self.target_fps = target_fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
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
                    # mss returns BGRA on Windows; drop alpha → BGR for OpenCV
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
# VARIANT TRACKER
# =============================================================================

class VariantTracker:
    """
    CSRT-based object tracking with automatic variant capture.
    
    OpenCV's CSRT (Channel and Spatial Reliability) tracker:
    - Learns a DISCRIMINATIVE model: object pixels vs background pixels
    - Uses spatial reliability maps: center = object, edges = background
    - Adapts its model each frame to handle gradual appearance changes
    - Handles moderate rotation, scale changes, and partial occlusion
    
    Unlike raw template matching, CSRT won't drift onto the background
    because it actively models what IS the object and what ISN'T.
    
    Drift detection: Periodically compares tracked region against saved
    variants. If similarity drops too low, CSRT has drifted and we
    reinitialize from a full-frame variant search.
    
    Variant saving: When the tracked region looks sufficiently different
    from ALL saved variants, it's a new visual appearance worth keeping.
    """

    def __init__(self, initial_frame_bgr: np.ndarray,
                 initial_crop_gray: np.ndarray,
                 initial_bbox: Tuple[int, int, int, int]):
        """
        Args:
            initial_frame_bgr: Full frame (BGR) for CSRT initialization
            initial_crop_gray: Grayscale crop for variant library
            initial_bbox: (x, y, w, h) on screen where object was selected
        """
        self.w, self.h = initial_bbox[2], initial_bbox[3]
        self.bbox = initial_bbox
        self.tracking = True

        # CSRT tracker
        self._csrt = _create_csrt()
        self._csrt.init(initial_frame_bgr, initial_bbox)

        # Variant library (grayscale crops for comparison)
        self.saved_grays: List[np.ndarray] = [initial_crop_gray.copy()]
        self.initial_gray = initial_crop_gray.copy()  # Never modified — ground truth

        # Timing
        self.frames_since_save = 0
        self.frames_since_drift_check = 0

        # Stats
        self.frame_count = 0
        self.variant_count = 1
        self.lost_count = 0
        self.reacquired_count = 0
        self.drift_resets = 0

        # New variant buffer (consumed by caller)
        self._new_variants: List[Tuple[np.ndarray, np.ndarray, Tuple]] = []

    @property
    def new_variants(self) -> List[Tuple[np.ndarray, np.ndarray, Tuple]]:
        """Pop new variants that were auto-saved since last check."""
        out = self._new_variants
        self._new_variants = []
        return out

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Process one frame. Returns (x, y, w, h, confidence) or None if lost.
        """
        self.frame_count += 1
        self.frames_since_save += 1
        self.frames_since_drift_check += 1

        if self.tracking:
            return self._track(frame_bgr)
        else:
            return self._reacquire(frame_bgr)

    def _track(self, frame_bgr: np.ndarray):
        """Use CSRT to track, check for drift, auto-save variants."""
        success, bbox_raw = self._csrt.update(frame_bgr)

        if not success:
            self.tracking = False
            self.lost_count += 1
            return None

        # CSRT returns (x, y, w, h) as floats
        x, y, w, h = [int(v) for v in bbox_raw]
        fh, fw = frame_bgr.shape[:2]

        # Bounds check
        if x < 0 or y < 0 or x + w > fw or y + h > fh or w < 5 or h < 5:
            self.tracking = False
            self.lost_count += 1
            return None

        self.bbox = (x, y, w, h)

        # Get current crop
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        crop_gray = frame_gray[y:y + h, x:x + w].copy()
        crop_bgr = frame_bgr[y:y + h, x:x + w].copy()

        # --- Drift detection ---
        # Periodically verify tracked region still resembles the object
        if self.frames_since_drift_check >= DRIFT_CHECK_INTERVAL:
            self.frames_since_drift_check = 0
            drift_score = self._best_similarity(crop_gray)

            if drift_score < DRIFT_THRESHOLD:
                # CSRT has drifted — tracked region no longer looks like any variant
                print(f"    ! Drift detected (score {drift_score:.2f}), attempting reacquisition...")
                self.tracking = False
                self.drift_resets += 1
                return self._reacquire(frame_bgr)

        # --- Variant saving ---
        # Check if appearance changed enough from ALL saved variants
        if self.frames_since_save >= MIN_FRAMES_BETWEEN_SAVES:
            best_sim = self._best_similarity(crop_gray)

            if best_sim < SAVE_THRESHOLD:
                # New distinct appearance — save it
                self.saved_grays.append(crop_gray.copy())
                self._new_variants.append((crop_bgr, crop_gray, self.bbox))
                self.variant_count += 1
                self.frames_since_save = 0

        # Confidence: compare current crop to initial template
        conf = self._similarity(crop_gray, self.initial_gray)
        return (x, y, w, h, float(conf))

    def _reacquire(self, frame_bgr: np.ndarray):
        """
        Search full frame using all saved variants to find the object.
        If found, reinitialize CSRT at that location.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = frame_gray.shape[:2]
        best_val = 0.0
        best_box = None

        for variant in self.saved_grays:
            vh, vw = variant.shape[:2]
            if fh < vh or fw < vw:
                continue
            result = cv2.matchTemplate(frame_gray, variant, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(result)
            if mv > best_val:
                best_val = mv
                best_box = (ml[0], ml[1], vw, vh)

        if best_val >= REACQUIRE_THRESHOLD and best_box is not None:
            x, y, w, h = best_box
            self.bbox = best_box

            # Reinitialize CSRT at the found location
            self._csrt = _create_csrt()
            self._csrt.init(frame_bgr, best_box)

            self.tracking = True
            self.reacquired_count += 1
            self.frames_since_drift_check = 0

            print(f"    ✓ Reacquired (score {best_val:.2f}) at ({x}, {y})")
            return (*best_box, float(best_val))

        return None

    def _best_similarity(self, crop_gray: np.ndarray) -> float:
        """Highest match score of crop against all saved variants."""
        best = 0.0
        for v in self.saved_grays:
            best = max(best, self._similarity(crop_gray, v))
        return best

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute TM_CCOEFF_NORMED similarity between two crops (resize if needed)."""
        h, w = a.shape[:2]
        if h < 3 or w < 3:
            return 0.0
        br = cv2.resize(b, (w, h))
        result = cv2.matchTemplate(a, br, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])

    def get_stats_str(self) -> str:
        return (f"Frames: {self.frame_count}  |  "
                f"Variants: {self.variant_count}  |  "
                f"Lost: {self.lost_count}  |  "
                f"Reacq: {self.reacquired_count}  |  "
                f"Drift resets: {self.drift_resets}")


# =============================================================================
# TEMPLATE MATCHING (multi-variant, for test mode + runtime)
# =============================================================================

def find_multi_variant_matches(
    frame_gray: np.ndarray,
    variant_grays: List[np.ndarray],
    threshold: float = TEST_THRESHOLD,
    max_matches: int = 20,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Search frame for ALL variants, merge results with NMS.
    Returns [(x, y, w, h, confidence, variant_idx), ...].
    """
    fh, fw = frame_gray.shape[:2]
    all_boxes, all_confs, all_vidx = [], [], []

    for vi, tpl in enumerate(variant_grays):
        th, tw = tpl.shape[:2]
        if fh < th or fw < tw:
            continue
        result = cv2.matchTemplate(frame_gray, tpl, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for py, px in zip(*locs):
            all_boxes.append([int(px), int(py), int(px + tw), int(py + th)])
            all_confs.append(float(result[py, px]))
            all_vidx.append(vi)

    if not all_boxes:
        return []

    boxes = np.array(all_boxes)
    confs = np.array(all_confs)
    vidxs = np.array(all_vidx)

    # NMS
    order = np.argsort(-confs)
    keep, suppressed = [], set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        if len(keep) >= max_matches:
            break
        bx1, by1, bx2, by2 = boxes[idx]
        for oi in order:
            if oi in suppressed or oi == idx:
                continue
            ox1, oy1, ox2, oy2 = boxes[oi]
            ix1, iy1 = max(bx1, ox1), max(by1, oy1)
            ix2, iy2 = min(bx2, ox2), min(by2, oy2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = ((bx2 - bx1) * (by2 - by1) +
                         (ox2 - ox1) * (oy2 - oy1) - inter)
                if union > 0 and inter / union > 0.3:
                    suppressed.add(oi)

    return [(int(boxes[i][0]), int(boxes[i][1]),
             int(boxes[i][2] - boxes[i][0]), int(boxes[i][3] - boxes[i][1]),
             float(confs[i]), int(vidxs[i])) for i in keep]


# =============================================================================
# OVERLAY WIDGET (transparent, click-through)
# =============================================================================

class OverlayWidget(QWidget):
    """
    Fullscreen transparent overlay. QPainter drawing, click-through.
    Uses Qt.WA_TranslucentBackground (true per-pixel alpha).
    """

    def __init__(self, screen_geo: QRect):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint |
            Qt.WindowTransparentForInput | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(screen_geo)

        # Draw state
        self.track_box: Optional[Tuple[int, int, int, int, float]] = None
        self.matches: List[Tuple[int, int, int, int, float]] = []
        self.match_label: str = ""
        self.status_text: str = ""

    def clear_all(self):
        self.track_box = None
        self.matches = []
        self.match_label = ""
        self.status_text = ""
        self.update()

    def set_track_box(self, x, y, w, h, conf):
        self.track_box = (x, y, w, h, conf)
        self.update()

    def set_matches(self, matches, label=""):
        self.matches = [(m[0], m[1], m[2], m[3], m[4]) for m in matches]
        self.match_label = label
        self.update()

    def set_status(self, text):
        self.status_text = text
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # Status bar
        if self.status_text:
            sw = self.width()
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(COL_STATUS_BG))
            p.drawRect(0, 6, sw, 40)
            p.setFont(QFont("Arial", 13, QFont.Bold))
            p.setPen(QPen(COL_STATUS_TEXT))
            p.drawText(QRect(0, 6, sw, 40), Qt.AlignCenter, self.status_text)

        # Tracking box
        if self.track_box:
            x, y, w, h, conf = self.track_box
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(COL_TRACK_FILL))
            p.drawRect(x, y, w, h)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(COL_TRACK_BOX, 3))
            p.drawRect(x, y, w, h)
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.drawText(x, y - 6, f"Tracking {conf:.0%}")

        # Test matches
        if self.matches:
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.setBrush(Qt.NoBrush)
            for x, y, w, h, conf in self.matches:
                col = COL_MATCH if conf >= DEFAULT_MATCH_THRESHOLD else COL_MATCH_WEAK
                p.setPen(QPen(col, 3))
                p.drawRect(x, y, w, h)
                lbl = f"{self.match_label} {conf:.0%}" if self.match_label else f"{conf:.0%}"
                p.drawText(x + 2, y - 6, lbl)

        p.end()


# =============================================================================
# REVIEW WINDOW — frozen frame + rectangle selection
# =============================================================================

class ReviewWindow(QWidget):
    """
    Displays a frozen screenshot. User draws a rectangle around the object.
    Emits selection_confirmed with (x, y, w, h) in ORIGINAL image coordinates.
    """

    selection_confirmed = pyqtSignal(int, int, int, int)
    selection_cancelled = pyqtSignal()

    def __init__(self, frame_bgr: np.ndarray, object_name: str):
        super().__init__()
        self.setWindowTitle(f"Select '{object_name}' — draw rectangle around it")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.frame_bgr = frame_bgr
        self.img_h, self.img_w = frame_bgr.shape[:2]

        # Scale to fit ~80% of screen
        screen = QApplication.primaryScreen().geometry()
        max_w = int(screen.width() * 0.8)
        max_h = int(screen.height() * 0.8)
        self.scale = min(max_w / self.img_w, max_h / self.img_h, 1.0)
        self.disp_w = int(self.img_w * self.scale)
        self.disp_h = int(self.img_h * self.scale)

        # Convert BGR to QPixmap
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.scale < 1.0:
            rgb = cv2.resize(rgb, (self.disp_w, self.disp_h))
        qimg = QImage(rgb.data, self.disp_w, self.disp_h,
                      self.disp_w * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)

        # Selection state
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._selecting = False

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image label
        self.img_label = QLabel()
        self.img_label.setPixmap(self.pixmap)
        self.img_label.setFixedSize(self.disp_w, self.disp_h)
        self.img_label.setMouseTracking(True)
        self.img_label.mousePressEvent = self._on_press
        self.img_label.mouseMoveEvent = self._on_move
        self.img_label.mouseReleaseEvent = self._on_release
        self.img_label.paintEvent = self._paint_selection
        layout.addWidget(self.img_label)

        # Buttons
        btn_row = QHBoxLayout()
        self.lbl_info = QLabel("Click and drag to select the object")
        self.lbl_info.setStyleSheet(f"color: {DARK_TEXT}; padding: 4px;")
        btn_row.addWidget(self.lbl_info, 1)

        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet("background-color: #2ca02c; color: white; "
                                       "padding: 8px 20px; font-weight: bold;")
        self.btn_confirm.clicked.connect(self._confirm)
        btn_row.addWidget(self.btn_confirm)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #d62728; color: white; "
                                  "padding: 8px 20px; font-weight: bold;")
        btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

        self.setStyleSheet(f"background-color: {DARK_BG};")
        self.resize(self.disp_w, self.disp_h + 50)

    def _on_press(self, event):
        self._start = event.pos()
        self._end = event.pos()
        self._selecting = True

    def _on_move(self, event):
        if self._selecting:
            self._end = event.pos()
            self.img_label.update()

    def _on_release(self, event):
        self._end = event.pos()
        self._selecting = False
        self.img_label.update()

        # Check if selection is valid
        rect = self._get_display_rect()
        if rect and rect.width() > 5 and rect.height() > 5:
            self.btn_confirm.setEnabled(True)
            ow = int(rect.width() / self.scale)
            oh = int(rect.height() / self.scale)
            self.lbl_info.setText(f"Selected: {ow}x{oh}px — Click Confirm or redraw")
        else:
            self.btn_confirm.setEnabled(False)

    def _get_display_rect(self) -> Optional[QRect]:
        if self._start and self._end:
            return QRect(self._start, self._end).normalized()
        return None

    def _paint_selection(self, event):
        # Draw base pixmap
        p = QPainter(self.img_label)
        p.drawPixmap(0, 0, self.pixmap)

        # Draw selection rectangle
        rect = self._get_display_rect()
        if rect and rect.width() > 2 and rect.height() > 2:
            # Dim outside selection
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 120)))
            # Top
            p.drawRect(0, 0, self.disp_w, rect.top())
            # Bottom
            p.drawRect(0, rect.bottom(), self.disp_w, self.disp_h - rect.bottom())
            # Left
            p.drawRect(0, rect.top(), rect.left(), rect.height())
            # Right
            p.drawRect(rect.right(), rect.top(), self.disp_w - rect.right(), rect.height())

            # Selection outline
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(0, 255, 0), 2))
            p.drawRect(rect)

            # Size label
            ow = int(rect.width() / self.scale)
            oh = int(rect.height() / self.scale)
            p.setFont(QFont("Arial", 10, QFont.Bold))
            p.setPen(QPen(QColor(0, 255, 0)))
            p.drawText(rect.left(), rect.top() - 6, f"{ow}x{oh}px")

        p.end()

    def _get_original_rect(self) -> Optional[Tuple[int, int, int, int]]:
        rect = self._get_display_rect()
        if not rect:
            return None
        x = max(0, int(rect.x() / self.scale))
        y = max(0, int(rect.y() / self.scale))
        w = min(self.img_w - x, int(rect.width() / self.scale))
        h = min(self.img_h - y, int(rect.height() / self.scale))
        return (x, y, w, h)

    def _confirm(self):
        sel = self._get_original_rect()
        if sel and sel[2] > 5 and sel[3] > 5:
            self.selection_confirmed.emit(*sel)
            self.close()

    def _cancel(self):
        self.selection_cancelled.emit()
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._confirm()
        elif event.key() == Qt.Key_Escape:
            self._cancel()


# =============================================================================
# HOTKEY BRIDGE (pynput → Qt signals)
# =============================================================================

class HotkeyBridge(QObject):
    f2_pressed = pyqtSignal()
    f3_pressed = pyqtSignal()
    f4_pressed = pyqtSignal()
    f5_pressed = pyqtSignal()
    esc_pressed = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)


# =============================================================================
# CONTROL PANEL
# =============================================================================

class ControlPanel(QMainWindow):
    """
    Main window. Manages calibration state machine.
    
    States:
        IDLE       — waiting for user
        SELECTING  — review window open, user drawing rectangle
        RECORDING  — tracking object, auto-saving variants
        TEST       — multi-variant matching on live screen
    """

    S_IDLE = "idle"
    S_SELECTING = "selecting"
    S_RECORDING = "recording"
    S_TEST = "test"

    def __init__(self, store: CalibrationStore, capture: ScreenCapture,
                 overlay: OverlayWidget, bridge: HotkeyBridge,
                 padding: int = DEFAULT_PADDING):
        super().__init__()
        self.store = store
        self.capture = capture
        self.overlay = overlay
        self.bridge = bridge
        self.padding = padding
        self.state = self.S_IDLE

        # Current calibration state
        self.current_cfg: Optional[ObjectConfig] = None
        self.current_state_label: str = "default"
        self.frozen_frame: Optional[np.ndarray] = None
        self.tracker: Optional[VariantTracker] = None
        self.review_win: Optional[ReviewWindow] = None
        self._init_crop_gray: Optional[np.ndarray] = None
        self._init_bbox: Optional[Tuple[int, int, int, int]] = None

        # Test mode
        self.test_variants: List[np.ndarray] = []
        self.test_name: str = ""

        self._build_ui()
        self._connect_signals()

        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(OVERLAY_REDRAW_MS)

    def _build_ui(self):
        self.setWindowTitle("Object Calibration v4.0")
        self.setFixedSize(460, 660)
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
                font-family: Consolas; font-size: 10px;
            }}
            QListWidget::item:selected {{ background-color: #533483; }}
            QPushButton {{
                border: none; border-radius: 4px; padding: 6px 14px;
                font-size: 11px; font-weight: bold; color: white;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Title
        title = QLabel("Object Calibration v4")
        title.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {DARK_TEXT};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Metadata ---
        meta = QGroupBox("Object Metadata")
        mg = QGridLayout(meta)

        mg.addWidget(QLabel("Name:"), 0, 0)
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("e.g. health_potion")
        mg.addWidget(self.inp_name, 0, 1)

        mg.addWidget(QLabel("Action:"), 1, 0)
        self.inp_action = QLineEdit()
        self.inp_action.setPlaceholderText("e.g. heal, attack")
        mg.addWidget(self.inp_action, 1, 1)

        mg.addWidget(QLabel("Category:"), 2, 0)
        self.inp_category = QLineEdit()
        self.inp_category.setPlaceholderText("e.g. consumable, npc")
        mg.addWidget(self.inp_category, 2, 1)

        mg.addWidget(QLabel("State:"), 3, 0)
        self.inp_state = QLineEdit()
        self.inp_state.setPlaceholderText("e.g. full, half_empty, depleted")
        self.inp_state.setText("default")
        mg.addWidget(self.inp_state, 3, 1)

        layout.addWidget(meta)

        # --- Actions ---
        self.btn_freeze = QPushButton("Freeze Frame + Select  (F2)")
        self.btn_freeze.setStyleSheet("background-color: #2ca02c;")
        self.btn_freeze.clicked.connect(self._on_freeze)
        layout.addWidget(self.btn_freeze)

        rec_row = QHBoxLayout()
        self.btn_record = QPushButton("Start Recording  (F3)")
        self.btn_record.setStyleSheet("background-color: #1f77b4;")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._on_start_recording)
        rec_row.addWidget(self.btn_record)

        self.btn_stop = QPushButton("Stop  (F4)")
        self.btn_stop.setStyleSheet("background-color: #d62728;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        rec_row.addWidget(self.btn_stop)
        layout.addLayout(rec_row)

        # --- Status ---
        sg = QGroupBox("Status")
        sl = QVBoxLayout(sg)
        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setStyleSheet("color: #00ff00; font-family: Consolas; font-size: 11px;")
        sl.addWidget(self.lbl_state)
        self.lbl_info = QLabel("Enter metadata, position camera, press F2 to freeze.")
        self.lbl_info.setStyleSheet(f"color: {DARK_DIM}; font-family: Consolas; font-size: 10px;")
        self.lbl_info.setWordWrap(True)
        sl.addWidget(self.lbl_info)
        layout.addWidget(sg)

        # --- Object list ---
        lg = QGroupBox("Calibrated Objects")
        ll = QVBoxLayout(lg)
        self.obj_list = QListWidget()
        self._refresh_list()
        ll.addWidget(self.obj_list)

        br = QHBoxLayout()
        b_test = QPushButton("Test (F5)")
        b_test.setStyleSheet("background-color: #ff7f0e;")
        b_test.clicked.connect(self._on_test_selected)
        br.addWidget(b_test)

        b_add = QPushButton("Add Variants")
        b_add.setStyleSheet("background-color: #1f77b4;")
        b_add.clicked.connect(self._on_add_variants)
        br.addWidget(b_add)

        b_del = QPushButton("Delete")
        b_del.setStyleSheet("background-color: #d62728;")
        b_del.clicked.connect(self._on_delete)
        br.addWidget(b_del)
        ll.addLayout(br)
        layout.addWidget(lg)

        # Hotkey ref
        hk = QLabel("F2=Freeze  F3=Record  F4=Stop  F5=Test  Esc=Cancel")
        hk.setStyleSheet(f"color: #555; font-family: Consolas; font-size: 9px;")
        hk.setAlignment(Qt.AlignCenter)
        layout.addWidget(hk)

    def _connect_signals(self):
        self.bridge.f2_pressed.connect(self._on_freeze)
        self.bridge.f3_pressed.connect(self._on_start_recording)
        self.bridge.f4_pressed.connect(self._on_stop)
        self.bridge.f5_pressed.connect(self._on_test_selected)
        self.bridge.esc_pressed.connect(self._on_cancel)

    def _refresh_list(self):
        self.obj_list.clear()
        for name in self.store.list_objects():
            self.obj_list.addItem(self.store.get_object_summary(name))

    def _set_status(self, state: str, info: str):
        self.lbl_state.setText(state)
        self.lbl_info.setText(info)

    def _selected_name(self) -> Optional[str]:
        row = self.obj_list.currentRow()
        names = self.store.list_objects()
        return names[row] if 0 <= row < len(names) else None

    # ---- F2: Freeze + Select ----

    def _on_freeze(self):
        if self.state not in (self.S_IDLE,):
            return

        name = self.inp_name.text().strip().replace(" ", "_").lower()
        name = "".join(c for c in name if c.isalnum() or c == "_")
        if not name:
            self._set_status("ERROR", "Enter a name first!")
            return

        self.current_state_label = self.inp_state.text().strip() or "default"

        # Load existing or create new config
        existing = self.store.load_config(name)
        if existing:
            self.current_cfg = existing
            # Update action/category if user changed them
            action = self.inp_action.text().strip()
            category = self.inp_category.text().strip()
            if action:
                self.current_cfg.action = action
            if category:
                self.current_cfg.category = category
        else:
            self.current_cfg = ObjectConfig(
                name,
                self.inp_action.text().strip(),
                self.inp_category.text().strip(),
            )

        # Freeze frame
        self.frozen_frame = self.capture.get_frame()
        if self.frozen_frame is None:
            self._set_status("ERROR", "No frame captured! Is the screen visible?")
            return

        self.state = self.S_SELECTING

        # Open review window
        self.review_win = ReviewWindow(self.frozen_frame, name)
        self.review_win.selection_confirmed.connect(self._on_selection_confirmed)
        self.review_win.selection_cancelled.connect(self._on_selection_cancelled)
        self.review_win.show()

        self._set_status("SELECTING",
                         "Draw a rectangle around the object in the frozen frame.\n"
                         "Click Confirm when done.")
        print(f"\n  Freeze frame captured. Select '{name}' in review window.")

    def _on_selection_confirmed(self, x: int, y: int, w: int, h: int):
        """User confirmed rectangle selection on frozen frame."""
        if not self.current_cfg or self.frozen_frame is None:
            return

        print(f"  Selection: ({x}, {y}) {w}x{h}")

        cfg = self.current_cfg
        cfg.initial_bbox_size = (w, h)
        cfg.capture_resolution = (self.frozen_frame.shape[1], self.frozen_frame.shape[0])

        # Crop initial template (cursor-free!)
        crop_bgr = self.frozen_frame[y:y + h, x:x + w].copy()
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Save as first variant (or next variant if adding to existing)
        vid = cfg.next_variant_id()
        vi = self.store.save_variant_images(cfg.name, vid, crop_bgr, crop_gray)
        vi.label = "initial"
        vi.state = self.current_state_label
        vi.source_frame_idx = 0
        vi.bbox_on_frame = (x, y, w, h)
        cfg.variants.append(vi)
        self.store.save_config(cfg)

        # Store selection data for tracker init at F3 time
        # (CSRT needs a live frame, not the frozen one)
        self._init_crop_gray = crop_gray
        self._init_bbox = (x, y, w, h)
        self.tracker = None  # Created at F3 with live frame

        self.state = self.S_IDLE  # back to idle, waiting for F3
        self.btn_record.setEnabled(True)
        self._refresh_list()

        self._set_status("READY TO RECORD",
                         f"Saved variant for '{cfg.name}' ({w}x{h}px). "
                         f"Total: {len(cfg.variants)} variant(s).\n"
                         f"Press F2 again from a different angle to add more,\n"
                         f"or press F3 to start recording (all variants pre-loaded).")
        print(f"  ✓ Variant saved: {vid}  (total: {len(cfg.variants)})")

    def _on_selection_cancelled(self):
        self.state = self.S_IDLE
        self._set_status("IDLE", "Selection cancelled. Press F2 to try again.")

    # ---- F3: Start Recording ----

    def _on_start_recording(self):
        if self.state != self.S_IDLE or not hasattr(self, '_init_crop_gray'):
            return
        if self._init_crop_gray is None:
            self._set_status("ERROR", "No selection made. Press F2 first.")
            return

        # Grab a LIVE frame for CSRT initialization
        live_frame = self.capture.get_frame()
        if live_frame is None:
            self._set_status("ERROR", "No frame available!")
            return

        # Find the object in the live frame using the initial template
        # (camera shouldn't have moved since F2, but this handles minor changes)
        live_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        th, tw = self._init_crop_gray.shape[:2]

        if live_gray.shape[0] >= th and live_gray.shape[1] >= tw:
            result = cv2.matchTemplate(live_gray, self._init_crop_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= 0.5:
                init_bbox = (max_loc[0], max_loc[1], tw, th)
            else:
                # Fallback to original position
                init_bbox = self._init_bbox
        else:
            init_bbox = self._init_bbox

        # Initialize CSRT tracker on the live frame
        self.tracker = VariantTracker(live_frame, self._init_crop_gray, init_bbox)

        # Load ALL existing variants into the tracker's library
        # This means multiple F2 screenshots from different angles all
        # contribute to drift detection and reacquisition during recording
        existing_grays = self.store.load_all_variant_grays(self.current_cfg.name)
        preloaded = 0
        for eg in existing_grays:
            # Don't duplicate the initial crop (already in saved_grays)
            if eg.shape == self._init_crop_gray.shape:
                diff = cv2.absdiff(eg, cv2.resize(self._init_crop_gray, (eg.shape[1], eg.shape[0])))
                if diff.mean() < 5.0:
                    continue
            self.tracker.saved_grays.append(eg)
            self.tracker.variant_count += 1
            preloaded += 1

        if preloaded > 0:
            print(f"  Pre-loaded {preloaded} existing variants into tracker")

        total_variants = len(self.tracker.saved_grays)

        self.state = self.S_RECORDING
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.overlay.clear_all()
        self.overlay.show()
        self.overlay.set_status(
            f"RECORDING: {self.current_cfg.name}  ({total_variants} variants loaded)  —  "
            f"Rotate camera slowly. F4 to stop."
        )

        self._set_status("RECORDING",
                         f"Tracking '{self.current_cfg.name}' "
                         f"({total_variants} variants pre-loaded).\n"
                         f"Rotate camera slowly — new variants auto-saved.\n"
                         f"Press F4 when done.")
        print(f"\n  Recording started for '{self.current_cfg.name}' "
              f"({total_variants} variants in library)")

    # ---- F4: Stop ----

    def _on_stop(self):
        if self.state == self.S_RECORDING:
            self._stop_recording()
        elif self.state == self.S_TEST:
            self._stop_test()

    def _stop_recording(self):
        """Stop recording, save session info, enter test mode."""
        if not self.tracker or not self.current_cfg:
            self._go_idle()
            return

        cfg = self.current_cfg
        stats = self.tracker.get_stats_str()

        # Save session metadata
        cfg.recording_sessions.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.current_state_label,
            "frames_processed": self.tracker.frame_count,
            "variants_saved": self.tracker.variant_count,
            "lost_count": self.tracker.lost_count,
        })
        self.store.save_config(cfg)
        self._refresh_list()

        print(f"\n  Recording stopped. {stats}")
        print(f"  Total variants for '{cfg.name}': {len(cfg.variants)}")

        # Auto-enter test mode
        self._start_test(cfg.name)

    def _stop_test(self):
        self._go_idle()
        print("  Test mode stopped.")

    # ---- F5: Test ----

    def _on_test_selected(self):
        if self.state == self.S_TEST:
            self._stop_test()
            return
        if self.state != self.S_IDLE:
            return

        name = self._selected_name()
        if not name:
            self._set_status("SELECT", "Select an object from the list first.")
            return
        self._start_test(name)

    def _start_test(self, name: str):
        variants = self.store.load_all_variant_grays(name)
        if not variants:
            self._set_status("ERROR", f"No variants for '{name}'")
            return

        self.test_variants = variants
        self.test_name = name
        self.state = self.S_TEST

        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(False)

        self.overlay.clear_all()
        self.overlay.show()
        self.overlay.set_status(
            f"TEST: {name} ({len(variants)} variants)  —  F4 to stop"
        )
        self._set_status("TEST MODE",
                         f"Searching for '{name}' using {len(variants)} variants.\n"
                         f"Green = strong match, Yellow = weak.\n"
                         f"Press F4 to stop.")
        print(f"\n  Test mode: '{name}' with {len(variants)} variants")

    # ---- Cancel / Idle ----

    def _on_cancel(self):
        if self.state == self.S_SELECTING and self.review_win:
            self.review_win.close()
        self._go_idle()

    def _go_idle(self):
        self.state = self.S_IDLE
        self.tracker = None
        self._init_crop_gray = None
        self._init_bbox = None
        self.test_variants = []
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.overlay.clear_all()
        self.overlay.hide()
        self._set_status("IDLE", "Enter metadata, position camera, press F2.")

    # ---- Add Variants to existing object ----

    def _on_add_variants(self):
        name = self._selected_name()
        if not name:
            return
        cfg = self.store.load_config(name)
        if not cfg:
            return
        # Populate fields and let user do F2 → select → F3 → record
        self.inp_name.setText(cfg.name)
        self.inp_action.setText(cfg.action)
        self.inp_category.setText(cfg.category)
        self.inp_state.setText("")
        self.inp_state.setFocus()
        self._set_status("ADD VARIANTS",
                         f"Adding to '{name}' ({len(cfg.variants)} existing variants).\n"
                         f"Enter state label, then F2 to freeze + select.")

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            return
        reply = QMessageBox.question(self, "Delete",
                                     f"Delete '{name}' and all its variants?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.store.delete_object(name)
            self._refresh_list()

    # ---- Periodic Tick ----

    def _tick(self):
        try:
            if self.state == self.S_RECORDING:
                self._tick_recording()
            elif self.state == self.S_TEST:
                self._tick_test()
        except Exception as e:
            print(f"  Tick error: {e}")

    def _tick_recording(self):
        if not self.tracker or not self.current_cfg:
            return

        frame = self.capture.get_frame()
        if frame is None:
            return

        result = self.tracker.process_frame(frame)

        # Save any new auto-captured variants
        for crop_bgr, crop_gray, bbox in self.tracker.new_variants:
            cfg = self.current_cfg
            vid = cfg.next_variant_id()
            vi = self.store.save_variant_images(cfg.name, vid, crop_bgr, crop_gray)
            vi.label = f"auto_frame_{self.tracker.frame_count}"
            vi.state = self.current_state_label
            vi.source_frame_idx = self.tracker.frame_count
            vi.bbox_on_frame = bbox
            cfg.variants.append(vi)
            self.store.save_config(cfg)
            print(f"    + Variant {vid} saved (frame {self.tracker.frame_count})")

        # Update overlay
        if result:
            x, y, w, h, conf = result
            self.overlay.set_track_box(x, y, w, h, conf)
        else:
            self.overlay.track_box = None

        stats = self.tracker.get_stats_str()
        self.overlay.set_status(
            f"REC: {self.current_cfg.name}  |  {stats}  |  F4 stop"
        )
        self._set_status("RECORDING", stats)

    def _tick_test(self):
        if not self.test_variants:
            return

        frame = self.capture.get_frame()
        if frame is None:
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = find_multi_variant_matches(
            frame_gray, self.test_variants, threshold=TEST_THRESHOLD
        )

        self.overlay.clear_all()
        self.overlay.set_matches(matches, self.test_name)
        self.overlay.set_status(
            f"TEST: {self.test_name}  —  "
            f"{len(matches)} match{'es' if len(matches) != 1 else ''}  "
            f"({len(self.test_variants)} variants)  |  F4 stop"
        )

        if matches:
            confs = [m[4] for m in matches]
            self._set_status("TEST MODE",
                             f"Found {len(matches)} match(es)\n"
                             f"Best: {max(confs):.1%}  Worst: {min(confs):.1%}")
        else:
            self._set_status("TEST MODE",
                             f"No matches for '{self.test_name}'.\n"
                             f"Try adding more variants or lowering threshold.")

    # ---- Cursor position ----

    @staticmethod
    def _get_cursor_pos() -> Tuple[int, int]:
        if IS_WINDOWS:
            from ctypes import wintypes
            pt = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)
        return (0, 0)

    def closeEvent(self, event):
        self.overlay.close()
        event.accept()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Object Calibration Tool v4.0 — variant-based visual database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Flow:
  1. Type name/action/category/state in panel
  2. Position camera, cursor AWAY from object
  3. F2 — freezes screen, opens review window
  4. Draw rectangle around object on frozen image (cursor-free!)
  5. Confirm — initial template saved
  6. F3 — start recording, slowly rotate camera 360°
     System auto-tracks object and saves variants when appearance changes
  7. F4 — stop recording → enters test mode
  8. F4 — stop test mode

  Repeat with different state labels for object states.
  Use "Add Variants" to record more angles/zoom levels.

Hotkeys: F2=Freeze  F3=Record  F4=Stop  F5=Test  Esc=Cancel
""")
    parser.add_argument("--calibration_dir", type=str, default=DEFAULT_CAL_DIR)
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    args = parser.parse_args()

    app = QApplication(sys.argv)

    store = CalibrationStore(args.calibration_dir)
    capture = ScreenCapture(CAPTURE_FPS)
    capture.start()

    screen_geo = app.primaryScreen().geometry()
    overlay = OverlayWidget(screen_geo)
    bridge = HotkeyBridge()

    panel = ControlPanel(store, capture, overlay, bridge, args.padding)

    # pynput listeners → bridge signals
    def on_key(key):
        try:
            m = {
                keyboard.Key.f2: bridge.f2_pressed,
                keyboard.Key.f3: bridge.f3_pressed,
                keyboard.Key.f4: bridge.f4_pressed,
                keyboard.Key.f5: bridge.f5_pressed,
                keyboard.Key.esc: bridge.esc_pressed,
            }
            sig = m.get(key)
            if sig:
                sig.emit()
        except Exception:
            pass

    kb = keyboard.Listener(on_press=on_key)
    kb.daemon = True
    kb.start()

    panel.show()

    print("\n" + "=" * 60)
    print("OBJECT CALIBRATION TOOL v4.0")
    print("=" * 60)
    print(f"Directory: {args.calibration_dir}")
    print(f"Objects:   {len(store.list_objects())}")
    print("Hotkeys:   F2=Freeze  F3=Record  F4=Stop  F5=Test  Esc=Cancel")
    print("=" * 60)

    def cleanup():
        capture.stop()
        kb.stop()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()