# YOLO Game Object Detection Pipeline — Build Plan v3

## Context

Building a Python machine vision system for game automation. Previous approach used OpenCV template matching + SAM2 tracking, which failed on 3D games due to camera rotation, anti-aliasing jitter, and color ambiguity (grey rocks vs green grass). YOLO is the standard escalation path.

**Hardware:** RTX 3060 Ti (8GB VRAM), Windows
**Existing code:** Screen capture via mss (30fps), PyQt5 overlay (transparent click-through), freeze-frame UI with rectangle selection, hotkey system (pynput)
**Goal:** Detect and locate game objects (NPCs, items, resources, UI elements) in real-time from screen capture, robust to camera rotation, zoom, lighting changes
**Tooling:** Fully local. No cloud services, no paid tiers, no accounts.

---

## Architecture Overview

```
Phase 0: Define class taxonomy (before any code or capture)
Phase 1: Capture screenshots while playing (automated, hotkey-triggered)
Phase 2: Bootstrap annotations (LabelImg + model-assisted)
Phase 3: Train YOLOv8-nano on local GPU
Phase 4: Active learning loop (detect → human review → retrain)
Phase 5: Runtime integration (YOLO detection → value reading post-process)
```

The key insight: **annotation is the bottleneck, not training.** Every design decision optimizes for less manual labeling.

### Single detection system

YOLO handles everything — 3D world objects AND UI elements. No parallel template matching system. UI elements are the easiest targets YOLO can learn: static position, consistent appearance, always the same size. They appear in nearly every captured frame, so YOLO gets hundreds of examples even from a small seed set.

Where detected objects contain **readable values** (health bar fill level, item stack counts, tooltip text, cooldown timers), a lightweight post-processing step runs OCR or pixel analysis on the YOLO-cropped region. YOLO answers "where is this thing", post-processing answers "what does it say."

```
Screen Capture
      │
      ▼
YOLO detection (all objects: world + UI)
      │
      ▼
ByteTrack (persistent IDs)
      │
      ├──▶ Overlay (draw all boxes)
      ├──▶ Automation (click targets, pathfinding)
      │
      └──▶ Value Reader (post-processing on YOLO crops)
              ├── OCR (text: tooltip content, item names, damage numbers)
              ├── Bar fill analysis (health/mana/stamina percentage)
              └── Digit recognition (stack counts, cooldown timers, currency)
```

---

## Phase 0: Class Taxonomy

Do this before writing any code or capturing any screenshots. Spend 20 minutes answering these questions — it saves hours of rework later.

### What to decide

Write down every object your automation needs to detect. For each one, answer:

1. **Does the automation need to distinguish subtypes?** If yes, separate classes. If not, merge.
   - "health_potion" and "mana_potion" → separate classes if bot uses them differently
   - "tree_oak" and "tree_pine" → single "tree" class if bot just avoids them

2. **Does this object have visually distinct states that require different behavior?**
   - "ore_node" vs "ore_node_depleted" → separate if bot should skip depleted nodes
   - "door_open" vs "door_closed" → separate if bot needs to know which to interact with

3. **Does this object carry readable values the automation needs?**
   - Health bar → YOLO detects it, value reader extracts fill percentage
   - Item stack → YOLO detects the item, value reader extracts the number
   - Tooltip → YOLO detects tooltip region, OCR extracts the text
   - Mark these in the taxonomy so the value reading pipeline knows which classes need post-processing

### Output

```yaml
# class_taxonomy.yaml — define BEFORE any annotation
classes:
  # 3D world objects
  0: health_potion
  1: mana_potion
  2: ore_node
  3: ore_node_depleted
  4: npc_merchant
  5: tree
  6: enemy
  7: loot_drop

  # UI elements (same YOLO model, same annotation pass)
  8: health_bar
  9: mana_bar
  10: minimap
  11: inventory_slot
  12: action_button
  13: tooltip
  14: item_stack          # item icon with visible count number
  15: damage_number       # floating combat text

# Classes that need value reading post-processing
value_reading:
  health_bar: bar_fill        # extract percentage from pixel fill
  mana_bar: bar_fill
  item_stack: digit_ocr       # extract stack count number
  damage_number: digit_ocr    # extract damage value
  tooltip: text_ocr           # extract full text content
  action_button: bar_fill     # cooldown timer (radial fill)

notes:
  - "Start with 8-12 classes. Add more in later training cycles."
  - "UI elements are trivial for YOLO — annotate them in the same pass as world objects."
  - "Every class with readable values gets a value_reading entry."
```

### Rules of thumb

- **Start with 8-12 classes.** More than 15 with <200 images means too few examples per class.
- **Start coarse, split later.** Train with "potion" first, split into "health_potion" and "mana_potion" after model proves it can find them.
- **UI classes are free annotations.** They appear in almost every frame. If you annotate them in your 30-40 seed images, YOLO sees them hundreds of times across the full training set. They'll converge to near-perfect accuracy in the first training cycle.
- **Value reading classes need extra thought.** The YOLO box must tightly crop the readable region. For a health bar, the box should cover exactly the bar fill area (not the whole HUD frame), because the value reader needs clean input.

---

## Phase 1: Screenshot Capture Tool

### What it does

Saves full-resolution screenshots while you play. No annotation yet — just raw frames with maximum visual variety.

### Requirements

- **Hotkey (F6):** saves current frame as PNG to `datasets/raw/`
- **Auto-capture mode:** toggled by hotkey, two speeds:
  - **Slow mode (F7 toggle):** one frame every 3 seconds — general gameplay
  - **Fast mode (F8 toggle):** one frame every 0.5 seconds — deliberate camera rotation around a specific object
- **Deduplication:** dHash (8×9 resize → horizontal gradient → 64-bit hash, reject if hamming distance < 5)
- **Filename format:** `frame_20250214_153022_001.png` (timestamp + sequence)
- **Overlay counter:** display running frame count
- **Target:** 150-250 frames covering different camera angles, zoom levels, times of day, object states

### Implementation notes

- Reuse existing `ScreenCapture` class (mss-based, thread-safe)
- dHash needs only `cv2.resize` + numpy. No `opencv-contrib-python` dependency:
  ```python
  def dhash(image, hash_size=8):
      resized = cv2.resize(image, (hash_size + 1, hash_size))
      gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
      diff = gray[:, 1:] > gray[:, :-1]
      return sum(2**i for i, v in enumerate(diff.flatten()) if v)

  def hamming(h1, h2):
      return bin(h1 ^ h2).count('1')
  ```
- Store at game resolution (1920×1080). YOLO resizes internally during training.

### Capture strategy

**Positive examples (~80% of frames):**
1. Walk around normally — objects at various distances
2. Rotate camera 360° around key objects (fast auto-capture)
3. Zoom in and out
4. Different game states: day/night, weather, combat, menus open
5. Frames with MULTIPLE target objects visible
6. Same class across different instances (different merchants, different trees)
7. **UI-heavy frames:** inventory open, tooltips visible, health bars at various fill levels, combat numbers floating — these train UI classes for free

**Negative examples (~20% of frames):**
8. Empty terrain, sky, walls — no target objects
9. Loading screens
10. Confusing backgrounds (rocks that look like ore, bushes that look like NPCs)

Negatives are critical with small datasets. False positives are usually the bigger early problem. Negatives require zero annotation — just capture them and include them with empty label files.

---

## Phase 2: Annotation (Fully Local)

### Tool: LabelImg

```bash
pip install labelImg
labelImg datasets/raw/ classes.txt
```

**Key shortcuts:** `W` = draw box, `D` = next image, `A` = previous, `Ctrl+S` = save, `Del` = delete box.

Set save format to YOLO (menu → Change Save Format → YOLO). Creates one `.txt` per image: `class x_center y_center width height` (normalized 0-1).

### Step 2a: Manual seed annotations (30-40 images)

**Selection strategy:**
- Pick frames where objects are clearly visible, unoccluded, well-lit
- At least 3-5 examples of every class (including UI classes)
- 2-3 negative frames (no objects → save empty `.txt` files)
- Include frames with multiple objects for spatial context
- Variety: different zoom levels, angles, backgrounds

**UI annotation tip:** When annotating UI elements, draw tight boxes. For health bars, box the fill region specifically, not the decorative frame around it. For item stacks with numbers, box the entire item+number as one unit — the value reader will crop tighter later. For tooltips, box the whole tooltip panel.

**Time estimate:** 45-90 minutes for 30-40 frames.

### Step 2b: Organize into YOLO dataset format

```
datasets/game_objects/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`dataset_manager.py` handles 80/20 split and incremental additions.

### Step 2c: Train initial model on seed set

```bash
yolo detect train \
  model=yolov8n.pt \
  data=training/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=4 \
  freeze=10 \
  patience=30 \
  project=runs/train \
  name=game_v1
```

- `freeze=10` — protect COCO backbone features while head learns your classes. Remove once dataset > 200 images.
- `patience=30` — early stopping. Default 50 is too long for tiny datasets.
- **Training time:** ~15-30 minutes on 3060 Ti.

### Step 2d: Model-assisted annotation

```
1. Run trained model on unlabeled screenshots
2. Model proposes bounding boxes + class labels + confidence
3. Save proposals as pre-filled YOLO .txt files
4. Save visualizations (frame + drawn boxes) for quick human triage
5. Human reviews in LabelImg: correct mistakes, add missed objects
6. Add to training set, retrain
7. Repeat until mAP50 > 0.7
```

**Confidence thresholds for proposals:**
- **≥ 0.85:** auto-accept (skip review)
- **0.4 – 0.85:** needs review (pre-filled labels, human corrects)
- **< 0.4:** discard proposal

Each review+retrain cycle improves mAP by 5-15%.

### Step 2e: Hard example mining

- Log frames where confidence is 0.3-0.7 (uncertain)
- Log false positives (high confidence, wrong class or no object)
- **10 hard examples teach more than 100 easy ones**

---

## Phase 3: Training

### Model: YOLOv8-nano

| Model | Params | Speed (3060 Ti) | Use case |
|-------|--------|-----------------|----------|
| **YOLOv8n** | 3.2M | **~2ms** | **Default** |
| YOLOv8s | 11.2M | ~4ms | If nano isn't enough |

**Why v8 over v5:** anchor-free head (better with small datasets), native ByteTrack, active development, `ultralytics` package IS v8.

### Training configs

```bash
# Small dataset (<100 images)
yolo detect train \
  model=yolov8n.pt \
  data=training/data.yaml \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 \
  freeze=10 patience=30 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.3 \
  project=runs/train name=game_v1

# Larger dataset (200+ images)
yolo detect train \
  model=yolov8n.pt \
  data=training/data.yaml \
  epochs=150 imgsz=640 batch=16 device=0 workers=4 \
  patience=40 \
  project=runs/train name=game_v2
  # freeze removed — enough data for full fine-tuning

# If small objects are missed
# bump resolution, drop batch size to fit VRAM
yolo detect train \
  model=yolov8n.pt \
  data=training/data.yaml \
  epochs=150 imgsz=1280 batch=8 device=0 workers=4 \
  patience=40 \
  project=runs/train name=game_v2_highres
```

### Parameter reasoning

| Parameter | Value | Why |
|-----------|-------|-----|
| `freeze=10` | Small datasets only | Protects COCO features while head trains |
| `patience=30` | Early stopping | Prevents overfitting on small datasets |
| `hsv_h=0.015` | Reduced hue jitter | Games have controlled color palettes |
| `imgsz=640` | Default | Bump to 1280 if small objects missed |

### Built-in augmentation (replaces Roboflow)

YOLO applies automatically during training — zero configuration:
- Mosaic (4 images → 1), mixup, HSV jitter, flips, scale, translate, copy-paste

### When to stop

- mAP50 plateaus for 20+ epochs → done
- Train loss drops but val mAP stalls → overfitting, need more data
- Typical: 50-80 epochs small datasets, 100-150 larger ones

---

## Phase 4: Active Learning Loop

### Architecture

```
Live Game → YOLO detect → Overlay + Automation
                │
                └──▶ Auto-save interesting frames
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      uncertain/    novel/     user_flagged/
      (0.3-0.7)    (new scene)   (F9 hotkey)
           │            │            │
           └────────────┼────────────┘
                        ▼
                  Batch review (LabelImg)
                        ▼
                     Retrain
```

### Auto-save triggers

| Trigger | Condition | Why |
|---------|-----------|-----|
| Uncertain detection | Confidence 0.3-0.7 | What model struggles with |
| Novel scene | dHash distance > 15 from last 10 saves | New areas, lighting |
| User hotkey (F9) | Manual press on miss/false positive | Human catches what metrics can't |
| Periodic | Every 30 seconds | Representative distribution |

### Deduplication: frame-hash-based, not time-based

Before saving, compute dHash. If hamming distance < 8 from any frame saved in the last 30 seconds, skip. This deduplicates similar views while keeping genuinely different scenes regardless of timing.

### Review workflow

1. Accumulate 50+ frames
2. Run `propose_labels.py` — pre-fill YOLO annotations from current model
3. Review in LabelImg — correct, add missed, delete false
4. Run `dataset_manager.py` — incorporate into training set
5. Retrain

---

## Phase 5: Runtime Integration

### YOLO detection loop

```python
from ultralytics import YOLO

model = YOLO("runs/train/game_v1/weights/best.pt")

# Single frame inference
results = model(frame_bgr, conf=0.5, iou=0.45, device=0)

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
```

### Built-in tracking

```python
results = model.track(frame_bgr, persist=True, tracker="bytetrack.yaml")

for box in results[0].boxes:
    track_id = int(box.id[0]) if box.id is not None else -1
    # Persistent ID across frames. No custom tracker needed.
```

### Per-class confidence thresholds

```python
CLASS_THRESHOLDS = {
    "health_potion": 0.6,
    "ore_node": 0.5,
    "npc_merchant": 0.7,
    "health_bar": 0.8,      # UI elements should be very high confidence
    "tooltip": 0.7,
    "item_stack": 0.75,
}

for box in results[0].boxes:
    class_name = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    threshold = CLASS_THRESHOLDS.get(class_name, 0.5)
    if conf >= threshold:
        # accept detection
```

### Click targeting

```python
# (0.5, 0.5) = center, (0.5, 0.9) = bottom center
CLICK_OFFSETS = {
    "health_potion": (0.5, 0.5),
    "tree": (0.5, 0.85),       # click base, not canopy
    "npc_merchant": (0.5, 0.7), # click body, not name tag
}
```

### Value Reading Pipeline (post-processing on YOLO crops)

YOLO tells you WHERE an object is. The value reader tells you WHAT IT SAYS.

This runs only on classes marked in `value_reading` in the taxonomy. It operates on the cropped region from YOLO's bounding box — a small image, typically 50-200px wide. Because the crop is small and pre-located, every method runs in <5ms per crop.

#### Three value reading methods

**1. Bar fill analysis** — health bars, mana bars, cooldown timers

No OCR needed. Measure how far a colored region extends within the bounding box.

```python
def read_bar_fill(crop_bgr, bar_color_range_hsv, direction="horizontal"):
    """
    Measure fill percentage of a colored bar.

    Args:
        crop_bgr: YOLO-cropped region containing the bar
        bar_color_range_hsv: ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
        direction: "horizontal" (left-to-right fill) or "vertical" (bottom-to-top)

    Returns:
        float 0.0-1.0 representing fill percentage
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, bar_color_range_hsv[0], bar_color_range_hsv[1])

    if direction == "horizontal":
        # Project mask onto x-axis, find rightmost filled column
        col_sums = mask.sum(axis=0)
        filled_cols = np.where(col_sums > 0)[0]
        if len(filled_cols) == 0:
            return 0.0
        return (filled_cols[-1] + 1) / mask.shape[1]
    else:
        # Vertical: project onto y-axis, find topmost filled row
        row_sums = mask.sum(axis=1)
        filled_rows = np.where(row_sums > 0)[0]
        if len(filled_rows) == 0:
            return 0.0
        return 1.0 - (filled_rows[0] / mask.shape[0])
```

**Calibration:** each bar class needs its HSV color range and fill direction defined once. This is a one-time manual step — eyedropper the bar color in an image editor, convert to HSV, set a range. Store in config:

```yaml
bar_configs:
  health_bar:
    color_hsv_low: [0, 150, 100]
    color_hsv_high: [10, 255, 255]
    direction: horizontal
  mana_bar:
    color_hsv_low: [100, 150, 100]
    color_hsv_high: [130, 255, 255]
    direction: horizontal
```

**Speed:** <1ms per bar. HSV conversion + inRange + column sum on a 200×30px crop is trivial.

**2. Digit OCR** — stack counts, damage numbers, currency, cooldown seconds

For reading numbers rendered in game fonts. Three options, in order of preference:

| Method | Speed (per crop) | Accuracy on game digits | Setup |
|--------|-----------------|------------------------|-------|
| **Windows OCR** (winocr) | 5-15ms | Good on standard fonts | `pip install winocr`, Windows only |
| **Tesseract** (PSM 7, digit whitelist) | 15-30ms | Good with preprocessing | System install + `pip install pytesseract` |
| **EasyOCR** (GPU) | 3-5ms GPU | Decent, loses case | `pip install easyocr` (~100MB download) |

**Recommended: Tesseract with digit whitelist** for broadest compatibility.

```python
def read_digits(crop_bgr):
    """
    Extract numeric value from a YOLO-cropped region.
    Preprocessing pipeline optimized for game-rendered numbers.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small crops — Tesseract needs ~30px character height
    if gray.shape[0] < 40:
        scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Binarize (handles both light-on-dark and dark-on-light text)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If text is light on dark background, invert so Tesseract sees dark-on-white
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Add white border (Tesseract struggles with text touching edges)
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT, value=255)

    # OCR with digit-only whitelist
    text = pytesseract.image_to_string(
        binary,
        config='--psm 7 -c tessedit_char_whitelist=0123456789'
    ).strip()

    try:
        return int(text)
    except ValueError:
        return None
```

**Key preprocessing steps (each one matters for game text):**
- Upscale 2-3× with INTER_CUBIC — game digits are often tiny (12-20px)
- Otsu binarization — adaptive threshold handles any background
- Auto-invert — detects whether text is light-on-dark and normalizes
- White border padding — Tesseract fails when characters touch image edges
- PSM 7 — single text line mode, much faster and more accurate than full page
- Digit whitelist — eliminates letter confusion (O vs 0, l vs 1)

**3. Text OCR** — tooltips, item names, chat messages

Same engine as digit OCR but without the whitelist, and with additional preprocessing for game text styling (outlines, shadows, colored text on gradient backgrounds).

```python
def read_text(crop_bgr):
    """
    Extract text content from a YOLO-cropped region (tooltips, names, etc).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if gray.shape[0] < 40:
        scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE for semi-transparent or faded text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Morphological closing to remove text outlines/shadows common in games
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT, value=255)

    text = pytesseract.image_to_string(binary, config='--psm 7').strip()
    return text if text else None
```

#### Value reading integration

The value reader runs AFTER YOLO detection, ONLY on classes that have a `value_reading` entry in the taxonomy. It adds a `value` field to each detection result.

```python
VALUE_READERS = {
    "bar_fill": read_bar_fill,
    "digit_ocr": read_digits,
    "text_ocr": read_text,
}

# After YOLO inference
for box in results[0].boxes:
    class_name = model.names[int(box.cls[0])]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Check if this class needs value reading
    reader_type = VALUE_READING_CONFIG.get(class_name)
    if reader_type:
        crop = frame_bgr[y1:y2, x1:x2]
        reader_fn = VALUE_READERS[reader_type]

        if reader_type == "bar_fill":
            value = reader_fn(crop, BAR_CONFIGS[class_name])
        else:
            value = reader_fn(crop)

        # value is now: 0.73 (bar fill), 42 (stack count), "Iron Sword" (tooltip text)
```

#### Value reading performance budget

| Method | Time per crop | When it runs |
|--------|--------------|--------------|
| Bar fill (HSV + column sum) | <1ms | Every frame (cheap enough) |
| Digit OCR (Tesseract, PSM 7) | 15-30ms | Only when crop changes (dHash cache) |
| Text OCR (Tesseract, PSM 7) | 15-30ms | Only when crop changes (dHash cache) |

**OCR caching:** compute dHash of each crop. If identical to last frame's crop for the same tracked object, return cached text. Tooltips and stack counts don't change every frame — caching reduces OCR calls by 90%+. Bar fill analysis is cheap enough to skip caching.

```python
class ValueCache:
    """Cache OCR results by crop hash. Avoid re-reading unchanged values."""

    def __init__(self):
        self._cache = {}  # track_id → (dhash, value)

    def get_or_read(self, track_id, crop_bgr, reader_fn, *args):
        h = dhash(crop_bgr)
        cached = self._cache.get(track_id)
        if cached and hamming(cached[0], h) < 5:
            return cached[1]
        value = reader_fn(crop_bgr, *args)
        self._cache[track_id] = (h, value)
        return value
```

### Complete runtime performance budget

| Stage | Time | Notes |
|-------|------|-------|
| Screen capture (mss/BetterCam) | 4-8ms | DXGI duplication |
| YOLO inference (v8n, 640px, FP16) | 2-4ms | 3060 Ti |
| ByteTrack | <1ms | Built into ultralytics |
| Bar fill reading (2-3 bars) | <1ms | HSV threshold, every frame |
| Digit/text OCR (1-2 crops, cached) | ~2ms amortized | 15-30ms actual, runs ~10% of frames |
| Overlay update | 2-3ms | PyQt5 QPainter |
| **Total** | **~12-20ms** | **50-80 FPS** |

---

## File Structure

```
project/
├── capture/
│   └── capture_tool.py          # Screenshot capture + dedup + hotkeys
├── datasets/
│   ├── raw/                     # Unlabeled screenshots
│   ├── game_objects/            # YOLO-format training dataset
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── review/                  # Frames pending human review
│       ├── uncertain/
│       ├── novel/
│       ├── user_flagged/
│       └── periodic/
├── training/
│   ├── data.yaml                # YOLO dataset config
│   ├── class_taxonomy.yaml      # Class definitions + value reading config
│   └── runs/                    # Training outputs
├── inference/
│   ├── detector.py              # YOLO detection + ByteTrack
│   ├── value_reader.py          # Bar fill + digit OCR + text OCR + caching
│   ├── active_learner.py        # Auto-save interesting frames
│   └── overlay.py               # PyQt5 overlay (reuse existing)
├── tools/
│   ├── dataset_manager.py       # Train/val split, incremental additions
│   ├── propose_labels.py        # Model → pre-filled YOLO .txt files
│   ├── visualize_predictions.py # Draw predictions for human triage
│   └── hard_example_miner.py    # Find uncertain detections
└── classes.txt                  # Class names for LabelImg
```

---

## Implementation Order

### Sprint 0: Planning (1 hour)

1. Define class taxonomy with value reading annotations
2. Write `classes.txt` for LabelImg
3. Write `data.yaml`
4. Define bar color HSV ranges for any bar-fill classes

### Sprint 1: Capture + Manual Seed (Day 1)

1. Build `capture_tool.py` — hotkey screenshot saver with dHash dedup
2. Capture 150-200 frames (20% negatives, include UI-heavy frames)
3. Install LabelImg: `pip install labelImg`
4. Annotate 30-40 seed images (world objects AND UI elements in same pass)
5. Build `dataset_manager.py` — 80/20 train/val split

### Sprint 2: First Training + Bootstrap (Day 1-2)

1. Train YOLOv8n on seed set
2. Build `propose_labels.py` + `visualize_predictions.py`
3. Review + correct 50 more frames in LabelImg
4. Retrain. Repeat until mAP50 > 0.7

### Sprint 3: Runtime Detection (Day 2-3)

1. Build `detector.py` — YOLO + ByteTrack wrapper
2. Integrate with PyQt5 overlay
3. Per-class confidence thresholds + click offsets

### Sprint 4: Value Reading (Day 3-4)

1. Build `value_reader.py` — bar fill + digit OCR + text OCR
2. Implement `ValueCache` (dHash-based OCR caching)
3. Wire into detection loop (post-process YOLO crops)
4. Calibrate bar HSV ranges for each bar class
5. Test OCR preprocessing on actual game crops, tune thresholds

### Sprint 5: Active Learning (Day 4-5)

1. Build `active_learner.py` — auto-save uncertain/novel/flagged frames
2. dHash-based dedup for auto-saves
3. Build `hard_example_miner.py`

### Sprint 6: Data Flywheel (Ongoing)

1. Play normally with active learner running
2. Batch review when 50+ frames accumulate
3. Retrain when 50+ reviewed frames added
4. Add new classes as needed
5. Tune per-class thresholds based on real performance
6. If small objects missed: retrain with `imgsz=1280 batch=8`

---

## Key Dependencies

```
pip install ultralytics        # YOLOv8 + ByteTrack
pip install labelImg            # Local annotation tool
pip install opencv-python       # Core CV
pip install PyQt5               # Overlay
pip install mss                 # Screen capture
pip install pynput              # Hotkeys
pip install pytesseract         # OCR (also install Tesseract binary)
```

Optional:
```
pip install bettercam           # Faster capture (240fps DXGI)
pip install winocr              # Windows OCR API (faster than Tesseract)
pip install easyocr             # GPU OCR alternative
```

Tesseract binary: download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.

---

## What to Reuse from Existing Code

| Component | Reuse? | Notes |
|-----------|--------|-------|
| `ScreenCapture` class (mss) | **Yes** | Core capture loop, thread-safe |
| `OverlayWidget` (PyQt5) | **Yes** | Draw YOLO boxes + values |
| Hotkey system (pynput) | **Yes** | Add F6-F9 for capture + active learning |
| Freeze frame / ReviewWindow | **Partially** | Useful for manual annotation assist |
| CalibrationStore | **No** | Replaced by YOLO dataset format |
| Template matching | **No** | Replaced by YOLO |
| SAM2 / CSRT tracking | **No** | Replaced by ByteTrack |
| Convex hull / boundary tracing | **No** | YOLO gives bounding boxes |
