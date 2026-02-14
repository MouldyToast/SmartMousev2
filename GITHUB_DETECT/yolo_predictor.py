"""
yolo_predictor.py — YOLO inference + background retraining for LabelImg integration

Used by the patched LabelImg to:
1. Auto-predict bounding boxes when opening an unlabeled image
2. Periodically retrain in the background as the user confirms annotations

Requirements:
    pip install ultralytics
"""

import os
import threading
from pathlib import Path


class YOLOPredictor:
    """
    Wraps ultralytics YOLO for predict-and-write + background retraining.

    Usage:
        predictor = YOLOPredictor("yolov8n.pt", "training/data.yaml")
        predictor.predict_and_write("frame_001.png", "frame_001.txt")
        predictor.retrain_background("training/data.yaml", epochs=10)
    """

    def __init__(self, weights="yolov8n.pt", data_yaml=None, conf=0.25):
        from ultralytics import YOLO
        self._YOLO = YOLO
        self.weights_path = weights
        self.model = YOLO(weights)
        self.data_yaml = data_yaml
        self.conf = conf
        self._training = False
        self._training_lock = threading.Lock()
        self._retrain_count = 0

    @property
    def is_training(self) -> bool:
        return self._training

    def predict_and_write(self, image_path: str, txt_path: str) -> int:
        """
        Run YOLO inference on image_path, write results as YOLO-format .txt.
        Returns number of boxes written.

        YOLO format: class_id x_center y_center width height (all normalized 0-1)
        """
        if self._training:
            return 0

        try:
            results = self.model(image_path, conf=self.conf, verbose=False)
        except Exception as e:
            print(f"  [YOLOPredictor] Inference failed: {e}")
            return 0

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            # Write empty file so LabelImg doesn't re-predict on revisit
            Path(txt_path).touch()
            return 0

        h, w = r.orig_shape
        count = 0
        with open(txt_path, 'w') as f:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1

        print(f"  [YOLOPredictor] {count} boxes → {os.path.basename(txt_path)}"
              f"  (conf>={self.conf})")
        return count

    def retrain_background(self, data_yaml=None, epochs=10, imgsz=640, batch=8):
        """
        Launch background retraining thread. Hot-swaps model weights on completion.
        No-op if already training.
        """
        with self._training_lock:
            if self._training:
                print("  [YOLOPredictor] Already training, skipping.")
                return
            self._training = True

        yaml_path = data_yaml or self.data_yaml
        if not yaml_path or not os.path.exists(yaml_path):
            print(f"  [YOLOPredictor] No data.yaml found at {yaml_path}")
            self._training = False
            return

        self._retrain_count += 1
        cycle = self._retrain_count

        def _train():
            try:
                print(f"\n  [YOLOPredictor] Retrain cycle {cycle} started "
                      f"(epochs={epochs}, imgsz={imgsz})")
                self.model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    verbose=False,
                    project="training/runs",
                    name=f"retrain_c{cycle}",
                    exist_ok=True,
                )
                # Hot-swap to new best weights
                best = Path(self.model.trainer.best)
                if best.exists():
                    self.model = self._YOLO(str(best))
                    self.weights_path = str(best)
                    print(f"  [YOLOPredictor] Retrain cycle {cycle} done → {best}")
                else:
                    print(f"  [YOLOPredictor] Retrain cycle {cycle} done, "
                          f"no best.pt found")
            except Exception as e:
                print(f"  [YOLOPredictor] Retrain failed: {e}")
            finally:
                self._training = False

        threading.Thread(target=_train, daemon=True).start()
