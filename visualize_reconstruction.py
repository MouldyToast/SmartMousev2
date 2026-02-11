"""
Visualize the path reconstruction formula against original recorded data.

Shows how the reconstruction algorithm transforms a recorded mouse path
when replayed at a different pixel distance than the original recording.

Three paths are overlaid:
  1. Original recording (raw offsets cumulated from origin)
  2. Reconstructed path at the SAME distance as the recording (sanity check)
  3. Reconstructed path at a USER-CHOSEN distance

This reveals how the linear backbone stretches/compresses the human wobble.
"""

import json
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

# --- Configuration ---
MOUSEDATA_FILE = "dreambot_example/mousedata.json"
DISTANCE_THRESHOLDS = [12, 18, 26, 39, 58, 87, 130, 190, 260, 360, 500]


def load_mouse_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_distance_bucket(distance):
    for t in DISTANCE_THRESHOLDS:
        if distance <= t:
            return str(t)
    return str(DISTANCE_THRESHOLDS[-1])


def cumulative_sum(offsets):
    """Return cumulative sums starting from 0."""
    result = []
    total = 0
    for v in offsets:
        total += v
        result.append(total)
    return result


def reconstruct_path(start_x, start_y, target_x, target_y, x_offsets, y_offsets):
    """
    Current reconstruction algorithm (identical in move.py and SmartMouseMultiDir.java).
    Returns list of (x, y) tuples.
    """
    path_len = min(len(x_offsets), len(y_offsets))
    if not path_len:
        return [(target_x, target_y)]

    total_offset_x = sum(x_offsets)
    total_offset_y = sum(y_offsets)

    # Linear adjustment = what the straight line needs to cover
    # AFTER subtracting what the offsets already contribute
    linear_dx = target_x - start_x - total_offset_x
    linear_dy = target_y - start_y - total_offset_y

    path = []
    for i in range(path_len):
        progress = (i + 1) / path_len

        current_linear_x = start_x + linear_dx * progress
        current_linear_y = start_y + linear_dy * progress

        cum_offset_x = sum(x_offsets[:i + 1])
        cum_offset_y = sum(y_offsets[:i + 1])

        new_x = current_linear_x + cum_offset_x
        new_y = current_linear_y + cum_offset_y
        path.append((new_x, new_y))

    return path


def original_path_from_offsets(x_offsets, y_offsets):
    """
    Reconstruct what the original recorded path looked like:
    just cumulate the offsets from (0, 0).
    """
    cx, cy = 0.0, 0.0
    path = [(cx, cy)]
    for dx, dy in zip(x_offsets, y_offsets):
        cx += dx
        cy += dy
        path.append((cx, cy))
    return path


def compute_recording_distance(x_offsets, y_offsets):
    """The actual pixel distance the recording covered (endpoint)."""
    total_x = sum(x_offsets)
    total_y = sum(y_offsets)
    return math.hypot(total_x, total_y)


def compute_recording_angle(x_offsets, y_offsets):
    """Angle of the recording's net displacement in degrees."""
    total_x = sum(x_offsets)
    total_y = sum(y_offsets)
    return math.degrees(math.atan2(total_y, total_x))


class ReconstructionVisualizer:
    def __init__(self, mouse_data):
        self.mouse_data = mouse_data
        self.current_bucket = None
        self.current_direction = None
        self.current_path_index = 0
        self.replay_distance = None

        # Build the figure
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("Mouse Path Reconstruction Visualizer", fontsize=14, fontweight="bold")

        # Main plot area
        self.ax = self.fig.add_axes([0.08, 0.28, 0.60, 0.65])

        # Info text area
        self.info_ax = self.fig.add_axes([0.72, 0.28, 0.26, 0.65])
        self.info_ax.axis("off")

        # Controls area at bottom
        # Bucket selector
        self.fig.text(0.08, 0.20, "Distance Bucket:", fontsize=10, fontweight="bold")
        bucket_labels = [str(t) for t in DISTANCE_THRESHOLDS]
        self.bucket_radio_ax = self.fig.add_axes([0.08, 0.02, 0.15, 0.17])
        self.bucket_radio = mwidgets.RadioButtons(self.bucket_radio_ax, bucket_labels, active=5)
        self.bucket_radio.on_clicked(self._on_bucket_change)

        # Direction selector
        self.fig.text(0.26, 0.20, "Direction:", fontsize=10, fontweight="bold")
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.dir_radio_ax = self.fig.add_axes([0.26, 0.02, 0.10, 0.17])
        self.dir_radio = mwidgets.RadioButtons(self.dir_radio_ax, directions, active=2)
        self.dir_radio.on_clicked(self._on_direction_change)

        # Path index selector
        self.fig.text(0.39, 0.20, "Path #:", fontsize=10, fontweight="bold")
        self.path_radio_ax = self.fig.add_axes([0.39, 0.02, 0.06, 0.17])
        self.path_radio = None

        # Replay distance slider
        self.fig.text(0.50, 0.20, "Replay Distance (px):", fontsize=10, fontweight="bold")
        self.slider_ax = self.fig.add_axes([0.50, 0.14, 0.40, 0.04])
        self.slider = mwidgets.Slider(self.slider_ax, "", 1, 600, valinit=100, valstep=1)
        self.slider.on_changed(self._on_slider_change)

        # Angle slider for replay direction
        self.fig.text(0.50, 0.09, "Replay Angle (deg):", fontsize=10, fontweight="bold")
        self.angle_slider_ax = self.fig.add_axes([0.50, 0.03, 0.40, 0.04])
        self.angle_slider = mwidgets.Slider(self.angle_slider_ax, "", -180, 180, valinit=0, valstep=1)
        self.angle_slider.on_changed(self._on_slider_change)

        # Initialize
        self.current_bucket = "87"
        self.current_direction = "E"
        self._update_path_selector()
        self._update_plot()

    def _on_bucket_change(self, label):
        self.current_bucket = label
        self.current_path_index = 0
        self._update_path_selector()
        self._update_angle_from_direction()
        self._update_plot()

    def _on_direction_change(self, label):
        self.current_direction = label
        self.current_path_index = 0
        self._update_path_selector()
        self._update_angle_from_direction()
        self._update_plot()

    def _on_path_index_change(self, label):
        self.current_path_index = int(label)
        self._update_angle_from_direction()
        self._update_plot()

    def _on_slider_change(self, val):
        self._update_plot()

    def _update_angle_from_direction(self):
        """Set the angle slider to match the recording's actual angle."""
        path_data = self._get_current_path_data()
        if path_data:
            x_off, y_off, _ = path_data
            angle = compute_recording_angle(x_off, y_off)
            self.angle_slider.set_val(round(angle))

    def _update_path_selector(self):
        """Rebuild the path index radio buttons for current bucket/direction."""
        paths = self._get_paths_for_selection()
        num_paths = len(paths) if paths else 0

        self.path_radio_ax.clear()
        self.path_radio_ax.axis("off")

        if num_paths > 0:
            labels = [str(i) for i in range(num_paths)]
            self.path_radio_ax.axis("on")
            self.path_radio = mwidgets.RadioButtons(self.path_radio_ax, labels, active=0)
            self.path_radio.on_clicked(self._on_path_index_change)
        else:
            self.path_radio = None

        # Update slider to match recording distance
        path_data = self._get_current_path_data()
        if path_data:
            x_off, y_off, _ = path_data
            rec_dist = compute_recording_distance(x_off, y_off)
            self.slider.set_val(max(1, round(rec_dist)))

    def _get_paths_for_selection(self):
        bucket_data = self.mouse_data.get(self.current_bucket, {})
        return bucket_data.get(self.current_direction, [])

    def _get_current_path_data(self):
        paths = self._get_paths_for_selection()
        if not paths or self.current_path_index >= len(paths):
            return None
        p = paths[self.current_path_index]
        if len(p) >= 3:
            return p[0], p[1], p[2]
        elif len(p) == 2:
            return p[0], p[1], [8.0] * len(p[0])
        return None

    def _update_plot(self):
        self.ax.clear()
        self.info_ax.clear()
        self.info_ax.axis("off")

        path_data = self._get_current_path_data()
        if path_data is None:
            self.ax.set_title("No data for this bucket/direction")
            self.fig.canvas.draw_idle()
            return

        x_offsets, y_offsets, time_deltas = path_data

        # --- Original recorded path (from origin) ---
        orig_path = original_path_from_offsets(x_offsets, y_offsets)
        orig_xs = [p[0] for p in orig_path]
        orig_ys = [p[1] for p in orig_path]
        rec_dist = compute_recording_distance(x_offsets, y_offsets)
        rec_angle = compute_recording_angle(x_offsets, y_offsets)

        # --- Reconstructed at SAME distance as recording (sanity) ---
        target_same_x = sum(x_offsets)
        target_same_y = sum(y_offsets)
        same_path = reconstruct_path(0, 0, target_same_x, target_same_y, x_offsets, y_offsets)
        same_xs = [0] + [p[0] for p in same_path]
        same_ys = [0] + [p[1] for p in same_path]

        # --- Reconstructed at USER-CHOSEN distance and angle ---
        replay_dist = self.slider.val
        replay_angle_deg = self.angle_slider.val
        replay_angle_rad = math.radians(replay_angle_deg)
        replay_target_x = replay_dist * math.cos(replay_angle_rad)
        replay_target_y = replay_dist * math.sin(replay_angle_rad)

        replay_path = reconstruct_path(0, 0, replay_target_x, replay_target_y, x_offsets, y_offsets)
        replay_xs = [0] + [p[0] for p in replay_path]
        replay_ys = [0] + [p[1] for p in replay_path]

        # --- Straight line for reference ---
        self.ax.plot(
            [0, replay_target_x], [0, replay_target_y],
            '--', color='gray', alpha=0.4, linewidth=1, label='Straight line (replay target)'
        )

        # --- Plot all three paths ---
        self.ax.plot(orig_xs, orig_ys, 'o-', color='#2196F3', markersize=3,
                     linewidth=2, alpha=0.8, label=f'Original recording ({rec_dist:.1f}px)')
        self.ax.plot(orig_xs[-1], orig_ys[-1], 's', color='#2196F3', markersize=10, zorder=5)

        self.ax.plot(same_xs, same_ys, 'x--', color='#4CAF50', markersize=4,
                     linewidth=1.5, alpha=0.7, label=f'Reconstructed @ same dist ({rec_dist:.1f}px)')

        self.ax.plot(replay_xs, replay_ys, 'o-', color='#F44336', markersize=3,
                     linewidth=2, alpha=0.8, label=f'Reconstructed @ {replay_dist:.0f}px, {replay_angle_deg:.0f}°')
        self.ax.plot(replay_xs[-1], replay_ys[-1], 's', color='#F44336', markersize=10, zorder=5)

        # Start point
        self.ax.plot(0, 0, '*', color='black', markersize=15, zorder=10, label='Start (0,0)')

        # Target markers
        self.ax.plot(replay_target_x, replay_target_y, 'x', color='#F44336',
                     markersize=12, markeredgewidth=3, zorder=10)

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        self.ax.set_title(
            f"Bucket: {self.current_bucket}  |  Direction: {self.current_direction}  |  Path #{self.current_path_index}",
            fontsize=12
        )
        self.ax.legend(loc='best', fontsize=8)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)

        # --- Info panel ---
        total_offset_x = sum(x_offsets)
        total_offset_y = sum(y_offsets)
        linear_dx = replay_target_x - total_offset_x
        linear_dy = replay_target_y - total_offset_y
        linear_dist = math.hypot(linear_dx, linear_dy)

        stretch_ratio = replay_dist / rec_dist if rec_dist > 0 else float('inf')

        # Per-step linear contribution
        num_steps = len(x_offsets)
        per_step_linear_x = linear_dx / num_steps if num_steps else 0
        per_step_linear_y = linear_dy / num_steps if num_steps else 0
        per_step_linear_mag = math.hypot(per_step_linear_x, per_step_linear_y)

        # Average offset magnitude per step
        avg_offset_mag = sum(math.hypot(x, y) for x, y in zip(x_offsets, y_offsets)) / num_steps if num_steps else 0

        info_lines = [
            "RECORDING INFO",
            f"  Steps: {num_steps}",
            f"  Net displacement: ({total_offset_x}, {total_offset_y})",
            f"  Recording distance: {rec_dist:.1f}px",
            f"  Recording angle: {rec_angle:.1f}°",
            f"  Avg offset/step: {avg_offset_mag:.2f}px",
            "",
            "RECONSTRUCTION",
            f"  Replay distance: {replay_dist:.0f}px",
            f"  Replay angle: {replay_angle_deg:.0f}°",
            f"  Target: ({replay_target_x:.1f}, {replay_target_y:.1f})",
            "",
            "LINEAR BACKBONE",
            f"  adjustedDx: {linear_dx:.1f}",
            f"  adjustedDy: {linear_dy:.1f}",
            f"  Backbone length: {linear_dist:.1f}px",
            f"  Per-step linear: {per_step_linear_mag:.2f}px",
            f"  Per-step offset:  {avg_offset_mag:.2f}px",
            "",
            "STRETCH ANALYSIS",
            f"  Distance ratio: {stretch_ratio:.2f}x",
            f"  Linear/Offset ratio: {per_step_linear_mag / avg_offset_mag:.2f}x" if avg_offset_mag > 0 else "  Linear/Offset ratio: N/A",
        ]

        if abs(stretch_ratio - 1.0) < 0.01:
            info_lines.append("  => Near-perfect replay")
        elif stretch_ratio > 1.5:
            info_lines.append("  => STRETCHED: wobble washed out")
        elif stretch_ratio < 0.5:
            info_lines.append("  => COMPRESSED: backbone reversed")
        else:
            info_lines.append("  => Moderate distortion")

        self.info_ax.text(
            0.0, 1.0, "\n".join(info_lines),
            transform=self.info_ax.transAxes,
            fontsize=8.5, fontfamily="monospace",
            verticalalignment="top"
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    data = load_mouse_data(MOUSEDATA_FILE)
    viz = ReconstructionVisualizer(data)
    viz.show()


if __name__ == "__main__":
    main()
