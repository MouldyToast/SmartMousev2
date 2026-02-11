"""
Visualize the path reconstruction formula against original recorded data.

Shows how the reconstruction algorithm transforms a recorded mouse path
when replayed at a different pixel distance than the original recording.

Three paths are overlaid:
  1. Original recording (raw offsets cumulated from origin)
  2. Reconstructed path at the SAME distance as the recording (sanity check)
  3. Reconstructed path at a USER-CHOSEN distance

Includes animated playback using the real recorded timing deltas so you
can see BOTH the shape and the speed/pacing of the movement.
"""

import json
import math
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation

# --- Configuration ---
MOUSEDATA_FILE = "dreambot_example/mousedata.json"
DISTANCE_THRESHOLDS = [12, 18, 26, 39, 58, 87, 130, 190, 260, 360, 500]
ANIMATION_FPS = 60  # Frame rate for playback


def load_mouse_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_distance_bucket(distance):
    for t in DISTANCE_THRESHOLDS:
        if distance <= t:
            return str(t)
    return str(DISTANCE_THRESHOLDS[-1])


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
    """Reconstruct the original recorded path by cumulating offsets from (0,0)."""
    cx, cy = 0.0, 0.0
    path = [(cx, cy)]
    for dx, dy in zip(x_offsets, y_offsets):
        cx += dx
        cy += dy
        path.append((cx, cy))
    return path


def compute_recording_distance(x_offsets, y_offsets):
    total_x = sum(x_offsets)
    total_y = sum(y_offsets)
    return math.hypot(total_x, total_y)


def compute_recording_angle(x_offsets, y_offsets):
    total_x = sum(x_offsets)
    total_y = sum(y_offsets)
    return math.degrees(math.atan2(total_y, total_x))


def build_cumulative_times_sec(time_deltas_ms):
    """Convert per-step ms deltas into cumulative seconds: [0, t0, t0+t1, ...]."""
    cum = [0.0]
    for dt in time_deltas_ms:
        cum.append(cum[-1] + dt / 1000.0)
    return cum


def interpolate_position(path_with_origin, cum_times, elapsed):
    """
    Given elapsed seconds, find the interpolated (x, y) position along the path.
    path_with_origin has N+1 points (origin + N steps).
    cum_times has N+1 entries (0 + N deltas cumulated).
    """
    if elapsed <= 0:
        return path_with_origin[0]
    if elapsed >= cum_times[-1]:
        return path_with_origin[-1]

    # Find the segment we're in
    for i in range(1, len(cum_times)):
        if elapsed <= cum_times[i]:
            t0 = cum_times[i - 1]
            t1 = cum_times[i]
            seg_duration = t1 - t0
            if seg_duration <= 0:
                frac = 1.0
            else:
                frac = (elapsed - t0) / seg_duration
            x0, y0 = path_with_origin[i - 1]
            x1, y1 = path_with_origin[i]
            return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)

    return path_with_origin[-1]


def trail_up_to(path_with_origin, cum_times, elapsed):
    """Return the portion of the path visited so far (for drawing the trail)."""
    xs, ys = [], []
    for i, (x, y) in enumerate(path_with_origin):
        if cum_times[i] <= elapsed:
            xs.append(x)
            ys.append(y)
        else:
            # Add the interpolated current position as the trail tip
            pos = interpolate_position(path_with_origin, cum_times, elapsed)
            xs.append(pos[0])
            ys.append(pos[1])
            break
    return xs, ys


class ReconstructionVisualizer:
    def __init__(self, mouse_data):
        self.mouse_data = mouse_data
        self.current_bucket = None
        self.current_direction = None
        self.current_path_index = 0

        # Animation state
        self.anim = None
        self.is_playing = False
        self.anim_elapsed = 0.0  # seconds into the playback

        # Build the figure
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("Mouse Path Reconstruction Visualizer", fontsize=14, fontweight="bold")

        # Main plot area
        self.ax = self.fig.add_axes([0.08, 0.28, 0.60, 0.65])

        # Info text area
        self.info_ax = self.fig.add_axes([0.72, 0.28, 0.26, 0.65])
        self.info_ax.axis("off")

        # --- Controls at bottom ---

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

        # Play button
        self.play_btn_ax = self.fig.add_axes([0.47, 0.14, 0.06, 0.05])
        self.play_btn = mwidgets.Button(self.play_btn_ax, "Play")
        self.play_btn.on_clicked(self._on_play_clicked)

        # Speed slider
        self.fig.text(0.54, 0.20, "Playback Speed:", fontsize=10, fontweight="bold")
        self.speed_slider_ax = self.fig.add_axes([0.54, 0.14, 0.15, 0.04])
        self.speed_slider = mwidgets.Slider(self.speed_slider_ax, "", 0.1, 5.0, valinit=1.0, valstep=0.1)

        # Replay distance slider
        self.fig.text(0.72, 0.20, "Replay Distance (px):", fontsize=10, fontweight="bold")
        self.slider_ax = self.fig.add_axes([0.72, 0.14, 0.25, 0.04])
        self.slider = mwidgets.Slider(self.slider_ax, "", 1, 600, valinit=100, valstep=1)
        self.slider.on_changed(self._on_slider_change)

        # Angle slider
        self.fig.text(0.54, 0.09, "Replay Angle (deg):", fontsize=10, fontweight="bold")
        self.angle_slider_ax = self.fig.add_axes([0.54, 0.03, 0.43, 0.04])
        self.angle_slider = mwidgets.Slider(self.angle_slider_ax, "", -180, 180, valinit=0, valstep=1)
        self.angle_slider.on_changed(self._on_slider_change)

        # Initialize
        self.current_bucket = "87"
        self.current_direction = "E"
        self._update_path_selector()
        self._update_static_plot()

    # --- Event handlers ---

    def _on_bucket_change(self, label):
        self._stop_animation()
        self.current_bucket = label
        self.current_path_index = 0
        self._update_path_selector()
        self._update_angle_from_direction()
        self._update_static_plot()

    def _on_direction_change(self, label):
        self._stop_animation()
        self.current_direction = label
        self.current_path_index = 0
        self._update_path_selector()
        self._update_angle_from_direction()
        self._update_static_plot()

    def _on_path_index_change(self, label):
        self._stop_animation()
        self.current_path_index = int(label)
        self._update_angle_from_direction()
        self._update_static_plot()

    def _on_slider_change(self, val):
        if not self.is_playing:
            self._update_static_plot()

    def _on_play_clicked(self, event):
        if self.is_playing:
            self._stop_animation()
        else:
            self._start_animation()

    # --- Data helpers ---

    def _update_angle_from_direction(self):
        path_data = self._get_current_path_data()
        if path_data:
            x_off, y_off, _ = path_data
            angle = compute_recording_angle(x_off, y_off)
            self.angle_slider.set_val(round(angle))

    def _update_path_selector(self):
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

    def _compute_all_paths(self):
        """Compute original + reconstructed paths and timing for current selection."""
        path_data = self._get_current_path_data()
        if path_data is None:
            return None

        x_offsets, y_offsets, time_deltas = path_data

        # Original path (with origin prepended)
        orig_path = original_path_from_offsets(x_offsets, y_offsets)

        # Cumulative time for original (same timing applies)
        cum_times = build_cumulative_times_sec(time_deltas)

        # Recording stats
        rec_dist = compute_recording_distance(x_offsets, y_offsets)
        rec_angle = compute_recording_angle(x_offsets, y_offsets)

        # Reconstructed at same distance
        target_same_x = sum(x_offsets)
        target_same_y = sum(y_offsets)
        same_path_raw = reconstruct_path(0, 0, target_same_x, target_same_y, x_offsets, y_offsets)
        same_path = [(0, 0)] + same_path_raw

        # Reconstructed at user-chosen distance/angle
        replay_dist = self.slider.val
        replay_angle_deg = self.angle_slider.val
        replay_angle_rad = math.radians(replay_angle_deg)
        replay_target_x = replay_dist * math.cos(replay_angle_rad)
        replay_target_y = replay_dist * math.sin(replay_angle_rad)

        replay_path_raw = reconstruct_path(0, 0, replay_target_x, replay_target_y, x_offsets, y_offsets)
        replay_path = [(0, 0)] + replay_path_raw

        return {
            "x_offsets": x_offsets,
            "y_offsets": y_offsets,
            "time_deltas": time_deltas,
            "orig_path": orig_path,
            "same_path": same_path,
            "replay_path": replay_path,
            "cum_times": cum_times,
            "rec_dist": rec_dist,
            "rec_angle": rec_angle,
            "replay_dist": replay_dist,
            "replay_angle_deg": replay_angle_deg,
            "replay_target_x": replay_target_x,
            "replay_target_y": replay_target_y,
            "total_time": cum_times[-1] if cum_times else 0,
        }

    # --- Static plot (no animation) ---

    def _update_static_plot(self):
        self.ax.clear()
        self.info_ax.clear()
        self.info_ax.axis("off")

        data = self._compute_all_paths()
        if data is None:
            self.ax.set_title("No data for this bucket/direction")
            self.fig.canvas.draw_idle()
            return

        self._draw_full_paths(data)
        self._draw_info_panel(data)
        self.fig.canvas.draw_idle()

    def _draw_full_paths(self, data):
        """Draw all three paths fully (non-animated view)."""
        orig = data["orig_path"]
        same = data["same_path"]
        replay = data["replay_path"]

        # Straight line reference
        self.ax.plot(
            [0, data["replay_target_x"]], [0, data["replay_target_y"]],
            '--', color='gray', alpha=0.4, linewidth=1, label='Straight line (replay target)'
        )

        # Original recording
        self.ax.plot([p[0] for p in orig], [p[1] for p in orig],
                     'o-', color='#2196F3', markersize=3, linewidth=2, alpha=0.8,
                     label=f'Original recording ({data["rec_dist"]:.1f}px)')
        self.ax.plot(orig[-1][0], orig[-1][1], 's', color='#2196F3', markersize=10, zorder=5)

        # Reconstructed at same distance
        self.ax.plot([p[0] for p in same], [p[1] for p in same],
                     'x--', color='#4CAF50', markersize=4, linewidth=1.5, alpha=0.7,
                     label=f'Reconstructed @ same dist ({data["rec_dist"]:.1f}px)')

        # Reconstructed at replay distance
        self.ax.plot([p[0] for p in replay], [p[1] for p in replay],
                     'o-', color='#F44336', markersize=3, linewidth=2, alpha=0.8,
                     label=f'Reconstructed @ {data["replay_dist"]:.0f}px, {data["replay_angle_deg"]:.0f}°')
        self.ax.plot(replay[-1][0], replay[-1][1], 's', color='#F44336', markersize=10, zorder=5)

        # Start + target markers
        self.ax.plot(0, 0, '*', color='black', markersize=15, zorder=10, label='Start (0,0)')
        self.ax.plot(data["replay_target_x"], data["replay_target_y"], 'x', color='#F44336',
                     markersize=12, markeredgewidth=3, zorder=10)

        self._style_main_axes(data)

    def _style_main_axes(self, data):
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

    def _draw_info_panel(self, data, elapsed=None):
        self.info_ax.clear()
        self.info_ax.axis("off")

        x_offsets = data["x_offsets"]
        y_offsets = data["y_offsets"]
        total_offset_x = sum(x_offsets)
        total_offset_y = sum(y_offsets)
        rec_dist = data["rec_dist"]
        rec_angle = data["rec_angle"]
        replay_dist = data["replay_dist"]
        replay_angle_deg = data["replay_angle_deg"]
        replay_target_x = data["replay_target_x"]
        replay_target_y = data["replay_target_y"]

        linear_dx = replay_target_x - total_offset_x
        linear_dy = replay_target_y - total_offset_y
        linear_dist = math.hypot(linear_dx, linear_dy)
        stretch_ratio = replay_dist / rec_dist if rec_dist > 0 else float('inf')

        num_steps = len(x_offsets)
        per_step_linear_mag = math.hypot(linear_dx / num_steps, linear_dy / num_steps) if num_steps else 0
        avg_offset_mag = sum(math.hypot(x, y) for x, y in zip(x_offsets, y_offsets)) / num_steps if num_steps else 0

        total_time = data["total_time"]

        lines = [
            "RECORDING INFO",
            f"  Steps: {num_steps}",
            f"  Net displacement: ({total_offset_x}, {total_offset_y})",
            f"  Recording distance: {rec_dist:.1f}px",
            f"  Recording angle: {rec_angle:.1f}°",
            f"  Avg offset/step: {avg_offset_mag:.2f}px",
            f"  Total duration: {total_time*1000:.0f}ms",
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
        ]

        if avg_offset_mag > 0:
            lines.append(f"  Linear/Offset ratio: {per_step_linear_mag / avg_offset_mag:.2f}x")

        if abs(stretch_ratio - 1.0) < 0.01:
            lines.append("  => Near-perfect replay")
        elif stretch_ratio > 1.5:
            lines.append("  => STRETCHED: wobble washed out")
        elif stretch_ratio < 0.5:
            lines.append("  => COMPRESSED: backbone reversed")
        else:
            lines.append("  => Moderate distortion")

        if elapsed is not None:
            speed = self.speed_slider.val
            lines += [
                "",
                "PLAYBACK",
                f"  Elapsed: {elapsed*1000:.0f}ms / {total_time*1000:.0f}ms",
                f"  Speed: {speed:.1f}x",
            ]

        self.info_ax.text(
            0.0, 1.0, "\n".join(lines),
            transform=self.info_ax.transAxes,
            fontsize=8.5, fontfamily="monospace",
            verticalalignment="top"
        )

    # --- Animation ---

    def _start_animation(self):
        self._stop_animation()

        self._anim_data = self._compute_all_paths()
        if self._anim_data is None:
            return

        self.is_playing = True
        self.anim_elapsed = 0.0
        self.play_btn.label.set_text("Stop")

        # Pre-compute axis limits from the static full plot so they don't jump
        self.ax.clear()
        self._draw_full_paths(self._anim_data)
        self._xlim = self.ax.get_xlim()
        self._ylim = self.ax.get_ylim()

        frame_interval_ms = 1000.0 / ANIMATION_FPS

        self.anim = FuncAnimation(
            self.fig,
            self._anim_frame,
            interval=frame_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self.fig.canvas.draw_idle()

    def _stop_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        self.is_playing = False
        self.play_btn.label.set_text("Play")
        self._update_static_plot()

    def _anim_frame(self, frame_num):
        data = self._anim_data
        if data is None:
            self._stop_animation()
            return

        speed = self.speed_slider.val
        dt = (1.0 / ANIMATION_FPS) * speed
        self.anim_elapsed += dt

        total_time = data["total_time"]
        elapsed = self.anim_elapsed

        # Stop when both paths are done
        if elapsed > total_time + 0.3:
            self._stop_animation()
            return

        cum_times = data["cum_times"]
        orig_path = data["orig_path"]
        replay_path = data["replay_path"]

        # Clamp elapsed for position lookups
        t = min(elapsed, total_time)

        # Get current cursor positions
        orig_pos = interpolate_position(orig_path, cum_times, t)
        replay_pos = interpolate_position(replay_path, cum_times, t)

        # Get trails drawn so far
        orig_trail_xs, orig_trail_ys = trail_up_to(orig_path, cum_times, t)
        replay_trail_xs, replay_trail_ys = trail_up_to(replay_path, cum_times, t)

        # Redraw
        self.ax.clear()

        # Ghost of full paths (faint)
        self.ax.plot([p[0] for p in orig_path], [p[1] for p in orig_path],
                     '-', color='#2196F3', alpha=0.15, linewidth=1)
        self.ax.plot([p[0] for p in replay_path], [p[1] for p in replay_path],
                     '-', color='#F44336', alpha=0.15, linewidth=1)

        # Straight line reference
        self.ax.plot(
            [0, data["replay_target_x"]], [0, data["replay_target_y"]],
            '--', color='gray', alpha=0.3, linewidth=1
        )

        # Animated trails (solid, growing)
        self.ax.plot(orig_trail_xs, orig_trail_ys,
                     '-', color='#2196F3', linewidth=2.5, alpha=0.8,
                     label=f'Original ({data["rec_dist"]:.0f}px)')
        self.ax.plot(replay_trail_xs, replay_trail_ys,
                     '-', color='#F44336', linewidth=2.5, alpha=0.8,
                     label=f'Reconstructed ({data["replay_dist"]:.0f}px)')

        # Cursor dots
        self.ax.plot(orig_pos[0], orig_pos[1], 'o', color='#2196F3',
                     markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=15)
        self.ax.plot(replay_pos[0], replay_pos[1], 'o', color='#F44336',
                     markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=15)

        # Start marker
        self.ax.plot(0, 0, '*', color='black', markersize=15, zorder=10)

        # Target markers
        self.ax.plot(orig_path[-1][0], orig_path[-1][1], 's',
                     color='#2196F3', markersize=8, alpha=0.5, zorder=5)
        self.ax.plot(data["replay_target_x"], data["replay_target_y"], 's',
                     color='#F44336', markersize=8, alpha=0.5, zorder=5)

        # Time indicator
        self.ax.set_title(
            f"Bucket: {self.current_bucket}  |  Dir: {self.current_direction}  |  "
            f"Path #{self.current_path_index}  |  "
            f"Time: {t*1000:.0f}ms / {total_time*1000:.0f}ms  ({speed:.1f}x)",
            fontsize=12
        )

        self.ax.set_xlim(self._xlim)
        self.ax.set_ylim(self._ylim)
        self.ax.legend(loc='best', fontsize=8)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)

        # Update info panel with elapsed time
        self._draw_info_panel(data, elapsed=t)

    def show(self):
        plt.show()


def main():
    data = load_mouse_data(MOUSEDATA_FILE)
    viz = ReconstructionVisualizer(data)
    viz.show()


if __name__ == "__main__":
    main()
