"""
A script for simulating human-like mouse movements using pre-recorded path and timing data.

This script uses data from a JSON file, which contains mouse movement offsets AND timing deltas
categorized by distance and direction, to generate realistic paths with authentic timing patterns.
It includes a visualization overlay to see the mouse's path in real-time.

NEW FEATURES:
- Uses actual recorded timing data instead of synthetic easing functions
- Replays human movement patterns with exact recorded timing variance
- Backward compatible with old data format (falls back to synthetic timing)
- Enhanced visualization showing movement progression
"""

import json
import math
import random
import time
import tkinter as tk
from typing import List, Optional, Tuple

from pynput.mouse import Controller

# --- Configuration Constants ---
MOUSE_DATA_FILE = "mousedata.json"
NUM_RANDOM_POINTS = 10 # Number of random points to generate
VISUALIZATION_DOT_SIZE = 6 # Size of the visualization dot
RANDOM_POINT_PADDING = 300 # Padding from screen edges for random point generation
ENABLE_MOUSE_TRAIL = False # Toggle to show mouse trail
USE_RANDOM_POINTS = False  # Toggle to use randomly generated points instead of predefined path

# Distance thresholds for bucketing movement paths.
# Each threshold defines the upper bound for a distance category.
DISTANCE_THRESHOLDS = [12, 18, 26, 39, 58, 87, 130, 190, 260, 360, 500]

# Note: Synthetic timing profiles removed - we now use recorded timing data

# --- Helper Functions ---
def get_path_distance_bucket(distance: float) -> str:
    """Finds the appropriate distance bucket from DISTANCE_THRESHOLDS."""
    for threshold in DISTANCE_THRESHOLDS:
        if distance <= threshold:
            return str(threshold)
    return str(DISTANCE_THRESHOLDS[-1])

def angle_to_direction(angle_deg: float) -> str:
    """Converts an angle in degrees to one of 8 cardinal directions."""
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    # Shift angle so E is at 0, then map to an index 0-7
    # Each slice is 45 degrees wide. 22.5 is the offset.
    index = round(((angle_deg % 360) + 22.5) / 45) % 8
    return directions[index]

# --- Core Classes ---
class VisualizationOverlay:
    """A transparent, full-screen overlay to visualize mouse movements."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.5, '-topmost', True)
        self.root.overrideredirect(True) # No window decorations

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='black')
        self.canvas.pack(fill='both', expand=True)
        self.root.bind('<Escape>', lambda e: self.root.destroy())

    def draw_dot(self, x: int, y: int, color: str = 'red'):
        """Draws a dot at the given coordinates."""
        r = VISUALIZATION_DOT_SIZE / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline='')
        self.root.update()

    def run(self):
        """Starts the Tkinter main loop."""
        print("Visualization running. Press ESC to close.")
        self.root.mainloop()

class HumanMouseMover:
    """Handles the logic for moving the mouse in a human-like fashion."""

    def __init__(self, mouse_data_file: str):
        self.mouse = Controller()
        try:
            with open(mouse_data_file, "r") as f:
                self.mouse_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading mouse data from {mouse_data_file}: {e}")
            raise

    def move_to(self, target_x: int, target_y: int, overlay: Optional[VisualizationOverlay] = None):
        """Moves the mouse from its current position to the target coordinates."""
        start_x, start_y = self.mouse.position
        distance = math.hypot(target_x - start_x, target_y - start_y)

        if distance < 1: # No movement needed
            return

        print(f"Moving from ({start_x}, {start_y}) to ({target_x}, {target_y}) | Distance: {distance:.1f}")

        # 1. Select a path profile based on distance and direction
        angle = math.degrees(math.atan2(target_y - start_y, target_x - start_x))
        direction = angle_to_direction(angle)
        path_profile = self._pick_path_profile(distance, direction)

        if not path_profile:
            print(f"No path profile found for dist={distance:.1f}, dir={direction}. Jumping.")
            self.mouse.position = (target_x, target_y)
            if overlay:
                overlay.draw_dot(target_x, target_y, 'yellow') # Yellow for jumps
            return

        # 2. Reconstruct the full path from the offset profile
        x_offsets, y_offsets, time_deltas = path_profile
        path = self._reconstruct_path(start_x, start_y, target_x, target_y, x_offsets, y_offsets)
        
        # Debug timing information
        max_timing = max(time_deltas) if time_deltas else 0
        total_timing = sum(time_deltas) if time_deltas else 0
        print(f"  └─ Path selected. Direction: {direction}, Steps: {len(path)}")
        print(f"     Timing: {total_timing:.3f}s total, max step: {max_timing:.3f}s")

        # 3. Execute the movement using recorded timing
        for i, (px, py) in enumerate(path):
            self.mouse.position = (px, py)
            if overlay and ENABLE_MOUSE_TRAIL:
                overlay.draw_dot(int(px), int(py), 'lime')

            # Use exact recorded timing without variance
            if i < len(time_deltas):
                sleep_duration = max(0.001, time_deltas[i])  # Enforce a minimum of 1ms

                # Debug output for significant pauses
                if sleep_duration > 0.1:  # More than 100ms
                    print(f"    Pause detected: {sleep_duration:.3f}s at step {i+1}/{len(path)}")
                elif sleep_duration > 0.05:  # More than 50ms
                    print(f"    Slow step: {sleep_duration:.3f}s at step {i+1}/{len(path)}")

                time.sleep(sleep_duration)

        # Ensure the final position is exact
        self.mouse.position = (target_x, target_y)
        if overlay:
            overlay.draw_dot(target_x, target_y, 'red')

    def _pick_path_profile(self, distance: float, direction: str) -> Optional[Tuple[List[int], List[int], List[float]]]:
        """Picks a random path offset profile from the loaded data."""
        dist_bucket = get_path_distance_bucket(distance)
        paths_in_bucket = self.mouse_data.get(dist_bucket, {})
        available_paths = paths_in_bucket.get(direction, [])
        if not available_paths:
            return None
        
        selected_path = random.choice(available_paths)
        
        # Handle backward compatibility
        if len(selected_path) == 2:
            # Old format: only position offsets
            x_offsets, y_offsets = selected_path
            # Generate synthetic timing (uniform intervals)
            time_deltas = [0.008] * len(x_offsets)  # 8ms default
            return x_offsets, y_offsets, time_deltas
        elif len(selected_path) == 3:
            # New format: includes timing data
            x_offsets, y_offsets, time_deltas = selected_path
            # Convert time deltas from milliseconds to seconds
            time_deltas_sec = [dt / 1000.0 for dt in time_deltas]
            return x_offsets, y_offsets, time_deltas_sec
        else:
            return None

    @staticmethod
    def _reconstruct_path(start_x, start_y, target_x, target_y, x_offsets, y_offsets) -> List[Tuple[float, float]]:
        """
        Builds the absolute coordinate path from start to target using offset data.
        The algorithm ensures the path ends exactly at the target by distributing
        the remaining straight-line distance across all points after accounting for
        the recorded offsets.
        """
        path_len = min(len(x_offsets), len(y_offsets))
        if not path_len:
            return [(target_x, target_y)]

        total_offset_x = sum(x_offsets)
        total_offset_y = sum(y_offsets)

        # The straight-line distance that needs to be covered
        # in addition to the recorded offsets.
        linear_dx = target_x - start_x - total_offset_x
        linear_dy = target_y - start_y - total_offset_y

        path = []
        for i in range(path_len):
            progress = (i + 1) / path_len
            
            # Interpolate the linear part of the movement
            current_linear_x = start_x + linear_dx * progress
            current_linear_y = start_y + linear_dy * progress

            # Add the cumulative human-like offsets
            cumulative_offset_x = sum(x_offsets[:i + 1])
            cumulative_offset_y = sum(y_offsets[:i + 1])

            new_x = current_linear_x + cumulative_offset_x
            new_y = current_linear_y + cumulative_offset_y
            path.append((new_x, new_y))

        return path




def generate_random_points(num_points: int, screen_width: int, screen_height: int) -> List[Tuple[int, int]]:
    """Generates a list of random points within screen bounds, avoiding edges."""
    def compute_bounds(size: int, padding: int) -> Tuple[int, int]:
        left = padding
        right = size - padding
        if right < left:
            # Reduce padding to fit within the size; fallback to centered bounds if still invalid
            adjusted = max(0, min(padding, max(0, (size - 1) // 2)))
            left = adjusted
            right = size - adjusted
            if right < left:
                # As a last resort, pin to center
                center = max(0, size // 2)
                left = center
                right = center
        return left, right

    min_x, max_x = compute_bounds(screen_width, RANDOM_POINT_PADDING)
    min_y, max_y = compute_bounds(screen_height, RANDOM_POINT_PADDING)

    return [
        (
            random.randint(min_x, max_x),
            random.randint(min_y, max_y)
        )
        for _ in range(num_points)
    ]

def main():
    """Main function to run the mouse movement demonstration."""
    try:
        # --- Setup ---
        overlay = VisualizationOverlay()
        mover = HumanMouseMover(MOUSE_DATA_FILE)

        # --- Define Path ---
        # Use a predefined list of points for a consistent demo
        points_to_visit = [
            (500, 500), (600, 650), (700, 700), (500, 500), (750, 700), (500, 500),
            (600, 650), (700, 700), (500, 500), (750, 700), (500, 500), (600, 650),
            (700, 700), (500, 500), (750, 700), (500, 500), (600, 650), (700, 700),
            (500, 500), (750, 700), (500, 500), (600, 650), (700, 700), (500, 500),
            (750, 700), (500, 500), (600, 650), (700, 700), (500, 500), (750, 700),
            (500, 500), (600, 650), (700, 700), (500, 500), (750, 700), (500, 500),
            (600, 650), (700, 700), (500, 500), (750, 700), (500, 500), (600, 650),
            (700, 700), (500, 500), (750, 700), (500, 500), (600, 650), (700, 700),
        ]

        # increase/decrease randomly by 10%
        points_to_visit = [(int(tx * random.uniform(0.9, 1.1)), int(ty * random.uniform(0.9, 1.1))) for tx, ty in points_to_visit]

        # Or generate random points
        if USE_RANDOM_POINTS:
            screen_w = overlay.root.winfo_screenwidth()
            screen_h = overlay.root.winfo_screenheight()
            points_to_visit = generate_random_points(NUM_RANDOM_POINTS, screen_w, screen_h)

        # --- Execution ---
        for i, (tx, ty) in enumerate(points_to_visit):
            print("-" * 30)
            print(f"Moving to point {i+1}/{len(points_to_visit)}: ({tx}, {ty})")
            mover.move_to(tx, ty, overlay)
            time.sleep(random.uniform(0.1, 0.3)) # Pause between movements

        print("\nMovement sequence complete.")
        overlay.run()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()