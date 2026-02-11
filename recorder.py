"""
Enhanced mouse movement recorder with improved timing visualization.

This version provides:
- Real-time timing feedback during recording
- Movement speed visualization
- Better quality control for recorded data
- Statistics about recorded movements
"""

import tkinter as tk
import random
import math
import json
import os
import time
from pynput.mouse import Controller

MOUSE_DATA_FILE = "mousedata_raw.json"
DOT_RADIUS = 10
SAMPLING_INTERVAL = 4  # 4ms sampling rate

# Define thresholds in ascending order
DISTANCE_THRESHOLDS = [12, 18, 26, 39, 58, 87, 130, 190, 260, 360, 500]

# We want to cover these 8 directions
ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Number of samples per orientation for each threshold
SAMPLES_PER_ORIENTATION = 1  # Increased for better data quality

def load_mousedata():
    if os.path.exists(MOUSE_DATA_FILE):
        with open(MOUSE_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_mousedata(data):
    with open(MOUSE_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def path_to_offsets_and_timings(positions_with_timestamps):
    """
    Extract position offsets and timing deltas from timestamped positions.
    Now properly handles pauses during movement by accumulating timing.
    Returns: ([x_offsets], [y_offsets], [time_deltas_ms])
    """
    if len(positions_with_timestamps) < 2:
        return [[], [], []]
    
    x_offsets, y_offsets, time_deltas = [], [], []
    accumulated_time = 0  # Track time during pauses
    
    for i in range(1, len(positions_with_timestamps)):
        prev_x, prev_y, prev_time = positions_with_timestamps[i - 1]
        curr_x, curr_y, curr_time = positions_with_timestamps[i]
        
        dx, dy = curr_x - prev_x, curr_y - prev_y
        dt = curr_time - prev_time  # Time delta in milliseconds
        
        if dx != 0 or dy != 0:
            # Movement detected - record it with accumulated timing
            x_offsets.append(dx)
            y_offsets.append(dy)
            # Include any accumulated pause time with this movement
            total_time = dt + accumulated_time
            time_deltas.append(total_time)
            accumulated_time = 0  # Reset accumulator
        else:
            # No movement - accumulate the timing for the next movement
            accumulated_time += dt
    
    # If we end with accumulated time but no final movement, 
    # add it to the last recorded timing (if any)
    if accumulated_time > 0 and len(time_deltas) > 0:
        time_deltas[-1] += accumulated_time
    
    return [x_offsets, y_offsets, time_deltas]

def analyze_movement_quality(positions_with_timestamps):
    """Analyze the quality of recorded movement data."""
    if len(positions_with_timestamps) < 3:
        return "Too few points"
    
    # Calculate average speed, acceleration changes, smoothness
    total_distance = 0
    speed_changes = 0
    time_deltas = []
    
    for i in range(1, len(positions_with_timestamps)):
        prev_x, prev_y, prev_time = positions_with_timestamps[i - 1]
        curr_x, curr_y, curr_time = positions_with_timestamps[i]
        
        distance = math.hypot(curr_x - prev_x, curr_y - prev_y)
        dt = curr_time - prev_time
        total_distance += distance
        time_deltas.append(dt)
        
        if i > 1 and dt > 0:
            speed = distance / dt
            prev_speed = math.hypot(positions_with_timestamps[i-1][0] - positions_with_timestamps[i-2][0],
                                   positions_with_timestamps[i-1][1] - positions_with_timestamps[i-2][1]) / (prev_time - positions_with_timestamps[i-2][2])
            if abs(speed - prev_speed) / max(speed, prev_speed, 0.1) > 0.5:  # >50% speed change
                speed_changes += 1
    
    total_time = positions_with_timestamps[-1][2] - positions_with_timestamps[0][2]
    avg_speed = total_distance / max(total_time, 1)
    smoothness = 1.0 - (speed_changes / max(len(positions_with_timestamps) - 2, 1))
    
    return f"Speed: {avg_speed:.1f}px/ms, Smoothness: {smoothness:.2f}, Points: {len(positions_with_timestamps)}"

def distance_range_for_threshold_index(i):
    """
    Given index i in DISTANCE_THRESHOLDS, return (low, high].
    For i=0 => (0, 12]
    For i=1 => (12, 18], etc.
    """
    if i == 0:
        return (DISTANCE_THRESHOLDS[0], DISTANCE_THRESHOLDS[0])
    low = DISTANCE_THRESHOLDS[i - 1]
    high = DISTANCE_THRESHOLDS[i]
    return (low, high)

def get_threshold_for_distance(distance):
    for threshold in DISTANCE_THRESHOLDS:
        if distance <= threshold:
            return threshold
    return DISTANCE_THRESHOLDS[-1]

def generate_point_in_orientation(x1, y1, orientation, low, high, width, height):
    """
    Generate (x2, y2) so that:
      - The distance from (x1, y1) is in (low, high].
      - The angle is ~ the given orientation (with some slack).
      - The final point is within the rectangle [margin, width-margin] x [margin, height-margin].
    """
    # Center angles (in degrees) for each of the 8 directions
    center_angles = {
        "N": 90,
        "NE": 45,
        "E": 0,
        "SE": 315,  # or -45
        "S": 270,
        "SW": 225,
        "W": 180,
        "NW": 135
    }

    angle_slack_deg = 15
    center_angle = center_angles[orientation]
    max_attempts = 2000
    margin = 60

    for _ in range(max_attempts):
        dist = random.uniform(low, high)
        # pick an angle around the center ± slack
        angle_deg = random.uniform(center_angle - angle_slack_deg, center_angle + angle_slack_deg)
        angle_rad = math.radians(angle_deg)

        dx = dist * math.cos(angle_rad)
        # In Tk y grows downward, so we invert sin to keep directions consistent
        dy = -dist * math.sin(angle_rad)

        x2 = x1 + dx
        y2 = y1 + dy

        # Check boundaries
        if margin <= x2 <= width - margin and margin <= y2 <= height - margin:
            return int(x2), int(y2), dist

    # If we couldn't find a suitable point, fallback => None
    return None

class EnhancedCoverageDotRecorderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Coverage Dot Path Recorder")
        self.width = 900
        self.height = 800
        self.master.geometry(f"{self.width}x{self.height}")
        
        # Create main frame and canvas
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info panel at top
        self.info_frame = tk.Frame(main_frame, height=100, bg='lightgray')
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.info_frame.pack_propagate(False)
        
        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info labels
        self.progress_label = tk.Label(self.info_frame, text="Progress: 0/0", font=("Arial", 12, "bold"))
        self.progress_label.pack()
        
        self.task_label = tk.Label(self.info_frame, text="Task: Ready", font=("Arial", 10))
        self.task_label.pack()
        
        self.timing_label = tk.Label(self.info_frame, text="Timing: Ready", font=("Arial", 10))
        self.timing_label.pack()
        
        self.quality_label = tk.Label(self.info_frame, text="Quality: Ready", font=("Arial", 10))
        self.quality_label.pack()

        self.mousedata = load_mousedata()
        self.positions = []
        self.is_recording = False
        self.last_sample_time = 0
        self.recording_start_time = 0
        
        # Create a list of all combinations to record
        self.combinations_to_record = self.create_combinations_list()
        random.shuffle(self.combinations_to_record)  # Randomize order
        self.current_combination_index = 0
        self.total_combinations = len(self.combinations_to_record)
        self.total_paths_recorded = 0
        
        # For continuity between dots
        self.last_end_position = None
        
        self.mouse_controller = Controller()

        self.start_dot = None
        self.end_dot = None

        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.master.bind("<Escape>", lambda e: self.quit_app())
        self.master.bind("<space>", lambda e: self.skip_current())
        self.master.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.schedule_position_sampling()
        self.spawn_next_dot_pair()

    def create_combinations_list(self):
        """Create a list of all threshold and orientation combinations to record"""
        combinations = []
        for threshold_idx in range(len(DISTANCE_THRESHOLDS)):
            for orient in ORIENTATIONS:
                for sample in range(SAMPLES_PER_ORIENTATION):
                    combinations.append({
                        'threshold_index': threshold_idx,
                        'orientation': orient,
                        'sample': sample
                    })
        return combinations

    def skip_current(self):
        """Skip the current recording and move to next."""
        if self.is_recording:
            self.is_recording = False
            self.update_timing_display("Skipped")
        self.next_step()

    def current_orientation(self):
        if self.combinations_to_record:
            return self.combinations_to_record[self.current_combination_index]['orientation']
        return ORIENTATIONS[0]  # Fallback
        
    def spawn_next_dot_pair(self):
        self.canvas.delete("all")
        self.start_dot = None
        self.end_dot = None
        self.is_recording = False
        self.positions = []

        # Ensure geometry is updated so canvas dimensions are correct
        self.master.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if self.current_combination_index >= len(self.combinations_to_record):
            print("All combinations collected! Exiting.")
            self.quit_app()
            return

        # Get the current combination
        current = self.combinations_to_record[self.current_combination_index]
        threshold_index = current['threshold_index']
        orientation = current['orientation']
        sample_number = current['sample']

        # Get the current distance range
        low, high = distance_range_for_threshold_index(threshold_index)
        
        # Update progress display
        progress = f"{self.current_combination_index+1}/{self.total_combinations}"
        self.progress_label.config(text=f"Progress: {progress}")
        self.task_label.config(
            text=f"Distance: {low}-{high}px, Direction: {orientation}, Sample: {sample_number+1}/{SAMPLES_PER_ORIENTATION}"
        )
        
        margin = 60

        # Helper: choose a safe random point inside the canvas
        def safe_random_point():
            if canvas_w > 2 * margin and canvas_h > 2 * margin:
                rx = random.randint(margin, canvas_w - margin)
                ry = random.randint(margin, canvas_h - margin)
                return rx, ry
            # Fallback to canvas center if canvas is too small
            return max(margin, canvas_w // 2), max(margin, canvas_h // 2)
        
        # Use the last end position as the new start position if available and in-bounds
        if self.last_end_position:
            lx, ly = self.last_end_position
            if margin <= lx <= max(margin, canvas_w - margin) and margin <= ly <= max(margin, canvas_h - margin):
                base_x, base_y = lx, ly
            else:
                base_x, base_y = safe_random_point()
        else:
            # First run, use a random position within the canvas
            base_x, base_y = safe_random_point()

        # Generate end point in the correct direction and distance range
        result = generate_point_in_orientation(
            base_x, base_y, orientation, low, high, 
            canvas_w, canvas_h
        )
        
        # If generation fails, try with a random start point
        if not result:
            base_x, base_y = safe_random_point()
            result = generate_point_in_orientation(
                base_x, base_y, orientation, low, high,
                canvas_w, canvas_h
            )
            
        # If still fails, try again later
        if not result:
            self.timing_label.config(text="Failed to place dots. Trying again...")
            self.master.after(500, self.spawn_next_dot_pair)
            return
            
        end_x, end_y, actual_distance = result
        
        # Create dots
        start_id = self.create_dot(base_x, base_y, "red")
        end_id = self.create_dot(end_x, end_y, "blue")
        
        # Draw a line between them
        self.canvas.create_line(base_x, base_y, end_x, end_y, fill="lightgray", dash=(4, 4))
        
        # Store dot info
        self.start_dot = (start_id, base_x, base_y)
        self.end_dot = (end_id, end_x, end_y)
        
        # Display distance
        actual_angle = math.degrees(math.atan2(base_y - end_y, end_x - base_x))
        self.timing_label.config(
            text=f"Distance: {actual_distance:.1f}px, Angle: {actual_angle:.1f}°"
        )
        self.quality_label.config(text="Ready - Click red dot to start")

    def create_dot(self, x, y, color):
        return self.canvas.create_oval(
            x - DOT_RADIUS, y - DOT_RADIUS,
            x + DOT_RADIUS, y + DOT_RADIUS,
            fill=color, outline="black", width=2
        )

    def schedule_position_sampling(self):
        if self.is_recording:
            now = time.time() * 1000
            if now - self.last_sample_time >= SAMPLING_INTERVAL:
                x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
                self.positions.append((x, y, now))  # Store timestamp with position
                self.last_sample_time = now
                
                # Update real-time timing display
                elapsed = (now - self.recording_start_time) / 1000.0
                self.update_timing_display(f"Recording... {elapsed:.2f}s, {len(self.positions)} points")
        
        self.master.after(1, self.schedule_position_sampling)

    def update_timing_display(self, message):
        self.timing_label.config(text=message)

    def canvas_to_global(self, cx, cy):
        window_left = self.master.winfo_rootx()
        window_top = self.master.winfo_rooty()
        canvas_left = self.canvas.winfo_x()
        canvas_top = self.canvas.winfo_y()
        return (window_left + canvas_left + cx, window_top + canvas_top + cy)

    def on_mouse_click(self, event):
        x, y = event.x, event.y

        # Start dot logic
        if not self.is_recording and self.start_dot:
            _, sx, sy = self.start_dot
            if math.hypot(x - sx, y - sy) <= DOT_RADIUS + 3:
                # Teleport mouse to start dot
                global_pos = self.canvas_to_global(sx, sy)
                self.mouse_controller.position = global_pos
                time.sleep(0.05)

                self.is_recording = True
                self.recording_start_time = time.time() * 1000
                start_time = self.recording_start_time
                self.positions = [(sx, sy, start_time)]  # Include timestamp for start position
                self.last_sample_time = start_time
                self.canvas.itemconfig(self.start_dot[0], fill="green")
                self.update_timing_display("Recording... Click blue dot to finish.")
                self.quality_label.config(text="Recording in progress...")
                return

        # End dot logic
        if self.is_recording and self.end_dot:
            _, ex, ey = self.end_dot
            if math.hypot(x - ex, y - ey) <= DOT_RADIUS + 3:
                self.is_recording = False
                
                # Analyze movement quality
                quality_info = analyze_movement_quality(self.positions)
                self.quality_label.config(text=f"Quality: {quality_info}")
                
                self.save_path(self.start_dot, self.end_dot)
                self.total_paths_recorded += 1
                
                # Store this end position for the next start position
                self.last_end_position = (ex, ey)
                
                # Brief delay before next movement
                self.master.after(1000, self.next_step)
                return

    def next_step(self):
        # Move to the next combination
        self.current_combination_index += 1
        
        if self.current_combination_index >= len(self.combinations_to_record):
            print("All combinations collected! Exiting.")
            self.quit_app()
            return
            
        self.spawn_next_dot_pair()

    def save_path(self, start_dot, end_dot):
        sx, sy = start_dot[1], start_dot[2]
        ex, ey = end_dot[1], end_dot[2]
        dx, dy = ex - sx, ey - sy
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        offsets = path_to_offsets_and_timings(self.positions)
        
        # Get current combination info
        current = self.combinations_to_record[self.current_combination_index]
        threshold_index = current['threshold_index']
        threshold = DISTANCE_THRESHOLDS[threshold_index]
        orientation = current['orientation']
        
        entry = {
            "distance": distance,
            "angle_deg": angle,
            "orientation": orientation,
            "offsets": offsets,
            "recorded_at": time.time(),
            "points_count": len(self.positions),
            "total_time_ms": self.positions[-1][2] - self.positions[0][2] if len(self.positions) > 1 else 0
        }
        self.mousedata.setdefault(str(threshold), []).append(entry)
        save_mousedata(self.mousedata)
        print(f"[Saved] distance={distance:.1f}px, angle={angle:.1f}° → threshold {threshold}, dir={orientation}")
        print(f"Progress: {self.current_combination_index+1}/{self.total_combinations}")

    def quit_app(self):
        print(f"Exiting. {self.total_paths_recorded} paths recorded.")
        save_mousedata(self.mousedata)
        self.master.destroy()

def main():
    root = tk.Tk()
    app = EnhancedCoverageDotRecorderApp(root)
    print("Enhanced Mouse Movement Recorder")
    print("Instructions:")
    print("- Click red dot to start recording")
    print("- Move mouse to blue dot and click to finish")
    print("- Press SPACE to skip current recording")
    print("- Press ESC to exit")
    root.mainloop()

if __name__ == "__main__":
    main() 