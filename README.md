# SmartMouse V2

## Overview

This project has been enhanced to record and replay **actual human timing patterns** instead of using synthetic easing functions. This makes the mouse movements significantly more realistic and human-like.

## What Changed?

### Before (Synthetic Timing)
- Used mathematical easing functions (cubic, quartic, elastic, bounce)
- Applied speed profiles based on distance categories
- Timing was artificial and predictable
- Did not reflect actual human movement patterns

### After (Recorded Timing)
- Records actual time deltas between each mouse position sample
- Replays movements using the recorded timing data
- Adds small variance (±5%) to prevent perfect repeatability
- Maintains authentic human acceleration and deceleration patterns

## Key Benefits

1. **More Realistic Movement**: Movements now follow actual human timing patterns
2. **Natural Acceleration/Deceleration**: No more artificial easing curves
3. **Individual Variation**: Each recorded path has unique timing characteristics
4. **Better Detection Resistance**: Harder to detect as automated due to natural timing

## New Data Format

The mouse data now includes timing information:

```json
{
  "distance_threshold": {
    "direction": [
      [
        [x_offset1, x_offset2, ...],      // Position offsets (unchanged)
        [y_offset1, y_offset2, ...],      // Position offsets (unchanged)
        [time_delta1, time_delta2, ...]   // NEW: Timing data in milliseconds
      ]
    ]
  }
}
```

## Core Files
- **`recorder.py`**: Now records timestamps with each position sample
- **`parser.py`**: Processes timing data and maintains backward compatibility
- **`move.py`**: Uses recorded timing instead of synthetic easing functions

## Example Dataset

For convenience, a small example dataset is included at `dreambot_example/mousedata.json`.
You can use this directly with `SmartMouseMultiDir.java` to get started quickly.
However, this dataset is limited in size and variety; for best results, it's recommended
to record your own data with `recorder.py` and process it with `parser.py` to generate
your personalized `mousedata.json`.

## How to Use

### 1. Record Mouse Movements
```bash
# Record human-like mouse movements with timing data
python recorder.py
```

This step creates `mousedata_raw.json`.

### 2. Process Raw Data
```bash
# Convert mousedata_raw.json to structured movement data in mousedata.json
python parser.py
```

### 3. Test Movements
```bash
# Test the processed movements
python move.py
```

### 4. Use in DreamBot
The `mousedata.json` file can now be used directly in SmartMouseMultiDir.java for more realistic human-like movements.


## Backward Compatibility

The system is fully backward compatible:
- Old data files without timing work fine (falls back to synthetic timing)
- Parser handles both old and new formats automatically
- No existing data needs to be regenerated (but new data will be better)

## Technical Details

### Recording Process
1. Samples mouse position every 4ms during movement
2. Records timestamp with each position sample
3. Calculates time deltas between consecutive samples
4. Stores timing data alongside position offsets

### Playback Process
1. Loads recorded position offsets and timing deltas
2. Reconstructs movement path using existing algorithm
3. Uses recorded timing with small random variance (±5%)
4. Falls back to synthetic timing if timing data unavailable

### Quality Improvements
- Movement speed analysis during recording
- Smoothness metrics to detect poor recordings
- Real-time feedback to improve data quality
- Statistics on recorded movements

## Performance Impact

- **Recording**: Minimal overhead (just timestamp storage)
- **Playback**: Slightly more efficient (no easing calculations)
- **Storage**: Small increase (one timing value per position sample)

## Conclusion

These improvements make the mouse movement system significantly more human-like by using actual recorded timing patterns instead of synthetic mathematical functions. The movements are now indistinguishable from real human mouse usage while maintaining all the flexibility and customization of the original system. 