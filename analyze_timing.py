"""
Analyze mousedata.json for timing correlations to inform reconstruction scaling.

Produces analysis and plots covering:
  1. Total duration vs net distance (macro: how does total time scale with distance?)
  2. Total duration vs arc length (does path length matter more than net distance?)
  3. Step count vs distance (do longer movements have more steps?)
  4. Per-step timing vs per-step displacement (micro: local speed correlation)
  5. Speed profile phases (acceleration / cruise / deceleration patterns)
  6. Timing distribution per distance bucket
  7. Per-step speed (px/ms) over normalized progress (velocity profile shape)
  8. Instantaneous speed vs timing delta scatter (the core local relationship)

Run: python3 analyze_timing.py [path_to_mousedata.json]
Default: dreambot_example/mousedata.json
"""

import json
import math
import sys
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib/numpy not found — text-only analysis will be produced.")


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

DISTANCE_THRESHOLDS = [12, 18, 26, 39, 58, 87, 130, 190, 260, 360, 500]


def load_mouse_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Per-path feature extraction
# ─────────────────────────────────────────────

def extract_path_features(x_offsets, y_offsets, time_deltas, bucket, direction):
    """Extract all measurable features from a single recorded path."""
    n = len(x_offsets)

    # Per-step displacement magnitudes
    step_dists = [math.hypot(x_offsets[i], y_offsets[i]) for i in range(n)]

    # Net displacement (start to end)
    net_x = sum(x_offsets)
    net_y = sum(y_offsets)
    net_dist = math.hypot(net_x, net_y)

    # Total arc length (actual distance cursor traveled)
    arc_length = sum(step_dists)

    # Timing
    total_time_ms = sum(time_deltas)
    avg_step_time = total_time_ms / n if n else 0

    # Cumulative time at each step
    cum_time = [0.0]
    for dt in time_deltas:
        cum_time.append(cum_time[-1] + dt)

    # Per-step speed (px/ms)
    step_speeds = []
    for i in range(n):
        dt = time_deltas[i]
        if dt > 0:
            step_speeds.append(step_dists[i] / dt)
        else:
            step_speeds.append(0.0)

    # Phase analysis (thirds)
    third = n // 3
    phases = {}
    if third > 0:
        for label, s, e in [("accel", 0, third), ("cruise", third, 2*third), ("decel", 2*third, n)]:
            phase_dist = sum(step_dists[s:e])
            phase_time = sum(time_deltas[s:e])
            phase_steps = e - s
            phase_speed = phase_dist / phase_time if phase_time > 0 else 0
            phases[label] = {
                "dist": phase_dist,
                "time": phase_time,
                "steps": phase_steps,
                "speed": phase_speed,
            }

    # Path curvature: ratio of arc length to net distance
    # 1.0 = perfectly straight, higher = more curved/wobbly
    curvature_ratio = arc_length / net_dist if net_dist > 0 else float("inf")

    return {
        "bucket": int(bucket),
        "direction": direction,
        "num_steps": n,
        "net_dist": net_dist,
        "arc_length": arc_length,
        "total_time_ms": total_time_ms,
        "avg_step_time_ms": avg_step_time,
        "curvature_ratio": curvature_ratio,
        "step_dists": step_dists,
        "step_times": list(time_deltas),
        "step_speeds": step_speeds,
        "cum_time": cum_time,
        "phases": phases,
        "net_x": net_x,
        "net_y": net_y,
    }


def extract_all_features(mouse_data):
    """Extract features from every path in the dataset."""
    all_paths = []
    for bucket in mouse_data:
        for direction in mouse_data[bucket]:
            for path_data in mouse_data[bucket][direction]:
                if len(path_data) >= 3:
                    x_off, y_off, timing = path_data[0], path_data[1], path_data[2]
                elif len(path_data) == 2:
                    x_off, y_off = path_data[0], path_data[1]
                    timing = [8.0] * len(x_off)
                else:
                    continue
                features = extract_path_features(x_off, y_off, timing, bucket, direction)
                all_paths.append(features)
    return all_paths


# ─────────────────────────────────────────────
# Correlation helpers
# ─────────────────────────────────────────────

def pearson_r(xs, ys):
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def fit_power_law(xs, ys):
    """Fit y = a * x^b using log-log linear regression. Returns (a, b, r)."""
    if not HAS_MATPLOTLIB:
        return 0, 0, 0
    log_x = np.log(np.array(xs, dtype=float))
    log_y = np.log(np.array(ys, dtype=float))
    # Linear regression on log-log
    n = len(log_x)
    sum_lx = np.sum(log_x)
    sum_ly = np.sum(log_y)
    sum_lx2 = np.sum(log_x ** 2)
    sum_lxly = np.sum(log_x * log_y)
    b = (n * sum_lxly - sum_lx * sum_ly) / (n * sum_lx2 - sum_lx ** 2)
    log_a = (sum_ly - b * sum_lx) / n
    a = np.exp(log_a)
    r = pearson_r(log_x.tolist(), log_y.tolist())
    return a, b, r


def fit_linear(xs, ys):
    """Fit y = a + b*x. Returns (a, b, r)."""
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if ss_xx == 0:
        return mean_y, 0, 0
    b = ss_xy / ss_xx
    a = mean_y - b * mean_x
    r = pearson_r(xs, ys)
    return a, b, r


# ─────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────

def print_report(all_paths):
    print("=" * 80)
    print("MOUSE TIMING CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"Total paths analyzed: {len(all_paths)}")
    print()

    # ── 1. Macro: total time vs distance ──
    print("─" * 60)
    print("1. TOTAL DURATION vs NET DISTANCE")
    print("─" * 60)
    dists = [p["net_dist"] for p in all_paths]
    times = [p["total_time_ms"] for p in all_paths]
    r = pearson_r(dists, times)
    print(f"   Pearson r = {r:.4f}")

    # Filter for positive values for power law
    valid = [(d, t) for d, t in zip(dists, times) if d > 0 and t > 0]
    if valid and HAS_MATPLOTLIB:
        vd, vt = zip(*valid)
        a, b, r_log = fit_power_law(list(vd), list(vt))
        print(f"   Power law fit: duration = {a:.2f} * distance^{b:.4f}")
        print(f"   Log-log r = {r_log:.4f}")
        la, lb, lr = fit_linear(list(vd), list(vt))
        print(f"   Linear fit: duration = {la:.2f} + {lb:.4f} * distance")
        print(f"   Linear r = {lr:.4f}")
    print()

    # ── 2. Total time vs arc length ──
    print("─" * 60)
    print("2. TOTAL DURATION vs ARC LENGTH")
    print("─" * 60)
    arcs = [p["arc_length"] for p in all_paths]
    r_arc = pearson_r(arcs, times)
    print(f"   Pearson r = {r_arc:.4f}")
    if HAS_MATPLOTLIB:
        valid_a = [(a, t) for a, t in zip(arcs, times) if a > 0 and t > 0]
        if valid_a:
            va, vt = zip(*valid_a)
            a2, b2, r2 = fit_power_law(list(va), list(vt))
            print(f"   Power law fit: duration = {a2:.2f} * arc^{b2:.4f}")
            print(f"   Log-log r = {r2:.4f}")
    print()

    # ── 3. Step count vs distance ──
    print("─" * 60)
    print("3. STEP COUNT vs NET DISTANCE")
    print("─" * 60)
    steps = [p["num_steps"] for p in all_paths]
    r_steps = pearson_r(dists, steps)
    print(f"   Pearson r = {r_steps:.4f}")
    la, lb, lr = fit_linear(dists, steps)
    print(f"   Linear fit: steps = {la:.2f} + {lb:.4f} * distance")
    print(f"   Linear r = {lr:.4f}")
    print()

    # ── 4. Per-step: timing vs displacement (LOCAL correlation) ──
    print("─" * 60)
    print("4. PER-STEP TIMING vs PER-STEP DISPLACEMENT (local)")
    print("─" * 60)
    all_step_dists = []
    all_step_times = []
    for p in all_paths:
        all_step_dists.extend(p["step_dists"])
        all_step_times.extend(p["step_times"])
    r_local = pearson_r(all_step_dists, all_step_times)
    print(f"   Pearson r = {r_local:.4f} (across {len(all_step_dists)} total steps)")
    print()

    # Break down by bucket
    print("   Per-bucket local correlation:")
    by_bucket = defaultdict(lambda: ([], []))
    for p in all_paths:
        for sd, st in zip(p["step_dists"], p["step_times"]):
            by_bucket[p["bucket"]][0].append(sd)
            by_bucket[p["bucket"]][1].append(st)
    for bucket in sorted(by_bucket.keys()):
        sd_list, st_list = by_bucket[bucket]
        r_b = pearson_r(sd_list, st_list)
        avg_speed = sum(d/t if t > 0 else 0 for d, t in zip(sd_list, st_list)) / len(sd_list) if sd_list else 0
        print(f"     Bucket {bucket:>3}: r={r_b:>7.4f}  steps={len(sd_list):>5}  avg_speed={avg_speed:.3f} px/ms")
    print()

    # ── 5. Phase analysis ──
    print("─" * 60)
    print("5. MOVEMENT PHASE SPEED ANALYSIS (accel / cruise / decel)")
    print("─" * 60)
    phase_data = defaultdict(lambda: {"speeds": [], "time_fracs": [], "dist_fracs": []})
    for p in all_paths:
        if not p["phases"]:
            continue
        total_t = p["total_time_ms"]
        total_d = p["arc_length"]
        if total_t <= 0 or total_d <= 0:
            continue
        for phase_name in ["accel", "cruise", "decel"]:
            ph = p["phases"][phase_name]
            phase_data[phase_name]["speeds"].append(ph["speed"])
            phase_data[phase_name]["time_fracs"].append(ph["time"] / total_t)
            phase_data[phase_name]["dist_fracs"].append(ph["dist"] / total_d)

    for phase_name in ["accel", "cruise", "decel"]:
        pd_entry = phase_data[phase_name]
        n = len(pd_entry["speeds"])
        if n == 0:
            continue
        avg_speed = sum(pd_entry["speeds"]) / n
        avg_time_frac = sum(pd_entry["time_fracs"]) / n
        avg_dist_frac = sum(pd_entry["dist_fracs"]) / n
        print(f"   {phase_name:>6}: avg_speed={avg_speed:.4f} px/ms | "
              f"time_frac={avg_time_frac:.3f} | dist_frac={avg_dist_frac:.3f}")
    print()

    # ── 6. Duration stats per bucket ──
    print("─" * 60)
    print("6. TIMING DISTRIBUTION PER BUCKET")
    print("─" * 60)
    by_bucket_paths = defaultdict(list)
    for p in all_paths:
        by_bucket_paths[p["bucket"]].append(p)

    print(f"   {'Bucket':>6} | {'Count':>5} | {'Net Dist':>10} | {'Arc Len':>10} | "
          f"{'Total ms':>10} | {'Steps':>6} | {'Curv':>5} | {'Speed px/ms':>11}")
    print(f"   {'-'*6:>6}-+-{'-'*5:>5}-+-{'-'*10:>10}-+-{'-'*10:>10}-+-"
          f"{'-'*10:>10}-+-{'-'*6:>6}-+-{'-'*5:>5}-+-{'-'*11:>11}")
    for bucket in sorted(by_bucket_paths.keys()):
        paths = by_bucket_paths[bucket]
        n = len(paths)
        avg_net = sum(p["net_dist"] for p in paths) / n
        avg_arc = sum(p["arc_length"] for p in paths) / n
        avg_time = sum(p["total_time_ms"] for p in paths) / n
        avg_steps = sum(p["num_steps"] for p in paths) / n
        avg_curv = sum(p["curvature_ratio"] for p in paths) / n
        avg_speed = avg_arc / avg_time if avg_time > 0 else 0
        print(f"   {bucket:>6} | {n:>5} | {avg_net:>10.1f} | {avg_arc:>10.1f} | "
              f"{avg_time:>10.1f} | {avg_steps:>6.0f} | {avg_curv:>5.2f} | {avg_speed:>11.4f}")
    print()

    # ── 7. Key question: does speed (px/ms) change with distance? ──
    print("─" * 60)
    print("7. AVERAGE SPEED vs DISTANCE (does speed scale with distance?)")
    print("─" * 60)
    avg_speeds = [p["arc_length"] / p["total_time_ms"] if p["total_time_ms"] > 0 else 0 for p in all_paths]
    r_speed_dist = pearson_r(dists, avg_speeds)
    print(f"   Pearson r (net_dist vs avg_speed) = {r_speed_dist:.4f}")
    r_speed_arc = pearson_r(arcs, avg_speeds)
    print(f"   Pearson r (arc_len vs avg_speed) = {r_speed_arc:.4f}")

    for bucket in sorted(by_bucket_paths.keys()):
        paths = by_bucket_paths[bucket]
        speeds = [p["arc_length"] / p["total_time_ms"] for p in paths if p["total_time_ms"] > 0]
        if speeds:
            avg_s = sum(speeds) / len(speeds)
            min_s = min(speeds)
            max_s = max(speeds)
            print(f"     Bucket {bucket:>3}: avg_speed={avg_s:.4f} px/ms  "
                  f"range=[{min_s:.4f}, {max_s:.4f}]")
    print()

    # ── 8. Per-step speed over normalized progress ──
    print("─" * 60)
    print("8. VELOCITY PROFILE SHAPE (speed at 10% intervals of progress)")
    print("─" * 60)
    print("   Normalized position within path vs average speed at that point.")
    print("   This shows the acceleration/cruise/deceleration curve shape.")
    print()

    # Group by bucket, compute avg speed at each decile of progress
    for bucket in sorted(by_bucket_paths.keys()):
        paths = by_bucket_paths[bucket]
        decile_speeds = defaultdict(list)
        for p in paths:
            n = p["num_steps"]
            for i in range(n):
                progress = (i + 1) / n  # 0..1
                decile = min(9, int(progress * 10))
                if p["step_times"][i] > 0:
                    speed = p["step_dists"][i] / p["step_times"][i]
                    decile_speeds[decile].append(speed)

        line = f"   Bucket {bucket:>3}: "
        for d in range(10):
            vals = decile_speeds.get(d, [])
            avg = sum(vals) / len(vals) if vals else 0
            line += f"{avg:.3f} "
        print(line)

    print()
    print("   (Columns = 0-10%, 10-20%, ..., 90-100% of path progress)")
    print()

    # ── 9. Summary: which correlations are strongest? ──
    print("─" * 60)
    print("9. CORRELATION SUMMARY")
    print("─" * 60)
    correlations = [
        ("Total time vs net distance", pearson_r(dists, times)),
        ("Total time vs arc length", pearson_r(arcs, times)),
        ("Step count vs net distance", pearson_r(dists, steps)),
        ("Per-step time vs per-step dist", pearson_r(all_step_dists, all_step_times)),
        ("Avg speed vs net distance", pearson_r(dists, avg_speeds)),
        ("Avg speed vs arc length", pearson_r(arcs, avg_speeds)),
        ("Arc length vs net distance", pearson_r(dists, arcs)),
        ("Curvature vs net distance", pearson_r(dists, [p["curvature_ratio"] for p in all_paths])),
    ]
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for label, r_val in correlations:
        strength = "STRONG" if abs(r_val) > 0.7 else "moderate" if abs(r_val) > 0.4 else "weak"
        print(f"   r={r_val:>7.4f} [{strength:>8}]  {label}")

    print()
    print("=" * 80)


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def generate_plots(all_paths):
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available).")
        return

    by_bucket = defaultdict(list)
    for p in all_paths:
        by_bucket[p["bucket"]].append(p)

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Mouse Movement Timing Correlation Analysis", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)

    colors_by_bucket = {}
    cmap = plt.cm.viridis
    for i, bucket in enumerate(sorted(by_bucket.keys())):
        colors_by_bucket[bucket] = cmap(i / max(1, len(by_bucket) - 1))

    # ── Plot 1: Total duration vs net distance ──
    ax1 = fig.add_subplot(gs[0, 0])
    for bucket in sorted(by_bucket.keys()):
        paths = by_bucket[bucket]
        ds = [p["net_dist"] for p in paths]
        ts = [p["total_time_ms"] for p in paths]
        ax1.scatter(ds, ts, c=[colors_by_bucket[bucket]], label=f"{bucket}",
                    s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Fit line
    dists_all = [p["net_dist"] for p in all_paths if p["net_dist"] > 0]
    times_all = [p["total_time_ms"] for p in all_paths if p["net_dist"] > 0]
    if dists_all:
        a, b, _ = fit_power_law(dists_all, times_all)
        x_fit = np.linspace(min(dists_all), max(dists_all), 100)
        y_fit = a * x_fit ** b
        ax1.plot(x_fit, y_fit, 'r--', linewidth=2, label=f"Power: {a:.1f}*d^{b:.3f}")

    ax1.set_xlabel("Net Distance (px)")
    ax1.set_ylabel("Total Duration (ms)")
    ax1.set_title("1. Total Duration vs Net Distance")
    ax1.legend(fontsize=6, ncol=3)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Total duration vs arc length ──
    ax2 = fig.add_subplot(gs[0, 1])
    for bucket in sorted(by_bucket.keys()):
        paths = by_bucket[bucket]
        arcs = [p["arc_length"] for p in paths]
        ts = [p["total_time_ms"] for p in paths]
        ax2.scatter(arcs, ts, c=[colors_by_bucket[bucket]], label=f"{bucket}",
                    s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    arcs_all = [p["arc_length"] for p in all_paths if p["arc_length"] > 0]
    times_all2 = [p["total_time_ms"] for p in all_paths if p["arc_length"] > 0]
    if arcs_all:
        a2, b2, _ = fit_power_law(arcs_all, times_all2)
        x_fit2 = np.linspace(min(arcs_all), max(arcs_all), 100)
        y_fit2 = a2 * x_fit2 ** b2
        ax2.plot(x_fit2, y_fit2, 'r--', linewidth=2, label=f"Power: {a2:.1f}*a^{b2:.3f}")

    ax2.set_xlabel("Arc Length (px)")
    ax2.set_ylabel("Total Duration (ms)")
    ax2.set_title("2. Total Duration vs Arc Length")
    ax2.legend(fontsize=6, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Per-step timing vs per-step displacement ──
    ax3 = fig.add_subplot(gs[1, 0])
    for bucket in sorted(by_bucket.keys()):
        step_d, step_t = [], []
        for p in by_bucket[bucket]:
            step_d.extend(p["step_dists"])
            step_t.extend(p["step_times"])
        ax3.scatter(step_d, step_t, c=[colors_by_bucket[bucket]], label=f"{bucket}",
                    s=5, alpha=0.3)

    ax3.set_xlabel("Step Displacement (px)")
    ax3.set_ylabel("Step Timing (ms)")
    ax3.set_title("3. Per-Step Timing vs Displacement")
    ax3.legend(fontsize=6, ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, 15)
    ax3.set_ylim(-1, 80)

    # ── Plot 4: Average speed vs distance ──
    ax4 = fig.add_subplot(gs[1, 1])
    for bucket in sorted(by_bucket.keys()):
        paths = by_bucket[bucket]
        ds = [p["net_dist"] for p in paths]
        speeds = [p["arc_length"] / p["total_time_ms"] if p["total_time_ms"] > 0 else 0 for p in paths]
        ax4.scatter(ds, speeds, c=[colors_by_bucket[bucket]], label=f"{bucket}",
                    s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax4.set_xlabel("Net Distance (px)")
    ax4.set_ylabel("Average Speed (px/ms)")
    ax4.set_title("4. Average Speed vs Distance")
    ax4.legend(fontsize=6, ncol=3)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Velocity profile (speed over normalized progress) ──
    ax5 = fig.add_subplot(gs[2, 0])
    num_bins = 50
    for bucket in sorted(by_bucket.keys()):
        bin_speeds = defaultdict(list)
        for p in by_bucket[bucket]:
            n = p["num_steps"]
            for i in range(n):
                progress = (i + 1) / n
                b_idx = min(num_bins - 1, int(progress * num_bins))
                if p["step_times"][i] > 0:
                    bin_speeds[b_idx].append(p["step_dists"][i] / p["step_times"][i])

        xs = [(b_idx + 0.5) / num_bins for b_idx in range(num_bins)]
        ys = [np.mean(bin_speeds[b_idx]) if bin_speeds[b_idx] else 0 for b_idx in range(num_bins)]
        ax5.plot(xs, ys, '-', color=colors_by_bucket[bucket], linewidth=1.5,
                 alpha=0.8, label=f"{bucket}")

    ax5.set_xlabel("Normalized Progress (0=start, 1=end)")
    ax5.set_ylabel("Speed (px/ms)")
    ax5.set_title("5. Velocity Profile Shape by Bucket")
    ax5.legend(fontsize=6, ncol=3)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Timing profile (ms per step over normalized progress) ──
    ax6 = fig.add_subplot(gs[2, 1])
    for bucket in sorted(by_bucket.keys()):
        bin_times = defaultdict(list)
        for p in by_bucket[bucket]:
            n = p["num_steps"]
            for i in range(n):
                progress = (i + 1) / n
                b_idx = min(num_bins - 1, int(progress * num_bins))
                bin_times[b_idx].append(p["step_times"][i])

        xs = [(b_idx + 0.5) / num_bins for b_idx in range(num_bins)]
        ys = [np.mean(bin_times[b_idx]) if bin_times[b_idx] else 0 for b_idx in range(num_bins)]
        ax6.plot(xs, ys, '-', color=colors_by_bucket[bucket], linewidth=1.5,
                 alpha=0.8, label=f"{bucket}")

    ax6.set_xlabel("Normalized Progress (0=start, 1=end)")
    ax6.set_ylabel("Step Timing (ms)")
    ax6.set_title("6. Timing Profile Shape by Bucket (ms per step)")
    ax6.legend(fontsize=6, ncol=3)
    ax6.grid(True, alpha=0.3)

    # ── Plot 7: Step count vs net distance ──
    ax7 = fig.add_subplot(gs[3, 0])
    for bucket in sorted(by_bucket.keys()):
        paths = by_bucket[bucket]
        ds = [p["net_dist"] for p in paths]
        ss = [p["num_steps"] for p in paths]
        ax7.scatter(ds, ss, c=[colors_by_bucket[bucket]], label=f"{bucket}",
                    s=40, alpha=0.7, edgecolors='white', linewidth=0.5)

    all_dists_pos = [p["net_dist"] for p in all_paths]
    all_steps = [p["num_steps"] for p in all_paths]
    la, lb, _ = fit_linear(all_dists_pos, all_steps)
    x_fit3 = np.linspace(min(all_dists_pos), max(all_dists_pos), 100)
    y_fit3 = la + lb * x_fit3
    ax7.plot(x_fit3, y_fit3, 'r--', linewidth=2, label=f"Linear: {la:.1f}+{lb:.3f}*d")

    ax7.set_xlabel("Net Distance (px)")
    ax7.set_ylabel("Step Count")
    ax7.set_title("7. Step Count vs Net Distance")
    ax7.legend(fontsize=6, ncol=3)
    ax7.grid(True, alpha=0.3)

    # ── Plot 8: Phase speeds comparison ──
    ax8 = fig.add_subplot(gs[3, 1])
    phase_names = ["accel", "cruise", "decel"]
    for bucket in sorted(by_bucket.keys()):
        paths = by_bucket[bucket]
        phase_speeds = {ph: [] for ph in phase_names}
        for p in paths:
            for ph in phase_names:
                if ph in p["phases"]:
                    phase_speeds[ph].append(p["phases"][ph]["speed"])

        avgs = [np.mean(phase_speeds[ph]) if phase_speeds[ph] else 0 for ph in phase_names]
        x_pos = [0, 1, 2]
        ax8.scatter(x_pos, avgs, c=[colors_by_bucket[bucket]], s=60, alpha=0.7,
                    edgecolors='white', linewidth=0.5, zorder=5)
        ax8.plot(x_pos, avgs, '-', color=colors_by_bucket[bucket], alpha=0.4, linewidth=1)

    ax8.set_xticks([0, 1, 2])
    ax8.set_xticklabels(["Accel\n(first 1/3)", "Cruise\n(middle 1/3)", "Decel\n(last 1/3)"])
    ax8.set_ylabel("Average Speed (px/ms)")
    ax8.set_title("8. Speed by Movement Phase (each dot = bucket avg)")
    ax8.grid(True, alpha=0.3, axis='y')

    plt.savefig("timing_analysis.png", dpi=150, bbox_inches="tight")
    print("Plots saved to timing_analysis.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "dreambot_example/mousedata.json"

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    print(f"Loading data from: {filepath}")
    mouse_data = load_mouse_data(filepath)
    all_paths = extract_all_features(mouse_data)

    print_report(all_paths)
    generate_plots(all_paths)


if __name__ == "__main__":
    main()
