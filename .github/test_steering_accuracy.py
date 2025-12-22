#!/usr/bin/env python3
"""
Steering Accuracy Test Script for CI

Measures angular steering accuracy by comparing actuator output angle vs actual steering angle.
Reports steering performance across different speed groups.
Supports comparing two routes and filtering segments where steering is not pressed.

Usage:
  # Test single route
  python .github/test_steering_accuracy.py --route ROUTE_ID --segments 0-5

  # Compare two routes
  python .github/test_steering_accuracy.py --before ROUTE1 --after ROUTE2 --segments 7-13

  # With caching (for CI)
  python .github/test_steering_accuracy.py --before ROUTE1 --after ROUTE2 --segments 7-13 --cache

  # Generate plots
  python .github/test_steering_accuracy.py --before ROUTE1 --after ROUTE2 --segments 7-13 --plot
"""
import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure repo root is in path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openpilot.tools.lib.logreader import LogReader, openpilotci_source, comma_api_source


@dataclass
class SpeedGroup:
    name: str
    min_speed: float  # m/s
    max_speed: float  # m/s
    description: str


SPEED_GROUPS = [
    SpeedGroup("crawl", 0, 5, "0-18 km/h / 0-11 mph"),
    SpeedGroup("slow", 5, 15, "18-54 km/h / 11-34 mph"),
    SpeedGroup("medium", 15, 25, "54-90 km/h / 34-56 mph"),
    SpeedGroup("fast", 25, 35, "90-126 km/h / 56-78 mph"),
    SpeedGroup("veryfast", 35, 45, "126-162 km/h / 78-101 mph"),
    SpeedGroup("autobahn", 45, 100, "162+ km/h / 101+ mph"),
]


@dataclass
class SteeringStats:
    """Statistics for steering accuracy at a given angle bin."""
    count: int = 0
    total_error: float = 0.0
    max_error: float = 0.0
    total_abs_desired: float = 0.0
    total_abs_actual: float = 0.0
    undershoot_count: int = 0  # actual < desired (magnitude)
    overshoot_count: int = 0   # actual > desired (magnitude)
    exact_count: int = 0


@dataclass
class SegmentResult:
    """Result from analyzing a single segment."""
    segment_num: int
    valid: bool
    steering_pressed_frames: int = 0
    total_frames: int = 0
    active_frames: int = 0
    stats_by_speed: dict = field(default_factory=dict)


def get_speed_group(v_ego: float) -> Optional[SpeedGroup]:
    """Get the speed group for a given velocity."""
    for group in SPEED_GROUPS:
        if group.min_speed <= v_ego < group.max_speed:
            return group
    return None


def analyze_segment(route: str, segment: int, skip_steering_pressed: bool = True) -> SegmentResult:
    """Analyze a single segment for steering accuracy.

    Returns None if segment should be skipped (steering pressed during segment).
    """
    result = SegmentResult(
        segment_num=segment,
        valid=True,
        stats_by_speed={group.name: defaultdict(SteeringStats) for group in SPEED_GROUPS}
    )

    segment_route = f"{route}/{segment}"

    try:
        lr = LogReader(segment_route, sources=[openpilotci_source, comma_api_source], sort_by_time=True)
    except Exception as e:
        print(f"  Segment {segment}: Failed to load - {e}")
        result.valid = False
        return result

    # Collect messages
    car_state = None
    car_output = None
    selfdrive_state = None
    steering_pressed_ever = False

    for msg in lr:
        msg_type = msg.which()

        if msg_type == "carState":
            car_state = msg.carState
            result.total_frames += 1

            if car_state.steeringPressed:
                result.steering_pressed_frames += 1
                steering_pressed_ever = True

        elif msg_type == "carOutput":
            car_output = msg.carOutput

        elif msg_type == "selfdriveState":
            selfdrive_state = msg.selfdriveState

        # Process when we have all required messages synced on carState update
        if msg_type == "carState" and car_state is not None and car_output is not None and selfdrive_state is not None:
            # Skip if not active or standstill
            if not selfdrive_state.active or car_state.standstill:
                continue

            result.active_frames += 1

            v_ego = car_state.vEgo
            speed_group = get_speed_group(v_ego)
            if speed_group is None:
                continue

            # Get angles
            actual_angle = car_state.steeringAngleDeg
            desired_angle = car_output.actuatorsOutput.steeringAngleDeg

            # Calculate error
            error = abs(desired_angle - actual_angle)

            # Bin by desired angle magnitude (rounded to integer)
            angle_bin = int(abs(round(desired_angle, 0)))

            # Update stats
            stats = result.stats_by_speed[speed_group.name][angle_bin]
            stats.count += 1
            stats.total_error += error
            stats.max_error = max(stats.max_error, error)
            stats.total_abs_desired += abs(desired_angle)
            stats.total_abs_actual += abs(actual_angle)

            # Determine over/undershoot
            if abs(actual_angle - desired_angle) < 0.05:
                stats.exact_count += 1
            elif abs(actual_angle) < abs(desired_angle):
                stats.undershoot_count += 1
            else:
                stats.overshoot_count += 1

    # Skip segment if steering was pressed at all during segment
    if skip_steering_pressed and steering_pressed_ever:
        result.valid = False
        print(f"  Segment {segment}: Skipped (steering pressed in {result.steering_pressed_frames}/{result.total_frames} frames)")
        return result

    print(f"  Segment {segment}: {result.active_frames} active frames, {result.total_frames} total frames")
    return result


def merge_stats(all_results: list[SegmentResult]) -> dict:
    """Merge stats from multiple segments."""
    merged = {group.name: defaultdict(SteeringStats) for group in SPEED_GROUPS}

    for result in all_results:
        if not result.valid:
            continue
        for speed_name, angle_stats in result.stats_by_speed.items():
            for angle_bin, stats in angle_stats.items():
                m = merged[speed_name][angle_bin]
                m.count += stats.count
                m.total_error += stats.total_error
                m.max_error = max(m.max_error, stats.max_error)
                m.total_abs_desired += stats.total_abs_desired
                m.total_abs_actual += stats.total_abs_actual
                m.undershoot_count += stats.undershoot_count
                m.overshoot_count += stats.overshoot_count
                m.exact_count += stats.exact_count

    return merged


def print_stats(stats: dict, title: str):
    """Pretty print steering statistics."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    total_frames = 0
    total_error_sum = 0.0

    for group in SPEED_GROUPS:
        angle_stats = stats[group.name]
        if not angle_stats:
            continue

        print(f"\n  Speed Group: {group.name:10s} ({group.description})")
        print(f"  {'-' * 95}")
        print(f"  {'Angle':>6s} | {'Avg Err':>8s} | {'Max Err':>8s} | {'Under%':>7s} | {'Exact%':>7s} | {'Over%':>7s} | {'Samples':>8s}")
        print(f"  {'-' * 95}")

        group_frames = 0
        group_error = 0.0

        for angle_bin in sorted(angle_stats.keys()):
            s = angle_stats[angle_bin]
            if s.count == 0:
                continue

            avg_error = s.total_error / s.count
            under_pct = (s.undershoot_count / s.count) * 100
            exact_pct = (s.exact_count / s.count) * 100
            over_pct = (s.overshoot_count / s.count) * 100

            print(f"  {angle_bin:5d}° | {avg_error:7.2f}° | {s.max_error:7.2f}° | "
                  f"{under_pct:6.1f}% | {exact_pct:6.1f}% | {over_pct:6.1f}% | {s.count:8d}")

            group_frames += s.count
            group_error += s.total_error

        if group_frames > 0:
            print(f"  {'-' * 95}")
            print(f"  {'TOTAL':>6s} | {group_error/group_frames:7.2f}° | {'-':>8s} | "
                  f"{'-':>7s} | {'-':>7s} | {'-':>7s} | {group_frames:8d}")

        total_frames += group_frames
        total_error_sum += group_error

    if total_frames > 0:
        print(f"\n  {'=' * 95}")
        print(f"  OVERALL: {total_error_sum/total_frames:.3f}° avg error across {total_frames} samples")
        print(f"  {'=' * 95}")

    return total_frames, total_error_sum / total_frames if total_frames > 0 else 0.0


def print_comparison(stats_before: dict, stats_after: dict):
    """Print comparison between two routes."""
    print(f"\n{'=' * 100}")
    print(f"  COMPARISON: BEFORE vs AFTER")
    print(f"{'=' * 100}")

    for group in SPEED_GROUPS:
        before_angles = stats_before[group.name]
        after_angles = stats_after[group.name]

        # Get all angle bins that exist in either
        all_bins = set(before_angles.keys()) | set(after_angles.keys())
        if not all_bins:
            continue

        print(f"\n  Speed Group: {group.name:10s} ({group.description})")
        print(f"  {'-' * 95}")
        print(f"  {'Angle':>6s} | {'Before':>10s} | {'After':>10s} | {'Change':>10s} | {'Before N':>9s} | {'After N':>9s}")
        print(f"  {'-' * 95}")

        for angle_bin in sorted(all_bins):
            b = before_angles.get(angle_bin, SteeringStats())
            a = after_angles.get(angle_bin, SteeringStats())

            b_avg = b.total_error / b.count if b.count > 0 else 0
            a_avg = a.total_error / a.count if a.count > 0 else 0

            if b.count > 0 and a.count > 0:
                change = a_avg - b_avg
                change_str = f"{change:+.2f}°"
                if change < -0.1:
                    change_str += " ✓"  # improved
                elif change > 0.1:
                    change_str += " ✗"  # worse
            else:
                change_str = "N/A"

            b_str = f"{b_avg:.2f}°" if b.count > 0 else "-"
            a_str = f"{a_avg:.2f}°" if a.count > 0 else "-"

            print(f"  {angle_bin:5d}° | {b_str:>10s} | {a_str:>10s} | {change_str:>10s} | {b.count:>9d} | {a.count:>9d}")


@dataclass
class TimeSeries:
    """Time series data for plotting."""
    timestamps: list = field(default_factory=list)
    desired_angles: list = field(default_factory=list)
    actual_angles: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    speeds: list = field(default_factory=list)


def collect_time_series(route: str, segments: list[int]) -> TimeSeries:
    """Collect time series data for plotting."""
    ts = TimeSeries()
    base_time = None

    for segment in segments:
        segment_route = f"{route}/{segment}"
        try:
            lr = LogReader(segment_route, sources=[openpilotci_source, comma_api_source], sort_by_time=True)
        except Exception:
            continue

        car_state = None
        car_output = None
        selfdrive_state = None
        steering_pressed_in_segment = False

        # First pass: check if steering was pressed
        msgs_cache = []
        for msg in lr:
            msgs_cache.append(msg)
            if msg.which() == "carState" and msg.carState.steeringPressed:
                steering_pressed_in_segment = True
                break

        if steering_pressed_in_segment:
            continue

        # Second pass: collect data
        for msg in msgs_cache:
            msg_type = msg.which()
            mono_time = msg.logMonoTime

            if base_time is None:
                base_time = mono_time

            if msg_type == "carState":
                car_state = msg.carState
            elif msg_type == "carOutput":
                car_output = msg.carOutput
            elif msg_type == "selfdriveState":
                selfdrive_state = msg.selfdriveState

            if msg_type == "carState" and all([car_state, car_output, selfdrive_state]):
                if not selfdrive_state.active or car_state.standstill:
                    continue

                t = (mono_time - base_time) / 1e9  # seconds
                desired = car_output.actuatorsOutput.steeringAngleDeg
                actual = car_state.steeringAngleDeg
                error = desired - actual
                speed = car_state.vEgo

                ts.timestamps.append(t)
                ts.desired_angles.append(desired)
                ts.actual_angles.append(actual)
                ts.errors.append(error)
                ts.speeds.append(speed)

    return ts


def generate_plots(stats: dict, title: str, output_prefix: str):
    """Generate plots for steering accuracy analysis."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # Plot 1: Average error by angle bin (all speeds combined)
    ax1 = axes[0, 0]
    all_angles = defaultdict(lambda: {'error': 0, 'count': 0})
    for group in SPEED_GROUPS:
        for angle_bin, s in stats[group.name].items():
            if s.count > 0:
                all_angles[angle_bin]['error'] += s.total_error
                all_angles[angle_bin]['count'] += s.count

    angles = sorted(all_angles.keys())
    avg_errors = [all_angles[a]['error'] / all_angles[a]['count'] for a in angles]
    counts = [all_angles[a]['count'] for a in angles]

    bars = ax1.bar(angles, avg_errors, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Desired Angle (degrees)')
    ax1.set_ylabel('Average Error (degrees)')
    ax1.set_title('Average Steering Error by Angle')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error by speed group
    ax2 = axes[0, 1]
    speed_names = []
    speed_errors = []
    speed_counts = []
    for group in SPEED_GROUPS:
        total_err = sum(s.total_error for s in stats[group.name].values())
        total_cnt = sum(s.count for s in stats[group.name].values())
        if total_cnt > 0:
            speed_names.append(group.name)
            speed_errors.append(total_err / total_cnt)
            speed_counts.append(total_cnt)

    if speed_names:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(speed_names)))
        bars = ax2.bar(speed_names, speed_errors, color=colors)
        ax2.set_xlabel('Speed Group')
        ax2.set_ylabel('Average Error (degrees)')
        ax2.set_title('Average Error by Speed')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Under/Over/Exact distribution
    ax3 = axes[1, 0]
    total_under = sum(s.undershoot_count for g in stats.values() for s in g.values())
    total_exact = sum(s.exact_count for g in stats.values() for s in g.values())
    total_over = sum(s.overshoot_count for g in stats.values() for s in g.values())
    total = total_under + total_exact + total_over

    if total > 0:
        labels = ['Undershoot', 'Exact', 'Overshoot']
        sizes = [total_under, total_exact, total_over]
        colors = ['#ff6b6b', '#51cf66', '#339af0']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Steering Response Distribution')

    # Plot 4: Sample count by angle (heatmap-style)
    ax4 = axes[1, 1]
    if angles:
        ax4.bar(angles, counts, color='coral', alpha=0.7)
        ax4.set_xlabel('Desired Angle (degrees)')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Data Distribution by Angle')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_stats.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_stats.png")
    plt.close()


def generate_time_series_plot(ts: TimeSeries, title: str, output_prefix: str):
    """Generate time series plot showing desired vs actual steering."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    if not ts.timestamps:
        print("No time series data to plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    t = np.array(ts.timestamps)
    desired = np.array(ts.desired_angles)
    actual = np.array(ts.actual_angles)
    error = np.array(ts.errors)
    speed = np.array(ts.speeds)

    # Plot 1: Desired vs Actual angle
    ax1 = axes[0]
    ax1.plot(t, desired, 'b-', label='Desired', alpha=0.8, linewidth=0.8)
    ax1.plot(t, actual, 'r-', label='Actual', alpha=0.6, linewidth=0.8)
    ax1.set_ylabel('Steering Angle (deg)')
    ax1.set_title('Desired vs Actual Steering Angle')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error over time
    ax2 = axes[1]
    ax2.plot(t, error, 'g-', alpha=0.7, linewidth=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(t, error, 0, alpha=0.3, color='green')
    ax2.set_ylabel('Error (deg)')
    ax2.set_title('Steering Error (Desired - Actual)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speed
    ax3 = axes[2]
    ax3.plot(t, speed * 3.6, 'purple', alpha=0.7, linewidth=0.8)  # Convert to km/h
    ax3.set_ylabel('Speed (km/h)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Vehicle Speed')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_timeseries.png")
    plt.close()


def generate_comparison_plot(stats_before: dict, stats_after: dict, output_prefix: str):
    """Generate comparison plot between before and after."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Steering Accuracy Comparison: Before vs After', fontsize=14)

    # Aggregate data
    def aggregate_by_angle(stats):
        result = defaultdict(lambda: {'error': 0, 'count': 0})
        for group in SPEED_GROUPS:
            for angle_bin, s in stats[group.name].items():
                if s.count > 0:
                    result[angle_bin]['error'] += s.total_error
                    result[angle_bin]['count'] += s.count
        return result

    before_by_angle = aggregate_by_angle(stats_before)
    after_by_angle = aggregate_by_angle(stats_after)
    all_angles = sorted(set(before_by_angle.keys()) | set(after_by_angle.keys()))

    # Plot 1: Side-by-side bar chart of average error by angle
    ax1 = axes[0, 0]
    x = np.arange(len(all_angles))
    width = 0.35

    before_errors = [before_by_angle[a]['error'] / before_by_angle[a]['count']
                     if before_by_angle[a]['count'] > 0 else 0 for a in all_angles]
    after_errors = [after_by_angle[a]['error'] / after_by_angle[a]['count']
                    if after_by_angle[a]['count'] > 0 else 0 for a in all_angles]

    ax1.bar(x - width/2, before_errors, width, label='Before', color='#ff6b6b', alpha=0.7)
    ax1.bar(x + width/2, after_errors, width, label='After', color='#51cf66', alpha=0.7)
    ax1.set_xlabel('Desired Angle (degrees)')
    ax1.set_ylabel('Average Error (degrees)')
    ax1.set_title('Error Comparison by Angle')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_angles)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement/regression by angle
    ax2 = axes[0, 1]
    changes = []
    valid_angles = []
    for a in all_angles:
        if before_by_angle[a]['count'] > 0 and after_by_angle[a]['count'] > 0:
            b_err = before_by_angle[a]['error'] / before_by_angle[a]['count']
            a_err = after_by_angle[a]['error'] / after_by_angle[a]['count']
            change = b_err - a_err  # positive = improvement
            changes.append(change)
            valid_angles.append(a)

    if valid_angles:
        colors = ['#51cf66' if c > 0 else '#ff6b6b' for c in changes]
        ax2.bar(valid_angles, changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Desired Angle (degrees)')
        ax2.set_ylabel('Improvement (degrees)')
        ax2.set_title('Error Change by Angle (positive = better)')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Error by speed group comparison
    ax3 = axes[1, 0]

    def get_speed_errors(stats):
        result = []
        for group in SPEED_GROUPS:
            total_err = sum(s.total_error for s in stats[group.name].values())
            total_cnt = sum(s.count for s in stats[group.name].values())
            result.append(total_err / total_cnt if total_cnt > 0 else 0)
        return result

    speed_names = [g.name for g in SPEED_GROUPS]
    before_speed_errors = get_speed_errors(stats_before)
    after_speed_errors = get_speed_errors(stats_after)

    x = np.arange(len(speed_names))
    ax3.bar(x - width/2, before_speed_errors, width, label='Before', color='#ff6b6b', alpha=0.7)
    ax3.bar(x + width/2, after_speed_errors, width, label='After', color='#51cf66', alpha=0.7)
    ax3.set_xlabel('Speed Group')
    ax3.set_ylabel('Average Error (degrees)')
    ax3.set_title('Error by Speed Group')
    ax3.set_xticks(x)
    ax3.set_xticklabels(speed_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overall summary
    ax4 = axes[1, 1]
    total_before = sum(s.count for g in stats_before.values() for s in g.values())
    total_after = sum(s.count for g in stats_after.values() for s in g.values())
    err_before = sum(s.total_error for g in stats_before.values() for s in g.values())
    err_after = sum(s.total_error for g in stats_after.values() for s in g.values())

    avg_before = err_before / total_before if total_before > 0 else 0
    avg_after = err_after / total_after if total_after > 0 else 0

    labels = ['Before', 'After']
    values = [avg_before, avg_after]
    colors = ['#ff6b6b', '#51cf66']
    bars = ax4.bar(labels, values, color=colors, alpha=0.7)

    # Add percentage change annotation
    if avg_before > 0:
        pct_change = ((avg_before - avg_after) / avg_before) * 100
        change_text = f'{pct_change:+.1f}%' if pct_change != 0 else '0%'
        result_text = 'IMPROVED' if pct_change > 0 else ('REGRESSED' if pct_change < 0 else 'NO CHANGE')
        color = '#51cf66' if pct_change > 0 else ('#ff6b6b' if pct_change < 0 else 'gray')
        ax4.annotate(f'{result_text}\n{change_text}',
                     xy=(0.5, max(values) * 0.5),
                     ha='center', fontsize=14, fontweight='bold', color=color)

    ax4.set_ylabel('Average Error (degrees)')
    ax4.set_title('Overall Average Error')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax4.annotate(f'{val:.3f}°',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_comparison.png")
    plt.close()


def analyze_route(route: str, segments: list[int]) -> dict:
    """Analyze a route across specified segments."""
    print(f"\nAnalyzing route: {route}")
    print(f"Segments: {segments[0]}-{segments[-1]}")

    results = []
    for seg in segments:
        result = analyze_segment(route, seg)
        results.append(result)

    valid_segments = sum(1 for r in results if r.valid)
    print(f"\nValid segments: {valid_segments}/{len(segments)}")

    return merge_stats(results)


def parse_segments(seg_str: str) -> list[int]:
    """Parse segment range string like '7-13' into list of ints."""
    if '-' in seg_str:
        start, end = seg_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(seg_str)]


def main():
    parser = argparse.ArgumentParser(description='Steering accuracy measurement tool for angular steering')
    parser.add_argument('--route', help='Single route to analyze')
    parser.add_argument('--before', help='Route to use as baseline (for comparison)')
    parser.add_argument('--after', help='Route to compare against baseline')
    parser.add_argument('--segments', required=True, help='Segment range to analyze (e.g., "7-13" or "5")')
    parser.add_argument('--cache', action='store_true', help='Use cached data (for CI)')
    parser.add_argument('--plot', action='store_true', help='Generate plots (requires matplotlib)')
    parser.add_argument('--output-dir', default='.', help='Directory for plot output (default: current directory)')
    args = parser.parse_args()

    if args.cache:
        os.environ['FILEREADER_CACHE'] = '1'

    segments = parse_segments(args.segments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.route:
        # Single route analysis
        stats = analyze_route(args.route, segments)
        print_stats(stats, f"Steering Accuracy: {args.route}")

        if args.plot:
            route_name = args.route.replace('/', '_')
            output_prefix = str(output_dir / f"steering_{route_name}")
            generate_plots(stats, f"Steering Accuracy: {args.route}", output_prefix)

            # Also generate time series
            print("\nCollecting time series data for plotting...")
            ts = collect_time_series(args.route, segments)
            generate_time_series_plot(ts, f"Steering Time Series: {args.route}", output_prefix)

    elif args.before and args.after:
        # Comparison mode
        stats_before = analyze_route(args.before, segments)
        stats_after = analyze_route(args.after, segments)

        total_before, avg_before = print_stats(stats_before, f"BEFORE: {args.before}")
        total_after, avg_after = print_stats(stats_after, f"AFTER: {args.after}")

        print_comparison(stats_before, stats_after)

        # Summary
        print(f"\n{'=' * 100}")
        print(f"  SUMMARY")
        print(f"{'=' * 100}")
        print(f"  Before: {avg_before:.3f}° avg error ({total_before} samples)")
        print(f"  After:  {avg_after:.3f}° avg error ({total_after} samples)")
        if total_before > 0 and total_after > 0:
            improvement = avg_before - avg_after
            pct = (improvement / avg_before) * 100 if avg_before > 0 else 0
            if improvement > 0:
                print(f"  Result: {improvement:.3f}° improvement ({pct:.1f}% better)")
            else:
                print(f"  Result: {-improvement:.3f}° regression ({-pct:.1f}% worse)")
        print(f"{'=' * 100}\n")

        if args.plot:
            output_prefix = str(output_dir / "steering_comparison")

            # Generate individual stats plots
            before_name = args.before.replace('/', '_')
            after_name = args.after.replace('/', '_')
            generate_plots(stats_before, f"BEFORE: {args.before}", str(output_dir / f"steering_{before_name}"))
            generate_plots(stats_after, f"AFTER: {args.after}", str(output_dir / f"steering_{after_name}"))

            # Generate comparison plot
            generate_comparison_plot(stats_before, stats_after, output_prefix)

            # Generate time series for both
            print("\nCollecting time series data for plotting...")
            ts_before = collect_time_series(args.before, segments)
            ts_after = collect_time_series(args.after, segments)
            generate_time_series_plot(ts_before, f"BEFORE Time Series: {args.before}", str(output_dir / f"steering_{before_name}"))
            generate_time_series_plot(ts_after, f"AFTER Time Series: {args.after}", str(output_dir / f"steering_{after_name}"))

    else:
        parser.error("Must specify either --route or both --before and --after")


if __name__ == "__main__":
    main()
