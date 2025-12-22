#!/usr/bin/env python3
"""
Steering Accuracy Test - Measures how well the car follows steering commands.

Automatically finds valid driving segments (no steering override, active control).
Results are normalized for fair comparison across different routes/conditions.

Usage:
  python .github/test_steering_accuracy.py ROUTE_ID
  python .github/test_steering_accuracy.py ROUTE1 ROUTE2  # compare two routes
  python .github/test_steering_accuracy.py --plot ROUTE_ID  # generate plots
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openpilot.tools.lib.logreader import LogReader, openpilotci_source, comma_api_source

# Minimum samples required for valid analysis
MIN_SAMPLES = 1000
# Skip first N frames of each segment (warmup)
WARMUP_FRAMES = 50


def load_segment(route: str, seg: int) -> list:
    """Load messages from a single segment."""
    try:
        lr = LogReader(f"{route}/{seg}", sources=[openpilotci_source, comma_api_source], sort_by_time=True)
        return list(lr)
    except Exception:
        return []


def analyze_route(route: str) -> dict:
    """Analyze steering accuracy for a route. Auto-detects valid segments."""
    print(f"Analyzing: {route}")

    errors = []
    speeds = []
    angles = []

    seg = 0
    segments_used = 0
    segments_skipped_override = 0
    segments_skipped_inactive = 0

    while True:
        msgs = load_segment(route, seg)
        if not msgs:
            if seg > 0:
                break  # End of route
            seg += 1
            if seg > 5:  # Give up after 5 failed attempts
                break
            continue

        # Collect segment data
        car_state = None
        car_output = None
        selfdrive_state = None
        frame = 0
        seg_errors = []
        seg_speeds = []
        seg_angles = []
        steering_pressed_frames = 0
        total_frames = 0

        for msg in msgs:
            msg_type = msg.which()

            if msg_type == "carState":
                car_state = msg.carState
                total_frames += 1
                if car_state.steeringPressed:
                    steering_pressed_frames += 1

            elif msg_type == "carOutput":
                car_output = msg.carOutput

            elif msg_type == "selfdriveState":
                selfdrive_state = msg.selfdriveState

            # Process on carState updates when we have all data
            if msg_type == "carState" and car_state and car_output and selfdrive_state:
                frame += 1
                if frame < WARMUP_FRAMES:
                    continue

                # Skip frames with steering override
                if car_state.steeringPressed:
                    continue

                # Only use active, non-standstill frames
                if not selfdrive_state.active or car_state.standstill:
                    continue

                desired = car_output.actuatorsOutput.steeringAngleDeg
                actual = car_state.steeringAngleDeg
                speed = car_state.vEgo

                # Skip very low speed (unreliable)
                if speed < 3.0:
                    continue

                seg_errors.append(desired - actual)
                seg_speeds.append(speed)
                seg_angles.append(desired)

        # Decide whether to use this segment
        # Skip if >50% of frames had steering override
        override_ratio = steering_pressed_frames / max(total_frames, 1)
        if override_ratio > 0.5:
            segments_skipped_override += 1
        elif len(seg_errors) < 100:
            segments_skipped_inactive += 1
        else:
            errors.extend(seg_errors)
            speeds.extend(seg_speeds)
            angles.extend(seg_angles)
            segments_used += 1

        seg += 1

    if segments_used == 0:
        print("  No valid segments found!")
        print(f"  Skipped: {segments_skipped_override} override, {segments_skipped_inactive} inactive")
        return None

    print(f"  Segments: {segments_used} used, {segments_skipped_override} override, {segments_skipped_inactive} inactive")
    print(f"  Samples: {len(errors)}")

    if len(errors) < MIN_SAMPLES:
        print(f"  Not enough samples ({len(errors)} < {MIN_SAMPLES})")
        return None

    errors = np.array(errors)
    speeds = np.array(speeds)
    angles = np.array(angles)

    # Compute statistics
    abs_errors = np.abs(errors)

    return {
        "route": route,
        "samples": len(errors),
        "segments": segments_used,
        # Error statistics
        "mean_error": float(np.mean(errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "p95_abs_error": float(np.percentile(abs_errors, 95)),
        "max_abs_error": float(np.max(abs_errors)),
        "std_error": float(np.std(errors)),
        # Breakdown by condition
        "error_low_speed": float(np.mean(np.abs(errors[speeds < 15]))) if np.sum(speeds < 15) > 100 else None,
        "error_high_speed": float(np.mean(np.abs(errors[speeds >= 15]))) if np.sum(speeds >= 15) > 100 else None,
        "error_small_angle": float(np.mean(np.abs(errors[np.abs(angles) < 5]))) if np.sum(np.abs(angles) < 5) > 100 else None,
        "error_large_angle": float(np.mean(np.abs(errors[np.abs(angles) >= 5]))) if np.sum(np.abs(angles) >= 5) > 100 else None,
        # Speed/angle stats
        "avg_speed": float(np.mean(speeds)),
        "avg_abs_angle": float(np.mean(np.abs(angles))),
        # Raw data for plotting
        "_errors": errors,
        "_speeds": speeds,
        "_angles": angles,
    }


def print_results(results: dict, title: str = None):
    """Print formatted results."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    print(f"\n  Route: {results['route']}")
    print(f"  Samples: {results['samples']:,} ({results['segments']} segments)")
    print(f"  Avg speed: {results['avg_speed'] * 3.6:.1f} km/h")
    print(f"  Avg |angle|: {results['avg_abs_angle']:.1f}°")

    print("\n  Steering Error:")
    print(f"    Mean:     {results['mean_abs_error']:.3f}°")
    print(f"    Median:   {results['median_abs_error']:.3f}°")
    print(f"    95th %:   {results['p95_abs_error']:.3f}°")
    print(f"    Max:      {results['max_abs_error']:.2f}°")
    print(f"    Std dev:  {results['std_error']:.3f}°")
    print(f"    Bias:     {results['mean_error']:+.3f}°")

    if results['error_low_speed'] or results['error_high_speed']:
        print("\n  By Speed:")
        if results['error_low_speed']:
            print(f"    <54 km/h:  {results['error_low_speed']:.3f}°")
        if results['error_high_speed']:
            print(f"    ≥54 km/h:  {results['error_high_speed']:.3f}°")

    if results['error_small_angle'] or results['error_large_angle']:
        print("\n  By Angle:")
        if results['error_small_angle']:
            print(f"    <5°:       {results['error_small_angle']:.3f}°")
        if results['error_large_angle']:
            print(f"    ≥5°:       {results['error_large_angle']:.3f}°")


def print_comparison(r1: dict, r2: dict):
    """Print comparison between two routes."""
    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")

    def compare(name, key, lower_better=True):
        v1, v2 = r1[key], r2[key]
        if v1 is None or v2 is None:
            return
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        better = diff < 0 if lower_better else diff > 0
        symbol = "✓" if better else "✗" if abs(pct) > 5 else ""
        print(f"  {name:12s}  {v1:6.3f}° → {v2:6.3f}°  ({diff:+.3f}°, {pct:+.1f}%) {symbol}")

    print(f"\n  Route 1: {r1['route']}")
    print(f"  Route 2: {r2['route']}")
    print()

    compare("Mean", "mean_abs_error")
    compare("Median", "median_abs_error")
    compare("95th %", "p95_abs_error")
    compare("Bias", "mean_error", lower_better=False)

    # Overall verdict
    improvement = r1['mean_abs_error'] - r2['mean_abs_error']
    pct = (improvement / r1['mean_abs_error'] * 100) if r1['mean_abs_error'] != 0 else 0

    print(f"\n  {'─' * 56}")
    if abs(pct) < 3:
        print(f"  Result: No significant change ({pct:+.1f}%)")
    elif improvement > 0:
        print(f"  Result: IMPROVED by {improvement:.3f}° ({pct:.1f}% better)")
    else:
        print(f"  Result: REGRESSED by {-improvement:.3f}° ({-pct:.1f}% worse)")


def generate_plots(results: dict, output_prefix: str):
    """Generate analysis plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    errors = results['_errors']
    speeds = results['_speeds']
    angles = results['_angles']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Steering Accuracy: {results['route']}", fontsize=12)

    # Error histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (mean={results["mean_error"]:.3f}°)')

    # Error vs Speed
    ax = axes[0, 1]
    ax.scatter(speeds * 3.6, np.abs(errors), alpha=0.1, s=1)
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('|Error| (degrees)')
    ax.set_title('Error vs Speed')
    ax.set_ylim(0, min(5, results['p95_abs_error'] * 2))

    # Error vs Desired Angle
    ax = axes[1, 0]
    ax.scatter(angles, errors, alpha=0.1, s=1)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Desired Angle (degrees)')
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Error vs Desired Angle')

    # Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Samples: {results['samples']:,}
    Segments: {results['segments']}

    Mean |Error|: {results['mean_abs_error']:.3f}°
    Median |Error|: {results['median_abs_error']:.3f}°
    95th percentile: {results['p95_abs_error']:.3f}°
    Max |Error|: {results['max_abs_error']:.2f}°

    Bias (mean error): {results['mean_error']:+.3f}°
    Std deviation: {results['std_error']:.3f}°

    Avg Speed: {results['avg_speed'] * 3.6:.1f} km/h
    Avg |Angle|: {results['avg_abs_angle']:.1f}°
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150)
    print(f"Saved: {output_prefix}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Steering accuracy analysis')
    parser.add_argument('routes', nargs='+', help='Route ID(s) to analyze (1 or 2)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--cache', action='store_true', help='Use cached data')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    args = parser.parse_args()

    if args.cache:
        os.environ['FILEREADER_CACHE'] = '1'

    if len(args.routes) > 2:
        parser.error("Maximum 2 routes for comparison")

    # Analyze routes
    results = []
    for route in args.routes:
        r = analyze_route(route)
        if r is None:
            print(f"\nFailed to analyze {route}")
            sys.exit(1)
        results.append(r)

    # Print results
    if len(results) == 1:
        print_results(results[0], "STEERING ACCURACY")
    else:
        print_results(results[0], "ROUTE 1")
        print_results(results[1], "ROUTE 2")
        print_comparison(results[0], results[1])

    # Generate plots
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            route_name = r['route'].replace('/', '_')
            generate_plots(r, str(output_dir / f"steering_{route_name}"))

    print()


if __name__ == "__main__":
    main()
