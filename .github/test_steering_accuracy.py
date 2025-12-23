#!/usr/bin/env python3
"""
Steering Accuracy Test - Measures how well the car follows steering commands.

Automatically finds valid driving segments (no steering override, active control).

Usage:
  python test_steering_accuracy.py ROUTE_ID                    # analyze single route
  python test_steering_accuracy.py ROUTE1 ROUTE2               # compare two routes
  python test_steering_accuracy.py --from-yaml                 # test all routes from test_routes.yaml
  python test_steering_accuracy.py --plot ROUTE_ID             # generate plots
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openpilot.tools.lib.logreader import LogReader, openpilotci_source, comma_api_source

# Minimum valid samples needed for analysis
MIN_SAMPLES = 500
# Frames to skip at start of each segment
WARMUP_FRAMES = 50
# Minimum speed to consider (m/s)
MIN_SPEED = 3.0


def load_segment(route: str, seg: int, retries: int = 2) -> list:
    """Load a single segment with retry logic, returns empty list on failure."""
    import time
    for attempt in range(retries + 1):
        try:
            lr = LogReader(f"{route}/{seg}", sources=[openpilotci_source, comma_api_source], sort_by_time=True)
            return list(lr)
        except Exception as e:
            err_str = str(e).lower()
            # Check if this is a genuine "not found" vs a transient network error
            # LogsUnavailable can be thrown for both cases - inspect nested exception
            is_not_found = (
                "indexerror" in err_str or  # segment doesn't exist
                "segment range is not valid" in err_str or  # invalid segment
                ("not found" in err_str and "connectionerror" not in err_str and "maxretry" not in err_str)
            )
            if is_not_found:
                return []
            # Transient error - retry
            if attempt < retries:
                time.sleep(0.5)
                continue
            print(f"  Warning: segment {seg} failed after {retries + 1} attempts: {type(e).__name__}")
            return []
    return []


def analyze_route(route: str) -> dict | None:
    """Analyze steering accuracy for a route."""
    print(f"Analyzing: {route}")

    all_errors = []
    all_speeds = []
    all_angles = []
    segments_used = 0
    seg = 0
    consecutive_empty = 0

    while consecutive_empty < 3:
        msgs = load_segment(route, seg)
        seg += 1

        if not msgs:
            consecutive_empty += 1
            continue
        consecutive_empty = 0

        # Parse segment
        car_state = car_output = selfdrive_state = None
        seg_data = []
        frame = 0

        for msg in msgs:
            t = msg.which()
            if t == "carState":
                car_state = msg.carState
            elif t == "carOutput":
                car_output = msg.carOutput
            elif t == "selfdriveState":
                selfdrive_state = msg.selfdriveState

            # Collect data on carState updates
            if t == "carState" and car_state and car_output and selfdrive_state:
                frame += 1
                if frame < WARMUP_FRAMES:
                    continue

                # Skip: steering override, not active, standstill, low speed
                if car_state.steeringPressed:
                    continue
                if not selfdrive_state.active or car_state.standstill:
                    continue
                if car_state.vEgo < MIN_SPEED:
                    continue

                desired = car_output.actuatorsOutput.steeringAngleDeg
                actual = car_state.steeringAngleDeg
                seg_data.append((desired - actual, car_state.vEgo, desired))

        # Use segment if enough valid data
        if len(seg_data) >= 100:
            for err, spd, ang in seg_data:
                all_errors.append(err)
                all_speeds.append(spd)
                all_angles.append(ang)
            segments_used += 1

    if segments_used == 0 or len(all_errors) < MIN_SAMPLES:
        print(f"  Insufficient data: {len(all_errors)} samples from {segments_used} segments")
        return None

    errors = np.array(all_errors)
    speeds = np.array(all_speeds)
    angles = np.array(all_angles)
    abs_errors = np.abs(errors)

    print(f"  {len(errors):,} samples from {segments_used} segments")

    return {
        "route": route,
        "samples": len(errors),
        "segments": segments_used,
        "mean_error": float(np.mean(errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "p95_abs_error": float(np.percentile(abs_errors, 95)),
        "max_abs_error": float(np.max(abs_errors)),
        "std_error": float(np.std(errors)),
        "avg_speed_kmh": float(np.mean(speeds) * 3.6),
        "avg_abs_angle": float(np.mean(np.abs(angles))),
        "_errors": errors,
        "_speeds": speeds,
        "_angles": angles,
    }


def print_results(r: dict, title: str = None):
    """Print formatted results."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    print(f"\n  Route: {r['route']}")
    print(f"  Samples: {r['samples']:,} ({r['segments']} segments)")
    print(f"  Avg speed: {r['avg_speed_kmh']:.1f} km/h, Avg |angle|: {r['avg_abs_angle']:.1f}°")

    print("\n  Steering Error:")
    print(f"    Mean |error|: {r['mean_abs_error']:.3f}°")
    print(f"    Median:       {r['median_abs_error']:.3f}°")
    print(f"    95th %:       {r['p95_abs_error']:.3f}°")
    print(f"    Max:          {r['max_abs_error']:.2f}°")
    print(f"    Bias:         {r['mean_error']:+.3f}°")


def print_comparison(r1: dict, r2: dict):
    """Print comparison between two routes."""
    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  {r1['route']}")
    print("  vs")
    print(f"  {r2['route']}")

    def cmp(name, key):
        v1, v2 = r1[key], r2[key]
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        sym = "better" if diff < 0 else "worse" if diff > 0 else ""
        print(f"  {name:12s}  {v1:.3f}° → {v2:.3f}°  ({diff:+.3f}°, {pct:+.1f}%) {sym}")

    print()
    cmp("Mean", "mean_abs_error")
    cmp("Median", "median_abs_error")
    cmp("95th %", "p95_abs_error")

    imp = r1['mean_abs_error'] - r2['mean_abs_error']
    pct = (imp / r1['mean_abs_error'] * 100) if r1['mean_abs_error'] != 0 else 0
    print(f"\n  {'─' * 54}")
    if abs(pct) < 5:
        print(f"  Result: No significant change ({pct:+.1f}%)")
    elif imp > 0:
        print(f"  Result: IMPROVED by {imp:.3f}° ({pct:.1f}% better)")
    else:
        print(f"  Result: WORSE by {-imp:.3f}° ({-pct:.1f}%)")


def print_summary_table(results: list[dict]):
    """Print a summary table of all routes."""
    print(f"\n{'=' * 80}")
    print("  STEERING ACCURACY SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Route':<45} {'Mean':>8} {'Median':>8} {'95th%':>8} {'Samples':>10}")
    print(f"  {'-' * 77}")

    for r in results:
        name = r['route'].split('/')[-1][:44]
        print(f"  {name:<45} {r['mean_abs_error']:>7.3f}° {r['median_abs_error']:>7.3f}° " +
              f"{r['p95_abs_error']:>7.3f}° {r['samples']:>10,}")

    print(f"  {'-' * 77}")


def generate_plots(r: dict, output_prefix: str):
    """Generate analysis plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    errors = r['_errors']
    speeds = r['_speeds']
    angles = r['_angles']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Steering Accuracy: {r['route']}", fontsize=12)

    # Error histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Error (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (bias={r["mean_error"]:.3f}°)')

    # Error vs Speed
    ax = axes[0, 1]
    ax.scatter(speeds * 3.6, np.abs(errors), alpha=0.1, s=1)
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('|Error| (°)')
    ax.set_title('Error vs Speed')

    # Error vs Angle
    ax = axes[1, 0]
    ax.scatter(angles, errors, alpha=0.1, s=1)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Desired Angle (°)')
    ax.set_ylabel('Error (°)')
    ax.set_title('Error vs Desired Angle')

    # Stats
    ax = axes[1, 1]
    ax.axis('off')
    txt = f"""
    Samples: {r['samples']:,}
    Segments: {r['segments']}

    Mean |Error|: {r['mean_abs_error']:.3f}°
    Median: {r['median_abs_error']:.3f}°
    95th %: {r['p95_abs_error']:.3f}°
    Max: {r['max_abs_error']:.2f}°
    Bias: {r['mean_error']:+.3f}°

    Avg Speed: {r['avg_speed_kmh']:.1f} km/h
    """
    ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=10, va='top', family='monospace')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150)
    print(f"Saved: {output_prefix}.png")
    plt.close()


def load_routes_from_yaml() -> list[tuple[str, str]]:
    """Load routes from test_routes.yaml. Returns list of (route, description)."""
    yaml_path = Path(__file__).parent / "test_routes.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    routes = []
    for route_list in config.values():
        for entry in route_list:
            routes.append((entry['route'], entry.get('description', '')))
    return routes


def main():
    parser = argparse.ArgumentParser(description='Steering accuracy analysis')
    parser.add_argument('routes', nargs='*', help='Route ID(s) to analyze')
    parser.add_argument('--from-yaml', action='store_true', help='Test all routes from test_routes.yaml')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--cache', action='store_true', help='Use cached data')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    args = parser.parse_args()

    if args.cache:
        os.environ['FILEREADER_CACHE'] = '1'

    # Get routes to analyze
    if args.from_yaml:
        route_entries = load_routes_from_yaml()
        routes = [r[0] for r in route_entries]
        descriptions = {r[0]: r[1] for r in route_entries}
    elif args.routes:
        routes = args.routes
        descriptions = {}
    else:
        parser.error("Provide route(s) or use --from-yaml")

    if len(routes) > 2 and not args.from_yaml:
        parser.error("Max 2 routes for comparison (or use --from-yaml for batch)")

    # Analyze routes
    results = []
    for route in routes:
        desc = descriptions.get(route, '')
        if desc:
            print(f"\n--- {desc} ---")
        r = analyze_route(route)
        if r:
            results.append(r)

    if not results:
        print("\nNo routes analyzed successfully")
        if not args.from_yaml:
            sys.exit(1)
        return

    # Output
    if args.from_yaml:
        print_summary_table(results)
    elif len(results) == 1:
        print_results(results[0], "STEERING ACCURACY")
    else:
        print_results(results[0], "ROUTE 1")
        print_results(results[1], "ROUTE 2")
        print_comparison(results[0], results[1])

    # Plots
    if args.plot:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            name = r['route'].replace('/', '_')
            generate_plots(r, str(out / f"steering_{name}"))

    print()


if __name__ == "__main__":
    main()
