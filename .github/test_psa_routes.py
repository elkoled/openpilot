#!/usr/bin/env python3
"""
PSA Route Tests for CI
Tests routes from test_routes.yaml for hard CAN errors.
Skips warmup period to avoid false positives from missing messages at start.

Usage:
  pytest .github/test_psa_routes.py -v
  pytest .github/test_psa_routes.py -v -k "PSA_PEUGEOT_208"
"""
import copy
import sys
from pathlib import Path

import pytest
import yaml

# Ensure repo root is in path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import FRAME_FINGERPRINT, interfaces
from opendbc.car.psa.values import CAR as PSA
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader, openpilotci_source, comma_api_source

# Platform mapping
PLATFORMS = {
    "PSA_PEUGEOT_208": PSA.PSA_PEUGEOT_208,
    "PSA_PEUGEOT_508": PSA.PSA_PEUGEOT_508,
}

# Skip first N frames (CAN init period - avoids false errors from missing messages)
WARMUP_FRAMES = 100

# Consecutive canValid=False that triggers immediate failure (real CAN error)
CONSECUTIVE_ERROR_THRESHOLD = 5


def load_test_routes():
    """Load routes from YAML and generate pytest parameters."""
    config_path = REPO_ROOT / ".github" / "test_routes.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    test_cases = []
    for platform_name, routes in config.items():
        if platform_name not in PLATFORMS:
            continue
        for route_info in routes:
            route = route_info["route"]
            desc = route_info.get("description", route)
            test_cases.append(pytest.param(
                platform_name, route,
                id=f"{platform_name}-{desc}"
            ))
    return test_cases


def get_route_data(route: str):
    """Load CAN messages and fingerprint from route."""
    # Try segments 0, 1, 2 (like test_models.py)
    test_segs = (0, 1, 2)
    last_error = None

    for seg in test_segs:
        try:
            segment_range = f"{route}/{seg}"
            lr = LogReader(segment_range, sources=[openpilotci_source, comma_api_source], sort_by_time=True)
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise Exception(f"Could not load any segment for {route}: {last_error}")

    can_msgs = []
    fingerprint = gen_empty_fingerprint()
    car_fw = []

    for msg in lr:
        if msg.which() == "can":
            can = can_capnp_to_list((msg.as_builder().to_bytes(),))[0]
            can_msgs.append((can[0], [CanData(*c) for c in can[1]]))

            # Build fingerprint from first frames
            if len(can_msgs) <= FRAME_FINGERPRINT:
                for m in msg.can:
                    if m.src < 64:
                        fingerprint[m.src][m.address] = len(m.dat)

        elif msg.which() == "carParams":
            car_fw = msg.carParams.carFw

    return can_msgs, fingerprint, car_fw


@pytest.mark.parametrize("platform_name,route", load_test_routes())
def test_route(platform_name: str, route: str):
    """
    Test a single route for hard CAN errors.

    Only counts errors after warmup period to avoid false positives
    from missing CAN messages at the start of the route.
    """
    platform = PLATFORMS[platform_name]

    # Load route data
    can_msgs, fingerprint, car_fw = get_route_data(route)
    assert len(can_msgs) > WARMUP_FRAMES + 100, f"Not enough CAN data: {len(can_msgs)} frames"

    # Initialize car interface
    CarInterface = interfaces[platform]
    CP = CarInterface.get_params(platform, fingerprint, car_fw, False, False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, platform, fingerprint, car_fw, False, False, docs=False)

    assert CP.carFingerprint == platform, f"Fingerprint mismatch: {CP.carFingerprint} != {platform}"

    CI = CarInterface(CP, copy.deepcopy(CP_SP))
    CC = structs.CarControl().as_reader()
    CC_SP = structs.CarControlSP()

    consecutive_invalid = 0

    for i, msg in enumerate(can_msgs):
        CS, _ = CI.update(msg)
        CI.apply(CC, CC_SP, msg[0])

        # Skip warmup period
        if i < WARMUP_FRAMES:
            continue

        # Fail immediately on sustained canValid=False (real CAN error)
        if not CS.canValid:
            consecutive_invalid += 1
            if consecutive_invalid >= CONSECUTIVE_ERROR_THRESHOLD:
                pytest.fail(
                    f"CAN ERROR at frame {i}: {CONSECUTIVE_ERROR_THRESHOLD} consecutive "
                    f"canValid=False (would show CAN error on device)"
                )
        else:
            consecutive_invalid = 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
