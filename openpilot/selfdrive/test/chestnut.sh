#!/usr/bin/env bash
set -e

test -f /data/disable_openpilot_autostart
test "$(cat /data/params/d/IsOffroad)" = 1
python -c 'from openpilot.common.hardware import HARDWARE; from openpilot.selfdrive.modeld.helpers import chestnut_present; assert HARDWARE.get_device_type() == "mici" and chestnut_present()'

MODEL=openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl
trap 'rm -f "$MODEL"*' EXIT
rm -f "$MODEL.chunkmanifest"
scons -j2 --cache-disable "$MODEL.chunkmanifest"
test -s "$MODEL.chunkmanifest"
