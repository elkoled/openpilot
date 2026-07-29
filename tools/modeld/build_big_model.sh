#!/usr/bin/env bash
set -e

device="${1:?usage: $0 comma@device [output directory]}"
output="${2:-big-model}"
remote="${OPENPILOT_REMOTE_DIR:-/data/openpilot}"
model="openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl"

mkdir "$output"
ssh "$device" "cd $remote && BUILD_BIG_MODEL=1 scons $model.chunkmanifest"
scp "$device:$remote/$model.chunk*" "$output/"
