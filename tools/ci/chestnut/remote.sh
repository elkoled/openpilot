#!/usr/bin/env bash
set -euo pipefail

expected_host=$1
commit=$2
repository=$3
[[ "$expected_host" == comma-de2e7866 && "$commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$(hostname)" == "$expected_host" ]]
[[ -f /AGNOS && -f /data/disable_openpilot_autostart ]]
[[ ! -f /data/params/d/IsOnroad || "$(cat /data/params/d/IsOnroad)" == 0 ]]
if pgrep -f 'manager\.py|selfdrive\.modeld\.modeld|sunnypilot\.modeld_v2\.modeld' >/dev/null; then
  echo 'Normal openpilot processes are running; provision this as an idle CI bench first.'
  exit 1
fi

# Also excludes accidental overlap with a second SSH invocation on the same bench.
exec 9>/data/chestnut-ci.lock
flock -n 9
export GIT_LFS_SKIP_SMUDGE=1
export CI=1 PYTHONUNBUFFERED=1
export DEV=USB+AMD:LLVM FRAME_DEV=CPU FLOAT16=1 JIT_BATCH_SIZE=0 GMMU=0 TC_OPT=2 TC_MIN_GLOBALS=32
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="/usr/local/bin:$PATH"

mkdir -p /data/chestnut-ci-workspace
cd /data/chestnut-ci-workspace
session_file="/data/chestnut-ci-workspace/session-$commit.pid"
ps -o pgid= -p $$ | tr -d ' ' > "$session_file"
trap 'rm -f "$session_file"; if [[ -n "${CHESTNUT_RESULTS:-}" ]]; then rm -f "$CHESTNUT_RESULTS/big_driving_tinygrad.pkl"; fi' EXIT
if [[ ! -d source ]]; then
  git init source
  git -C source remote add origin "$repository"
fi
cd source
[[ "$(git remote get-url origin)" == "$repository" ]]
git fetch --depth=1 --no-tags --no-recurse-submodules origin "$commit"
git checkout --detach --force "$commit"
[[ "$(git rev-parse HEAD)" == "$commit" ]]
git submodule sync -- tinygrad_repo
git submodule update --init --depth=1 -- tinygrad_repo
ln -sfn tinygrad_repo/tinygrad tinygrad
export PYTHONPATH="$PWD"
export CHESTNUT_RESULTS="/data/chestnut-ci-workspace/reports/$commit"
mkdir -p "$CHESTNUT_RESULTS"
rm -f "$CHESTNUT_RESULTS"/*.xml "$CHESTNUT_RESULTS"/*.log "$CHESTNUT_RESULTS"/*.json
git rev-parse HEAD > "$CHESTNUT_RESULTS/commit.txt"
hostname > "$CHESTNUT_RESULTS/host.txt"

# Synthetic model CI needs only this model, not fleet data or the full LFS tree.
python3 tools/ci/chestnut/run.py preflight
git -c lfs.concurrenttransfers=1 -c lfs.transfer.maxretries=1 lfs pull \
  --include='openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx*' --exclude=''
python3 tools/ci/chestnut/run.py compile
python3 tools/ci/chestnut/run.py smoke --runs 20
