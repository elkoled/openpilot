#!/usr/bin/env bash
# compares every safety RX and TX result and the safety state on the car test routes, between BASE's opendbc and this one
set -e

cd "$(dirname "$0")/../../../.."
BASE=${1:-HEAD^}

base=$(git rev-parse "$BASE:opendbc_repo")
new=$(git rev-parse HEAD:opendbc_repo)
if [ "$base" = "$new" ]; then
  echo "opendbc is unchanged ($new), nothing to compare"
  exit 0
fi
echo "comparing opendbc $base -> $new"

W=$(mktemp -d)
trap 'git -C opendbc_repo worktree remove --force "$W/base" 2>/dev/null; rm -rf "$W"' EXIT
git -C opendbc_repo fetch -q --depth=1 origin "$base"
git -C opendbc_repo worktree add -q --detach "$W/base" "$base"
# the base's safety code, built like this opendbc builds libsafety
cp opendbc_repo/opendbc/safety/tests/libsafety/libsafety_py.py "$W/base/opendbc/safety/tests/libsafety/"

REPLAY=opendbc_repo/opendbc/safety/tests/safety_replay/route_replay.py
python3 "$REPLAY" dump "$W/routes"
PYTHONPATH="$W/base" python3 "$REPLAY" digest "$W/routes" "$W/base.json"
python3 "$REPLAY" digest "$W/routes" "$W/new.json"
python3 "$REPLAY" compare "$W/base.json" "$W/new.json"
