#!/usr/bin/env bash
set -euo pipefail
commit=$1
[[ "$commit" =~ ^[0-9a-f]{40}$ ]]
session_file="/data/chestnut-ci-workspace/session-$commit.pid"
[[ -f "$session_file" ]] || exit 0
group=$(cat "$session_file")
[[ "$group" =~ ^[0-9]+$ && "$group" -gt 1 ]]
# Verify the timeout supervisor still belongs to this exact build before killing.
command=$(ps -o args= -p "$group") || exit 0
[[ "$command" == *"timeout --signal=TERM --kill-after=30s 3600 bash -s"* && "$command" == *"$commit"* ]]
kill -TERM -- "-$group" || true
for _ in {1..10}; do
  if ! kill -0 -- "-$group" 2>/dev/null; then
    rm -f "$session_file"
    exit 0
  fi
  sleep 1
done
kill -KILL -- "-$group" || true
rm -f "$session_file"
