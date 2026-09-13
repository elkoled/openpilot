#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
cc -std=c11 -Wall -Wextra -Werror -pedantic -I"$here" \
  "$here/protocol.c" "$here/tests/test_protocol.c" -o "$build_dir/test_protocol"
"$build_dir/test_protocol"
