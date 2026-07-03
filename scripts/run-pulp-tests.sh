#!/usr/bin/env bash
# Builds bin/cbc, runs PuLP's own COIN_CMDTest suite against it, and diffs
# the failures against testdata/pulp_known_failures.txt. Exits non-zero only
# on a NEW failure (a regression); pre-known failures are reported, not fatal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PULP_VERSION="${PULP_VERSION:-3.3.2}"
VENV="$ROOT/.pulpenv"
KNOWN_FAILURES="$ROOT/testdata/pulp_known_failures.txt"
PATH_SHIM="$ROOT/.pulpenv-bin"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

installed_version="$(python -c 'import pulp; print(pulp.__version__)' 2>/dev/null || true)"
if [ "$installed_version" != "$PULP_VERSION" ]; then
  echo "==> installing pulp==$PULP_VERSION pytest"
  pip install --quiet "pulp==$PULP_VERSION" pytest
fi

echo "==> building bin/cbc"
go build -o "$ROOT/bin/cbc" ./cmd/cbc

mkdir -p "$PATH_SHIM"
ln -sf "$ROOT/bin/cbc" "$PATH_SHIM/cbc"

PULP_TESTS="$(python -c 'import pulp, os; print(os.path.dirname(pulp.__file__))')/tests/test_pulp.py"

echo "==> running PuLP's COIN_CMDTest suite"
set +e
OUTPUT="$(PATH="$PATH_SHIM:$PATH" python -m pytest "$PULP_TESTS" -k COIN_CMDTest -q -rf --no-header 2>&1)"
set -e
echo "$OUTPUT"

actual_failures="$(echo "$OUTPUT" | awk -F'::' '/^FAILED /{split($3,a," "); print $2"::"a[1]}' | sort -u)"
known_failures="$(grep -v '^#' "$KNOWN_FAILURES" 2>/dev/null | grep -v '^[[:space:]]*$' | sort -u || true)"

new_failures="$(comm -23 <(echo "$actual_failures") <(echo "$known_failures"))"
resolved="$(comm -13 <(echo "$actual_failures") <(echo "$known_failures"))"

if [ -n "$resolved" ]; then
  echo ""
  echo "==> these known failures now PASS — remove from $KNOWN_FAILURES:"
  echo "$resolved"
fi

if [ -n "$new_failures" ]; then
  echo ""
  echo "==> REGRESSION: unexpected new failures not in $KNOWN_FAILURES:"
  echo "$new_failures"
  exit 1
fi

echo ""
echo "==> OK: no failures beyond the known/documented ones"
