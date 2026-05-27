#!/usr/bin/env bash
# Promote a dated capture session into tools/reference-renders/baseline/.
#
# Usage:
#   scripts/promote-ref-baseline.sh session-20260527
#   scripts/promote-ref-baseline.sh tools/reference-renders/session-20260527
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:-session-$(date +%Y%m%d)}"
if [[ "$LABEL" == */* ]]; then
  SRC="$LABEL"
else
  SRC="$REPO_ROOT/tools/reference-renders/$LABEL"
fi
DST="$REPO_ROOT/tools/reference-renders/baseline"

if [[ ! -d "$SRC" ]]; then
  echo "Source dir missing: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"
shopt -s nullglob
pngs=("$SRC"/*.png)
if [[ ${#pngs[@]} -eq 0 ]]; then
  echo "No PNGs in $SRC" >&2
  exit 1
fi
cp -v "${pngs[@]}" "$DST/"
echo "[promote-ref-baseline] ${#pngs[@]} PNG(s) → $DST"
