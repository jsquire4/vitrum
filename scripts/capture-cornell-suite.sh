#!/usr/bin/env bash
#
# capture-cornell-suite.sh — render the 6 cornell-* scenarios via the
# benchmark-runner Playwright harness and write PNGs to a local folder.
#
# Quick post-sweep verification: you eyeball 6 images, confirm they look
# physically plausible, adopt as new baselines if happy.
#
# Requires:
#   - Real GPU passthrough (Chrome with hardware acceleration; SwiftShader
#     fallback produces black canvases — see plan/sweep-2026-05-12-followup.md
#     "Sweep verification capture").
#   - Playwright + Chromium installed (already in node_modules per
#     tools/benchmark-runner/).
#
# Usage:
#   scripts/capture-cornell-suite.sh                           # default 1280x720, 512spp, 8 bounces
#   scripts/capture-cornell-suite.sh --quick                   # 512x512, 64spp, 4 bounces (smoke)
#   scripts/capture-cornell-suite.sh --hero                    # 1920x1080, 2048spp, 12 bounces
#   scripts/capture-cornell-suite.sh --out /tmp/sweep-renders  # custom output dir
#   scripts/capture-cornell-suite.sh --only glass,spectral     # subset of scenarios
#   scripts/capture-cornell-suite.sh --bdpt                    # set vitrumBdpt=1 query param
#
set -euo pipefail

# Default settings
WIDTH=1280
HEIGHT=720
SPP=512
BOUNCES=8
SEED=12345
TIMEOUT_MS=120000
OUT_DIR="$(pwd)/tools/reference-renders/post-sweep-$(date +%Y%m%d)"
SCENARIOS_RAW="glass,caustic,spectral,layered,sss,parity"
EXTRA_QUERY=""

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      WIDTH=512; HEIGHT=512; SPP=64; BOUNCES=4
      shift
      ;;
    --hero)
      WIDTH=1920; HEIGHT=1080; SPP=2048; BOUNCES=12; TIMEOUT_MS=600000
      shift
      ;;
    --out)
      OUT_DIR="$2"; shift 2
      ;;
    --only)
      SCENARIOS_RAW="$2"; shift 2
      ;;
    --bdpt)
      EXTRA_QUERY="${EXTRA_QUERY}&vitrumBdpt=1"; shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1" >&2; usage
      ;;
  esac
done

mkdir -p "$OUT_DIR"

# Resolve repo root (script lives in scripts/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Start the example dev server in background.
echo "[capture-suite] Starting vite dev server (cornell-box)..."
( cd examples/cornell-box && exec npx vite --port 5173 --host 127.0.0.1 ) >/tmp/cornell-vite.log 2>&1 &
VITE_PID=$!

# Cleanup on exit (Ctrl-C, error, success).
cleanup() {
  if [[ -n "${VITE_PID:-}" ]] && kill -0 "$VITE_PID" 2>/dev/null; then
    echo "[capture-suite] Stopping vite (pid $VITE_PID)..."
    kill -TERM "$VITE_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$VITE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for vite to be ready (HTTP 200 on /).
echo -n "[capture-suite] Waiting for vite "
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:5173/ -o /dev/null 2>/dev/null; then
    echo " ready."
    break
  fi
  echo -n "."
  sleep 1
  if [[ $i -eq 30 ]]; then
    echo
    echo "[capture-suite] ERROR — vite never came up. Check /tmp/cornell-vite.log:" >&2
    tail -20 /tmp/cornell-vite.log >&2
    exit 1
  fi
done

# Build the URL with extra query (e.g., bdpt) if any.
BASE_URL="http://127.0.0.1:5173/"
if [[ -n "$EXTRA_QUERY" ]]; then
  # Strip leading & if present; the adapter appends ? + scenario params.
  EXTRA_QUERY="${EXTRA_QUERY#&}"
  # The adapter unconditionally treats VITRUM_CAPTURE_URL as the base; query
  # params from env are appended via URLSearchParams. To get our extra param
  # in, we encode it directly into the base URL.
  BASE_URL="${BASE_URL}?${EXTRA_QUERY}"
fi

# Iterate scenarios.
IFS=',' read -ra SCENARIOS <<< "$SCENARIOS_RAW"
SUMMARY=()
for scenario_short in "${SCENARIOS[@]}"; do
  scenario_id="cornell-${scenario_short}"
  out_png="${OUT_DIR}/${scenario_id}.png"
  echo "[capture-suite] → ${scenario_id} (${WIDTH}x${HEIGHT}, ${SPP} spp, ${BOUNCES} bounces)"
  start_ts=$(date +%s)

  set +e
  telemetry=$(VITRUM_OUTPUT_PNG="$out_png" \
    VITRUM_SCENARIO_ID="$scenario_id" \
    VITRUM_SEED="$SEED" \
    VITRUM_WIDTH="$WIDTH" VITRUM_HEIGHT="$HEIGHT" \
    VITRUM_BOUNCES="$BOUNCES" VITRUM_SPP="$SPP" \
    VITRUM_CAPTURE_TIMEOUT_MS="$TIMEOUT_MS" \
    VITRUM_CAPTURE_URL="$BASE_URL" \
    node ./tools/benchmark-runner/capture-adapter-playwright.mjs 2>&1 | tail -1)
  rc=$?
  set -e

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  if [[ $rc -eq 0 && -f "$out_png" ]]; then
    size=$(stat -c%s "$out_png" 2>/dev/null || stat -f%z "$out_png" 2>/dev/null || echo 0)
    SUMMARY+=("✓ ${scenario_id}  (${elapsed}s, ${size} bytes)  → ${out_png}")
  else
    SUMMARY+=("✗ ${scenario_id}  (${elapsed}s, exit ${rc})  telemetry: ${telemetry}")
  fi
done

# Final summary.
echo
echo "[capture-suite] Done. Output dir: ${OUT_DIR}"
for line in "${SUMMARY[@]}"; do
  echo "  $line"
done
echo
echo "[capture-suite] Eyeball the PNGs. If they look physically plausible:"
echo "  cp ${OUT_DIR}/*.png tools/reference-renders/baseline/"
echo
echo "[capture-suite] If anything looks wrong, check telemetry above + vite log:"
echo "  /tmp/cornell-vite.log"
