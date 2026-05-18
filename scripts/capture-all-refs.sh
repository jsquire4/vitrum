#!/usr/bin/env bash
#
# capture-all-refs.sh — produce the canonical reference-render set for a given
# branch by orchestrating Playwright captures across all example apps that
# implement the VITRUM_CAPTURE_READY protocol.
#
# This is the script `npm run capture:refs` invokes. It's the bit of
# infrastructure called "mandatory" in the premium-grade-refactor plan and
# referenced from CLAUDE.md "Testing protocol" — capture before + after for
# any algorithmic change so 60+ parallel feature branches can A/B at merge.
#
# Output layout:
#   tools/reference-renders/<label>/
#     cornell-glass.png        ← 6 cornell-box scenarios (via the existing
#     cornell-caustic.png        capture-cornell-suite.sh helper)
#     cornell-spectral.png
#     cornell-layered.png
#     cornell-sss.png
#     cornell-parity.png
#     hero-product-viz.png     ← procedural glass scene at 512 SPP
#     hero-viewer-realtime.png ← procedural chrome sphere, walkaround GI
#     hero-viewer-quality.png  ← same scene, path-traced
#
# Usage:
#   scripts/capture-all-refs.sh                            # default 1280x720
#                                                          # label = session-YYYYMMDD
#   scripts/capture-all-refs.sh --label W1-post            # custom label dir
#   scripts/capture-all-refs.sh --quick                    # 512x512 64spp smoke
#   scripts/capture-all-refs.sh --hero                     # 1920x1080 2048spp
#   scripts/capture-all-refs.sh --only cornell             # subset (cornell|hero-product|hero-viewer)
#   scripts/capture-all-refs.sh --diff baseline            # also diff against tools/reference-renders/baseline/
#
# Requires (same as capture-cornell-suite.sh):
#   - Real GPU passthrough (Chrome with hardware acceleration). SwiftShader
#     produces black canvases.
#   - npm install completed at workspace root.
#   - Playwright + chromium installed (`npx playwright install chromium`).
set -euo pipefail

# Defaults
WIDTH=1280
HEIGHT=720
SPP=512
BOUNCES=8
SEED=12345
TIMEOUT_MS=120000
LABEL="session-$(date +%Y%m%d)"
ONLY="all"
DIFF_AGAINST=""

usage() {
  sed -n '2,40p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)   WIDTH=512;  HEIGHT=512;  SPP=64;   BOUNCES=4;  shift ;;
    --hero)    WIDTH=1920; HEIGHT=1080; SPP=2048; BOUNCES=12; TIMEOUT_MS=600000; shift ;;
    --label)   LABEL="$2"; shift 2 ;;
    --only)    ONLY="$2";  shift 2 ;;
    --diff)    DIFF_AGAINST="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${REPO_ROOT}/tools/reference-renders/${LABEL}"
mkdir -p "$OUT_DIR"

echo "[capture-all-refs] Output dir: ${OUT_DIR}"
echo "[capture-all-refs] Resolution: ${WIDTH}x${HEIGHT}, ${SPP} SPP, ${BOUNCES} bounces"
echo "[capture-all-refs] Scope:      ${ONLY}"
echo

# ───────────────────────────────────────────────────────────────────────────
# Helper: start a vite dev server in the requested example, wait for it ready,
# print the PID. Caller is responsible for killing it.
# ───────────────────────────────────────────────────────────────────────────
start_vite() {
  local example_dir="$1"
  local port="$2"
  local log="/tmp/capture-${example_dir//\//-}.log"

  # All human-facing output goes to stderr; stdout is reserved for the PID so
  # callers can do `pid=$(start_vite ...)` safely.
  echo "[capture-all-refs] Starting vite in ${example_dir} on :${port}..." >&2
  ( cd "${example_dir}" && exec npx vite --port "${port}" --host 127.0.0.1 ) >"${log}" 2>&1 &
  local pid=$!

  # Wait for ready (HTTP 200 on /).
  printf '[capture-all-refs] Waiting for :%s ' "${port}" >&2
  for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${port}/" -o /dev/null 2>/dev/null; then
      echo " ready." >&2
      echo "$pid"
      return 0
    fi
    printf '.' >&2
    sleep 1
  done
  echo >&2
  echo "[capture-all-refs] ERROR — vite never came up. Tail of ${log}:" >&2
  tail -20 "${log}" >&2
  kill -TERM "$pid" 2>/dev/null || true
  return 1
}

stop_vite() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 1
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

# ───────────────────────────────────────────────────────────────────────────
# Capture one scenario via the existing Playwright adapter.
#
# Args:
#   $1 = scenario id (becomes the PNG filename and VITRUM_SCENARIO_ID)
#   $2 = vite port the example is listening on
#   $3 = optional extra URL query (e.g. 'vitrumPrefer=realtime'); already-set
#        vitrumAutoStart=1 is appended automatically.
# ───────────────────────────────────────────────────────────────────────────
capture_one() {
  local scenario_id="$1"
  local port="$2"
  local extra_query="${3:-}"
  local out_png="${OUT_DIR}/${scenario_id}.png"

  local url="http://127.0.0.1:${port}/?vitrumAutoStart=1"
  if [[ -n "${extra_query}" ]]; then
    url="http://127.0.0.1:${port}/?${extra_query}&vitrumAutoStart=1"
  fi

  echo "[capture-all-refs] → ${scenario_id} (${WIDTH}x${HEIGHT}, ${SPP} SPP)"
  local start_ts end_ts elapsed
  start_ts=$(date +%s)

  set +e
  VITRUM_OUTPUT_PNG="${out_png}" \
    VITRUM_SCENARIO_ID="${scenario_id}" \
    VITRUM_SEED="${SEED}" \
    VITRUM_WIDTH="${WIDTH}" VITRUM_HEIGHT="${HEIGHT}" \
    VITRUM_BOUNCES="${BOUNCES}" VITRUM_SPP="${SPP}" \
    VITRUM_CAPTURE_TIMEOUT_MS="${TIMEOUT_MS}" \
    VITRUM_CAPTURE_URL="${url}" \
    node ./tools/benchmark-runner/capture-adapter-playwright.mjs 2>&1 | tail -3
  local rc=$?
  set -e

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  if [[ $rc -eq 0 && -f "${out_png}" ]]; then
    local size
    size=$(stat -c%s "${out_png}" 2>/dev/null || stat -f%z "${out_png}" 2>/dev/null || echo 0)
    SUMMARY+=("OK   ${scenario_id}  (${elapsed}s, ${size} bytes)")
  else
    SUMMARY+=("FAIL ${scenario_id}  (${elapsed}s, exit ${rc})")
  fi
}

# ───────────────────────────────────────────────────────────────────────────
# Cleanup chain — gets re-armed each time a new vite is launched.
# ───────────────────────────────────────────────────────────────────────────
declare -a VITE_PIDS=()
cleanup() {
  for pid in "${VITE_PIDS[@]}"; do stop_vite "$pid"; done
}
trap cleanup EXIT INT TERM

SUMMARY=()

# ───────────────────────────────────────────────────────────────────────────
# Cornell-box (delegate to the existing helper for the 6 cornell scenarios).
# ───────────────────────────────────────────────────────────────────────────
if [[ "${ONLY}" == "all" || "${ONLY}" == "cornell" ]]; then
  echo "[capture-all-refs] === Cornell suite ==="
  args=(--out "${OUT_DIR}")
  if [[ "${WIDTH}" == "512" && "${SPP}" == "64" ]]; then
    args=(--quick "${args[@]}")
  elif [[ "${WIDTH}" == "1920" && "${SPP}" == "2048" ]]; then
    args=(--hero "${args[@]}")
  fi
  if ./scripts/capture-cornell-suite.sh "${args[@]}"; then
    SUMMARY+=("OK   cornell-suite  (6 scenarios)")
  else
    SUMMARY+=("FAIL cornell-suite")
  fi
fi

# ───────────────────────────────────────────────────────────────────────────
# Hero product viz — procedural glass scene.
# ───────────────────────────────────────────────────────────────────────────
if [[ "${ONLY}" == "all" || "${ONLY}" == "hero-product" ]]; then
  echo "[capture-all-refs] === Hero product viz ==="
  if pid=$(start_vite "examples/hero-product-viz" "5174"); then
    VITE_PIDS+=("$pid")
    capture_one "hero-product-viz" "5174"
    stop_vite "$pid"
    VITE_PIDS=("${VITE_PIDS[@]/$pid}")
  else
    SUMMARY+=("FAIL hero-product-viz (vite did not start)")
  fi
fi

# ───────────────────────────────────────────────────────────────────────────
# Hero viewer — procedural fallback scene, captured twice (realtime + quality).
# ───────────────────────────────────────────────────────────────────────────
if [[ "${ONLY}" == "all" || "${ONLY}" == "hero-viewer" ]]; then
  echo "[capture-all-refs] === Hero viewer (realtime + quality) ==="
  if pid=$(start_vite "examples/hero-viewer" "5175"); then
    VITE_PIDS+=("$pid")
    capture_one "hero-viewer-realtime" "5175" "vitrumPrefer=realtime"
    capture_one "hero-viewer-quality"  "5175" "vitrumPrefer=quality"
    stop_vite "$pid"
    VITE_PIDS=("${VITE_PIDS[@]/$pid}")
  else
    SUMMARY+=("FAIL hero-viewer (vite did not start)")
  fi
fi

# ───────────────────────────────────────────────────────────────────────────
# Optional: pixel-diff against a baseline directory.
# ───────────────────────────────────────────────────────────────────────────
if [[ -n "${DIFF_AGAINST}" ]]; then
  baseline_dir="${REPO_ROOT}/tools/reference-renders/${DIFF_AGAINST}"
  if [[ -d "${baseline_dir}" ]]; then
    echo
    echo "[capture-all-refs] Diffing ${OUT_DIR} vs ${baseline_dir}..."
    node ./tools/reference-renders/diff-baselines.mjs \
      --candidate "${OUT_DIR}" --baseline "${baseline_dir}" || true
  else
    echo "[capture-all-refs] WARN — baseline dir not found: ${baseline_dir}"
  fi
fi

# ───────────────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────────────
echo
echo "[capture-all-refs] Done. Output dir: ${OUT_DIR}"
for line in "${SUMMARY[@]}"; do
  echo "  $line"
done
echo
echo "[capture-all-refs] To adopt these as a new baseline:"
echo "  cp ${OUT_DIR}/*.png tools/reference-renders/baseline/"
echo
echo "[capture-all-refs] To compare a future capture against this one:"
echo "  scripts/capture-all-refs.sh --label post-fix --diff ${LABEL}"
