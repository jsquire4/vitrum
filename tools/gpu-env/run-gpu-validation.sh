#!/usr/bin/env bash
# Headless WebGPU validation for vitrum, running entirely inside WSL2.
#
# Uses Deno's native WebGPU (wgpu-native) over Mesa lavapipe (LLVMpipe Vulkan),
# which is the ONLY backend on this box that clears the hybrid storage-resource
# limits headlessly:  maxStorageBuffers/Textures = 1,000,000 (> 16/8 required),
# so hybridCanRun=true and ptWebgpuFullTier=true. (Chromium/Playwright's Dawn
# falls back to bundled SwiftShader = 10/4 -> hybridCanRun=false.)
#
# lavapipe is a CPU rasteriser: correct but slow. Use this for LIMIT-GATED
# correctness / validation, NOT for perf benchmarking. For wall-clock GPU perf
# you still need a real GPU adapter (native Windows Chrome on the dGPU; see
# tools/gpu-env/FINDINGS.md).
#
# Usage:
#   tools/gpu-env/run-gpu-validation.sh            # adapter probe + compute smoke
#   tools/gpu-env/run-gpu-validation.sh probe      # adapter limits only
#   tools/gpu-env/run-gpu-validation.sh smoke      # compute smoke only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

# --- locate Deno (userspace install) -----------------------------------------
DENO_BIN="${DENO_BIN:-$HOME/.deno/bin/deno}"
if ! command -v "$DENO_BIN" >/dev/null 2>&1; then
  if command -v deno >/dev/null 2>&1; then DENO_BIN="$(command -v deno)";
  else
    echo "ERROR: Deno not found. Install (userspace, no root):" >&2
    echo "  curl -fsSL https://deno.land/install.sh | DENO_INSTALL=\$HOME/.deno sh" >&2
    exit 3
  fi
fi

# --- pin the Vulkan ICD to lavapipe ------------------------------------------
LVP_ICD="${LVP_ICD:-/usr/share/vulkan/icd.d/lvp_icd.json}"
if [[ ! -f "$LVP_ICD" ]]; then
  echo "ERROR: lavapipe ICD not found at $LVP_ICD." >&2
  echo "  Install with: sudo apt-get install -y mesa-vulkan-drivers" >&2
  exit 3
fi
export VK_ICD_FILENAMES="$LVP_ICD"
export VK_DRIVER_FILES="$LVP_ICD"

run() { echo "=== $1 ==="; "$DENO_BIN" run --unstable-webgpu -A "$HERE/$2" 2>/dev/null; }

case "$MODE" in
  probe) run "WebGPU adapter probe (Deno + lavapipe)" probe-deno.ts ;;
  smoke) run "WebGPU compute smoke (Deno + lavapipe)" compute-smoke-deno.ts ;;
  all)
    run "WebGPU adapter probe (Deno + lavapipe)" probe-deno.ts
    echo
    run "WebGPU compute smoke (Deno + lavapipe)" compute-smoke-deno.ts
    ;;
  *) echo "usage: $0 [probe|smoke|all]" >&2; exit 2 ;;
esac
