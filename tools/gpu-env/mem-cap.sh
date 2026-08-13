#!/usr/bin/env bash
# Run a command under a hard memory cap so a runaway shader compile cannot take
# the whole machine (or the WSL2 VM) down with it.
#
# Why this exists: creating a compute pipeline for the full-tier path-trace
# megakernel (packages/pt-webgpu, ~708KB of composed WGSL, 485 functions) grows
# monotonically inside naga/wgpu and does not converge — measured past 14GB on
# both llvmpipe and hardware dzn. Unconstrained under WSL2 it exhausts the VM's
# entire memory + swap, the kernel OOM-killer escalates to init.scope, and the
# distro dies with Wsl/Service/E_UNEXPECTED.
#
# Capping turns "the machine dies" into "this one command exits 137", which is a
# normal, debuggable CI failure.
#
# Usage:   tools/gpu-env/mem-cap.sh <command> [args...]
# Tuning:  VITRUM_GPU_MEM_CAP=12G   (default 8G;  set to 0 or "off" to disable)
#
# Portability: on any host without systemd-run + cgroup v2 user delegation
# (macOS, most CI containers, native Windows) this is a transparent no-op that
# execs the command unchanged.

set -euo pipefail

CAP="${VITRUM_GPU_MEM_CAP:-8G}"

if [ "$#" -eq 0 ]; then
  echo "mem-cap.sh: no command given" >&2
  exit 2
fi

if [ "$CAP" = "0" ] || [ "$CAP" = "off" ]; then
  exec "$@"
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  exec "$@"
fi

# Verify user-scope delegation actually works before relying on it; a failed
# probe must not block the real command.
if ! systemd-run --user --scope -q -p MemoryMax="$CAP" true >/dev/null 2>&1; then
  echo "[mem-cap] systemd user scopes unavailable — running uncapped" >&2
  exec "$@"
fi

echo "[mem-cap] MemoryMax=$CAP (override with VITRUM_GPU_MEM_CAP)" >&2
set +e
systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 "$@"
STATUS=$?
set -e

if [ "$STATUS" -eq 137 ]; then
  echo "[mem-cap] command exceeded $CAP and was killed (exit 137)." >&2
  echo "[mem-cap] This is the runaway pipeline compile, not a flaky gate." >&2
fi
exit "$STATUS"
