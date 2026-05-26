# Gap-closure baseline renders (WG-0.2)

PNG baselines for `tools/benchmark-runner/run-gap-closure-verification.mjs`.

## Generate (WebGPU + Playwright required)

```bash
# Terminal A — not needed when using auto-start below
npm run dev --workspace @vitrum-examples/two-engines-one-scene

# Terminal B — seed pt-webgpu parity baseline (smoke resolution when VITRUM_CAPTURE_SMOKE=1)
#
# Requires a WebGPU adapter with maxStorageBuffersPerShaderStage ≥ 23 (SwiftShader
# caps at 10 — use a hardware-GPU machine for WG-0 PNG commits).
VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_GAP_SCENARIOS=ptwgpu-parity-material-fields \
VITRUM_CAPTURE_SMOKE=1 \
npm run benchmark:seed-wg0 --workspace @vitrum/benchmark-runner
```

Or with an already-running dev server:

```bash
VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_GAP_SCENARIOS=ptwgpu-parity-material-fields \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

## Verify

```bash
VITRUM_GPU_CAPTURE=1 \
VITRUM_GAP_SCENARIOS=ptwgpu-parity-material-fields \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

Commit `*.png` and optional `*.png.json` sidecars when visuals are approved.

## Scenario files

| File | Scenario |
|------|----------|
| `ptwgpu-parity-material-fields.png` | WG-0 pt-webgpu material parity (cornell via two-engines) |

Other gap-closure scenarios still use `VITRUM_CAPTURE_CMD` pointed at the cornell-box host capture page.
