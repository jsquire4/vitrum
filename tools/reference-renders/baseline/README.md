# Gap-closure baseline renders (WG-0.2)

PNG baselines for `tools/benchmark-runner/run-gap-closure-verification.mjs`.

## Generate (WebGPU + Playwright required)

```bash
# Terminal A — not needed when using auto-start below
npm run dev --workspace @vitrum-examples/two-engines-one-scene

# Terminal B — seed pt-webgpu parity baseline (smoke resolution when VITRUM_CAPTURE_SMOKE=1)
#
# Requires a WebGPU adapter with full pt-webgpu tier (≥10 storage buffers/stage
# and ≥5 storage textures; split bind groups). SwiftShader only gets lite tier —
# use a hardware-GPU machine (or `npm run benchmark:gpu-windows -- run-seed-wg0-baselines.mjs`).
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
| `rfe03-layered-front-back.png` | WG-4 layered front/back |
| `rfe07-11-sss-mixed-panels.png` | WG-5 translucent / SSS (smoke capture) |
| `rfe08-13-spectral-payload.png` | WG-2 spectral hero-λ (smoke capture) |
| `rfe14-thinfilm-angle-shift.png` | WG-2 thin-film stack (smoke capture) |
| `rfe09-bridge-global-cmf.png` | WG-2 CMF / spectral bridge (smoke capture) |
| `rfe05-caustic-strategy.png` | Caustic strategy variants (smoke capture; full tier) |

Other gap-closure scenarios still use `VITRUM_CAPTURE_CMD` pointed at the cornell-box host capture page.
