# Reference render baselines

Cornell + hero quick captures (`session-20260527`, 512×512 / 64 SPP) live here as `cornell-*.png` and `hero-*.png`.

Use the current fail-closed capture flow documented in `../README.md`:

1. Start a capture-capable example such as `npm run dev --workspace @vitrum-examples/pt-webgl2-direct`.
2. Run `npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner` with `VITRUM_GPU_CAPTURE=1`, `VITRUM_CAPTURE_CMD`, and `VITRUM_CAPTURE_URL` set.
3. After visual review, promote the approved session with `bash scripts/promote-ref-baseline.sh session-YYYYMMDD`.

## Gap-closure (WG-0.2)

PNG baselines for `tools/benchmark-runner/run-gap-closure-verification.mjs`.

## Generate (WebGPU + Playwright required)

```bash
# Terminal A
npm run dev --workspace @vitrum-examples/pt-webgpu-direct

# Terminal B — seed pt-webgpu parity baseline (smoke resolution when VITRUM_CAPTURE_SMOKE=1)
#
# Requires a WebGPU adapter with full pt-webgpu tier (≥10 storage buffers/stage
# and ≥5 storage textures; split bind groups). SwiftShader only gets lite tier —
# use a hardware-GPU machine.
VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_GAP_SCENARIOS=ptwgpu-parity-material-fields \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
VITRUM_CAPTURE_SMOKE=1 \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
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
