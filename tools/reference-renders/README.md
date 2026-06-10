# Reference renders

Reference PNGs are how we A/B-verify visual output of the renderer across
branches before merge. The capture flow is **the verification gate** called
"mandatory" for any change that touches a backend's visible output.

## Mechanical acceptance gates (no GPU)

These run in `npm run verify:mechanical` and pin harness contracts only — **not** visual quality.

| Gate | Command |
|------|---------|
| W8 RC off/on | `npm run benchmark:rc-acceptance-mechanical --workspace @vitrum/benchmark-runner` |
| RC behavior | `npm run benchmark:rc-behavior-mechanical --workspace @vitrum/benchmark-runner` |

Fixture PNGs live under `W8-rc-{off,on}/` and `bdpt-layered-mechanical/`. Replace
with real GPU captures when refreshing baselines.

## Quick start — capture refs on the current branch

GPU captures require a running example server and Playwright Chromium.
Start an example app in one terminal (e.g., `npm run dev --workspace @vitrum-examples/pt-webgl2-direct`),
then in another terminal:

```bash
VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

The `capture-adapter-playwright.mjs` adapter launches Playwright Chromium with
WebGPU flags (`--enable-unsafe-webgpu`; `--use-angle=vulkan` on Linux) and
awaits `globalThis.VITRUM_CAPTURE_READY === true` on the page before screenshotting.

See `tools/benchmark-runner/README.md` for the full capture adapter protocol and
env-var reference.

The H57 example apps under `examples/` implement the capture protocol:
- `examples/pt-webgl2-direct/` — pt-webgl2 backend
- `examples/pt-webgpu-direct/` — pt-webgpu backend
- `examples/create-engine/`, `examples/attach-vitrum/`, `examples/progressive/` — facade

Default resolution: **1280×720 @ 512 SPP, 8 bounces** (set via URL params).

## A/B diff against a baseline

```bash
# Compare a fresh capture dir vs the locked baseline:
node tools/reference-renders/diff-baselines.mjs \
  --candidate tools/reference-renders/session-YYYYMMDD \
  --baseline  tools/reference-renders/baseline
```

The diff tool reports:
- **`OK`** — SHA-256 match (bit-exact, expected for pure refactors).
- **`OK*`** — pixels differ but mean-absolute-diff is below tolerance (default `0.001`).
- **`DIFF`** — pixels differ outside tolerance. Either a bug or an intentional
  algorithmic change requiring visual sign-off in the PR body.
- **`MISS`** — baseline is missing the file. New scenarios need baselines generated separately.

Pixel-diff requires `pngjs` (installed at workspace root). Without it, the tool
falls back to size+SHA comparison only.

## Promoting a capture to a new baseline

Once a capture has visual sign-off (eyeballed on a real GPU, not SwiftShader):

```bash
cp tools/reference-renders/session-YYYYMMDD/*.png tools/reference-renders/baseline/
git add tools/reference-renders/baseline/
git commit -m "chore(refs): adopt session-YYYYMMDD captures as new baseline"
```

A promote helper is also available:

```bash
bash scripts/promote-ref-baseline.sh session-YYYYMMDD
```

## What the capture protocol requires from an example app

Any example added to the capture orchestrator must set these globals **after**
its engine reaches a deterministic converged state:

```ts
globalThis.VITRUM_CAPTURE_READY = true;          // sentinel Playwright awaits
globalThis.VITRUM_MS_PER_SAMPLE = 12.4;          // perf telemetry (optional)
globalThis.VITRUM_CAPTURE_TELEMETRY = {/*…*/};   // arbitrary JSON sidecar (optional)
globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c'; // which canvas to screenshot
```

It must also honour these URL params:
- `?vitrumScenario=<id>` — flip into capture mode
- `?vitrumAutoStart=1` — skip any "click to start" gate
- `?vitrumWidth`, `?vitrumHeight` — canvas size lock
- `?vitrumSpp`, `?vitrumBounces` — quality knobs

The Playwright adapter (`tools/benchmark-runner/capture-adapter-playwright.mjs`)
translates `VITRUM_*` env vars into the corresponding URL params automatically.

## Existing artifacts

- `baseline/` — the locked reference set. **24 PNG baselines committed** as of
  2026-06-09: pt-webgpu fidelity-matrix captures (rfe03, rfe05, rfe07, rfe08,
  rfe09, rfe14, ptwgpu-parity-material-fields, ptwgpu-spectral-hero,
  ptwgpu-sss-mixed-panels, ptwgpu-thinfilm-angle, ptwgpu-cauchy-dispersion,
  ptwgpu-layered-front), fork-vs-native A/B captures (cornell-glass,
  cornell-caustic, cornell-spectral, cornell-layered, cornell-sss,
  cornell-parity, cornell-bdpt-on, hero-product-viz, hero-viewer-quality,
  hero-viewer-realtime), plus mnee-glass-slab and rfe03-layered-front-back
  originals. See `baseline/README.md` for per-file status.
  **Note on rfe09-bridge-global-cmf.png:** this baseline was captured before
  commit `daa9716` (pt-webgpu within-leaf closest-hit fix); its post-`daa9716`
  radiometric status is unverified — re-capture before using as an enforcement gate.
- `post-sweep-20260512/` — sample capture set from the 2026-05-12
  post-sweep verification (6 Cornell scenarios at 1280×720).
- `pt-webgl-fidelity/` — pre-e14000c fork baselines kept as THREE-cutover A/B
  reference. See `pt-webgl-fidelity/README.md`.
- `W8-rc-{off,on}/`, `bdpt-layered-mechanical/` — mechanical fixture sets.

## CI gate

```bash
# fails if any scene drifts outside tolerance
VITRUM_GPU_CAPTURE=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
VITRUM_STRICT_GAP_CLOSURE=1 \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

GPU is required; SwiftShader produces black canvases and will report `DIFF` for
everything. Use a self-hosted GPU runner or a cloud GPU instance.

## Troubleshooting

- **Black PNGs** → no real GPU passthrough. Chrome fell back to SwiftShader.
  Re-run on a machine with hardware acceleration, or verify `--use-angle=vulkan`
  / `--enable-unsafe-webgpu` flags are present in the Playwright launch
  (see `capture-adapter-playwright.mjs`).

- **`Executable doesn't exist at .../chrome-headless-shell`** → Chromium not
  installed. Run `npx playwright install chromium`. This is a ~200 MB download
  and is intentionally not part of `npm install` (GPU tests are opt-in; see
  `CONTRIBUTING.md`).

- **Timeouts** → bump `VITRUM_CAPTURE_TIMEOUT_MS` or run with a smaller SPP target.

- **Sliders / camera moved mid-capture** → headless Chromium can fire stray pointer
  events. Lock inputs in capture mode (check `?vitrumAutoStart=1` is being read).
