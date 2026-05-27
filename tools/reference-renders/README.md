# Reference renders

Reference PNGs are how we A/B-verify visual output of the renderer across
session branches before merge. The capture flow is **the verification gate** the
premium-grade refactor plan calls "mandatory" for any change that touches a
backend's visible output (see `plan/premium-grade-refactor-20260517.md` §1.2).

## Mechanical acceptance gates (no GPU)

These run in `npm run verify:mechanical` and pin harness contracts only — **not** visual quality.

| Gate | Command |
|------|---------|
| W8 RC off/on | `npm run benchmark:rc-acceptance-mechanical` |
| BDPT vs layered | `npm run benchmark:bdpt-layered-mechanical` |

Fixture PNGs live under `W8-rc-{off,on}/` and `bdpt-layered-mechanical/`. Replace with real GPU captures when refreshing baselines (`benchmark:rc-acceptance-full`, `benchmark:bdpt-layered-refs`).

## Quick start — capture refs on the current branch

```bash
# From repo root, with workspace npm-installed and Playwright chromium pulled.
npm run capture:refs
```

This produces a labelled output directory under
`tools/reference-renders/session-YYYYMMDD/` containing:

| File                          | What it is                                                    |
|-------------------------------|---------------------------------------------------------------|
| `cornell-glass.png`           | Cornell glass-sphere scenario (pt-webgl)                      |
| `cornell-caustic.png`         | Cornell caustic test (manifold-NEE)                            |
| `cornell-spectral.png`        | Cornell spectral-rendering scenario                            |
| `cornell-layered.png`         | Cornell layered-BSDF test                                      |
| `cornell-sss.png`             | Cornell SSS / dispersion                                       |
| `cornell-parity.png`          | Cornell WebGPU-vs-WebGL parity                                 |
| `hero-product-viz.png`        | Hero product visualizer — procedural glass scene               |
| `hero-viewer-realtime.png`    | Hero glTF viewer — walkaround GI engine on a chrome sphere     |
| `hero-viewer-quality.png`     | Same scene, path-tracer engine                                 |

Default resolution: **1280×720 @ 512 SPP, 8 bounces**.

## Variants

```bash
npm run capture:refs:quick    # 512x512 @ 64 SPP — smoke test (~30s/scene)
npm run capture:refs:hero     # 1920x1080 @ 2048 SPP — high quality (~5-10 min/scene)
```

Custom label (recommended for branch-named captures):

```bash
./scripts/capture-all-refs.sh --label W1-pre
./scripts/capture-all-refs.sh --label W1-post --diff W1-pre
```

Subset:

```bash
./scripts/capture-all-refs.sh --only cornell        # cornell-box only
./scripts/capture-all-refs.sh --only hero-product   # product viz only
./scripts/capture-all-refs.sh --only hero-viewer    # glTF viewer only
```

## A/B diff against a baseline

```bash
# Compare a fresh capture vs the locked baseline:
node tools/reference-renders/diff-baselines.mjs \
  --candidate tools/reference-renders/session-20260517 \
  --baseline  tools/reference-renders/baseline

# Or built into the capture run:
./scripts/capture-all-refs.sh --label post-fix --diff baseline
```

The diff tool reports:
- **`OK`** — SHA-256 match (bit-exact, the expected outcome for pure
  refactors like W1).
- **`OK*`** — pixels differ but mean-absolute-diff is below tolerance
  (default `0.001`). Allowed for FP-rounding-level numerical drift.
- **`DIFF`** — pixels differ outside tolerance. Either a bug or an
  intentional algorithmic change requiring visual sign-off in the PR body.
- **`MISS`** — baseline is missing the file. New scenarios need baselines
  generated separately.

Pixel-diff requires `pngjs` (`npm i -D pngjs` at the workspace root). Without
it, the tool falls back to size+SHA comparison only, which still catches
bit-exact regressions.

## Promoting a capture to a new baseline

Once a capture has visual sign-off (eyeballed on a real GPU, not SwiftShader):

```bash
cp tools/reference-renders/session-20260517/*.png tools/reference-renders/baseline/
git add tools/reference-renders/baseline/
git commit -m "chore(refs): adopt session-20260517 captures as new baseline"
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
- `?vitrumScenario=<id>` — flips the page into capture mode
- `?vitrumAutoStart=1` — skips any "click to start" gate
- `?vitrumWidth`, `?vitrumHeight` — canvas size lock
- `?vitrumSpp`, `?vitrumBounces` — quality knobs

The Playwright adapter (`tools/benchmark-runner/capture-adapter-playwright.mjs`)
translates `VITRUM_*` env vars into the corresponding URL params automatically.

Currently implementing the protocol:
- `examples/cornell-box` (full implementation, multiple scenarios)
- `examples/hero-product-viz` (procedural scene, single scenario)
- `examples/hero-viewer` (built-in fallback hero scene + procedural)

## CI gate

For continuous protection, wire the capture + diff into CI:

```bash
# fails if any scene drifts outside tolerance
./scripts/capture-all-refs.sh --label ci-$GITHUB_SHA --diff baseline
```

The diff tool exits non-zero on divergence, so this is a one-line CI gate.
GPU is required; a self-hosted GPU runner or a cloud GPU instance is needed
(SwiftShader produces black canvases and will report `DIFF` for everything).

## Existing artifacts

- `baseline/` — the locked reference set. Currently sparse — only
  `rfe03-layered-front-back.png` is committed (RFE-03 verification).
  Run `npm run capture:refs` on a known-good GPU machine to populate.
- `post-sweep-20260512/` — sample capture set from the 2026-05-12
  post-sweep verification (6 cornell scenarios at 1280×720).
- `sweep-2026-05-11-diff-report.md` — A/B sign-off template for the
  2026-05-11 sweep. Mirror this format for future per-workstream sweeps
  (`W1-diff-report.md`, `W2-C3-diff-report.md`, etc.).

## Troubleshooting

- **Black PNGs** → no real GPU passthrough. Chrome fell back to SwiftShader.
  Re-run on a machine with hardware acceleration, or set
  `--use-angle=vulkan` / `--enable-unsafe-webgpu` in the Playwright launch
  args (see `tools/benchmark-runner/capture-adapter-playwright.mjs`).

- **`Executable doesn't exist at .../chrome-headless-shell`** → Chromium not
  installed. Run `npx playwright install chromium`. This is a 200MB download
  and is intentionally not part of `npm install` (see `CONTRIBUTING.md` for
  why GPU tests are opt-in).

- **Timeouts on `--hero` mode** → 2048 SPP at 1080p can take several
  minutes per scene. Bump `VITRUM_CAPTURE_TIMEOUT_MS` or run with a smaller
  SPP target.

- **Sliders / camera moved mid-capture** → headless Chromium can fire stray
  pointer events. Both `hero-product-viz` and `hero-viewer` lock their
  inputs in capture mode (sliders ignore input, orbit controls disabled).
  Verify the canvas-style width/height matches what's in the PNG header.
