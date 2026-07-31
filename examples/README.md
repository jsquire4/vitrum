# vitrum examples

Seven self-contained Vite apps demonstrate the `@vitrum/core` Scene contract
against each public API entry point; an eighth workspace provides their shared
Cornell scene. Every runnable app implements the **capture protocol** (see below)
so reference-render scripts can drive it headlessly.

---

## Apps at a glance

| Directory | Package name | What it demonstrates |
|-----------|-------------|----------------------|
| `attach-vitrum/` | `@vitrum-examples/attach-vitrum` | `attachVitrum()` lifecycle helper — RAF loop, ResizeObserver, camera matrices, `CameraLike` |
| `create-engine/` | `@vitrum-examples/create-engine` | `createEngine()` explicit construction + auto backend selection, then lifecycle handoff for backend-correct presentation |
| `vitrum-canvas/` | `@vitrum-examples/vitrum-canvas` | `<VitrumCanvas>` React component — drop-in mount, `onProgress` telemetry |
| `progressive/` | `@vitrum-examples/progressive` | `createProgressiveEngine()` — realtime walkaround while camera moves, lifecycle-presented handoff to converged PT when it settles |
| `gltf-viewer/` | `@vitrum-examples/gltf-viewer` | `loadGltfWithEngine()` — self-contained glTF asset, feature report, backend recommendation, controller attachment |
| `pt-webgpu-direct/` | `@vitrum-examples/pt-webgpu-direct` | Backend-direct WebGPU — host-owned device plus explicit offscreen-to-canvas presentation |
| `pt-webgl2-direct/` | `@vitrum-examples/pt-webgl2-direct` | Backend-direct WebGL2 — `createPTEngine_WebGL2()`, host-owned `WebGL2RenderingContext` |
| `cornell-scene/` | `@vitrum-examples/cornell-scene` | Shared scene factory (not a runnable app — exported as `createCornellScene()`) |

---

## Running an example

Each app is a standard Vite project.  From the monorepo root:

```bash
# Install workspace deps once (if not already done):
npm install

# Start the dev server for a specific example:
npm run dev --workspace=examples/attach-vitrum
# → opens on http://localhost:5173 by default
```

Or `cd` into the example directory and run `npm run dev` directly.

To build for preview:

```bash
npm run build --workspace=examples/attach-vitrum
npm run preview --workspace=examples/attach-vitrum
```

---

## URL params

All apps read quality/capture parameters from the URL query string.

| Param | Type | Default | Used by |
|-------|------|---------|---------|
| `vitrumSpp` | integer | `128` | All apps — PT sample target, or real-time frame target, for the capture signal |
| `vitrumBounces` | integer | `8` | `pt-webgpu-direct`, `pt-webgl2-direct` — `maxBounces` passed to the engine factory |

Examples:

```
http://localhost:5173/?vitrumSpp=64
http://localhost:5173/?vitrumSpp=256&vitrumBounces=16
```

The `gltf-viewer` app also exposes these debug fields:

| Field | Type | Set when |
|-------|------|----------|
| `VITRUM_GLTF_BACKEND` | string | `loadGltfWithEngine()` selects a backend |
| `VITRUM_GLTF_TEXTURE_REPORT` | `GltfTextureDecodeReport` | the embedded asset has been decoded and reported |
| `VITRUM_GLTF_WARNINGS` | string[] | adapter/controller warnings, if any |

---

## Capture protocol

Every runnable app (except `cornell-scene/`) implements the **vitrum capture
protocol** so reference-render and E2E harnesses can wait for convergence and
then read back the frame.

### `globalThis` fields set by the app

| Field | Type | Set when |
|-------|------|----------|
| `VITRUM_CAPTURE_READY` | `boolean` | An accumulating backend reaches `vitrumSpp`; a real-time backend renders the same number of frames |
| `VITRUM_CAPTURE_ERROR` | `string` | Construction, presentation, or render-loop failure prevents capture completion |
| `VITRUM_MS_PER_SAMPLE` | `number` | Updated every frame — `frameTimeMs / spp` ratio from `FrameStats` (omitted on the `vitrum-canvas` app which uses `onProgress`) |
| `VITRUM_HANDLE` | `AttachVitrumHandle` | Lifecycle-driven apps (`attach-vitrum`, `create-engine`, `progressive`, `gltf-viewer`) — allows DevTools / E2E disposal |
| `VITRUM_DISPOSE` | `() => void` | Backend-direct apps, plus `gltf-viewer`; releases each app's engine/presenter/device or imported asset resources as applicable |

### Usage from a headless harness

```js
// Puppet (Playwright / Puppeteer example):
await page.waitForFunction(() => globalThis.VITRUM_CAPTURE_READY === true);
const msPerSample = await page.evaluate(() => globalThis.VITRUM_MS_PER_SAMPLE);
```

### `progressive` example: phase awareness

The `progressive` app sets `VITRUM_CAPTURE_READY` only once the **converging
phase** has started (the pt-webgpu converged engine is the active renderer) AND
its `pt-spp` progress event reports `current >= vitrumSpp`.

Note that the terminal `HandoffPhase` value is `'converging'` — there is no
`'converged'` state.  Capture harnesses must wait for `'converging'` + SPP, not
a separate terminal phase.

---

## Sharp edges documented in the examples

Each example's source file opens with a `@remarks`-style header that records the
API sharp edges the author hit.  These are signal, not noise — they feed back into
the public API improvement queue.

Key findings (current as of 2026-07-29):

- **`CameraLike` is now exported from `@vitrum/engine`** (added 2026-06-10).
  The `attach-vitrum` and `vitrum-canvas` examples previously redefined the
  structural interface inline with a complaint comment; they now import it directly:
  ```ts
  import { attachVitrum, type CameraLike } from '@vitrum/engine';
  ```

- **Presentation follows `EngineCapabilities.presentationMode`.**
  Swapchain-required engines need a fresh per-frame view; offscreen engines such
  as `pt-webgpu` need their presentation source blitted to the canvas.
  `attachVitrum()` handles both modes; `pt-webgpu-direct/` demonstrates the
  host-owned `createOffscreenPresenter()` path.

- **`pt-webgpu-direct` and `pt-webgl2-direct` require `engine.setScene(scene)`
  before the first `renderFrame()`.** The facade (`createEngine`) handles this
  automatically; the backend-direct APIs do not.

- **`FrameStats.spp` comes from `engine.onFrame`, not `renderFrame()`.**
  `renderFrame()` returns `samplesAccumulated` in the `FrameOutput` object; per-frame
  timing lives in the `FrameStats` callback.

- **`HandoffPhase` has no `'converged'` terminal state** (`progressive` app).
  The converged engine is active while phase is `'converging'`.

For the black-frame debugging guide (swap-chain plumbing, missing setScene, context
lost, device limits, NaN pixels) see
[`../docs/debugging-black-frames.md`](../docs/debugging-black-frames.md).
