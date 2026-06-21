# vitrum

A WebGPU + WebGL2 **path tracing & global illumination engine** for the browser. Core-scene, host-agnostic, drop-in.

> **Status**: release-candidate track, private monorepo. Not yet on npm. Public release follows final packaging, ecosystem docs, and cross-host verification.

## What is vitrum

vitrum gives you two render modes under one API:

- **Real-time GI** — WebGPU layered DDGI + ReSTIR DI + per-channel SVGF + GTAO. Targets ~60 fps for sub-500k-triangle scenes on consumer dGPUs.
- **Converged path tracing** — native WebGL2 / WebGPU path tracing backends. Hero-quality renders that converge over seconds.

You don't pick the backend; vitrum picks for you based on your hardware and scene complexity. You hand it a `@vitrum/core` `Scene`, get back an `Engine`, and call `renderFrame()` on each frame.

## 5-line hello world

```ts
import { attachVitrum } from '@vitrum/engine/lifecycle';

const handle = await attachVitrum({ canvas, scene, camera });
// … later
handle.dispose();
```

`attachVitrum()` runs the requestAnimationFrame loop, attaches a ResizeObserver, pauses on tab-hidden, and disposes the engine cleanly. Use the lower-level `createEngine()` if you want to drive the loop yourself.

## React variant

```tsx
import { VitrumCanvas } from '@vitrum/engine/react';

<VitrumCanvas
  scene={scene}
  camera={camera}
  prefer="realtime"
  onFrame={(s) => setFps(Math.round(1000 / s.frameTimeMs))}
/>
```

`react` and `react-dom` are **optional** peer deps. Vanilla hosts pay nothing for them.

## Capability matrix

| Feature                       | walkaround-hybrid (WebGPU)  | pt-webgl2 (WebGL2)         | pt-webgpu (WebGPU) |
| ----------------------------- | --------------------------- | -------------------------- | ------------------ |
| **GI quality**                | real-time, single-bounce GI | converged, multi-bounce PT | converged, multi-bounce PT |
| **Bounce count**              | 1 (DDGI gives multi-bounce) | unlimited                  | unlimited |
| **Light types**               | point/spot/dir/area/sky (point+spot: analytic NEE in shade pass + DDGI probe bounce; rect-area: NEE) | point / dir / area / sky   | point / spot / dir / area / HDRI |
| **Materials**                 | PBR + transmission + atlas textures; rich lobes approximate; spectral/thin-film/layered fields unsupported | PBR + transmission + spectral/Disney/layered material fields; WebGL2 runtime A/B promotion pending for some fidelity rows | PBR + transmission + spectral/Disney/layered material fields; full-tier rows tracked in the fidelity matrix |
| **Caustics**                  | none (DDGI only)            | heuristic approximate (not Newton-solve MNEE)      | manifold NEE / photon-map modes |
| **BDPT**                      | not applicable              | implemented + host-driven for analytic-light subpaths; mesh/env sources fall back to unidirectional NEE/BSDF; browser runtime promotion pending | supported safe-default path; multi-vertex research mode remains opt-in |
| **Animation**                 | camera ✓ / lights limited / mesh ✓ (material + positions + transform; vertex/index-count via rebuild) | camera ✓ / lights ✓ / mesh ✓ (material + transform/positions native; topology/list via bounded texture refresh) | camera ✓ / lights ✓ / mesh ✓ (targeted BLAS/TLAS paths where available) |
| **Hardware**                  | WebGPU                      | WebGL2                     | WebGPU |
| **Convergence**               | re-renders every frame      | accumulates SPP            | accumulates SPP |

See [`plan/archive/animation-support-status.md`](./plan/archive/animation-support-status.md) for the full animation matrix with caveats.

## When to use which engine

`prefer: 'auto'` (default) gives you walkaround-hybrid on WebGPU + scenes < 500k tris, else a native path-tracing backend.

Set `prefer: 'realtime'` for interactive viewers, lighting designers, scrub-the-camera demos. Set `prefer: 'quality'` for hero renders, product visualization, anything you'd want to save as a 4K PNG.

## Public API

| Package                       | What you import                                        |
| ----------------------------- | ------------------------------------------------------ |
| `@vitrum/engine`              | `createEngine`, `attachVitrum`, `<VitrumCanvas>` (React subpath), `CameraLike` |
| `@vitrum/core`                | `Engine`, `Scene`, `FrameInput`, `FrameStats`, `ProgressStats`, `EngineError` types; `Engine.onError`, `Engine.captureFrame`, `Engine.pickPrimitive` |
| `@vitrum/gltf-adapter`        | `gltfToScene`, `loadGltfAsset`, `loadGltfAndDecodeTextures`, `analyzeGltfAsset` / backend ranking, `decodeSceneTextures`, `loadGltfForEngine`, `GltfSceneController` — glTF 2.0 / GLB ingestion, compatibility planning, texture decode diagnostics, and runtime animation/variant patches |
| `@vitrum/dev`                 | Debug overlays (FrameTimeHUD, MaterialInspector, …) — devDep only |

Backend packages (`@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl2`, `@vitrum/pt-webgpu`) are also installable directly if you need backend-specific knobs that the facade doesn't surface.

## Capability Contract Notes

`engine.capabilities` is now the authoritative behavior matrix. Hosts should gate behavior off:

- `supportedPrimitiveKinds`, `supportedEnvironmentKinds`, `supportedEmitterKinds`
- `incrementalPatchSupport` (granular patch-path truth, not just a boolean)
- `presentationMode` (`swapchain-required` vs `offscreen-texture`)
- `experimentalFeatures` (explicit non-final algorithm seams)

This removes prior silent divergence between advertised and actual backend behavior.

## Sweep Gate Commands

For local technical-maturity verification in this monorepo:

- `npm run typecheck` — workspace TypeScript gate
- `npm test` — workspace Vitest gate
- `npm run verify:mechanical` — typecheck + tests + RC mechanical benchmark gate

## Migration Notes (2026-05 Coherence Sweep)

- `FrameInput.swapChainView` / `swapChainFormat` now require branded handles from `asBackendTexture` / `asBackendTextureFormat`.
- `@vitrum/core` no longer exports `_resetCacheUnsafe`.
- Stained-glass-specific contracts moved under `@vitrum/stained-glass-extensions`.
- `@vitrum/walkaround-hybrid` package root is intentionally trimmed; treat non-documented root exports as unstable internals.

## Lifecycle (the white-whale insight)

The engine accepts a device handle but **does not own** the device's lifetime. You own when the device is created, when it's lost, when it's reset. The engine owns its GPU resources for as long as you say.

This is the design choice that makes the library survive Canvas remount, route changes, tab-visibility transitions. `<VitrumCanvas>` and `attachVitrum()` already wire this correctly — you only need to think about it if you build your own host shell.

## Examples

Seven Vite apps in [`examples/`](./examples/) demonstrate every public API entry point —
`attachVitrum`, `createEngine`, `<VitrumCanvas>`, `createProgressiveEngine`,
`createPTEngine_WebGPU` (backend-direct), and `createPTEngine_WebGL2` (backend-direct).
Each implements the capture protocol (`VITRUM_CAPTURE_READY` + `?vitrumSpp=N` URL param)
for headless reference-render scripts.

See **[`examples/README.md`](./examples/README.md)** for the full app inventory, how to
run each one (`npm run dev --workspace=examples/<name>`), URL params, and sharp-edge notes.

For diagnosing black renders (missing `swapChainView`, `setScene` not called, WebGL2
context loss, device-limit errors, NaN pixels) see
**[`docs/debugging-black-frames.md`](./docs/debugging-black-frames.md)**.

## Architecture

```
@vitrum/engine             Drop-in facade — createEngine, attachVitrum, VitrumCanvas
  ↓
@vitrum/core               Engine contract; types only, no GPU code
@vitrum/gltf-adapter       glTF 2.0 / GLB → @vitrum/core Scene (zero-dependency)
@vitrum/walkaround-hybrid  WebGPU DDGI + ReSTIR DI/GI + SVGF + GTAO; opt-in PPG/NRC/neural; composes RC
@vitrum/walkaround-rc      Radiance Cascades subsystem (cascade pyramid + raw GPU dispatch)
@vitrum/pt-webgl2          Native WebGL2 PT
@vitrum/pt-webgpu          WebGPU-native PT (full/lite tiers; row-level fidelity tracked in the matrix)
@vitrum/shared-bvh         Software BVH compute (CPU + GPU)
@vitrum/shared-samplers    Hammersley, light tree, hero-wavelength MIS, spectral helpers
@vitrum/shared-denoisers   À-trous, SVGF, OIDN bridge
@vitrum/scene-lighting     Host-side lighting state (time-of-day, sun, sky)
@vitrum/stained-glass-extensions  Stained-glass host contract extensions
@vitrum/dev                Debug overlays (devDep)
```

## Performance budgets

(Reference renders captured on RTX 4090, ANGLE D3D11.)

| Scene                       | Engine            | Resolution | Convergence target | Time / Frame   |
| --------------------------- | ----------------- | ---------- | ------------------ | -------------- |
| Cornell box (~30 tris)      | pt-webgl2         | 512×512    | 64 SPP             | validation pending |
| Cornell glass               | pt-webgl2         | 512×512    | 64 SPP             | validation pending |
| Cornell spectral (hero MIS) | pt-webgl2         | 512×512    | 64 SPP             | validation pending |
| Living room (~200k tris)    | walkaround-hybrid | 1080p      | real-time          | validation pending |

Current mechanical benchmark gates:

```bash
npm run benchmark:gap-closure-mechanical
npm run benchmark:rc-acceptance-mechanical
```

Bench reports include `p95FrameMs` and `estimatedGpuMemoryBytes` when the hybrid pipeline is initialized (~8 GB iGPU tier documented in PR-6 plan).

## What's novel here

- **Layered hybrid GI** — WebGPU pipeline combining diffuse probe GI (DDGI) with stochastic direct illumination (ReSTIR-DI) and a single-bounce indirect (ReSTIR-GI), denoised with per-channel SVGF + GTAO. ([packages/walkaround-hybrid/README.md](packages/walkaround-hybrid/README.md))
- **NormalMap-perturbed NEE shadow rays** — produces textured caustics through transmissive materials in pure NEE; ported into the native path-tracing stack.
- **Hybrid analytic-CSG + BVH-mesh intersection** — closed-form quadrics + triangle meshes in the same path-tracing kernel. Production renderers usually pick one or the other.
- **Hero-wavelength MIS** (Wilkie et al. 2014) — one-sample MIS across X/Y/Z CMFs ships in the WebGL2 PT spectral path; material spectral coefficients remain a known promotion tail.

## Built on prior work

See [CREDITS.md](./CREDITS.md) for the full attribution list (~30 papers + libraries). Headline dependencies:

- **three-gpu-pathtracer** (Garrett Johnson, MIT)
- **three-mesh-bvh** (Garrett Johnson, MIT)
- **DDGI** (Majercik et al., 2019)
- **Radiance Cascades** (Sannikov, 2023)
- **ReSTIR DI / GI** (Bitterli et al., 2020 / Ouyang et al., 2021)
- **SVGF** (Schied et al., 2017)
- **Hero Wavelength Spectral Sampling** (Wilkie et al., 2014)
- **Disney BSDF** (Burley, 2012)

## License

MIT. See [LICENSE](./LICENSE).
