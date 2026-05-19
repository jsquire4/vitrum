# vitrum

A WebGPU + WebGL2 **path tracing & global illumination engine** for the browser. THREE.js-friendly, host-agnostic, drop-in.

> **Status**: pre-alpha, private. Not yet on npm. Public release planned after the foundational packages stabilize.

## What is vitrum

vitrum gives you two render modes under one API:

- **Real-time GI** — WebGPU layered DDGI + ReSTIR DI + per-channel SVGF + GTAO. Targets ~60 fps for sub-500k-triangle scenes on consumer dGPUs.
- **Converged path tracing** — WebGL2 wrapping a forked `three-gpu-pathtracer`. Hero-quality renders that converge over seconds.

You don't pick the backend; vitrum picks for you based on your hardware and scene complexity. You hand it a `THREE.Scene`, get back an `Engine` that you call `renderFrame()` on each frame.

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

| Feature                       | walkaround-hybrid (WebGPU)  | pt-webgl (WebGL2)          |
| ----------------------------- | --------------------------- | -------------------------- |
| **GI quality**                | real-time, single-bounce GI | converged, multi-bounce PT |
| **Bounce count**              | 1 (DDGI gives multi-bounce) | unlimited                  |
| **Light types**               | point / dir / area / sky    | point / dir / area / sky   |
| **Materials**                 | PBR + transmission          | PBR + clearcoat + transmission + spectral hero-MIS |
| **Caustics**                  | none (DDGI only)            | manifold-NEE (opt-in)      |
| **Animation**                 | camera ✓ / lights limited / mesh transform ✗ | camera ✓ / lights ✓ / mesh ✗ |
| **Hardware**                  | WebGPU                      | WebGL2                     |
| **Convergence**               | re-renders every frame      | accumulates SPP            |

See [`plan/archive/animation-support-status.md`](./plan/archive/animation-support-status.md) for the full animation matrix with caveats.

## When to use which engine

`prefer: 'auto'` (default) gives you walkaround-hybrid on WebGPU + scenes < 500k tris, else pt-webgl.

Set `prefer: 'realtime'` for interactive viewers, lighting designers, scrub-the-camera demos. Set `prefer: 'quality'` for hero renders, product visualization, anything you'd want to save as a 4K PNG.

## Public API

| Package                       | What you import                                        |
| ----------------------------- | ------------------------------------------------------ |
| `@vitrum/engine`              | `createEngine`, `attachVitrum`, `<VitrumCanvas>` (React subpath) |
| `@vitrum/core`                | `Engine`, `Scene`, `FrameInput`, `FrameStats`, `ProgressStats` types |
| `@vitrum/three-bindings`      | `sceneFromThreeJS`, `loadGltfScene`                    |
| `@vitrum/dev`                 | Debug overlays (FrameTimeHUD, MaterialInspector, …) — devDep only |

Backend packages (`@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`) are also installable directly if you need backend-specific knobs that the facade doesn't surface.

## Lifecycle (the white-whale insight)

The engine accepts a device handle but **does not own** the device's lifetime. You own when the device is created, when it's lost, when it's reset. The engine owns its GPU resources for as long as you say.

This is the design choice that makes the library survive Canvas remount, route changes, tab-visibility transitions. `<VitrumCanvas>` and `attachVitrum()` already wire this correctly — you only need to think about it if you build your own host shell.

## Examples

- [`examples/cornell-box`](./examples/cornell-box) — minimal `@vitrum/pt-webgl` + `@vitrum/three-bindings` consumer (the regression-test scenes live here)
- [`examples/hero-viewer`](./examples/hero-viewer) — drag-drop glTF viewer
- [`examples/hero-lighting-designer`](./examples/hero-lighting-designer) — interactive lights + frame-time HUD
- [`examples/hero-product-viz`](./examples/hero-product-viz) — progressive PT product render with material editor

## Architecture

```
@vitrum/engine             Drop-in facade — createEngine, attachVitrum, VitrumCanvas
  ↓
@vitrum/core               Engine contract; types only, no GPU code
@vitrum/three-bindings     THREE.Scene → vitrum Scene + glTF loader
@vitrum/walkaround-hybrid  WebGPU DDGI + ReSTIR DI/GI + SVGF + GTAO + PPG + neural; composes RC
@vitrum/walkaround-rc      Radiance Cascades subsystem (cascade pyramid + GPU dispatch + receiver)
@vitrum/pt-webgl           WebGL2 PT (wraps three-gpu-pathtracer fork)
@vitrum/pt-webgpu          WebGPU-native PT (pre-alpha, internal)
@vitrum/shared-bvh         Software BVH compute (CPU + GPU)
@vitrum/shared-samplers    Sobol, Hammersley, light tree, hero-wavelength MIS
@vitrum/shared-denoisers   À-trous, SVGF, OIDN bridge
@vitrum/scene-lighting     Host-side lighting state (time-of-day, sun, sky)
@vitrum/dev                Debug overlays (devDep)
```

## Performance budgets

(Reference renders captured on RTX 4090, ANGLE D3D11.)

| Scene                       | Engine            | Resolution | Convergence target | Time / Frame   |
| --------------------------- | ----------------- | ---------- | ------------------ | -------------- |
| Cornell box (~30 tris)      | pt-webgl          | 512×512    | 64 SPP             | ~17 s total    |
| Cornell glass               | pt-webgl          | 512×512    | 64 SPP             | ~20 s total    |
| Cornell spectral (hero MIS) | pt-webgl          | 512×512    | 64 SPP             | ~20 s total    |
| Living room (~200k tris)    | walkaround-hybrid | 1080p      | real-time          | 14–22 ms / frame |

## What's novel here

- **Layered hybrid GI** — WebGPU pipeline combining diffuse probe GI (DDGI) with stochastic direct illumination (ReSTIR-DI) and a single-bounce indirect (ReSTIR-GI), denoised with per-channel SVGF + GTAO. ([packages/walkaround-hybrid/README.md](packages/walkaround-hybrid/README.md))
- **NormalMap-perturbed NEE shadow rays** — produces textured caustics through transmissive materials in pure NEE; lives in the `three-gpu-pathtracer` fork.
- **Hybrid analytic-CSG + BVH-mesh intersection** — closed-form quadrics + triangle meshes in the same path-tracing kernel. Production renderers usually pick one or the other.
- **Hero-wavelength MIS** (Wilkie et al. 2014) — one-sample MIS across X/Y/Z CMFs ships in the WebGL2 PT spectral path.

## Built on prior work

See [CREDITS.md](./CREDITS.md) for the full attribution list (~30 papers + libraries). Headline dependencies:

- **three.js** (Mr.doob et al., MIT)
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
