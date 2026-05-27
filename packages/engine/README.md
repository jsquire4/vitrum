# @vitrum/engine

Drop-in entry point: `createEngine()` picks a backend, derives defaults from scene AABB, and returns the `@vitrum/core` Engine contract. Vanilla `attachVitrum` + React `VitrumCanvas` helpers included.

## Hello world

```ts
import { attachVitrum } from '@vitrum/engine/lifecycle';

const handle = await attachVitrum({ canvas, scene, camera });
// … later
handle.dispose();
```

`attachVitrum` runs the requestAnimationFrame loop, attaches a `ResizeObserver`, pauses on tab-hidden, and disposes the engine cleanly. Use `createEngine()` directly if you want to drive the loop yourself.

## React

```tsx
import { VitrumCanvas } from '@vitrum/engine/react';

<VitrumCanvas
  scene={scene}
  camera={camera}
  prefer="realtime"
  onFrame={(s) => setFps(Math.round(1000 / s.frameTimeMs))}
/>
```

## Scene layout helpers

`auditSceneNeedsTlas(scene)` (from `@vitrum/core`, re-exported here) classifies multi-mesh / instanced layouts. `createEngine()` uses it for backend pick + walkaround `bvhMode: 'tlas'` defaults.

## Backend selection

`createEngine()` picks a backend from `CreateEngineOptions.prefer` (`'realtime' | 'quality' | 'quality-webgpu' | 'auto'`) plus runtime probing:
- `realtime` → `@vitrum/walkaround-hybrid` (WebGPU DDGI + ReSTIR + per-channel SVGF + GTAO)
- `quality` → `@vitrum/pt-webgl` (single merged BVH); **`pt-webgpu`** when the scene has multiple meshes or instancing and WebGPU is available (`auditSceneNeedsTlas`)
- `quality-webgpu` → `@vitrum/pt-webgpu` when WebGPU is available, else `pt-webgl`
- `auto` → `walkaround-hybrid` (&lt;500k tris, WebGPU) or **`pt-webgpu`** (≥500k tris, WebGPU), else `pt-webgl`
- Multi-mesh / instanced scenes on WebGL-only hosts still get `pt-webgl` with a `console.warn` from `createEngine()`
- Denoiser `'svgf-real'` → `realtime` (hybrid) or `quality-webgpu` (pt-webgpu full tier)

Use **`quality-webgpu`** to force the WebGPU path tracer on smaller scenes.

Backend-specific tuning goes through `CreateEngineOptions.advanced` (typed as partial backend options) — see each backend's options interface.

## Status

Pre-1.0. Companion to `@vitrum/core`.
