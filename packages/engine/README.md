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

## Backend selection

`createEngine()` picks a backend from `CreateEngineOptions.prefer` (`'realtime' | 'quality' | 'auto'`) plus runtime probing:
- `realtime` → `@vitrum/walkaround-hybrid` (WebGPU DDGI + ReSTIR + per-channel SVGF + GTAO)
- `quality` → `@vitrum/pt-webgl` (WebGL2 path tracing via the three-gpu-pathtracer fork)
- `auto` → picks based on `probeWebGPU()` + scene complexity (`@vitrum/engine/sceneAABB`)

Backend-specific tuning goes through `CreateEngineOptions.advanced` (typed as partial backend options) — see each backend's options interface.

## Status

Pre-1.0. Companion to `@vitrum/core`.
