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

For glTF/GLB hosts, pass `gltf` instead of a prebuilt core scene. The component
loads through the engine-aware `loadGltfWithEngine()` bridge, so `gltfOptions`
can carry decoder hooks, texture decode settings, compatibility modes, and
backend choices. The prepared engine and glTF controller are handed to the same
`attachVitrum` lifecycle, preserving RAF/resize/device-loss behavior while
keeping `prefer="auto"` so the default path is the progressive viewer.

```tsx
<VitrumCanvas
  gltf="/models/hero.glb"
  camera={camera}
  prefer="auto"
/>
```

## Scene layout helpers

`auditSceneNeedsTlas(scene)` (from `@vitrum/core`, re-exported here) classifies multi-mesh / instanced layouts. `createEngine()` uses it for backend pick + walkaround `bvhMode: 'tlas'` defaults.

## Backend selection

`createEngine()` picks a backend from `CreateEngineOptions.prefer` (`'realtime' | 'quality' | 'quality-webgpu' | 'auto'`) plus runtime probing:
- `realtime` → `@vitrum/walkaround-hybrid` (WebGPU DDGI + ReSTIR + per-channel SVGF + GTAO)
- `quality` → `@vitrum/pt-webgl2` (single merged BVH); **`pt-webgpu`** when the scene has multiple meshes or instancing and WebGPU is available (`auditSceneNeedsTlas`)
- `quality-webgpu` → `@vitrum/pt-webgpu` when WebGPU is available, else `pt-webgl2`
- `auto` (default) → progressive viewer (walkaround while moving, PT when settled) when WebGPU satisfies the shared-device limit union; else a single PT engine (full or lite by device) or `pt-webgl2`. Triangle count is not a cutoff.
- Multi-mesh / instanced scenes on WebGL-only hosts still get `pt-webgl2` with a `console.warn` from `createEngine()`

Use **`realtime`**, **`quality`**, or **`quality-webgpu`** to force a single engine.

Backend-specific tuning goes through `CreateEngineOptions.advanced` (typed as partial backend options) — see each backend's options interface.

## Status

Pre-1.0. Companion to `@vitrum/core`.
