# @vitrum/pt-webgl

WebGL2 **path tracing** backend implementing `@vitrum/core`’s **`Engine`** contract via the absorbed [`three-gpu-pathtracer`](../three-gpu-pathtracer/) workspace package.

## Renderer dependency

`package.json` pins the renderer as an in-repo workspace file dependency:

```json
"three-gpu-pathtracer": "file:../three-gpu-pathtracer"
```

The renderer package is intentionally kept package-shaped rather than folded
directly into `pt-webgl/src`, so the backend wrapper and path-tracer
implementation remain separately understandable. The old sibling checkout is
no longer required for vitrum development.

Peers: `three`, `three-mesh-bvh` (required by the path tracer). Optional: `xatlas-web` (UV unwrap path in the upstream library).

## API

- **`createPTEngine_WebGL2({ device })`** — `device` is a `THREE.WebGLRenderer` with a **WebGL2** context.
- **`setScene(scene)`** — accepts a `@vitrum/core` `Scene` (e.g. from `sceneFromThreeJS` in `@vitrum/three-bindings`).
- **`renderFrame(FrameInput)`** — drives **three-gpu-pathtracer**'s `WebGLPathTracer` (samples, bounces, resolution).
- **`causticStrategy` options** — bridged to fork-facing uniforms (`none`, `manifold-nee`, `photon-map`) with mode-distinct shader behavior. These fork modes are approximate rather than radiometrically promoted; `EngineCapabilities.causticStrategy` reports the selected strategy and `experimentalFeatures` includes `pt-webgl-caustic-approximate` when either non-`none` mode is active.
- **BDPT (Sprint 10c)** — opt in via `extensions['vitrum.ptWebgl.bdpt'] = true` + `extensions['vitrum.ptWebgl.bdptMaxLightBounces']` (1–3). Per frame: `engine.fillBdptLightPath(bdpt, frameSeed)` then `engine.bdptAdvanceFrame(bdpt.texture)` before `renderFrame()`. Hardware GL runs the fork GPU light-subpath pass (scratch RT + column upload; see fork `PathTracingRenderer.renderBdptLightSubpathPass`). SwiftShader/WSL headed captures validate the GPU subpath; Windows ANGLE may still need `vitrumBdptCpuFill=1` until the host driver path is verified. Use `npm run benchmark:bdpt-layered-refs-gpu` (WSL vite + Windows Playwright); `benchmark:bdpt-layered-refs-gpu-wsl` for WSL-only headed capture; promotion skips captures under 50KB (400KB for Windows hardware script).

Helpers: **`applyFrameToPerspectiveCamera`** for advanced integration, **`BdptLightPathBuffer`** + **`BdptLightPathBufferOptions`** for the BDPT host-side ping-pong texture. For the THREE-direction adapter (`vitrumSceneToThree`), depend on `@vitrum/three-bindings`.

## Stability

**Release-candidate track** — public `Engine` shapes follow `@vitrum/core`. Unsupported primitives/emitters are skipped or warned. **`updatePrimitive` with `{ material }` only** re-packs `MaterialsTexture` via `WebGLPathTracer.updateMaterials()` (no BVH rebuild, PR-8). **`updatePrimitive` with `{ transform }` only** updates the internal THREE mesh matrix and runs fork `PathTracingSceneGenerator` BVH refit (`GEOMETRY_ADJUSTED`) without full `setScene()` (PR-8b). **`{ positions }` only** (same vertex count, optional `normals`) patches the THREE `BufferAttribute` and regenerates merged geometry + BVH without full `setScene()` (PR-8c). Vertex/index-count and instance-count changes are absorbed via an internal geometry + BVH regeneration (`regenerateSceneGeometry`) without a host `setScene()`, per `incrementalPatchSupport.topology: true` (a co-present `material` patch routes to full rebuild). `updateEmitter` uses `updateLights()` when emitter-only (PR-8). See `EngineCapabilities.incrementalPatchSupport`.
