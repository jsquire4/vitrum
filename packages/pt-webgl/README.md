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
- **`causticStrategy` options** — bridged to fork-facing uniforms (`none`, `manifold-nee`, `photon-map`) with mode-distinct shader behavior. `EngineCapabilities.causticStrategy` reports the selected strategy.
- **BDPT (Sprint 10c)** — opt in via `extensions['vitrum.ptWebgl.bdpt'] = true` + `extensions['vitrum.ptWebgl.bdptMaxLightBounces']` (1–3). Then per frame: `const bdpt = new BdptLightPathBuffer({ maxLightBounces: 3 })` once, and `engine.bdptAdvanceFrame(bdpt.texture)` before each `renderFrame()`. See `examples/cornell-box` (`?vitrumBdpt=1`) for the wiring.

Helpers: **`applyFrameToPerspectiveCamera`** for advanced integration, **`BdptLightPathBuffer`** + **`BdptLightPathBufferOptions`** for the BDPT host-side ping-pong texture. For the THREE-direction adapter (`vitrumSceneToThree`), depend on `@vitrum/three-bindings`.

## Stability

**Pre-alpha** — public `Engine` shapes follow `@vitrum/core`; implementation details and supported scene subset may change. Unsupported primitives/emitters are skipped or called out in console warnings. Runtime visual/perf verification for the latest RFE paths remains required before release use.
