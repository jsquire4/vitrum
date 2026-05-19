# @vitrum/pt-webgl

WebGL2 **path tracing** backend implementing `@vitrum/core`’s **`Engine`** contract via **[three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer)** (use the project fork — see [CLAUDE.md](../../CLAUDE.md)).

## Fork dependency

`package.json` pins the renderer as:

```json
"three-gpu-pathtracer": "file:../../../three-gpu-pathtracer"
```

Adjust the relative path if your checkout layout differs. The intended layout is `vitrum/` and `three-gpu-pathtracer/` as **sibling** directories under the same parent.

This package is marked `private` while the dependency is a sibling `file:` pin.
For public npm release, replace this with a published fork package or a
commit-pinned git dependency and remove `private`.

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
