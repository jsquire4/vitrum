# @vitrum/pt-webgl

WebGL2 **path tracing** backend implementing `@vitrum/core`’s **`Engine`** contract via **[three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer)** (use the project fork — see [CLAUDE.md](../../CLAUDE.md)).

## Fork dependency

`package.json` pins the renderer as:

```json
"three-gpu-pathtracer": "file:../../../three-gpu-pathtracer"
```

Adjust the relative path if your checkout layout differs. The intended layout is `vitrum/` and `three-gpu-pathtracer/` as **sibling** directories under the same parent.

Peers: `three`, `three-mesh-bvh` (required by the path tracer). Optional: `xatlas-web` (UV unwrap path in the upstream library).

## API

- **`createPTEngine_WebGL2({ device })`** — `device` is a `THREE.WebGLRenderer` with a **WebGL2** context.
- **`setScene(scene)`** — accepts a `@vitrum/core` `Scene` (e.g. from `sceneFromThreeJS` in `@vitrum/three-bindings`).
- **`renderFrame(FrameInput)`** — drives **three-gpu-pathtracer**’s `WebGLPathTracer` (samples, bounces, resolution).

Helpers: **`vitrumSceneToThree`**, **`applyFrameToPerspectiveCamera`** for advanced integration.

## Stability

**Pre-alpha** — public `Engine` shapes follow `@vitrum/core`; implementation details and supported scene subset may change. Unsupported primitives/emitters are skipped or called out in console warnings.
