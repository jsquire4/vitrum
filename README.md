# vitrum

WebGPU + WebGL2 path tracing & global illumination engine for the browser. Backend-agnostic, scene-binding-agnostic.

> **Status**: pre-alpha, private. Not yet ready for prime time. Public release planned after the foundational packages stabilize.

## What is this

`vitrum` is a WebGPU + WebGL2 path tracing `vitrum` is the engine half of a SOTA browser rendering project. It is being developed under a host application that exercises the hard cases into a reusable, host-agnostic library. global illumination engine for the browser. Host-agnostic, scene-binding-agnostic.

The white-whale ambition: own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent API contract.

## Architecture

The package layout reflects the architectural separation between **what the engine consumes** (a scene), **how the engine renders** (a backend), and **who the host is** (any three.js / babylon.js / raw-WebGL/WebGPU app).

```
@vitrum/core                    Public façade: types, lifecycle contract, no GPU code
@vitrum/three-bindings          three.js scene → @vitrum/core Scene adapter
@vitrum/shared-bvh              Software BVH compute (WebGPU + WebGL2)
@vitrum/shared-samplers         Sobol, Hammersley, light tree, mixture PDF
@vitrum/shared-denoisers        À-trous, SVGF, BMFR, OIDN bridge
@vitrum/pt-webgl                WebGL2 path-tracer backend (wraps three-gpu-pathtracer fork)
@vitrum/pt-webgpu               WebGPU-native path-tracer backend (Phase 7+, the white whale)
@vitrum/walkaround-hybrid       WebGPU layered DDGI + RC + ReSTIR DI engine
```

## The contract (in three sentences)

The library accepts a **device handle** + a **Scene** and exposes an **Engine** with `setScene`, `renderFrame`, `pause`, `resume`, `reset`, `dispose`. The host owns the device's lifetime, the frame cadence, the scene state, and the camera state. The engine owns its GPU resources for as long as the host says.

That contract is what dissolves the "render mode toggle remounts the Canvas, accumulator vanishes" bug class structurally. The engine never assumes its lifetime matches a particular React mount — its lifetime is exactly `init` to `dispose`, owned by the host.

## What's already novel here

- **Layered hybrid GI in one WebGPU compute pipeline** — DDGI + Radiance Cascades + ReSTIR DI in additive composition. First of its kind in browser rendering.
- **NormalMap-perturbed NEE shadow rays** — produces textured caustics through transmissive materials in pure NEE. Originated as a patch on three-gpu-pathtracer; will land in `@vitrum/pt-webgl` and `@vitrum/pt-webgpu`.
- **Hybrid analytic-CSG + BVH-mesh intersection** — closed-form quadrics (came/solder, gemstones) + triangle meshes (glass, panels, room) in the same path-tracing kernel. Production renderers usually pick one or the other.

## Built on prior work

vitrum depends on the foundational work of many others. See [CREDITS.md](./CREDITS.md) for the full attribution list. Headline dependencies:

- **three.js** (Mr.doob et al., MIT)
- **three-gpu-pathtracer** (Garrett Johnson, MIT)
- **three-mesh-bvh** (Garrett Johnson, MIT)
- **DDGI** (Majercik et al., 2019)
- **Radiance Cascades** (Sannikov, 2024)
- **ReSTIR DI** (Bitterli et al., 2020)
- **Disney BSDF** (Burley, 2012)

## License

MIT. See [LICENSE](./LICENSE).
