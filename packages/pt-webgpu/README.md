# @vitrum/pt-webgpu

WebGPU-native path tracing backend prototype for `@vitrum/core`.

## Current status

This package is now functional (no longer a stub), but still pre-alpha:

- Progressive accumulation renderer (compute shader)
- CPU-built BVH with GPU BVH traversal
- Multi-bounce sampling (clamped by `maxBounces`)
- Material-driven diffuse/specular/emissive shading with a prototype transmission/refraction branch
- Extended packed-material payload path with bounded rich scattering/layered/thin-film/spectral fields (8 thin-film layers + 32 spectral samples per material)
- Procedural-sky environment lighting controls (scene-driven tint/sun direction)
- HDRI environment importance sampling when CPU-side HDRI payload provides `width`, `height`, and float RGB texel data
- Direct lighting for:
  - directional emitters
  - point emitters
  - spot emitters (cone-gated, inverse-square falloff)
  - rect-area emitters
  - mesh-area emitters (single-triangle representative sampling)
- Analytic primitive intersection for `sphere`, `box`, `capsule`, `cylinder`, and `h-channel-came`
- Scene patch API methods (`updatePrimitive`, `updateEmitter`) that currently route through a safe full-scene rebuild fallback
- Auxiliary output textures:
  - `normalDepth`
  - `albedo`
  - `variance`
  - `motionVectors`

## Known limitations

- Prototype BRDF/MIS path (not full Disney/OpenPBR fidelity yet)
- Published capabilities report `supportsIncrementalScene: false`; patch APIs currently rebuild full scene/BVH as a safe fallback
- No denoiser execution path wired yet (aux outputs are now available for integration)
- `causticStrategy` requests map to mode-distinct shader paths; runtime image/perf artifact capture remains blocked in this environment

## Intended next steps

- Replace prototype BRDF path with shared sampler/BSDF contracts
- Add richer emitter coverage and MIS
- Wire aux buffers + denoiser integration
- Add visual regression scenes for GPU-verified parity checks
