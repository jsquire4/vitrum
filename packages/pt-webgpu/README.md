# @vitrum/pt-webgpu

WebGPU-native path tracing backend prototype for `@vitrum/core`.

## Current status

This package is now functional (no longer a stub), but still pre-alpha:

- Progressive accumulation renderer (compute shader)
- CPU-built BVH with GPU BVH traversal
- Multi-bounce sampling (clamped by `maxBounces`)
- Material-driven diffuse/specular/emissive shading with a prototype transmission/refraction branch
- Procedural-sky environment lighting controls (scene-driven tint/sun direction)
- Direct lighting for:
  - directional emitters
  - point emitters
  - spot emitters (cone-gated, inverse-square falloff)
- Incremental scene patch API support (`updatePrimitive`, `updateEmitter`) via safe full-scene rebuild fallback
- Auxiliary output textures:
  - `normalDepth`
  - `albedo`
  - `variance`
  - `motionVectors`

## Known limitations

- Prototype BRDF/MIS path (not full Disney/OpenPBR fidelity yet)
- No analytic primitive intersection support yet
- Incremental updates currently rebuild full scene/BVH (safe fallback), not fine-grained GPU patching
- No denoiser execution path wired yet (aux outputs are now available for integration)
- HDRI environment maps are not sampled yet (prototype sky fallback)

## Intended next steps

- Replace prototype BRDF path with shared sampler/BSDF contracts
- Add richer emitter coverage and MIS
- Wire aux buffers + denoiser integration
- Add visual regression scenes for GPU-verified parity checks
