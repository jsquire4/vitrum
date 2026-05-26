# @vitrum/pt-webgpu

WebGPU-native experimental path tracing backend for `@vitrum/core`.

## Experimental boundary

This package is **experimental and internal-focused**. The production WebGL2
path tracer is `@vitrum/pt-webgl` (wraps the `three-gpu-pathtracer`
fork); `@vitrum/pt-webgpu` is the experimental WebGPU-native backend
that will eventually replace it once feature parity + GPU verification
land. Until then:

- **No public API stability**. Types, options, and binding shapes can
  change in any commit.
- **No GPU-verified visual reference**. The CPU/struct-layout audit
  (`plan/pt-webgpu-deep-audit.md`) is closed (all HIGH + MEDIUM + LOW
  findings fixed or NOT-A-BUG by 2026-05-19), but no end-to-end visual
  parity check against `@vitrum/pt-webgl` exists. Renders may be
  numerically correct yet visually different in ways the audit doesn't
  catch.
- **Not in `createEngine`'s `prefer: 'auto'` selection** — hosts that
  want pt-webgpu must opt in explicitly. The facade selects between
  `pt-webgl` (quality) and `walkaround-hybrid` (realtime) only.
- **Not for npm publish.** `private: true` is the publish-safety belt;
  see `RELEASING.md`.

## Implemented (verified clean)

This package is now functional (no longer a stub) and the deep-audit
findings are all closed; the experimental label is about productionisation,
not baseline correctness. What's implemented:

- Progressive accumulation renderer (compute shader)
- CPU-built BVH with GPU BVH traversal; CPU TLAS build via `buildSceneTlas()` (`scene/tlasBridge.ts`) for multi-instance follow-up
- Multi-bounce sampling (clamped by `maxBounces`)
- Material-driven diffuse/specular/emissive shading with an experimental transmission/refraction branch
- Extended packed-material payload path with bounded rich scattering/layered/thin-film/spectral fields (**22 vec4s / material**: 8 thin-film layers × `(ior, thicknessNm, extinctionCoefficient)` plus stack `incidentIor` / `angleDependent`, and 32 spectral samples)
- Procedural-sky environment lighting controls (scene-driven tint/sun direction)
- HDRI environment importance sampling when CPU-side HDRI payload provides `width`, `height`, and float RGB texel data
- Direct lighting for bounded **arrays** of emitters (counts in uniform `FrameParams`, payloads in storage buffers):
  - directional (single; legacy uniform path)
  - point emitters (up to 16)
  - spot emitters (up to 8)
  - rect-area emitters (up to 8)
  - mesh-area emitters (up to 8; first-triangle representative sampling per emitter)
- Analytic primitive intersection for `sphere`, `box`, `capsule`, `cylinder`, and `h-channel-came`
- Scene patch API methods (`updatePrimitive`, `updateEmitter`) with fast paths for transform/material/emitter changes and safe rebuild fallback for unsupported topology edits
- Auxiliary output textures:
  - `normalDepth`
  - `albedo`
  - `variance`
  - `motionVectors`

## Known limitations

- Hero-wavelength spectral parity with the WebGL fork is **not** claimed: thin-film evaluation uses **RGB wavelength probes** (see `plan/renderer-fidelity-matrix.md`).
- Experimental BRDF/MIS path — transmission hemisphere MIS uses a **simplified** PDF branch in `brdfDirectionalPdf`.
- Incremental patch support remains partial by design: `transform/material/emitter` fast paths are implemented; `positions/topology` still fall back to rebuild
- No denoiser execution path wired yet (aux outputs are now available for integration)
- `causticStrategy` requests map to mode-distinct shader paths; runtime image/perf artifact capture remains blocked in this environment

## Intended next steps

- Replace experimental BRDF path with shared sampler/BSDF contracts
- Add richer emitter coverage and MIS
- Wire aux buffers + denoiser integration
- Add visual regression scenes for GPU-verified parity checks
