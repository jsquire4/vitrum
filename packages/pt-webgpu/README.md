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
- **`createEngine({ prefer: 'auto' })`** selects pt-webgpu when WebGPU is
  available and the scene has **≥ 500k triangles**; use `prefer:
  'quality-webgpu'` to force pt-webgpu on smaller scenes. Below 500k tris,
  `auto` picks `walkaround-hybrid` for realtime GI.
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

## Adapter tiers (storage-buffer limits)

**If you have a normal discrete GPU, you already get the full path.** Nothing is
“turned off” in code — the engine calls `resolvePtWebgpuTraceTier(device)` at
construction and picks **`full`** whenever the device reports
`maxStorageBuffersPerShaderStage ≥ 23` and `maxStorageTexturesPerShaderStage ≥ 5`.
That layout includes TLAS, analytic shapes, HDRI, point/spot/rect/mesh area lights,
motion vectors, variance moments, and caustic strategies. Check the browser console
for `[vitrum/pt-webgpu] Full trace tier: …` on startup.

The **lite** tier exists only as a **CI / SwiftShader fallback** (often **10** /
**4** limits in headless Chromium on Linux). WSL2 without GPU passthrough frequently
hits lite even when the machine has a GPU elsewhere — use native Windows/macOS Chrome
or pass through the GPU to WSL if you need full tier there.

Optional override: `createPTEngine_WebGPU({ device, traceTier: 'full' })` fails fast
if the adapter cannot bind the full shader (so you know limits are wrong, not that
features are missing).

When the device cannot satisfy the full layout, the factory automatically selects
**lite** (`capabilities.experimentalFeatures` includes `pt-webgpu-lite-tier`):

| Tier | Limits | Features |
|------|--------|----------|
| **full** | ≥10 buffers/group, ≥5 textures | TLAS, analytics, HDRI, all emitter arrays, motion vectors, variance moments, caustics (3 bind groups) |
| **lite** | ≥8 buffers, ≥4 textures | Merged-mesh BVH, directional + procedural sky, core G-buffer aux |

Host device acquisition should use `ptWebgpuRequiredLimitsForAdapter(adapter)` (not the
full-only `PT_WEBGPU_REQUIRED_LIMITS`) so `requestDevice` succeeds on lite-capable
adapters. WG-0 baseline capture (`npm run benchmark:seed-wg0`) runs on lite tier in CI.

Shader entry points: `PT_WEBGPU_TRACE_WGSL` (full), `PT_WEBGPU_TRACE_LITE_WGSL` (lite).

## Parity with `@vitrum/pt-webgl` (contract surface)

Mechanical parity for the fork-backed WebGL2 path tracer is **implemented** for:

- Progressive path tracing, BVH (+ TLAS on full tier), multi-bounce clamp
- Packed materials (layers, thin-film stack, spectral grid, dispersion Abbe)
- Bounded multi-emitter direct lighting (full tier)
- Analytic shapes, procedural sky + HDRI (full tier)
- `updatePrimitive` / `updateEmitter` incremental APIs (see ledger)
- Hero-wavelength spectral (opt-in extension), Cauchy IOR at hero λ, layered MIS, translucent SSS gate
- `denoiser: 'oidn-final'` with aux readback

**Denoisers on pt-webgpu:** `'none'`, `'oidn-final'`, `'svgf-real'` (full tier).

**BDPT (WG-7):** `extensions['vitrum.ptWebgpu.bdpt'] = true`, `bdptMaxLightBounces` 1–3, optional `engine.bdptAdvanceFrame(view)`; CPU-filled light-path texture + `evaluateBdptConnection` in the full-tier kernel.

Visual sign-off uses `npm run benchmark:gap-closure` on a WebGPU-capable host (`plan/WG-signoff-2026-05-26.md`).

## Known limitations

- **Lite tier** disables TLAS, analytic shapes, HDRI texel buffers, point/spot/area lights,
  motion vectors, and caustic strategies regardless of scene content.
- **Hero-wavelength spectral** is opt-in: `extensions['vitrum.ptWebgpu.spectralHeroWavelength']`.
- **Gap-closure RFE scenarios** (`rfe03`, `rfe07`, `rfe08`, …) need hardware capture; `ptwgpu-parity-material-fields` has a committed baseline PNG.
- Incremental `positions`/`normals` require unchanged vertex count; topology edits full-repack.
## Polish commands

```bash
npm run benchmark:gap-closure-mechanical
npm run benchmark:gap-closure-gpu --workspace @vitrum/benchmark-runner   # refresh smoke baselines
npm run benchmark:pr-hybrid-refs --workspace @vitrum/benchmark-runner    # PR-D6 PNG dirs (hybrid GPU)
```
