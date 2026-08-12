# @vitrum/pt-webgpu

Peer WebGPU-native path tracing backend for `@vitrum/core`, with full/lite
adapter tiers and row-level fidelity tracking.

## Fidelity boundary

`@vitrum/pt-webgpu` is a peer WebGPU-native path tracing backend that runs
alongside `@vitrum/pt-webgl2`, serving the same contract surface at different
quality/capability tiers. Stable options fail before allocation when the chosen
adapter tier or estimator composition cannot implement them.

- **GPU-verified per-feature.** The
  CPU/struct-layout audit (`plan/archive/pt-webgpu-deep-audit-archived-2026-05-28.md`)
  is closed (all HIGH + MEDIUM + LOW findings fixed or NOT-A-BUG by 2026-05-19),
  and as of 2026-06-04 **ten rendering rows are `supported`** in
  `plan/renderer-fidelity-matrix.md` — each GPU-validated on dzn (RTX 4090) against a
  deterministic reference (analytic / forward-traced) with a committed, sha-pinned
  baseline PNG (hero-λ spectral, spectral Beer–Lambert, thin-film TMM, Cauchy dispersion,
  layered front/back, SSS, multi-emitter, material-fields, MNEE caustic, BDPT). What does
  two backends intentionally retain different estimator and adapter profiles;
  support is stated per row rather than inferred from pixel identity.
- **`createEngine({ prefer: 'auto' })`** selects pt-webgpu when WebGPU is
  available and the scene has **≥ 500k triangles**; use `prefer:
  'quality-webgpu'` to force pt-webgpu on smaller scenes. Below 500k tris,
  `auto` picks `walkaround-hybrid` for realtime GI.
- **Not for npm publish.** `private: true` is the publish-safety belt;
  see `RELEASING.md`.

## Implemented (verified clean)

The deep-audit findings are closed. Implemented paths include:

- Progressive accumulation renderer (compute shader)
- CPU-built BVH with GPU BVH traversal; CPU TLAS build via `buildSceneTlas()` (`scene/tlasBridge.ts`) for multi-instance follow-up
- Multi-bounce sampling (clamped by `maxBounces`)
- Material-driven diffuse/specular/emissive/transmission/refraction shading
- Extended packed-material payload path with bounded rich scattering/layered/thin-film/spectral fields (**29 vec4s / material**: 8 thin-film layers × `(ior, thicknessNm, extinctionCoefficient)` plus stack `incidentIor` / `angleDependent`, 32 spectral samples, WS4 volumetric absorption coefficient σ_a, Disney/KHR scalar lobes, Jakob-Hanika spectral reflectance coefficients, `KHR_materials_specular`, and KHR volume thickness)
- Full-tier material texture path for readable `TextureRef` handles: base color/emissive sRGB maps, normal/bump with authored/generated tangent frames, roughness/metallic/AO/light/alpha/transmission/thickness maps, per-map uv0/uv1 selection plus transform/wrap metadata, and clearcoat/sheen/iridescence/specular extension-lobe maps. See the promise ledger for native vs approximate rows.
- Procedural-sky environment lighting controls (scene-driven tint/sun direction)
- HDRI environment importance sampling when CPU-side HDRI payload provides `width`, `height`, and float RGB texel data
- Direct lighting for bounded **arrays** of emitters (counts in uniform `FrameParams`, payloads in storage buffers):
  - directional (single; legacy uniform path)
  - point emitters (up to 16)
  - spot emitters (up to 8)
  - rect-area emitters (up to 8)
  - mesh-area emitters (up to 8; first-triangle representative sampling per emitter)
- **Many-light importance sampling (full tier, WS2):** the per-event light pick is power-weighted via a `@vitrum/shared-samplers` light tree (group(3) storage buffer) instead of a uniform random pick — power × spatial-proximity descent (Conty Estévez & Kulla 2018) over a power-as-cost median-split tree (Shirley et al. 1996). The NEE divides by the tree's branch-product selection pdf (unbiased); the lite tier keeps the uniform pick. Built with ≥ 2 selectable lights; falls back to uniform below that or when degenerate.
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
`maxStorageBuffersPerShaderStage ≥ 34` and `maxStorageTexturesPerShaderStage ≥ 5`
(`restirPtReuse` raises the buffer floor to 38).
That layout includes TLAS, analytic shapes, HDRI, point/spot/rect/mesh area lights,
motion vectors, variance moments, caustic strategies, and full-tier material texture
arrays/descriptors. Check the browser console
for `[vitrum/pt-webgpu] Full trace tier: …` on startup.

Caustic truthfulness: `causticStrategy: 'manifold-nee'` is the **validated
reference** caustic path — GPU-A/B'd against a forward-traced oracle it recovers
~98.7% of the true caustic energy, fires on 99.4% of caustic pixels, and is
scale-invariant (no world-unit magic constant). `causticStrategy: 'photon-map'`
is now a **real SPPM (stochastic progressive photon mapping)** path (Hachisuka &
Jensen 2009; 2026-06-10, commit `06910e2`): persistent spatial hash grid in
group-3 bindings 6–8, photon-emission pass from point/spot lights through the BVH,
camera-hit gather with α=2/3 progressive radius shrink seeded from the scene AABB.
The prior per-pixel 32-photon approximation with hardcoded `gatherRadius=0.35` and
`×1.25` brightness fudge is gone. Full-tier only (warn+degrade on lite). Radiometric
The deterministic radiometric harness records convergence and bounded mean error
against the manifold-NEE reference in `tools/radiometric-ab/results-sppm.json`.

The **lite** tier exists only as a **CI / SwiftShader fallback** (often **10** /
**4** limits in headless Chromium on Linux). WSL2 without GPU passthrough frequently
hits lite even when the machine has a GPU elsewhere — use native Windows/macOS Chrome
or pass through the GPU to WSL if you need full tier there.

Optional override: `createPTEngine_WebGPU({ device, traceTier: 'full' })` fails fast
if the adapter cannot bind the full shader (so you know limits are wrong, not that
features are missing).

When the device cannot satisfy the full layout, the factory automatically selects
**lite** (`capabilities.activeFeatures` includes `pt-webgpu-lite-tier`):

| Tier | Limits | Features |
|------|--------|----------|
| **full** | ≥32 storage buffers/stage, ≥5 storage textures/stage; opt-ins add BDPT +1, one-edge reuse +4, CWBVH +5 | TLAS, analytics, HDRI, all emitter arrays, material texture arrays/descriptors, motion vectors, variance moments, caustics (4 bind groups) |
| **lite** | ≥8 buffers, ≥4 textures | Merged-mesh BVH, directional + procedural sky, core G-buffer aux |

Host device acquisition should use `ptWebgpuRequiredLimitsForAdapter(adapter)` (not the
full-only `PT_WEBGPU_REQUIRED_LIMITS`) so `requestDevice` succeeds on lite-capable
adapters. WG-0 baseline capture (`npm run benchmark:seed-wg0`) runs on lite tier in CI.

Shader entry points: `PT_WEBGPU_TRACE_WGSL` (full), `PT_WEBGPU_TRACE_LITE_WGSL` (lite).

## Parity with `@vitrum/pt-webgl2` (contract surface)

Mechanical parity for the native WebGL2 path tracer is **implemented** for:

- Progressive path tracing, BVH (+ TLAS on full tier), multi-bounce clamp
- Packed materials (layers, thin-film stack, spectral grid, dispersion Abbe)
- Bounded multi-emitter direct lighting (full tier)
- Analytic shapes, procedural sky + HDRI (full tier)
- **Skinned-mesh pose solving (2026-06-10):** `solveSkin` runs at ingestion + re-runs on `bones`/morphTargets patches. Ledger grade `skinned-mesh: 'native'`.
- `updatePrimitive` / `updateEmitter` incremental APIs (see ledger)
- Hero-wavelength spectral (opt-in extension), Cauchy IOR at hero λ, layered MIS
- Volumetric subsurface scattering (WS4): homogeneous participating-media random walk — free-flight distance sampling (`t = -ln(1-ξ)/σ_t`), Henyey-Greenstein phase scatter, single-scatter albedo σ_s/σ_t, in-medium next-event estimation with phase↔light power-heuristic MIS, and specular-chain Beer-Lambert extinction in the caustic path. σ_t = σ_a (from `attenuationColor`/`attenuationDistance`, or the spectral curve when authored) + σ_s (`scatteringCoefficient(RGB)`); g = `scatteringAnisotropy`. BDPT carries the same medium-stack state, free-flight vertices, directional densities, and connection-edge transmittance. The compatibility (lite) tier keeps Beer-Lambert absorption only (no walk).
- `denoiser: 'auto'` / `'oidn-final'` with aux readback

**Denoisers on pt-webgpu:** `'none'`, `'auto'`, `'oidn-final'`. Omitted / `'none'` leaves the accumulator unfiltered. `'auto'` always resolves to `'oidn-final'`: host `oidn.modelUrl` if provided, otherwise the pinned Intel RT HDR alb+nrm ONNX (pmndrs/denoiser-weights `models-v1` via jsDelivr). Explicit `'oidn-final'` uses the same default when `modelUrl` is omitted. The `onnxruntime-web` optional peer is still required at the first denoise cycle. Any other explicit mode (incl. `'svgf-real'`) fails construction instead of silently selecting another estimator — SVGF is a real-time 1-spp filter, the wrong regime for a converged tracer.

**BDPT (WG-7):** `bdpt: true` with `bdptOptions.maxLightBounces` 1–8
(default 2) builds one invocation-private light subpath per camera invocation.
The full-tier kernel evaluates bounded eye↔light connections and the native
`s=n-1,t=1` light-subpath-to-camera strategy. Arbitrary-pixel t=1 contributions
use a per-frame atomic RGB splat buffer and a second resolve entry point; the
ordinary eye sample is staged into the same buffer so accumulation and variance
see one complete base+splat sample per pixel. Both sides of the MIS partition
use the same perspective-camera `Pdf_We = 1/(A cos³θ)`. CPU oracles pin camera
projection/importance algebra, finite-area endpoint radiometry, medium
sentinels, diffuse and glossy light tracing, bounded strategy ownership, and
the full Veach recurrence. Enabling BDPT raises the full-tier storage-buffer
floor from 32 to 33.

Visual sign-off uses `npm run benchmark:gap-closure` on a WebGPU-capable host (`plan/archive/WG-signoff-2026-05-26-archived-2026-05-28.md`).

## Known limitations

- **BEHAVIOR CHANGE (2026-06-10):** this backend previously returned raw linear HDR in `primaryRadiance`. The contract default (`aces` tonemap @ exposure 1.0 @ sRGB output) now applies. Adjoint/OIDN readbacks remain linear. To get raw HDR: `quality: { tonemap: 'none', outputColorSpace: 'linear' }`.
- **Lite tier** disables TLAS, analytic shapes, full-tier material texture bindings,
  explicit mesh-area emitters, motion vectors, and caustic strategies regardless
  of scene content. It does support static mesh/skinned/instanced primitives,
  directional/point/spot/rect-area/disc-area emitters, and HDRI/procedural-sky
  environments through the lite sampled-texture packing path. The lite support
  matrix reports material-map rows as unsupported so strict glTF compatibility can
  reject before render.
- **Hero-wavelength spectral** is opt-in: `extensions['vitrum.ptWebgpu.spectralHeroWavelength']`.
- **Gap-closure RFE scenarios** (`rfe03`, `rfe07`, `rfe08`, …) need hardware capture; `ptwgpu-parity-material-fields` has a committed baseline PNG.
- Incremental `positions`/`normals` (same vertex count) patch in place; vertex/index-count and instance-count changes are absorbed via a targeted BLAS/TLAS repack (`incrementalPatchSupport.topology: true`).
- **Material-lobe proof boundary:** full-tier texture-map plumbing is implemented, but some
  extension-lobe rows remain graded `approximate` until inverse/adjoint gradients and
  material-furnace/reference A/B prove the same texture-modulated parameters across the
  sampled eye, ReSTIR-PT, and BDPT paths. Private adjoint kernels cover a wider local
  direct-light derivative domain for validation, but the public path-replay contract is
  deliberately narrower: full tier, one bounce, material `emissive` only. Lite and every
  other field/transport regime must use explicit finite difference; requesting path replay
  outside the certified domain throws before creating a session or mutating scene values.
- **`denoiser: 'auto'` / `'oidn-final'` default-resolves the ONNX URL** — `auto`
  always becomes `oidn-final`. Omitted `oidn.modelUrl` uses the pinned Intel
  RT HDR alb+nrm ONNX (`DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL`). Hosts may
  override with a bundled or self-hosted file. The `onnxruntime-web` optional
  peer is still required at the first denoise cycle; a missing runtime or
  unfetchable model fails the async final-pass and is reported through the
  denoiser error state. Preview/walkaround denoisers stay à-trous/SVGF.

## Polish commands

```bash
npm run benchmark:gap-closure-mechanical
npm run benchmark:gap-closure   # WebGPU-capable host; refreshes the active gap-closure verification suite
```
