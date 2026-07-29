# Renderer fidelity matrix

Last graded: 2026-07-24 (direct implementation audit).

This matrix reports the implemented renderer contract for `@vitrum/pt-webgl2`
(native WebGL2) and the full tier of `@vitrum/pt-webgpu` (WebGPU-native). The
former fork-backed `@vitrum/pt-webgl` backend was removed with commit `e14000c`.

Classification is based on the code path that actually runs: public option or
scene-field acceptance, CPU packing/upload, shader consumption, estimator
ownership/PDF logic, fail-closed validation, and executable tests. Captures and
host-specific runs remain useful regression evidence, but are not prerequisites
for an implementation-maturity grade and are intentionally outside this matrix.

## Legend

| Tag | Meaning |
|-----|---------|
| supported | Implemented end to end and executable within the stated contract/envelope. |
| approximate | Implemented end to end with a deliberate, material model simplification stated in the row. |
| unsupported | Not selectable on that backend, or explicitly rejected before rendering. |

## Feature rows

| Feature | pt-webgl2 (WebGL2) | pt-webgpu full tier (WebGPU) | Source and executable evidence | Contract / modeled envelope |
|---------|--------------------|--------------------------------|--------------------------------|-----------------------------|
| Hero-wavelength + CMF accumulation | supported | supported | pt-webgl2: `gl/frameUniformsPacker.ts`, `gl/uploadFrameUniforms.ts`, `glsl/renderMain.glsl.ts`, `__tests__/uploadGapGuard.test.ts`, `glsl/spectralGlslParity.test.ts`; pt-webgpu: `__tests__/heroWavelengthPlumbing.test.ts`, `__tests__/spectralProductionClosure.test.ts` | One sampled hero wavelength is reconstructed once through the CIE CMFs; RGB mode remains separately selectable. |
| Spectral Beer–Lambert (packed μ) | supported | supported | pt-webgl2: `scene/materialsTexture.ts`, `glsl/render/attenuate_hit_function.glsl.js`, `materialsTexture.test.ts`; pt-webgpu: `scene/materialPacking.ts`, `scenePack.materials.test.ts`, `spectralProductionClosure.test.ts` | pt-webgl2 stores the authored curve on a 32-sample wavelength grid; pt-webgpu consumes its packed spectral material representation. |
| Multi-layer thin-film TMM | supported | supported | pt-webgl2: `scene/materialsTexture.ts`, `glsl/shader/bsdf/thin_film_tmm.glsl.js`, `thinFilmLayerLimit.test.ts`; pt-webgpu: `thinFilmProductionClosure.test.ts`, `wgslContract.test.ts`; `core/src/engine/promiseLedger.ts` publishes both limits | Fail-closed layer cap: 35 layers on pt-webgl2 and 8 on pt-webgpu. Per-wavelength transfer-matrix evaluation is used inside the active BSDF. |
| Cauchy / Abbe dispersion | supported | supported | pt-webgl2: `gl/uploadFrameUniforms.ts`, `glsl/shader/bsdf/bsdf_functions.glsl.js`, `uploadGapGuard.test.ts`, `spectralGlslParity.test.ts`; pt-webgpu: `scene/materialPacking.ts`, `wgsl/pathTrace/bsdf.wgsl.ts`, `spectralProductionClosure.test.ts` | Active when spectral rendering is enabled; authored Abbe/dispersion strength alters the wavelength-dependent transmission IOR and its matched PDF. |
| Layered front/back + transmission MIS | supported | supported | core: `scene/material.ts`; pt-webgl2: `scene/materialsTexture.ts`, `glsl/render/get_surface_record_function.glsl.js`, `glsl/shader/bsdf/bsdf_functions.glsl.js`, `materialStrideParity.test.ts`; pt-webgpu: `wgsl/pathTrace/material.wgsl.ts`, `wgsl/pathTrace/bsdf.wgsl.ts`, `wgslContract.test.ts` | Supported against the core contract: infinitesimally thin absorption/tint layers without multiple scattering. Face selection, nested normal payloads, Walter transmission sampling, and matched selection/PDF terms are consumed. |
| SSS / translucent panels | approximate | supported | pt-webgl2: `scene/materialsTexture.ts`, `glsl/shader/bsdf/bsdf_functions.glsl.js`, `composeTraceGlsl.test.ts`; pt-webgpu: `volumetricSss.test.ts`, `wgslContract.test.ts` | pt-webgl2 deliberately uses one back-face single-scatter event with a scalar free-flight majorant, per-channel σs/σt albedo, and HG phase. pt-webgpu uses its native per-channel volume transport path. |
| Multi-emitter direct lighting | supported | supported | pt-webgl2: `scene/lightsTexture.ts`, `scene/meshAreaLights.ts`, `glsl/neeEstimator.test.ts`, `scene/meshAreaMis.test.ts`, `composeTraceGlsl.test.ts`; pt-webgpu: `scene/emitterPacking.ts`, `scenePack.emitters.test.ts`, `lightTreeImportance.test.ts`, `wgslContract.test.ts` | Both backends use exact selector PDFs for their power-weighted analytic and mesh-triangle streams and account for environment/distant families in the estimator denominator. |
| Cornell/core material fixture parity | supported | supported | pt-webgl2: `scene/materialsTexture.test.ts`, `scene/materialStrideParity.test.ts`, `glsl/composeTraceGlsl.test.ts`; pt-webgpu: `__tests__/scenePack.materials.test.ts`, `__tests__/wgslContract.test.ts` | This row is strictly the named core fixture, not a blanket claim that every `MaterialSpec` field has identical semantics. Field-level differences remain authoritative in `BACKEND_PROMISE_LEDGER`. |
| Manifold next-event estimation (MNEE) | unsupported | supported | pt-webgl2: `options.ts` and `options.validate.ts` accept only the `bdpt` caustic strategy; pt-webgpu: `scene/mneeFacetCandidates.ts`, `wgsl/pathTrace/mneeNewton.wgsl.ts`, `mneeBoundedChain.test.ts`, `mneeFacetCandidates.test.ts`, `mneeEstimatorInvariance.test.ts` | pt-webgpu solves chains of 1–8 planar, geometric-normal mesh/instanced/skinned delta interfaces. Analytic interfaces, varying interface normals, and normal/bump/layer-normal mapped interfaces fail closed before upload. Volume scattering is outside MNEE. |
| Progressive photon mapping (SPPM; `photon-map`) | unsupported | supported | pt-webgl2: strict option union/validator has no photon-map value; pt-webgpu: `wgsl/pathTrace/caustic.wgsl.ts`, `wgsl/pathTrace/sppmBindings.wgsl.ts`, `sppmProductionClosure.test.ts`, `sppmPhotonEmission.test.ts` | Full-tier pt-webgpu only. Persistent progressive surface and homogeneous-medium state uses separate disk/sphere density updates and fails construction/dispatch when required buffers or pipelines are unavailable. |
| SVGF-real denoiser | unsupported | unsupported | pt-webgl2: `options.ts`, `options.validate.ts`, `engineContract.test.ts`; pt-webgpu: `index.ts`, `unsupportedDenoiserDegrade.test.ts`; core promise ledger | Both converged tracers accept `none`, `auto`, or `oidn-final`. Realtime-only denoiser names are rejected rather than silently degraded. SVGF remains available to the realtime hybrid backend. |
| BDPT (eye↔light connections) | supported | supported | pt-webgl2: `glsl/renderMain.glsl.ts`, `glsl/render/bdpt_connection.glsl.js`, `__tests__/bdptProductionEstimator.test.ts`; pt-webgpu: `wgsl/pathTrace/kernel.wgsl.ts`, `wgsl/bdpt/{bdptConnection,bdptCameraSplat}.wgsl.ts`, `__tests__/{bdptEstimatorOwnership,bdptCameraSplatCpu,bdptCameraSplatWiring}.test.ts`; both option validators | Bounded general BDPT with 1–8 stored light vertices (defaults: pt-webgl2 4, pt-webgpu 2). Finite c=0 and c≥1 surface/medium connections use Veach power-heuristic MIS. pt-webgpu also executes `s=n-1,t=1` light-subpath-to-camera splats with matched perspective-camera densities and atomic arbitrary-pixel accumulation. Ordinary-eye finite NEE is disabled under BDPT while distant families retain a disjoint ownership partition. |

## Maintainer gate

Any future grade change must identify the selectable public contract, the
packer/upload path, the consuming shader branch, estimator ownership/PDF where
applicable, the modeled envelope or rejection boundary, and an executable test.
`npm run renderer-fidelity-proof-check` pins the named source/evidence paths and
required identifiers; it does not execute the cited tests. Focused workspace
tests (or `npm test`) execute those tests, while `npm run typecheck` checks the
typed contracts.
