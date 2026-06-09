# WS5 — renderFrame loop, frame-params packer, trace-tier, and the feature port

> Ties WS1–WS4 together into a working accumulating engine, then ports the fork's vitrum-specific extensions feature-by-feature.

## 1. `renderFrame` — the accumulation loop (mirror pt-webgpu `index.ts:674-873`, adapted to FBO ping-pong)

```ts
renderFrame(input: FrameInput): FrameOutput {
  this.#guardLive('renderFrame');
  if (!this.#inInverseRender) this.#lastFrameInput = input;
  const t0 = performance.now?.() ?? Date.now();
  const gpu = this.#gpu;

  // paused → return current accum without drawing (mirror pt-webgpu paused fast-out)
  if (this.#slot.get() === 'paused') return this.#frameOutputOrSkip(gpu, input, t0);

  const q = input.quality ?? {};
  this.#activeBounces = Math.max(1, Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit));
  const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
  const res = q.resolutionFactor ?? 1;
  const w = Math.max(1, Math.floor(input.viewport.width * res));
  const h = Math.max(1, Math.floor(input.viewport.height * res));

  if (gpu.ensureAccumResources(w, h)) this.#samplesAccumulated = 0;   // recreate → reset counter
  gpu.ensureProgram(this.#traceFeatures());
  if (this.#sceneTextures == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };

  // already converged → fast-out without drawing (this is how accumulation terminates)
  if (this.#samplesAccumulated >= targetSpp)
    return this.#frameRendered(gpu.resultTexture(), this.#samplesAccumulated, true, t0, targetSpp);

  // pack + upload the FrameParams UBO (§2)
  const params = packFrameParams(this.#engineConfig(), this.#sceneInputs(), input, w, h);
  uploadUbo(this.#gl, gpu.paramsUbo, params);

  // one accumulation draw (WS2 §6) — regime chosen from EXT_float_blend + backgroundAlpha
  gpu.drawAccumStep(this.#sceneTextures, this.#regime(input), input.frameSeed);
  this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);

  const isConverged = this.#samplesAccumulated >= targetSpp;
  if (this.#postDenoiser && isConverged) this.#postDenoiser.kickIfReady(/* color, albedo, normalDepth readback, w, h */);

  return this.#frameRendered(gpu.resultTexture(), this.#samplesAccumulated, isConverged, t0, targetSpp);
}

#frameRendered(tex, samples, isConverged, t0, target): FrameRendered {
  this.#emitProgress({ kind: 'pt-spp', current: samples, target, fraction: Math.min(samples / target, 1) });
  this.#emitFrame({ frameTimeMs: (performance.now?.() ?? Date.now()) - t0, spp: samples });
  return {
    kind: 'rendered',
    primaryRadiance: asBackendTexture<'pt-webgl2', WebGLTexture>(tex),
    ...(this.#gpu.normalDepthTex ? { normalDepth: asBackendTexture<'pt-webgl2', WebGLTexture>(this.#gpu.normalDepthTex) } : {}),
    ...(this.#gpu.albedoTex ? { albedo: asBackendTexture<'pt-webgl2', WebGLTexture>(this.#gpu.albedoTex) } : {}),
    samplesAccumulated: samples, isConverged,
  };
}
```
Telemetry shapes are fixed by core: `ProgressStats.kind ∈ {'pt-spp','denoiser-converge','ddgi-warmup'}`; `FrameStats.frameTimeMs` required. Use `'pt-spp'`.

## 2. `frameParamsPacker.ts` + generated layout (adopt the codegen freebie)

Mirror pt-webgpu's `packFrameParams` + `frameParamsLayout.generated.ts` (`tools/generate-wgsl-layouts.mjs`). Generate a **GLSL `FrameParams` std140 UBO struct** + the matching `FrameParamsSlot` packer offsets from one source, so the GLSL block and the TS packer can't drift. Fields = the per-frame scalars/matrices from the schema (WS4 §1): `resolution`, `bounces`, `transmissiveBounces`, `seed`, `filterGlossyFactor`, `uRadianceClamp`, `cameraWorldMatrix`, `invProjectionMatrix`, the spectral scalars (`uSpectralRendering`, `iorCauchy*`, the CMF integrals), `uCausticStrategy`, `uMnee*`, light/env counts, `uBdpt*`. The big arrays (`uCmfX/Y/Z`[81], CDFs[82]) and the samplers stay as direct uniforms/textures (UBO std140 array padding is wasteful for them).
```ts
export function packFrameParams(cfg: FrameParamsEngineConfig, sb: FrameParamsSceneInputs, input: FrameInput, w: number, h: number): ArrayBuffer {
  // compute cameraWorldMatrix = inverse(viewMatrix); invProjectionMatrix = inverse(projMatrix) (throw if singular)
  // allocate std140-sized ArrayBuffer; write each scalar at FrameParamsSlot.<name>; .set(mat4) at the mat slots; return.
}
```

## 3. `traceTier.ts` — the WebGL2 capability gate (mirror `selectPtWebgpuTraceTier`)

pt-webgpu gates on storage-buffer/texture limits; WebGL2 has no SSBOs, so gate on the relevant WebGL2 caps:
```ts
export function selectWebGl2TraceTier(gl: WebGL2RenderingContext): 'full' | 'lite' {
  const floatColor = !!gl.getExtension('EXT_color_buffer_float');   // RGBA32F renderable — required at all
  const floatBlend = !!gl.getExtension('EXT_float_blend');          // additive HDR accumulation (Regime 1/3)
  const drawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;        // MRT G-buffer
  const texUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number;    // sampler budget
  const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;         // BVH/material square texture cap
  if (!floatColor) throw new Error('pt-webgl2: EXT_color_buffer_float required (RGBA32F render targets)');
  if (drawBuffers >= 3 && texUnits >= 12 && maxTexSize >= 8192) return 'full';
  return 'lite';   // single-output (no MRT aux), smaller textures, fewer bounces, no HDRI
}
export function resolveWebGl2TraceTier(gl, force?: 'full'|'lite') { /* force overrides, else select; throws if force:'full' unmet */ }
```
`lite` tier feeds graceful degradation (the robustness win, [`08`](./08-freebies-and-future.md)): `supportsAuxBuffers:false`, drop HDRI from `PT_WEBGL2_SUPPORT`, cap bounces, `rgba16f` accumulation if `rgba32f` isn't renderable.

## 4. Incremental patches (`updatePrimitive`/`updateEmitter`/`updateEnvironment`/`updateLighting`)

Mirror pt-webgl's `scenePatch.ts` semantics but THREE-free, driving `@vitrum/shared-bvh`'s incremental entry points (`rebuildPrimitiveBlas`, `refitTlasTransforms`, `rebuildTlasReuseBlas` — `scenePack.ts:954/1161/1235`):
- `updatePrimitive(id, {transform})` → `refitTlasTransforms` (no rebuild) → re-upload the changed TLAS instance matrices texture → `reset()`.
- `updatePrimitive(id, {material})` → re-pack just that material's 85px slot → `gl.texSubImage2D` the materials texture → `reset()`.
- `updatePrimitive(id, {positions})` → `rebuildPrimitiveBlas` → re-upload BVH + position textures → `reset()`.
- `updateEmitter` → re-pack the 6px light slot. `updateEnvironment` → rebuild equirect+CDF. `updateLighting` → update sun/env scalars in the UBO.
All invalidate accumulation (`reset()`), matching the contract's "incremental edits invalidate accumulation."

## 5. The feature port (S3) — each is its own A/B gate

The GLSL for every feature is KEPT (WS4); each feature is "drive the right uniforms + wire the host pass." Where pt-webgpu has a validated peer, borrow its deterministic math harness as an independent oracle.

| Feature | Uniforms / passes to wire | pt-webgpu parity opportunity |
|---|---|---|
| **Spectral hero-λ** | `uSpectralRendering=1`, `uCmfX/Y/Z`[81], CDFs[82], `iorCauchyA/B/C`, `u_jakobCoeffs` (Jakob-Hanika RGB→spectrum). The `spectral_accumulator.glsl` kernel is kept. | pt-webgpu's spectral path is validated; reuse its CMF tables + a deterministic spectral A/B. |
| **MNEE caustics** | `uCausticStrategy∈{1,2}`, `uMneeMaxIterations/ChainLength`. Kept `mneeNewton`-equivalent GLSL. | **Replace** the fork's phenomenological `pow(dot,10)` with pt-webgpu's validated manifold-NEE math (the audit's P1.7 fix) — port the WGSL kernel to GLSL. |
| **BDPT** | `FEATURE_BDPT=1` define, `uBdpt*` uniforms, the light-subpath scratch-RT pass (`PathTracingRenderer.js:496-561`) → an RGBA32F W×3 target, column-copy between bounces (WebGL forbids sampling the active RT). | pt-webgpu BDPT is complete; use it as the reference for connection-MIS weights. |
| **Jakob-Hanika upsampling** | `u_jakobCoeffs` from the RGB→spectrum LUT (CPU). | shared with spectral. |
| **Additive accumulation** | `FEATURE_ADDITIVE_ACCUM=1` define + Regime 1 blend (`ONE/ONE`, host clears to 0). Needs `EXT_float_blend`. | n/a (WebGL-specific). |

## 6. WS5 done-when
- The diffuse spine renders a converging Cornell (SPP increments, `isConverged` flips at `targetSpp`, telemetry emits `pt-spp`).
- `packFrameParams` matches a golden std140 layout; the generated GLSL UBO struct compiles.
- Incremental patches re-upload only the changed texture + `reset()` (no full rebuild) — unit-tested via a GL-call spy (mirror `rcMergedRefit`/`positionsRefitTlas`).
- Each S3 feature passes its A/B vs the fork baseline (WS6).
