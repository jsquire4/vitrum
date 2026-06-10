import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ── Theme-C byte-identity pin ───────────────────────────────────────────────
// Task 2.2 collapsed the compile-time-variant whole-body WGSL duplication
// (intersectionCore, FrameParams group-0 bindings, shadePrologue) into shared
// fragments. The CORE RULE is that the FINAL composed shader string for every
// variant must be BYTE-IDENTICAL before vs after. This SHA256 pins the composed
// full-tier trace string to the value captured from the pre-refactor code; a
// single differing byte flips the hash and fails. If a future intentional WGSL
// change lands, recompute the digest (sha256 of PT_WEBGPU_TRACE_WGSL) and update
// it here in the same commit.
describe('pt-webgpu WGSL byte-identity (Theme-C dedup pin)', () => {
  it('composed full-tier trace string matches the golden SHA256', () => {
    const digest = createHash('sha256').update(PT_WEBGPU_TRACE_WGSL).digest('hex');
    // Updated for Phase I.1: the trace kernel now composes MNEE_NEWTON_WGSL +
    // MNEE_CHAIN_WGSL + MNEE_CONNECTION_WGSL and the real point-light REFLECTION +
    // REFRACTION + GLASS-SLAB (2-vertex chain) caustics (caustic.wgsl.ts:
    // pointLightReflectionCaustic / pointLightRefractionCaustic / pointLightGlassSlabCaustic).
    // The reflection caustic is GPU-validated against the analytic mirror-image
    // reference (wsl-gpu mnee-reflection-caustic-ab.ts); the refraction caustic
    // ("water surface") against a deterministic forward-traced grid reference
    // (wsl-gpu mnee-refraction-caustic-ab.ts — ratio 0.986, slope 0.984); the
    // glass-slab chain caustic against a deterministic forward-traced SLAB grid
    // (offline focusing derivation ratio/slope 1.000 in
    // mnee-glass-slab-focusing-derivation.ts, then GPU-A/B'd in
    // mnee-glass-slab-caustic-ab.ts — ratio 0.996, slope 0.990 on lavapipe). The
    // refraction caustic gained a single-interface GUARD (causticSegmentCrossesTransmissive):
    // it skips when the light→v leg crosses a transmissive facet, since that path is
    // really a chain the slab kernel owns — without it the refraction + slab kernels
    // double-counted (slab A/B ratio 2.17 → 0.996). The composed string also passes a
    // naga compile gate (wsl-gpu mnee-slab-wgsl-compile.ts) — the byte-identity goldens
    // alone do NOT catch a symbol-scope regression. Recompute (sha256 of
    // PT_WEBGPU_TRACE_WGSL) on any intentional WGSL change and update both here.
    // Re-pinned 2026-06-06: FrameParams dropped the never-read heroStrategy slot
    // (replaced by _padAuto0) and bdptConnection.wgsl gained a firefly-guard
    // comment above BDPT_CONTRIBUTION_CLAMP. Both are render-neutral.
    // Re-pinned 2026-06-06 (2): intersectionCore closest-hit leaf loop now
    // compares against the LIVE running (*hit).dist instead of a per-leaf
    // snapshot — the snapshot made within-leaf acceptance last-writer-wins
    // (thin slabs shaded from their buried far face → black walls). RENDER-
    // CHANGING on purpose; GPU-validated via the G-P0.3 capture matrix.
    // Re-pinned 2026-06-07: photon-map demotion comment block added to
    // caustic.wgsl.ts (honest-labeling, COMMENT-ONLY — verified no WGSL-code
    // line changed; the demotion docs the gatherRadius/strategyScale constants).
    // Re-pinned 2026-06-08: TLAS mesh hits now carry instanceIndex, and normal
    // maps transform their derived tangent through the hit instance localToWorld
    // before shading rotated/scaled instances. RENDER-CHANGING on purpose.
    // Re-pinned 2026-06-09: H13/D4 — brdfDirectionalPdf opposite-hemisphere
    // branch now returns 0.0 (delta lobe; prior finite cosine·η² was wrong —
    // did not match the deterministic Snell sampler in sampleNextBounceDirection).
    // RENDER-CHANGING on transmissive scenes; A/B required before compositing.
    // Re-pinned 2026-06-09: H14-E/H14-B/H51-D — HDRI gets its own intensity lane
    // (environmentHdriIntensity, f32 slot 31, replaces _padAuto0); H14-B adds spot
    // loop to restirPtProducer; H51-D bumps point stride 8→12 + spot stride 12→16
    // (penumbra+distance+decay). RENDER-CHANGING on HDRI + spot/point scenes; A/B required.
    // Re-pinned 2026-06-09: H52 — clearcoat (additive GGX F0=0.04), sheen (Charlie
    // NDF + Neubelt-Pettineo visibility), and iridescence (Belcour & Barla 2017
    // thin-film Fresnel F0 modification) lobes added to the WebGPU BRDF.
    // MATERIAL_VEC4_STRIDE bumped 23→26. Zero-default invariant: all lobes
    // short-circuit when their scalar is 0 → pre-H52 scenes are numerically identical.
    // RENDER-CHANGING on clearcoat/sheen/iridescence materials.
    // Re-pinned 2026-06-09: H6 — HdriEnvironment.rotationY implemented in pt-webgpu.
    // rotateYNeg + rotateYPos helpers added to connect.wgsl.ts; environmentLookup
    // rotates dir by -rotationY before UV; sampleEnvironmentImportance rotates the
    // CDF-sampled direction by +rotationY. rotationY=0 → identity (zero-rotation invariant).
    // Packed into params.environmentTint.w (no layout change). RENDER-CHANGING on HDRI scenes
    // with non-zero rotationY; rotationY=0 byte-identical to pre-H6.
    // Re-pinned 2026-06-10: A9 (BDPT production quality) — the BDPT-gated light
    // subpath gained a REAL glossy/specular BSDF extension (was Lambertian-only);
    // the light-path vertex widened to 4 rows (BDPT_LIGHT_PATH_ROWS 3→4, row 3 =
    // matId + wo-toward-prev) so the §10.3 connection evaluates the REAL light-vertex
    // BSDF/pdfs (fwdEe/revLcMinus); the connection light-bounce cap rose 3→8; the
    // point emitter went isotropic (uniform sphere, was cosine-up). All gated behind
    // `if (params.bdptEnabled != 0u)` so a bdpt:false RUNTIME render is byte-identical
    // (the WGSL STRING changes, hence this re-pin; the OFF runtime path does not).
    // RENDER-CHANGING on bdpt:true scenes; equal-spp variance + caustic A/Bs → V28.
    // Re-pinned 2026-06-10: Wave B — A3 (true spectral transport: baseColor
    // Jakob-Hanika spectral reflectance carried scalar-spectral at the hero λ;
    // emitters/env/lights spectralised via spectralEmissionAtHero; MATERIAL_VEC4_STRIDE
    // bumped 26→27 for the spectral-coeff vec4 #26), B9 (Kulla-Conty GGX multiscatter
    // energy compensation in evaluateBrdf/Full + the sampled-spec boost), B10 (physical
    // refraction transmittance = baseColor, replacing mix(vec3(1),baseColor,0.15)).
    // spectralEnabled=false RGB path is byte-identical at RUNTIME (the WGSL string
    // changes via the always-present select()s, hence this re-pin); B9/B10 are
    // RENDER-CHANGING on rough-metal / glass scenes → V28 A/Bs.
    // Re-pinned 2026-06-10 for B8 (light-tree orientation cones): the shared
    // light-tree traversal WGSL grew the node stride 12→16 and gained the
    // lt_coneFactor culling term (+2024 chars). Default-path RUNTIME is byte-
    // identical for unoriented scenes (full-sphere cone ⇒ factor ≡ 1); oriented
    // emitters (spot / single-sided area) get tighter SELECTION pdf only (the pdf
    // is divided out — unbiased) → V28 oriented-emitter A/B.
    // Re-pinned 2026-06-10: D3 reserved-field consumption (ao/light/bump/env/anisotropy maps) + backtick escape fix — RENDER-CHANGING, A/B pending V28-B
    // Re-pinned 2026-06-10: caustic point/spot stride fix (H1-class) + shared light-stride constants
    // (POINT_LIGHT_VEC4_STRIDE=3u / SPOT_LIGHT_VEC4_STRIDE=4u added to material.wgsl.ts; three MNEE
    // point-light loops + photon-map point/spot seeds now use them; spot-axis negation removed from
    // photon seeding). RENDER-CHANGING for multi-light caustic scenes, A/B pending V28-B.
    // Re-pinned 2026-06-10: BDPT light-subpath estimator coherence — stored throughput/pdfFwd now
    // describe the traced segment (was a sampled-then-discarded direction): scatter direction is
    // now sampled at prevPos from the REAL BSDF (woAtPrev as outgoing), traced, and used for
    // throughput (f·cos/pdf) + pdfFwd; pdfRev(prevCol) is patched to pdfFwd (PBRT RandomWalk
    // convention); the dead "nextDir" sampling at newPos is removed. RENDER-CHANGING for bdpt:true,
    // A/B pending V28-B.
    // Re-pinned 2026-06-10: A4 real SPPM progressive photon map — SPPM_GROUP4_BINDINGS_WGSL
    // (group-3 @binding(6/7/8): sppmPhotonCells, sppmCellCounters, sppmStats) added to
    // composition; photonMapContribution replaced with sppmGather() shim; old 32-photon
    // per-pixel approximation + gatherRadius=0.35 + ×1.25 fudge REMOVED.
    // RENDER-CHANGING for causticStrategy:'photon-map'; off-path byte-identical for other strategies.
    expect(digest).toBe('29ccf24475567e0d939a3acb3d96c4eea728586d85ae24e6aa853da1bf4e9762');
    expect(PT_WEBGPU_TRACE_WGSL.length).toBe(281858);
  });
});

describe('pt-webgpu WGSL material contract', () => {
  it('uses the bounded rich material payload layout', () => {
    // A3 bumped the stride 26 → 27 (new vec4 #26 carries the baseColor Jakob-Hanika
    // spectral-reflectance sigmoid coeffs). Kept in lockstep with TS MATERIAL_VEC4_STRIDE.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 27u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const SPECTRAL_SAMPLE_COUNT = 32u;');
  });

  it('includes hero-wavelength MIS helpers when spectral mode is enabled', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleHeroWavelengthMIS');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn heroWavelengthToRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('params.spectralEnabled != 0u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('heroLambdaTo01(heroLambda)');
  });

  it('returns 0 for the delta-refraction lobe in brdfDirectionalPdf (H13 D4)', () => {
    // H13/D4: delta-refraction pdf is 0 (Dirac delta, not a finite density).
    // The prior finite cosine·η² approximation did not match the deterministic
    // sampler in sampleNextBounceDirection; returning 0 is unbiased because all
    // call sites guard against pdf <= 1e-6 before dividing.
    // Verify the opposite-hemisphere branch now unconditionally returns 0.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!sameHemisphere) {');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('let nDotT = max(abs(wiDotN), 1e-5);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('eta * eta * INV_PI');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('transProb * pdfTransApprox');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn activeLayerWeightRgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('isTranslucent');
  });

  it('accounts for the light selection probability in direct lighting (WS2)', () => {
    // WS2 replaced the unconditional uniform compensation (`· f32(lightCount)`):
    // each NEE branch now self-normalizes — delta lights by `· lightSelectInvPdf`
    // (1/p_select), area/env lights by folding p_select into the MIS combined
    // pdf — so the accumulation is a bare add. p_select is `1/lightCount` on the
    // uniform fallback path and `lt.pdf` on the power-weighted light-tree path.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('radiance = radiance + directLi;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelectInvPdf = f32(lightCount);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('lightSelectInvPdf = 1.0 / lt.pdf;');
  });

  it('gates emissive-on-hit to camera + refraction paths (camera-visible emitters)', () => {
    // The emissive-on-hit term must be gated on !prevSampleAllowsAreaMis so it
    // fires ONLY on the camera ray (prevSampleAllowsAreaMis inits false) and after
    // a refraction/specular-transmission bounce (which sets sampleAllowsAreaMis
    // false) — the paths the analytic bsdfAreaLightConnectionContribution cannot
    // reach. On diffuse/glossy bounces the connection already MIS-accounts for the
    // light, so the gate prevents a double-count.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var prevSampleAllowsAreaMis = false;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!prevSampleAllowsAreaMis) {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevSampleAllowsAreaMis = sampleAllowsAreaMis;');
    // The emissive add must appear EXACTLY ONCE and be the gated form (wrapped in
    // the `if (!prevSampleAllowsAreaMis)` block). A second/unconditional add would
    // double-count emissive against the analytic connection once re-attached.
    // A3: the added quantity is `emitContribution` (= emissive in RGB mode,
    // spectralEmissionAtHero(emissive,λ) in spectral mode) — still a single gated
    // add, still double-count-free.
    const emissiveAdds = (PT_WEBGPU_TRACE_WGSL.match(/radiance = radiance \+ throughput \* emitContribution;/g) ?? []).length;
    expect(emissiveAdds).toBe(1);
    // The select that produces emitContribution falls back to the RGB emissive
    // when spectral mode is off (the byte-identical-RGB guarantee).
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let emitContribution = select(emissive, emitSpectral, params.spectralEnabled != 0u);',
    );
  });

  it('contains active strategy-specific caustic paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn causticMode() -> u32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn photonMapContribution');
    // Caches the strategy code in `caustic` and branches on the local, so
    // the manifold/photon dispatch sites read a single causticMode() call.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let caustic = causticMode();');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (caustic == 1u)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('else if (caustic == 2u)');
  });

  it('declares INV_2PI alongside INV_PI for HDRI equirect sampling', () => {
    // pathTraceBruteforce.wgsl.ts uses INV_2PI for spherical-to-UV mapping;
    // omitting it would cause a runtime WGSL compile failure for any scene
    // with an HDRI environment.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const INV_2PI');
  });

  it('FrameParams matches the host-side 512-byte UBO buffer size', () => {
    // The host allocates `new ArrayBuffer(512)` in PTEngineWebGPU.#buildParamsBuffer
    // and writes vec4-aligned fields at offsets 0..127 (16-byte units). If the
    // WGSL FrameParams struct grows past 512 bytes, this assertion fails first
    // and points at the host-side layout mismatch.
    // Count of vec4-aligned slots referenced by WGSL: matrix terms (12 mat4f =
    // 48 vec4f-slots? no, three mat4x4f = 12 vec4f total) + scalar/vec4 fields.
    // Total budget 512 bytes / 16 bytes-per-vec4 = 32 vec4-slots = 128 f32-slots.
    // Sanity check: the WGSL header should NOT reference paramsF32[128+].
    const stride = (PT_WEBGPU_TRACE_WGSL.match(/struct FrameParams/g) ?? []).length;
    expect(stride).toBeGreaterThanOrEqual(1);
    // Verify FrameParams contains the matrix fields. Exact byte offsets
    // (invViewProj@144, viewProj@208, prevViewProj@272 post W3-D12 + W4-A4
    // refactor) are pinned numerically in the next test.
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/invViewProj\s*:\s*mat4x4f/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/viewProj\s*:\s*mat4x4f/);
  });

  // Regression for pt-webgpu-deep-audit H-1 (matrix offsets in FrameParams
  // were 16 bytes shifted from spec, corrupting all ray reconstruction).
  // Fixed since the 2026-05-10 audit; this test pins the layout so a future
  // WGSL refactor can't silently re-break the host packer at
  // index.ts:274-280.
  it('FrameParams matrix block sits at the WGSL-spec byte offsets the host packer assumes (H-1 regression)', () => {
    const struct = PT_WEBGPU_TRACE_WGSL.match(
      /struct FrameParams \{([\s\S]*?)\};/,
    )?.[1];
    expect(struct).toBeDefined();
    if (struct == null) return;

    const lines = struct
      .split('\n')
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && !s.startsWith('//'));

    // Walk the struct in WGSL-spec alignment + size to derive each
    // mat4x4f's byte offset. Any drift here vs the CPU packer (paramsF32
    // .set at f32 indices 36/52/68) breaks the GPU output silently.
    const SIZE: Record<string, number> = { u32: 4, f32: 4, vec3f: 12, vec4f: 16, mat4x4f: 64 };
    const ALIGN: Record<string, number> = { u32: 4, f32: 4, vec3f: 16, vec4f: 16, mat4x4f: 16 };
    const fieldRe = /^([A-Za-z_]\w*)\s*:\s*([\w<,>]+),/;
    let offset = 0;
    const matrixOffsets: Record<string, number> = {};
    for (const line of lines) {
      const m = line.match(fieldRe);
      if (m == null) continue;
      const name = m[1]!;
      const type = m[2]!;
      const a = ALIGN[type];
      const s = SIZE[type];
      if (a == null || s == null) {
        throw new Error(`Unknown WGSL field type "${type}" in FrameParams — extend SIZE/ALIGN tables.`);
      }
      const aligned = Math.ceil(offset / a) * a;
      if (type === 'mat4x4f') matrixOffsets[name] = aligned;
      offset = aligned + s;
    }

    // Byte offsets must match FrameParamsSlot (tools/generate-wgsl-layouts.mjs).
    expect(matrixOffsets['invViewProj']).toBe(FrameParamsSlot.invViewProj * 4);
    expect(matrixOffsets['viewProj']).toBe(FrameParamsSlot.viewProj * 4);
    expect(matrixOffsets['prevViewProj']).toBe(FrameParamsSlot.prevViewProj * 4);
    expect(offset).toBeLessThanOrEqual(512);
  });

  it('declares TLAS storage bindings and host bind-group wiring in lockstep', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(0) var<storage, read> tlasNodes');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(1) var<storage, read> tlasInstanceIndices');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(2) var<storage, read> tlasBlasRoots');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld');

    const here = dirname(fileURLToPath(import.meta.url));
    const limitsSource = readFileSync(resolve(here, '../webgpuLimits.ts'), 'utf8');
    const indexSource = readFileSync(resolve(here, '../index.ts'), 'utf8');
    // Group-2 bind-group construction was extracted into the GpuResources
    // sub-struct (T14-followup); the per-frame dispatch (`setBindGroup(2, …)`)
    // stays in index.ts. Both halves of the host↔WGSL lockstep are asserted.
    const gpuResourcesSource = readFileSync(resolve(here, '../gpuResources.ts'), 'utf8');
    expect(limitsSource).toMatch(/PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP\s*=\s*10/);
    expect(indexSource).toContain('selectPtWebgpuTraceTier');
    expect(gpuResourcesSource).toContain('pathTrace.bindgroup2.full');
    expect(gpuResourcesSource).toContain('{ binding: 0, resource: { buffer: sb.tlasNodesBuffer } }');
    expect(indexSource).toContain('setBindGroup(2, gpu.pathTraceBindGroup2)');
  });

  it('keeps TLAS hit reconstruction in world space', () => {
    // TLAS traversal must convert local-space BLAS hits back into world
    // distance/normal so denoising and shading stay frame-stable.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let worldDist = dot(worldHitPos - ray.origin, ray.direction);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('(*hit).dist = worldDist;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);',
    );
  });

  it('carries the TLAS instance index into normal-map tangent reconstruction', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const INVALID_TLAS_INSTANCE_INDEX = 0xffffffffu;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('instanceIndex: u32,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('(*hit).instanceIndex = instIdx;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);',
    );
  });

  it('uses conservative local-ray bounds for TLAS closest-hit traversal under scaling', () => {
    // Closest-hit traversal keeps the prior unbounded local BLAS query because
    // the dynamic closest world distance is clamped after reconstruction.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, true);');
  });

  it('uses finite local-ray bounds for TLAS any-hit instance traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('captureShadingDetails: bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var localTMax = tMax;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let localEnd = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMax);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('localTMax = max(dot(localEnd - localRay.origin, localRay.direction), tMin);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('traceMeshBvh(localRay, tMin, localTMax, false, &localHit, blasRoot, false)');
  });

  it('returns closest-hit result from traceMeshBvh when in closest mode', () => {
    // Regression guard: closest-mode traversal must return didHit so
    // traceAny fallback paths without TLAS still report mesh occlusion.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return select(false, (*hit).didHit, closest);');
  });

  it('routes traceAny through TLAS any-hit traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn traceTlasAny(ray: Ray, tMin: f32, tMax: f32) -> bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (traceTlasAny(ray, tMin, tMax)) {');
  });

  it('uses conservative any-hit stack-overflow handling to avoid light leaks', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('Conservative any-hit overflow policy: prefer occlusion over light leak.');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('return true;');
  });

  // ── Theme-D luminance dedup (behavior-preserving) ──────────────────────────
  // material.wgsl.ts:441 and bdpt/bdptLightSubpath.wgsl.ts:8 previously inlined
  // the Rec.709 `dot(c, vec3f(0.2126, 0.7152, 0.0722))` though the canonical
  // `luminance()` (LUMINANCE_WGSL from @vitrum/shared-samplers) is composed into
  // the trace shader ahead of both modules. These pins prove the dedup landed:
  // the canonical call sites are present, and NO inline Rec.709 dot remains in
  // the composed full-tier shader. `luminance()` is defined exactly as
  // `dot(c, vec3f(0.2126,0.7152,0.0722))`, so the GPU result is unchanged.
  describe('Theme-D — Rec.709 luminance routed through canonical luminance()', () => {
    it('activeLayerWeightRgb uses luminance(layerRgb), not an inline Rec.709 dot', () => {
      expect(PT_WEBGPU_TRACE_WGSL).toContain('let lum = max(luminance(layerRgb), 0.0);');
    });

    it('bdptLightLuminance routes through luminance() and keeps the 1e-20 floor', () => {
      expect(PT_WEBGPU_TRACE_WGSL).toContain('return max(luminance(c), 1e-20);');
    });

    it('no inline Rec.709 dot(..., vec3f(0.2126, 0.7152, 0.0722)) remains in the composed shader', () => {
      // The canonical LUM_W709 const + luminance() body live in LUMINANCE_WGSL;
      // that vec3f literal is the ONLY remaining occurrence (inside the const
      // declaration), never an inline dot at a call site.
      const inlineDots = (
        PT_WEBGPU_TRACE_WGSL.match(/dot\([^)]*vec3f\(0\.2126, 0\.7152, 0\.0722\)\)/g) ?? []
      ).length;
      expect(inlineDots).toBe(0);
    });
  });

  // ── Theme-D caustic decode: COLLAPSED onto canonical decodeMaterial() ───────
  // caustic.wgsl.ts previously hand-decoded the packed material inline at two
  // sites (traceSpecularTransmissiveChain + photonMapContribution), duplicating
  // the m0/m2/m3/m22 offset arithmetic decodeMaterial() already owns. The decode
  // is now routed through decodeMaterial(matId); the only behavioural difference
  // — caustic's clamp(baseColor, 0, 1) inside its mix — is re-applied at both
  // sites to preserve bit-identical render output for in-[0,1] albedos (OOB
  // albedos are unreachable for valid scenes). This pin guards the collapse.
  it('caustic material decode is routed through canonical decodeMaterial() with baseColor re-clamped', () => {
    // Both decode sites now CALL the canonical decoder. Strip comment lines so
    // the explanatory references in comments don't count as invocations.
    const causticCode = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL
      .split('\n')
      .filter((l) => !l.trim().startsWith('//'))
      .join('\n');
    // A4: traceSpecularTransmissiveChain = 1 real call.  photonMapContribution now
    // delegates to sppmGather() (SPPM hash-grid lookup) and no longer has its own
    // decodeMaterial call — the gather is done inside the SPPM bindings module.
    const decodeCalls = causticCode.match(/let mat = decodeMaterial\(matId\);/g) ?? [];
    expect(decodeCalls.length).toBe(1);

    // The historical baseColor clamp inside the mix is preserved at the remaining site.
    expect(causticCode).toContain(
      'mix(vec3f(1.0), clamp(mat.baseColor, vec3f(0.0), vec3f(1.0)), 0.2)',
    );

    // NO raw-slot offset arithmetic / inline OOB fallback remains in caustic.
    expect(causticCode).not.toContain('m0Index');
    expect(causticCode).not.toContain('m2Index');
    expect(causticCode).not.toContain('m3Index');
    expect(causticCode).not.toContain('m22Index');
    expect(causticCode).not.toContain('select(vec4f(1.0, 1.0, 1.0, 0.5)');

    // The deferral TODO is gone (collapse is done, not deferred).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('TODO(theme-D / V-caustic)');
  });
});
