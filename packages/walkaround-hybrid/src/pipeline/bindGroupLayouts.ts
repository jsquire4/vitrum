/**
 * Bind group layout factories — one function per layout used by the
 * WalkaroundGPUPipeline. Each function is memoised on first call via the
 * `cache` argument (a plain object on the class) so the GPUDevice only
 * allocates each layout once.
 *
 * Layouts:
 *   frame   — per-frame G-buffer textures + reservoir buffers + HDR output
 *   scene   — static BVH + emitter buffers
 *   ubo     — 256-byte WalkaroundUBO uniform
 *   atrous  — denoiser I/O textures + per-pass UBO
 *   accum   — temporal accumulator I/O textures + AccumUBO
 *   composite — final blit (fragment stage, unfilterable-float + sampler)
 *   hybridLayers — DDGI atlas textures + grid uniform (shade pass slot 3)
 */

export interface BGLCache {
  frame?: GPUBindGroupLayout;
  scene?: GPUBindGroupLayout;
  ubo?: GPUBindGroupLayout;
  atrous?: GPUBindGroupLayout;
  composite?: GPUBindGroupLayout;
  accum?: GPUBindGroupLayout;
  hybridLayers?: GPUBindGroupLayout;
  /** Sprint 9 — sample-budget pass bind group layout. */
  sampleBudget?: GPUBindGroupLayout;
  /** Sprint 9 — resolve pass bind group layout. */
  resolve?: GPUBindGroupLayout;
  /** Motion-vector generation pass bind group layout. */
  motionVectors?: GPUBindGroupLayout;
  /** Sprint 15 — GTAO half-res compute pass bind group layout. */
  gtao?: GPUBindGroupLayout;
  /** Sprint 15 — GTAO bilateral upsample pass bind group layout. */
  gtaoUpsample?: GPUBindGroupLayout;
  /** Sprint 17 — GI temporal-reuse pass bind group layout. */
  temporalGi?: GPUBindGroupLayout;
  /** Sprint 17 — GI spatial-reuse pass bind group layout. */
  spatialGi?: GPUBindGroupLayout;
  /** Sprint 18 — indirect-blur + combine pass bind group layout. */
  indirectCombine?: GPUBindGroupLayout;
  /** Sprint 18 follow-up — indirect-channel pre-atrous temporal accumulator. */
  indirectTemporalAccum?: GPUBindGroupLayout;
}

export function getFrameBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.frame) return cache.frame;
  cache.frame = device.createBindGroupLayout({
    label: 'frame-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
      // gNormalDepth — written by shade pass (normal in xyz, primary-hit
      // distance in w); read by the à-trous denoiser for edge stopping.
      // Declared in all four compute pass bind groups (RIS / temporal /
      // spatial / shade) for layout compatibility, but only shade actually
      // writes to it. Bound to the same texture in every dispatch.
      { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // Sprint 16 — half-res GI reservoir (read_write storage buffer).
      // Written by risGiMain; read by shade.wgsl to compute Lo_indirect.
      // Sized for (W/2) × (H/2) × 80 bytes (RESERVOIR_GI_STRIDE × u32).
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      // Sprint 18 — indirect-channel HDR output (rgba16float storage). Written
      // by shade as `Lo_indirect × ao`. Other shaders that bind the frame BGL
      // (ris, temporal, spatial, risGi) do not reference this binding; only
      // shade declares it, but it must be present in the layout for compat.
      { binding: 12, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // Sprint 18 follow-up — total-radiance HDR output (rgba16float storage).
      // Written by shade as direct + indirect. Welford reads it so the variance
      // / tier estimate covers the full signal, not just the direct channel.
      { binding: 13, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // Item 24 — albedo demodulation (Schied 2017 §4.1). Written by shade
      // alongside hdrIndirectOut; read by indirectCombine to re-modulate the
      // denoised lighting signal back to full outgoing radiance.
      { binding: 14, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  return cache.frame;
}

export function getSceneBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.scene) return cache.scene;
  cache.scene = device.createBindGroupLayout({
    label: 'scene-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhNodes
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhIndex (vec4u: [0..2]=indices, [3]=RGBA8 raw attCol)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvhPositions
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // emitters
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // emitterCdf
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_beer (Beer-Lambert visible color)
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // tlasNodes
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // tlasInstanceIndices
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // tlasBlasRoots
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // tlasInstanceWorldToLocal (mat4 cols)
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // tlasInstanceLocalToWorld
    ],
  });
  return cache.scene;
}

export function getUboBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.ubo) return cache.ubo;
  cache.ubo = device.createBindGroupLayout({
    label: 'ubo-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // Sprint 15 — full-res GTAO occlusion factor (r16float), 1-frame lagged.
      // Sampled in shade to modulate the diffuse / indirect light terms.
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      // Sprint 9 / Original-#2 wire-in — per-pixel adaptive-sampling tier
      // (r32uint, written by sample-budget pass). risGi.wgsl reads it to
      // scale the RIS-GI candidate count (M_GI) per pixel — high-variance
      // pixels get more candidates, low-variance pixels get fewer. Other
      // pipelines that bind this BGL (ris/temporal/spatial/shade) declare
      // the slot for layout compatibility but do not reference the symbol.
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint' } },
    ],
  });
  return cache.ubo;
}

export function getAtrousBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.atrous) return cache.atrous;
  cache.atrous = device.createBindGroupLayout({
    label: 'atrous-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.atrous;
}

export function getCompositeBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.composite) return cache.composite;
  cache.composite = device.createBindGroupLayout({
    label: 'composite-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
    ],
  });
  return cache.composite;
}

export function getAccumBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.accum) return cache.accum;
  cache.accum = device.createBindGroupLayout({
    label: 'accum-bgl',
    entries: [
      // 0: currentAtrous (read)
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      // 1: prevAccum (read)
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      // 2: accumOut (write)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // 3: AccumUBO
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.accum;
}

/**
 * Phase 2B — combined hybrid-layers bind group (slot 3, shade only).
 * DDGI + RC inputs packed into one group because Lovelace's adapter
 * caps `maxBindGroups = 4` (verified empirically); a 5th group is
 * rejected. maxBindingsPerBindGroup is 1000 so 6 bindings is fine.
 * Layout:
 *   DDGI section
 *     0 — irradiance atlas (texture_2d<f32>, unfilterable)
 *     1 — visibility atlas (texture_2d<f32>, unfilterable)
 *     2 — non-filtering sampler
 *     3 — DDGI grid uniform (64 bytes)
 *   RC section (W8 Phase 3, 2026-05-18)
 *     4 — cascade-0 storage buffer (read-only) — `array<vec4f>` packed
 *         (probeX·probeY·probeZ·rays) entries. Always bound; a 1-vec4f
 *         placeholder backs the slot when RC is disabled.
 *     5 — RCParams uniform (64 bytes) — probeOrigin/roomSize/probeCount/
 *         raysPerProbe/rayGridSize + rcWeight + enabled bit. The
 *         `enabled == 0u` short-circuits sampleCascadeC0 to vec3f(0),
 *         so the same bind group works for rcEnabled=true and false
 *         without a pipeline recompile.
 *
 * shade.wgsl reads bindings 0-5; risGi.wgsl reads only 0-3. WebGPU
 * spec allows pipelines to reference a subset of layout entries, so
 * the unified BGL works for both. Bind groups must still provide
 * resources for all 6 entries (placeholders for unused).
 */
export function getHybridLayersBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.hybridLayers) return cache.hybridLayers;
  cache.hybridLayers = device.createBindGroupLayout({
    label: 'hybrid-layers-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.hybridLayers;
}

/**
 * Sprint 9 — sample-budget BGL. Matches the @group(0) bindings in
 * sampleBudget.wgsl.ts:
 *   0 — SampleBudgetUniforms ubo (thresholds + screen size)
 *   1 — variance source (rg32float, sampled, unfilterable)
 *   2 — tier output (r32uint, write-only storage)
 *   3 — SampleCountUniforms ubo (sample-count counter)
 */
export function getSampleBudgetBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.sampleBudget) return cache.sampleBudget;
  cache.sampleBudget = device.createBindGroupLayout({
    label: 'sample-budget-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32uint' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.sampleBudget;
}

/**
 * Sprint 9 — resolve BGL. Matches the @group(0) bindings in
 * resolve.wgsl.ts:
 *   0 — ResolveUniforms ubo (screen size + frame parity)
 *   1 — current radiance (rgba16float, sampled, unfilterable)
 *   2 — previous radiance (rgba16float, sampled, unfilterable)
 *   3 — motion vectors (rg32float, sampled, unfilterable)
 *   4 — resolved out (rgba16float, write-only storage)
 */
export function getResolveBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.resolve) return cache.resolve;
  cache.resolve = device.createBindGroupLayout({
    label: 'resolve-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  return cache.resolve;
}

/**
 * Motion-vectors pass BGL. Matches `motionVectors.wgsl.ts`:
 *   0 — gNormalDepth in (rgba16float sampled, unfilterable)
 *   1 — motion out (rg32float write-only storage)
 *   2 — WalkaroundUBO (uniform)
 */
export function getMotionVectorsBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.motionVectors) return cache.motionVectors;
  cache.motionVectors = device.createBindGroupLayout({
    label: 'motion-vectors-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rg32float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.motionVectors;
}

/**
 * Sprint 15 — GTAO half-res compute pass BGL. Matches `gtao.wgsl.ts`:
 *   0 — gNormalDepth (rgba16float, sampled)
 *   1 — aoHalf out (rgba16float, write-only storage — E1 multi-bounce: bumped from r16float)
 *   2 — GTAOUniforms ubo
 *   3 — gtao_albedo (rgba16float, sampled — E1 Jiménez 2016 §5.2 multi-bounce term)
 */
export function getGTAOBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.gtao) return cache.gtao;
  cache.gtao = device.createBindGroupLayout({
    label: 'gtao-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // E1 — hdrAlbedoOut (shade pass M9.C) wired for multi-bounce factor.
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  return cache.gtao;
}

/**
 * Sprint 15 — GTAO bilateral upsample BGL. Matches `gtaoUpsample.wgsl.ts`:
 *   0 — aoHalf in (rgba16float, sampled — carries per-channel multi-bounce AO)
 *   1 — gNormalDepth (rgba16float, sampled)
 *   2 — aoFull out (rgba16float, write-only storage — Tier-G fix: per-channel
 *       Jiménez 2016 §5.2 AO stored in `.rgb`; previously collapsed to a
 *       luminance scalar in `.r`)
 *   3 — GTAOUniforms (uniform, audit B3 — for bilateralDepthSigma)
 */
export function getGTAOUpsampleBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.gtaoUpsample) return cache.gtaoUpsample;
  cache.gtaoUpsample = device.createBindGroupLayout({
    label: 'gtao-upsample-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.gtaoUpsample;
}

/**
 * Sprint 17 — temporal GI BGL. Matches `temporalGi.wgsl.ts`:
 *   0 — reservoirGiCurrent  (storage, read_write)
 *   1 — reservoirGiPrevious (storage, read)
 *   2 — WalkaroundUBO       (uniform)
 */
export function getTemporalGiBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.temporalGi) return cache.temporalGi;
  cache.temporalGi = device.createBindGroupLayout({
    label: 'temporal-gi-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.temporalGi;
}

/**
 * Sprint 17 — spatial GI BGL. Two ping-pong bind groups share this layout:
 *   pass 1 → in = reservoirGiCurrent, out = reservoirGiSpatial
 *   pass 2 → in = reservoirGiSpatial, out = reservoirGiCurrent
 * Matches `spatialGi.wgsl.ts`:
 *   0 — input  reservoir (storage, read)
 *   1 — output reservoir (storage, read_write)
 *   2 — WalkaroundUBO    (uniform)
 */
export function getSpatialGiBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.spatialGi) return cache.spatialGi;
  cache.spatialGi = device.createBindGroupLayout({
    label: 'spatial-gi-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.spatialGi;
}

/**
 * Sprint 18 follow-up — indirect temporal accumulator BGL.  Matches
 * `indirectTemporalAccum.wgsl.ts`:
 *   0 — currentRaw (rgba16float sampled, hdrIndirectTexture)
 *   1 — prevAccum  (rgba16float sampled, previous frame's accumulator output)
 *   2 — outAccum   (rgba16float storage write, this frame's accumulator output)
 */
export function getIndirectTemporalAccumBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.indirectTemporalAccum) return cache.indirectTemporalAccum;
  cache.indirectTemporalAccum = device.createBindGroupLayout({
    label: 'indirect-temporal-accum-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  return cache.indirectTemporalAccum;
}

/**
 * Sprint 18 — indirect-combine BGL. Matches `indirectCombine.wgsl.ts`:
 *   0 — denoisedDirect (rgba16float, sampled, unfilterable)
 *   1 — hdrIndirect    (rgba16float, sampled, unfilterable)
 *   2 — combinedOut    (rgba16float, write-only storage)
 *   3 — albedo         (rgba16float, sampled, unfilterable) — Item 24
 *
 * W5-I2 cleanup (2026-05-18): the previous slot-2 `gNormalDepth` entry was
 * declared "for BGL compat" but never read by the shader; dropped along
 * with its host-side BGL entry + builder argument.
 */
export function getIndirectCombineBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.indirectCombine) return cache.indirectCombine;
  cache.indirectCombine = device.createBindGroupLayout({
    label: 'indirect-combine-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // Item 24 — albedo demodulation (Schied 2017 §4.1).
      // Re-modulates the denoised indirect lighting signal by albedo.
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  return cache.indirectCombine;
}

