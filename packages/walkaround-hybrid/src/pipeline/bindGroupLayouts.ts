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
 * rejected. maxBindingsPerBindGroup is 1000 so 4 bindings is fine.
 * Layout:
 *   DDGI section
 *     0 — irradiance atlas (texture_2d<f32>, unfilterable)
 *     1 — visibility atlas (texture_2d<f32>, unfilterable)
 *     2 — non-filtering sampler
 *     3 — DDGI grid uniform (64 bytes)
 *
 * Note: RC bindings were dropped — Lo_rc was computed and discarded;
 * the cascade buffers, params UBO, and setRCInputs wiring all retired
 * together. The RC subsystem remains live for the standalone 'rc'
 * walkaround engine.
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
    ],
  });
  return cache.hybridLayers;
}
