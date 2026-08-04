/**
 * Probe-update compute pass dispatch (W4c — extracted from probeUpdatePass.ts).
 *
 * D6.4: bind groups are cached by resource identity. Each dispatch function
 * lazily allocates its cache slot on first call and reuses the bind group on
 * subsequent calls as long as all participating resources are unchanged. When
 * any resource reference changes (BVH rebuild, atlas swap, emitter update, etc.)
 * the affected slot is invalidated automatically and a new bind group is created.
 */
import type { ProbeGrid } from './probeGrid.js';
import type { ProbeUpdateGpuState, DispatchBindGroupCache } from './probeUpdateGpuState.js';
import { getOrCreateBindGroup } from './probeUpdateGpuState.js';
import { DDGI_BORDER_UBO } from './probeUpdateUbos.js';

/** Allocate an empty keyed bind-group cache; called once on the first dispatch. */
function makeBgCache(): DispatchBindGroupCache {
  return new Map();
}

export interface ProbeUpdateBorderUboInput {
  probeCount: number;
  atlasWidth: number;
  atlasHeight: number;
  gridDimX: number;
  gridDimY: number;
  gridDimZ: number;
}

export function packProbeUpdateBorderUbo(input: ProbeUpdateBorderUboInput): ArrayBuffer {
  const data = new ArrayBuffer(DDGI_BORDER_UBO.sizeBytes);
  DDGI_BORDER_UBO.pack(new DataView(data), 0, {
    numProbes: input.probeCount,
    atlasWidth: input.atlasWidth,
    atlasHeight: input.atlasHeight,
    _pad0: 0,
    gridDimX: input.gridDimX,
    gridDimY: input.gridDimY,
    gridDimZ: input.gridDimZ,
    _pad1: 0,
  });
  return data;
}

export function uploadProbeUpdateBorderUbo(
  device: GPUDevice,
  gpu: ProbeUpdateGpuState,
  grid: ProbeGrid,
  atlas: GPUTexture,
): void {
  const data = packProbeUpdateBorderUbo({
    probeCount: grid.probeCount,
    atlasWidth: atlas.width,
    atlasHeight: atlas.height,
    gridDimX: grid.params.dims.x,
    gridDimY: grid.params.dims.y,
    gridDimZ: grid.params.dims.z,
  });
  device.queue.writeBuffer(gpu.borderVisUboBuf, 0, data);
}

/**
 * Copy the entire accepted irradiance image, including the packed probe-state
 * ring texels, into the next ping-pong target before classifying this frame's
 * stratum. The classifier overwrites only active state texels and the SH blend
 * overwrites only active 3x3 coefficient blocks, so this is the bit-for-bit
 * carry-forward guarantee for every non-updated probe.
 */
export function copyProbeIrradianceAndPackedStateForward(
  encoder: Pick<GPUCommandEncoder, 'copyTextureToTexture'>,
  readTexture: GPUTexture,
  writeTexture: GPUTexture,
): void {
  if (
    readTexture.width !== writeTexture.width ||
    readTexture.height !== writeTexture.height
  ) {
    throw new Error('DDGI irradiance ping-pong textures must have identical dimensions.');
  }
  encoder.copyTextureToTexture(
    { texture: readTexture },
    { texture: writeTexture },
    {
      width: readTexture.width,
      height: readTexture.height,
      depthOrArrayLayers: 1,
    },
  );
}

export function dispatchProbeUpdateRaysPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
  visReadTex: GPUTexture,
): void {
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;

  // Group 0: BVH geometry buffers (13 entries). Key every bound identity:
  // transform-only TLAS refits preserve the BLAS buffers and may replace only
  // the TLAS streams whose capacities grew.
  const raysG0 = getOrCreateBindGroup(c, 'raysG0', [
    gpu.bvhBuf,
    gpu.posBuf,
    gpu.idxBuf,
    gpu.normBuf,
    gpu.matIdBuf,
    gpu.tlasNodesBuf,
    gpu.tlasInstIdxBuf,
    gpu.tlasBlasRootsBuf,
    gpu.tlasW2lBuf,
    gpu.tlasL2wBuf,
    gpu.traceParamsBuf,
    gpu.opticalTriangleIdentityBuf,
    gpu.opticalInstanceBoundaryIdBasePlusOneBuf,
  ], () => gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.bvhBuf } },
      { binding: 1, resource: { buffer: gpu.posBuf } },
      { binding: 2, resource: { buffer: gpu.idxBuf } },
      { binding: 3, resource: { buffer: gpu.normBuf } },
      { binding: 4, resource: { buffer: gpu.matIdBuf } },
      { binding: 5, resource: { buffer: gpu.tlasNodesBuf } },
      { binding: 6, resource: { buffer: gpu.tlasInstIdxBuf } },
      { binding: 7, resource: { buffer: gpu.tlasBlasRootsBuf } },
      { binding: 8, resource: { buffer: gpu.tlasW2lBuf } },
      { binding: 9, resource: { buffer: gpu.tlasL2wBuf } },
      { binding: 10, resource: { buffer: gpu.traceParamsBuf } },
      { binding: 11, resource: { buffer: gpu.opticalTriangleIdentityBuf } },
      {
        binding: 12,
        resource: {
          buffer: gpu.opticalInstanceBoundaryIdBasePlusOneBuf,
        },
      },
    ],
  }));

  // Group 1: materials + lights + emitters + material-map atlas + authored tangent/color streams.
  // Invalidated on material/emitter upload or atlas/tangent/color replacement.
  const raysG1 = getOrCreateBindGroup(c, 'raysG1', [
    gpu.materialsBuf,
    gpu.lightsBuf,
    gpu.emitterTrisBuf,
    gpu.materialTextureAtlasView,
    gpu.materialTextureAtlasMetaView,
    gpu.bvhTangentTextureView,
    gpu.bvhVertexColorTextureView,
  ], () => gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: gpu.materialsBuf } },
      { binding: 1, resource: { buffer: gpu.lightsBuf } },
      // H18 Stage 2 — area-emitter NEE triangles (dummy 16-byte buf when count==0).
      { binding: 2, resource: { buffer: gpu.emitterTrisBuf } },
      { binding: 3, resource: gpu.materialTextureAtlasView },
      { binding: 4, resource: gpu.materialTextureAtlasMetaView },
      { binding: 5, resource: gpu.bvhTangentTextureView },
      { binding: 6, resource: gpu.bvhVertexColorTextureView },
    ],
  }));

  // Group 2: per-frame resources. Changes every atlas swap (irrReadTex ping-pongs),
  // on envMap update, or when rayResultsBuf / activeProbesBuf is reallocated.
  // activeProbesBuf is keyed explicitly: it reallocs alongside rayResultsBuf in
  // practice, but the cache must not rely on that coupling.
  const raysG2 = getOrCreateBindGroup(c, 'raysG2', [
    irrReadTex,
    visReadTex,
    gpu.rayResultsBuf,
    gpu.activeProbesBuf,
    gpu.envMapView,
  ], () => gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: irrReadTex.createView() },
      { binding: 4, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 5, resource: { buffer: gpu.frameParamsBuf } },
      // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
      // A 1×1 placeholder view is bound when hasEnv=0 so the bind group is
      // always valid; the WGSL sampleSkyColor gates on frameParams.hasEnv
      // before sampling, so the placeholder is never actually read.
      { binding: 6, resource: gpu.envMapView },
      { binding: 8, resource: visReadTex.createView() },
    ],
  }));

  const pass = encoder.beginComputePass({ label: 'ddgi-probe-rays' });
  pass.setPipeline(gpu.raysPipeline);
  pass.setBindGroup(0, raysG0);
  pass.setBindGroup(1, raysG1);
  pass.setBindGroup(2, raysG2);
  pass.dispatchWorkgroups(activeCount);
  pass.end();
}

/**
 * Classify and relocate the current probe stratum from the ray producer's
 * records. State is read from the accepted irradiance atlas and written to the
 * reserved ring texel in the next atlas, so one command buffer has no
 * read/write alias and publication can swap atomically.
 */
export function dispatchProbeClassifyRelocatePass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
  irrWriteTex: GPUTexture,
): void {
  if (!Number.isSafeInteger(activeCount) || activeCount < 0) {
    throw new RangeError('DDGI classifier active count must be a non-negative safe integer.');
  }
  if (activeCount === 0) return;
  const activeBindingBytes = activeCount * Uint32Array.BYTES_PER_ELEMENT;
  if (
    !Number.isSafeInteger(activeBindingBytes) ||
    activeBindingBytes > gpu.activeProbesBuf.size
  ) {
    throw new RangeError('DDGI classifier active prefix exceeds its GPU buffer.');
  }
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;
  const classifyG0 = getOrCreateBindGroup(c, 'classifyG0', [
    gpu.rayResultsBuf,
    gpu.activeProbesBuf,
    activeBindingBytes,
    irrReadTex,
    irrWriteTex,
  ], () => gpu.device.createBindGroup({
    layout: gpu.classifyRelocatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      // Bind only the just-uploaded prefix. The compute dispatch is rounded to
      // 64 lanes; arrayLength(activeProbes) must not expose stale capacity from
      // a larger prior stratum to those padded lanes.
      {
        binding: 1,
        resource: {
          buffer: gpu.activeProbesBuf,
          offset: 0,
          size: activeBindingBytes,
        },
      },
      { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 3, resource: irrReadTex.createView() },
      {
        binding: 4,
        resource: irrWriteTex.createView({
          format: 'rgba16float',
          mipLevelCount: 1,
        }),
      },
    ],
  }));

  const pass = encoder.beginComputePass({ label: 'ddgi-classify-relocate' });
  pass.setPipeline(gpu.classifyRelocatePipeline);
  pass.setBindGroup(0, classifyG0);
  pass.dispatchWorkgroups(Math.ceil(activeCount / 64), 1, 1);
  pass.end();
}

export function dispatchProbeUpdateBlendIrrPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
  irrWriteTex: GPUTexture,
): void {
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;

  // Group 0: stable buffer references. Invalidated on rayResultsBuf or
  // activeProbesBuf reallocation. The active-probe slice can grow by one when
  // probeCount is not divisible by the update divisor.
  const blendIrrG0 = getOrCreateBindGroup(c, 'blendIrrG0', [
    gpu.rayResultsBuf,
    gpu.activeProbesBuf,
  ], () => gpu.device.createBindGroup({
    layout: gpu.blendIrrPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
    ],
  }));

  // Group 1: atlas textures — changes every atlas swap (ping-pong) or resize.
  const blendIrrG1 = getOrCreateBindGroup(c, 'blendIrrG1', [
    irrReadTex,
    irrWriteTex,
  ], () => gpu.device.createBindGroup({
    layout: gpu.blendIrrPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: irrReadTex.createView() },
      { binding: 1, resource: irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
    ],
  }));

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-irr' });
  pass.setPipeline(gpu.blendIrrPipeline);
  pass.setBindGroup(0, blendIrrG0);
  pass.setBindGroup(1, blendIrrG1);
  pass.dispatchWorkgroups(activeCount, 1, 1);
  pass.end();
}

export function dispatchProbeUpdateBlendVisPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  visReadTex: GPUTexture,
  visWriteTex: GPUTexture,
): void {
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;

  // Group 0: stable buffer references. Mirrors blendIrr group 0 but for vis pipeline.
  const blendVisG0 = getOrCreateBindGroup(c, 'blendVisG0', [
    gpu.rayResultsBuf,
    gpu.activeProbesBuf,
  ], () => gpu.device.createBindGroup({
    layout: gpu.blendVisPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
    ],
  }));

  // Group 1: atlas textures — changes every atlas swap (ping-pong) or resize.
  const blendVisG1 = getOrCreateBindGroup(c, 'blendVisG1', [
    visReadTex,
    visWriteTex,
  ], () => gpu.device.createBindGroup({
    layout: gpu.blendVisPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: visReadTex.createView() },
      { binding: 1, resource: visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
    ],
  }));

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-vis' });
  pass.setPipeline(gpu.blendVisPipeline);
  pass.setBindGroup(0, blendVisG0);
  pass.setBindGroup(1, blendVisG1);
  pass.dispatchWorkgroups(activeCount, 1, 1);
  pass.end();
}

export function dispatchProbeUpdateBorderVisPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  probeCount: number,
  scratchTex: GPUTexture,
  writeAtlas: GPUTexture,
): void {
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;

  // Group 0: per-atlas. Invalidated when the atlas is reallocated or when the
  // write side of the ping-pong changes (scratchTex ≡ current-frame visWriteTex,
  // which swaps every frame — so this will miss once per atlas swap. That is
  // intentional: one extra BG create per frame beats reading stale views).
  const borderG0 = getOrCreateBindGroup(c, 'borderG0', [
    scratchTex,
    writeAtlas,
  ], () => gpu.device.createBindGroup({
    layout: gpu.borderVisPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: scratchTex.createView() },
      { binding: 1, resource: writeAtlas.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
      { binding: 2, resource: { buffer: gpu.borderVisUboBuf } },
    ],
  }));

  const pass = encoder.beginComputePass({ label: 'ddgi-border-vis' });
  pass.setPipeline(gpu.borderVisPipeline);
  pass.setBindGroup(0, borderG0);
  pass.dispatchWorkgroups(probeCount, 1, 1);
  pass.end();
}
