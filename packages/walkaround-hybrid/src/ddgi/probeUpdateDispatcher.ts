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
import { DDGI_BORDER_UBO } from './probeUpdateUbos.js';

/** Allocate an empty bind-group cache; called once on the first dispatch. */
function makeBgCache(): DispatchBindGroupCache {
  return {
    raysG0Key: null, raysG0: null,
    raysG1Key0: null, raysG1Key1: null, raysG1KeyAtlas: null, raysG1KeyAtlasMeta: null, raysG1KeyTangent: null, raysG1KeyVertexColor: null, raysG1: null,
    raysG2KeyTex: null, raysG2KeyBuf: null, raysG2KeyProbes: null, raysG2KeyEnv: null,
    raysG2IrrView: null, raysG2: null,
    blendIrrG0Key: null, blendIrrG0KeyProbes: null, blendIrrG0: null,
    blendIrrG1KeyRead: null, blendIrrG1KeyWrite: null,
    blendIrrG1ReadView: null, blendIrrG1WriteView: null, blendIrrG1: null,
    blendVisG0Key: null, blendVisG0KeyProbes: null, blendVisG0: null,
    blendVisG1KeyRead: null, blendVisG1KeyWrite: null,
    blendVisG1ReadView: null, blendVisG1WriteView: null, blendVisG1: null,
    borderG0KeyScratch: null, borderG0KeyWrite: null,
    borderG0ScratchView: null, borderG0WriteView: null, borderG0: null,
  };
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
  _which: 'irr' | 'vis', // retained for call-site clarity; always 'vis' (irr uses SH, no border pass)
): void {
  const data = packProbeUpdateBorderUbo({
    probeCount: grid.probeCount,
    atlasWidth: atlas.width,
    atlasHeight: atlas.height,
    gridDimX: grid.params.dims.x,
    gridDimY: grid.params.dims.y,
    gridDimZ: grid.params.dims.z,
  });
  // Irradiance is SH (seam-free, no border pass), so only the visibility border
  // UBO exists; `which` is retained for call-site clarity but is always 'vis'.
  const buf = gpu.borderVisUboBuf;
  device.queue.writeBuffer(buf, 0, data);
}

export function dispatchProbeUpdateRaysPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
): void {
  if (!gpu.bgCache) gpu.bgCache = makeBgCache();
  const c = gpu.bgCache;

  // Group 0: BVH geometry buffers (11 entries). Invalidated when BVH is rebuilt
  // (bvhBuf reference changes). All 11 buffers are rebuilt atomically, so keying
  // on bvhBuf alone is sufficient.
  if (c.raysG0Key !== gpu.bvhBuf) {
    c.raysG0 = gpu.device.createBindGroup({
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
      ],
    });
    c.raysG0Key = gpu.bvhBuf;
  }

  // Group 1: materials + lights + emitters + material-map atlas + authored tangent/color streams.
  // Invalidated on material/emitter upload or atlas/tangent/color replacement.
  if (
    c.raysG1Key0 !== gpu.materialsBuf ||
    c.raysG1Key1 !== gpu.emitterTrisBuf ||
    c.raysG1KeyAtlas !== gpu.materialTextureAtlasView ||
    c.raysG1KeyAtlasMeta !== gpu.materialTextureAtlasMetaView ||
    c.raysG1KeyTangent !== gpu.bvhTangentTextureView ||
    c.raysG1KeyVertexColor !== gpu.bvhVertexColorTextureView
  ) {
    c.raysG1 = gpu.device.createBindGroup({
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
    });
    c.raysG1Key0 = gpu.materialsBuf;
    c.raysG1Key1 = gpu.emitterTrisBuf;
    c.raysG1KeyAtlas = gpu.materialTextureAtlasView;
    c.raysG1KeyAtlasMeta = gpu.materialTextureAtlasMetaView;
    c.raysG1KeyTangent = gpu.bvhTangentTextureView;
    c.raysG1KeyVertexColor = gpu.bvhVertexColorTextureView;
  }

  // Group 2: per-frame resources. Changes every atlas swap (irrReadTex ping-pongs),
  // on envMap update, or when rayResultsBuf / activeProbesBuf is reallocated.
  // activeProbesBuf is keyed explicitly: it reallocs alongside rayResultsBuf in
  // practice, but the cache must not rely on that coupling.
  if (
    c.raysG2KeyTex !== irrReadTex ||
    c.raysG2KeyBuf !== gpu.rayResultsBuf ||
    c.raysG2KeyProbes !== gpu.activeProbesBuf ||
    c.raysG2KeyEnv !== gpu.envMapView
  ) {
    const irrView = irrReadTex.createView();
    c.raysG2 = gpu.device.createBindGroup({
      layout: gpu.raysPipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
        { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
        { binding: 2, resource: irrView },
        { binding: 3, resource: gpu.linearSampler },
        { binding: 4, resource: { buffer: gpu.gridParamsBuf } },
        { binding: 5, resource: { buffer: gpu.frameParamsBuf } },
        // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
        // A 1×1 placeholder view is bound when hasEnv=0 so the bind group is
        // always valid; the WGSL sampleSkyColor gates on frameParams.hasEnv
        // before sampling, so the placeholder is never actually read.
        // Trust-audit F3: NO sampler entry — the WGSL uses textureLoad and a
        // declared-but-unused sampler is stripped by layout:'auto', so passing
        // an entry for it failed bind-group validation every frame.
        { binding: 6, resource: gpu.envMapView },
      ],
    });
    c.raysG2IrrView = irrView;
    c.raysG2KeyTex = irrReadTex;
    c.raysG2KeyBuf = gpu.rayResultsBuf;
    c.raysG2KeyProbes = gpu.activeProbesBuf;
    c.raysG2KeyEnv = gpu.envMapView;
  }

  const pass = encoder.beginComputePass({ label: 'ddgi-probe-rays' });
  pass.setPipeline(gpu.raysPipeline);
  pass.setBindGroup(0, c.raysG0);
  pass.setBindGroup(1, c.raysG1);
  pass.setBindGroup(2, c.raysG2);
  pass.dispatchWorkgroups(activeCount);
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
  if (c.blendIrrG0Key !== gpu.rayResultsBuf || c.blendIrrG0KeyProbes !== gpu.activeProbesBuf) {
    c.blendIrrG0 = gpu.device.createBindGroup({
      layout: gpu.blendIrrPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
        { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
        { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
        { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
      ],
    });
    c.blendIrrG0Key = gpu.rayResultsBuf;
    c.blendIrrG0KeyProbes = gpu.activeProbesBuf;
  }

  // Group 1: atlas textures — changes every atlas swap (ping-pong) or resize.
  if (c.blendIrrG1KeyRead !== irrReadTex || c.blendIrrG1KeyWrite !== irrWriteTex) {
    const readView = irrReadTex.createView();
    const writeView = irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 });
    c.blendIrrG1 = gpu.device.createBindGroup({
      layout: gpu.blendIrrPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: readView },
        { binding: 1, resource: gpu.linearSampler },
        { binding: 2, resource: writeView },
      ],
    });
    c.blendIrrG1ReadView = readView;
    c.blendIrrG1WriteView = writeView;
    c.blendIrrG1KeyRead = irrReadTex;
    c.blendIrrG1KeyWrite = irrWriteTex;
  }

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-irr' });
  pass.setPipeline(gpu.blendIrrPipeline);
  pass.setBindGroup(0, c.blendIrrG0);
  pass.setBindGroup(1, c.blendIrrG1);
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
  if (c.blendVisG0Key !== gpu.rayResultsBuf || c.blendVisG0KeyProbes !== gpu.activeProbesBuf) {
    c.blendVisG0 = gpu.device.createBindGroup({
      layout: gpu.blendVisPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
        { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
        { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
        { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
      ],
    });
    c.blendVisG0Key = gpu.rayResultsBuf;
    c.blendVisG0KeyProbes = gpu.activeProbesBuf;
  }

  // Group 1: atlas textures — changes every atlas swap (ping-pong) or resize.
  if (c.blendVisG1KeyRead !== visReadTex || c.blendVisG1KeyWrite !== visWriteTex) {
    const readView = visReadTex.createView();
    const writeView = visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 });
    c.blendVisG1 = gpu.device.createBindGroup({
      layout: gpu.blendVisPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: readView },
        { binding: 1, resource: gpu.linearSampler },
        { binding: 2, resource: writeView },
      ],
    });
    c.blendVisG1ReadView = readView;
    c.blendVisG1WriteView = writeView;
    c.blendVisG1KeyRead = visReadTex;
    c.blendVisG1KeyWrite = visWriteTex;
  }

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-vis' });
  pass.setPipeline(gpu.blendVisPipeline);
  pass.setBindGroup(0, c.blendVisG0);
  pass.setBindGroup(1, c.blendVisG1);
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
  if (c.borderG0KeyScratch !== scratchTex || c.borderG0KeyWrite !== writeAtlas) {
    const scratchView = scratchTex.createView();
    const writeView = writeAtlas.createView({ format: 'rgba16float', mipLevelCount: 1 });
    c.borderG0 = gpu.device.createBindGroup({
      layout: gpu.borderVisPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: scratchView },
        { binding: 1, resource: writeView },
        { binding: 2, resource: { buffer: gpu.borderVisUboBuf } },
      ],
    });
    c.borderG0ScratchView = scratchView;
    c.borderG0WriteView = writeView;
    c.borderG0KeyScratch = scratchTex;
    c.borderG0KeyWrite = writeAtlas;
  }

  const pass = encoder.beginComputePass({ label: 'ddgi-border-vis' });
  pass.setPipeline(gpu.borderVisPipeline);
  pass.setBindGroup(0, c.borderG0);
  pass.dispatchWorkgroups(probeCount, 1, 1);
  pass.end();
}
