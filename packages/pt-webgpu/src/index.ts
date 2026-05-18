import type {
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineFactory,
  EngineOptions,
  EngineState,
  FrameInput,
  FrameOutput,
  GpuMemoryBreakdown,
  Scene,
  SceneEmitter,
  ScenePrimitive,
} from '@vitrum/core';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import { buildPackedScene, uploadPackedScene, PT_WEBGPU_ANALYTIC_SHAPES, type UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { PT_WEBGPU_TRACE_WGSL } from './wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export { summarizeScene };
export type { SceneSummary };

export interface PTEngineWebGPUOptions extends EngineOptions {
  readonly device: GPUDevice;
}

const PROTOTYPE_MAX_BOUNCES = 8;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;

interface StateSlot {
  readonly get: () => EngineState;
  readonly set: (s: EngineState) => void;
}

function makeStateSlot(initial: EngineState = 'initializing'): StateSlot {
  let s: EngineState = initial;
  return {
    get: () => s,
    set: (v) => {
      s = v;
    },
  };
}

class PTEngineWebGPU implements Engine {
  readonly #slot: StateSlot;
  readonly #device: GPUDevice;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;

  #scene: Scene | null = null;
  #sceneBuffers: UploadedSceneBuffers | null = null;
  #samplesAccumulated = 0;
  #activeBounces = 1;

  #accumTexture: GPUTexture | null = null;
  #accumView: GPUTextureView | null = null;
  #normalDepthTexture: GPUTexture | null = null;
  #normalDepthView: GPUTextureView | null = null;
  #albedoTexture: GPUTexture | null = null;
  #albedoView: GPUTextureView | null = null;
  #varianceTexture: GPUTexture | null = null;
  #varianceView: GPUTextureView | null = null;
  #motionVectorsTexture: GPUTexture | null = null;
  #motionVectorsView: GPUTextureView | null = null;
  #accumBuffer: GPUBuffer | null = null;
  #varianceMomentsBuffer: GPUBuffer | null = null;
  #accumBufferByteSize = 0;
  #accumWidth = 0;
  #accumHeight = 0;

  /** Reused bind group until scene buffers or accum views are recreated. */
  #pathTraceBindGroup: GPUBindGroup | null = null;

  #paramsBuffer: GPUBuffer | null = null;
  #computePipeline: GPUComputePipeline | null = null;
  #bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(opts: PTEngineWebGPUOptions, slot: StateSlot) {
    this.#slot = slot;
    this.#device = opts.device;
    this.#maxBouncesLimit = Math.max(1, Math.min(opts.maxBounces ?? 3, PROTOTYPE_MAX_BOUNCES));
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    const mneeIter = typeof causticOpts.mneeMaxIterations === 'number' ? causticOpts.mneeMaxIterations : 8;
    const mneeChain = typeof causticOpts.mneeMaxChainLength === 'number' ? causticOpts.mneeMaxChainLength : 3;
    this.#mneeMaxIterations = Math.max(1, mneeIter);
    this.#mneeMaxChainLength = Math.max(1, mneeChain);
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  get capabilities(): EngineCapabilities {
    return {
      supportsIncrementalScene: false, // Honest reporting — updatePrimitive/updateEmitter currently delegate to setScene; flip to true when real incremental patching lands.
      supportsMotionBlur: false,
      supportsAuxBuffers: true,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      // Slot 0 is the "unknown" sentinel; supported shapes start at index 1.
      supportedAnalyticShapes: new Set<string>(PT_WEBGPU_ANALYTIC_SHAPES.slice(1)),
      supportedEmitterKinds: new Set<string>(['directional', 'point', 'spot', 'rect-area', 'disc-area', 'mesh-area']),
      causticStrategy: this.#causticStrategy,
      // W3-D8 — this engine exposes `debug.estimatedGpuMemoryBytes()`.
      debugSurface: true,
    };
  }

  // ── Debug introspection (T3.G followup) ────────────────────────────────
  // Prototype backend exposes only the GPU-memory estimate for now — atlas
  // / BVH / pick / denoiser-toggle hooks are walkaround-hybrid concepts
  // that don't apply to a brute-force compute path tracer.
  readonly debug: EngineDebugSurface = {
    estimatedGpuMemoryBytes: (): GpuMemoryBreakdown | null => {
      const W = this.#accumWidth;
      const H = this.#accumHeight;
      // Pre-init / pre-renderFrame: no accum textures yet.
      if (W <= 0 || H <= 0 || this.#accumTexture == null) return null;

      // Per-texel bytes inferred from the actual format at allocation time
      // (see #ensureAccumResources: accum / normalDepth / albedo / variance
      // / motionVectors are all rgba16float = 8 bytes/texel).
      const RGBA16F = 8;
      const texPixels = W * H;
      const accumBytes        = texPixels * RGBA16F;
      const normalDepthBytes  = texPixels * RGBA16F;
      const albedoBytes       = texPixels * RGBA16F;
      const varianceBytes     = texPixels * RGBA16F;
      const motionBytes       = texPixels * RGBA16F;
      const accumBufBytes     = this.#accumBufferByteSize;
      const varMomentBufBytes = this.#varianceMomentsBuffer != null
        ? this.#accumBufferByteSize : 0;
      const paramsBufBytes    = this.#paramsBuffer != null ? 512 : 0;
      // Scene buffers (BVH, materials, indices, etc.) are owned by
      // UploadedSceneBuffers; size accounting there would require touching
      // the uploader — left for a follow-up so this commit stays focused.

      const commonTexBytes = accumBytes + normalDepthBytes + albedoBytes
        + varianceBytes + motionBytes;
      const commonBufBytes = accumBufBytes + varMomentBufBytes + paramsBufBytes;
      const total = commonTexBytes + commonBufBytes;

      return Object.freeze({
        total,
        byCategory: Object.freeze({
          common: total,
        }),
        byTextureFormat: Object.freeze({
          rgba16float: commonTexBytes,
        }),
        byBufferUsage: Object.freeze({
          storage: accumBufBytes + varMomentBufBytes,
          uniform: paramsBufBytes,
        }),
      });
    },
  };

  #assertLive(method: string): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
    if (this.#scene == null) {
      throw new Error(`${method}: call setScene() before ${method}`);
    }
  }

  /**
   * Pack the per-frame uniform buffer (512 bytes, vec4-aligned). Layout is
   * pinned and pathTraceBruteforce.wgsl's `params` struct reads from these
   * exact offsets. Callers must have already validated that #sceneBuffers
   * is non-null (renderFrame's preconditions handle this).
   *
   * Layout (336 bytes used out of a 512-byte buffer; trailing bytes are zero):
   *
   *   u32 slot 0..1   width, height
   *   u32 slot 2..3   frameIndex, frameSeed
   *   u32 slot 4..7   triangleCount, activeBounces, bvhNodeCount, analyticCount
   *   u32 slot 8..11  pointLightCount, spotLightCount, rectAreaLightCount,
   *                   meshAreaLightCount
   *   u32 slot 12..13 mneeMaxIterations, mneeMaxChainLength
   *   u32 slot 14..15 hasEnvironmentMap (0/1), causticStrategy
   *                   (0=none, 1=manifold-nee, 2=photon-map)
   *   u32 slot 16..17 environmentMapWidth, environmentMapHeight
   *   f32 slot 18     triIntersectEpsilon (default 1e-5; metre-scale)
   *   u32 slot 19     _pad1   (zero; reserved)
   *
   *   f32 slot 20..23 cameraPos.xyz + 1.0
   *   f32 slot 24..27 lightDir.xyz + averageDirectionalIrradiance
   *   f32 slot 28..31 environmentTint.xyz + 0   (.w unused, write 0)
   *   f32 slot 32..35 environmentSun.xyz (sun dir) + sun strength
   *
   *   f32 slot 36..51 invViewProj (mat4x4f, 16 floats)
   *   f32 slot 52..67 viewProj    (mat4x4f, 16 floats)
   *   f32 slot 68..83 prevViewProj(mat4x4f, 16 floats)
   *
   * Per-light data lives in dedicated storage buffers at bind slots 20..23;
   * see `packEmitterArrays` for the layout (8 f32 / point light, 12 / spot,
   * 16 / rect-area, 16 / mesh-area).
   */
  #buildParamsBuffer(input: FrameInput, width: number, height: number): ArrayBuffer {
    const sb = this.#sceneBuffers!;
    const vp = multiplyMat4(input.projMatrix, input.viewMatrix);
    const invVp = invertMat4(vp) ?? new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    const paramsArrayBuffer = new ArrayBuffer(512);
    const paramsU32 = new Uint32Array(paramsArrayBuffer);
    const paramsF32 = new Float32Array(paramsArrayBuffer);
    paramsU32[0] = width;
    paramsU32[1] = height;
    paramsU32[2] = input.frameIndex >>> 0;
    paramsU32[3] = input.frameSeed >>> 0;
    paramsU32[4] = sb.triangleCount >>> 0;
    paramsU32[5] = this.#activeBounces >>> 0;
    paramsU32[6] = sb.bvhNodeCount >>> 0;
    paramsU32[7] = sb.analyticCount >>> 0;
    paramsU32[8] = sb.pointLightCount >>> 0;
    paramsU32[9] = sb.spotLightCount >>> 0;
    paramsU32[10] = sb.rectAreaLightCount >>> 0;
    paramsU32[11] = sb.meshAreaLightCount >>> 0;
    paramsU32[12] = this.#mneeMaxIterations >>> 0;
    paramsU32[13] = this.#mneeMaxChainLength >>> 0;
    paramsU32[14] = sb.hasEnvironmentMap ? 1 : 0;
    paramsU32[15] =
      this.#causticStrategy === 'manifold-nee'
        ? 1
        : this.#causticStrategy === 'photon-map'
          ? 2
          : 0;
    paramsU32[16] = sb.environmentMapWidth >>> 0;
    paramsU32[17] = sb.environmentMapHeight >>> 0;
    paramsF32[18] = 1e-5; // triIntersectEpsilon: default metre-scale (D12)
    // Slot 19 (_pad1) is padding; zero-initialized by ArrayBuffer.
    paramsF32[20] = input.cameraPosition[0];
    paramsF32[21] = input.cameraPosition[1];
    paramsF32[22] = input.cameraPosition[2];
    paramsF32[23] = 1;
    paramsF32[24] = sb.directionalLight[0];
    paramsF32[25] = sb.directionalLight[1];
    paramsF32[26] = sb.directionalLight[2];
    paramsF32[27] =
      (sb.directionalIrradiance[0] +
        sb.directionalIrradiance[1] +
        sb.directionalIrradiance[2]) /
      3;
    paramsF32[28] = sb.environmentTint[0];
    paramsF32[29] = sb.environmentTint[1];
    paramsF32[30] = sb.environmentTint[2];
    paramsF32[31] = 0;
    paramsF32[32] = sb.environmentSunDirection[0];
    paramsF32[33] = sb.environmentSunDirection[1];
    paramsF32[34] = sb.environmentSunDirection[2];
    paramsF32[35] = sb.environmentSunStrength;
    paramsF32.set(invVp, 36);
    paramsF32.set(vp, 52);
    const prevVp = multiplyMat4(
      input.prevProjMatrix ?? input.projMatrix,
      input.prevViewMatrix ?? input.viewMatrix,
    );
    paramsF32.set(prevVp, 68);
    return paramsArrayBuffer;
  }

  #destroyAccumTexture(): void {
    this.#accumTexture?.destroy();
    this.#accumTexture = null;
    this.#accumView = null;
    this.#normalDepthTexture?.destroy();
    this.#normalDepthTexture = null;
    this.#normalDepthView = null;
    this.#albedoTexture?.destroy();
    this.#albedoTexture = null;
    this.#albedoView = null;
    this.#varianceTexture?.destroy();
    this.#varianceTexture = null;
    this.#varianceView = null;
    this.#motionVectorsTexture?.destroy();
    this.#motionVectorsTexture = null;
    this.#motionVectorsView = null;
    this.#accumBuffer?.destroy();
    this.#accumBuffer = null;
    this.#varianceMomentsBuffer?.destroy();
    this.#varianceMomentsBuffer = null;
    this.#accumBufferByteSize = 0;
    this.#accumWidth = 0;
    this.#accumHeight = 0;
  }

  #clearAccumBuffer(): void {
    if (this.#accumBuffer == null) return;
    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.clearAccum' });
    encoder.clearBuffer(this.#accumBuffer);
    if (this.#varianceMomentsBuffer != null) {
      encoder.clearBuffer(this.#varianceMomentsBuffer);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  #ensureAccumResources(width: number, height: number): void {
    const targetByteSize = width * height * 16;
    const textureReady =
      this.#accumTexture != null && this.#accumWidth === width && this.#accumHeight === height;
    const bufferReady = this.#accumBuffer != null && this.#accumBufferByteSize === targetByteSize;
    if (textureReady && bufferReady) {
      return;
    }
    this.#destroyAccumTexture();
    this.#accumTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.accum',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.#accumView = this.#accumTexture.createView();
    this.#normalDepthTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.normalDepth',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#normalDepthView = this.#normalDepthTexture.createView();
    this.#albedoTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.albedo',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#albedoView = this.#albedoTexture.createView();
    this.#varianceTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.variance',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#varianceView = this.#varianceTexture.createView();
    this.#motionVectorsTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.motionVectors',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#motionVectorsView = this.#motionVectorsTexture.createView();
    this.#accumBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.accum.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#varianceMomentsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.varianceMoments.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#accumBufferByteSize = targetByteSize;
    this.#accumWidth = width;
    this.#accumHeight = height;
    this.#samplesAccumulated = 0;
    this.#pathTraceBindGroup = null;
    this.#clearAccumBuffer();
  }

  #ensurePipeline(): void {
    if (this.#computePipeline != null && this.#bindGroupLayout != null && this.#paramsBuffer != null) {
      return;
    }
    this.#paramsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.params',
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = this.#device.createShaderModule({
      label: 'vitrum.pt-webgpu.pathTrace',
      code: PT_WEBGPU_TRACE_WGSL,
    });
    this.#computePipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.pathTrace.pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
      },
    });
    this.#bindGroupLayout = this.#computePipeline.getBindGroupLayout(0);
  }

  setScene(scene: Scene): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('setScene: engine is disposed');
    }
    const packed = buildPackedScene(scene);
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = uploadPackedScene(this.#device, packed);
    this.#pathTraceBindGroup = null;
    this.#scene = scene;
    const sceneSummary = summarizeScene(scene);
    if (sceneSummary.primitiveCount === 0) {
      console.warn('[vitrum/pt-webgpu] Empty scene provided; rendering sky-only fallback.');
    }
    for (const warning of packed.warnings) {
      console.warn(`[vitrum/pt-webgpu] ${warning}`);
    }
    this.reset();
  }

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    // #assertLive already throws when #scene is null; the non-null assertion
    // captures that invariant for the type checker.
    const nextScene = patchPrimitiveInScene(this.#scene!, id, patch);
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    const nextScene = patchEmitterInScene(this.#scene!, id, patch);
    this.setScene(nextScene);
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');

    if (this.#slot.get() === 'paused') {
      const pq = input.quality ?? {};
      const targetSppPaused = Math.min(pq.samplesTarget ?? 16, this.#maxSamplesLimit);
      return {
        primaryRadiance: this.#accumTexture,
        normalDepth: this.#normalDepthTexture ?? undefined,
        albedo: this.#albedoTexture ?? undefined,
        variance: this.#varianceTexture ?? undefined,
        motionVectors: this.#motionVectorsTexture ?? undefined,
        samplesAccumulated: this.#samplesAccumulated,
        isConverged: this.#samplesAccumulated >= targetSppPaused,
      };
    }

    const q = input.quality ?? {};
    this.#activeBounces = Math.max(1, Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit));
    const resolution = q.resolutionFactor ?? 1;
    const width = Math.max(1, Math.floor(input.viewport.width * resolution));
    const height = Math.max(1, Math.floor(input.viewport.height * resolution));

    this.#ensureAccumResources(width, height);
    this.#ensurePipeline();
    if (
      this.#accumView == null ||
      this.#normalDepthView == null ||
      this.#albedoView == null ||
      this.#varianceView == null ||
      this.#motionVectorsView == null ||
      this.#accumBuffer == null ||
      this.#varianceMomentsBuffer == null ||
      this.#paramsBuffer == null ||
      this.#computePipeline == null ||
      this.#bindGroupLayout == null ||
      this.#sceneBuffers == null
    ) {
      throw new Error('renderFrame: failed to initialize WebGPU pipeline resources');
    }

    const paramsArrayBuffer = this.#buildParamsBuffer(input, width, height);
    this.#device.queue.writeBuffer(this.#paramsBuffer, 0, paramsArrayBuffer);

    let bindGroup = this.#pathTraceBindGroup;
    if (bindGroup == null) {
      bindGroup = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup',
        layout: this.#bindGroupLayout,
        entries: [
          { binding: 0, resource: this.#accumView },
          { binding: 1, resource: { buffer: this.#paramsBuffer } },
          { binding: 2, resource: { buffer: this.#accumBuffer } },
          { binding: 3, resource: { buffer: this.#sceneBuffers.positionsBuffer } },
          { binding: 4, resource: { buffer: this.#sceneBuffers.indicesBuffer } },
          { binding: 5, resource: { buffer: this.#sceneBuffers.triMaterialIdsBuffer } },
          { binding: 6, resource: { buffer: this.#sceneBuffers.materialsBuffer } },
          { binding: 7, resource: { buffer: this.#sceneBuffers.bvhNodesBuffer } },
          { binding: 8, resource: { buffer: this.#sceneBuffers.normalsBuffer } },
          { binding: 9, resource: this.#normalDepthView },
          { binding: 10, resource: this.#albedoView },
          { binding: 11, resource: this.#varianceView },
          { binding: 12, resource: this.#motionVectorsView },
          { binding: 13, resource: { buffer: this.#varianceMomentsBuffer } },
          { binding: 14, resource: { buffer: this.#sceneBuffers.analyticHeadersBuffer } },
          { binding: 15, resource: { buffer: this.#sceneBuffers.analyticParamsBuffer } },
          { binding: 16, resource: { buffer: this.#sceneBuffers.analyticLocalToWorldBuffer } },
          { binding: 17, resource: { buffer: this.#sceneBuffers.analyticWorldToLocalBuffer } },
          { binding: 18, resource: { buffer: this.#sceneBuffers.environmentMapTexelsBuffer } },
          { binding: 19, resource: { buffer: this.#sceneBuffers.environmentMapCdfBuffer } },
          { binding: 20, resource: { buffer: this.#sceneBuffers.pointLightsBuffer } },
          { binding: 21, resource: { buffer: this.#sceneBuffers.spotLightsBuffer } },
          { binding: 22, resource: { buffer: this.#sceneBuffers.rectAreaLightsBuffer } },
          { binding: 23, resource: { buffer: this.#sceneBuffers.meshAreaLightsBuffer } },
        ],
      });
      this.#pathTraceBindGroup = bindGroup;
    }

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.pathTrace.encoder' });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.pathTrace.pass' });
    pass.setPipeline(this.#computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / WORKGROUP_SIZE),
      Math.ceil(height / WORKGROUP_SIZE),
      1,
    );
    pass.end();
    this.#device.queue.submit([encoder.finish()]);

    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    return {
      primaryRadiance: this.#accumTexture,
      normalDepth: this.#normalDepthTexture ?? undefined,
      albedo: this.#albedoTexture ?? undefined,
      variance: this.#varianceTexture ?? undefined,
      motionVectors: this.#motionVectorsTexture ?? undefined,
      samplesAccumulated: this.#samplesAccumulated,
      isConverged: this.#samplesAccumulated >= targetSpp,
    };
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#samplesAccumulated = 0;
    this.#clearAccumBuffer();
  }

  pause(): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('pause: engine is disposed');
    }
    this.#slot.set('paused');
  }

  resume(): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('resume: engine is disposed');
    }
    this.#slot.set('ready');
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#destroyAccumTexture();
    this.#pathTraceBindGroup = null;
    this.#paramsBuffer?.destroy();
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = null;
    this.#paramsBuffer = null;
    this.#computePipeline = null;
    this.#bindGroupLayout = null;
    this.#scene = null;
    this.#slot.set('disposed');
  }
}

export const createPTEngine_WebGPU: EngineFactory<PTEngineWebGPUOptions> = async (
  opts: PTEngineWebGPUOptions,
): Promise<Engine> => {
  if (opts.device == null || typeof (opts.device as GPUDevice).createCommandEncoder !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGPU: device must be a GPUDevice instance',
    );
  }
  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }
  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }
  if (maxBounces !== undefined && maxBounces > PROTOTYPE_MAX_BOUNCES) {
    console.warn(
      `[vitrum/pt-webgpu] maxBounces=${maxBounces} requested, clamping to prototype limit ${PROTOTYPE_MAX_BOUNCES}.`,
    );
  }
  if (opts.denoiser != null && opts.denoiser !== 'none') {
    console.warn(
      `[vitrum/pt-webgpu] denoiser="${opts.denoiser}" requested, but prototype backend has no denoiser integration yet.`,
    );
  }
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(opts, slot);
  slot.set('ready');
  return engine;
};
