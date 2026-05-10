import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineOptions,
  EngineState,
  FrameInput,
  FrameOutput,
  Scene,
  SceneEmitter,
  ScenePrimitive,
} from '@vitrum/core';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import { buildPackedScene, uploadPackedScene, type UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { PT_WEBGPU_TRACE_WGSL } from './wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import { HAMMERSLEY_WGSL } from './wgsl/hammersley.wgsl.js';
import { OCTAHEDRAL_WGSL } from './wgsl/octahedral.wgsl.js';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_WGSL };
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

  #paramsBuffer: GPUBuffer | null = null;
  #computePipeline: GPUComputePipeline | null = null;
  #bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(opts: PTEngineWebGPUOptions, slot: StateSlot) {
    this.#slot = slot;
    this.#device = opts.device;
    this.#maxBouncesLimit = Math.max(1, Math.min(opts.maxBounces ?? 3, PROTOTYPE_MAX_BOUNCES));
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#mneeMaxIterations = Math.max(1, opts.mneeMaxIterations ?? 8);
    this.#mneeMaxChainLength = Math.max(1, opts.mneeMaxChainLength ?? 3);
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
      supportedAnalyticShapes: new Set<string>([
        'sphere',
        'box',
        'capsule',
        'cylinder',
        'h-channel-came',
      ]),
      supportedEmitterKinds: new Set<string>(['directional', 'point', 'spot', 'rect-area', 'mesh-area']),
      causticStrategy: this.#causticStrategy,
    };
  }

  #assertLive(method: string): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
    if (this.#scene == null) {
      throw new Error(`${method}: call setScene() before ${method}`);
    }
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
    const scene = this.#scene;
    if (scene == null) {
      throw new Error('updatePrimitive: call setScene() before updatePrimitive');
    }
    const nextScene = patchPrimitiveInScene(scene, id, patch);
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    const scene = this.#scene;
    if (scene == null) {
      throw new Error('updateEmitter: call setScene() before updateEmitter');
    }
    const nextScene = patchEmitterInScene(scene, id, patch);
    this.setScene(nextScene);
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');

    if (this.#slot.get() === 'paused') {
      return {
        primaryRadiance: this.#accumTexture,
        normalDepth: this.#normalDepthTexture ?? undefined,
        albedo: this.#albedoTexture ?? undefined,
        variance: this.#varianceTexture ?? undefined,
        motionVectors: this.#motionVectorsTexture ?? undefined,
        samplesAccumulated: this.#samplesAccumulated,
        isConverged: this.#samplesAccumulated >= this.#maxSamplesLimit,
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
    paramsU32[4] = this.#sceneBuffers.triangleCount >>> 0;
    paramsU32[5] = this.#activeBounces >>> 0;
    paramsU32[6] = this.#sceneBuffers.bvhNodeCount >>> 0;
    paramsU32[7] = this.#sceneBuffers.analyticCount >>> 0;
    paramsF32[8] = input.cameraPosition[0];
    paramsF32[9] = input.cameraPosition[1];
    paramsF32[10] = input.cameraPosition[2];
    paramsF32[11] = 1;
    paramsF32[12] = this.#sceneBuffers.directionalLight[0];
    paramsF32[13] = this.#sceneBuffers.directionalLight[1];
    paramsF32[14] = this.#sceneBuffers.directionalLight[2];
    paramsF32[15] =
      (this.#sceneBuffers.directionalIrradiance[0] +
        this.#sceneBuffers.directionalIrradiance[1] +
        this.#sceneBuffers.directionalIrradiance[2]) /
      3;
    paramsF32[16] = this.#sceneBuffers.pointLightPosition[0];
    paramsF32[17] = this.#sceneBuffers.pointLightPosition[1];
    paramsF32[18] = this.#sceneBuffers.pointLightPosition[2];
    paramsF32[19] = this.#sceneBuffers.hasPointLight ? 1 : 0;
    paramsF32[20] = this.#sceneBuffers.pointLightRadiance[0];
    paramsF32[21] = this.#sceneBuffers.pointLightRadiance[1];
    paramsF32[22] = this.#sceneBuffers.pointLightRadiance[2];
    paramsF32[23] = this.#sceneBuffers.hasPointLight
      ? (this.#sceneBuffers.pointLightRadiance[0] +
          this.#sceneBuffers.pointLightRadiance[1] +
          this.#sceneBuffers.pointLightRadiance[2]) /
        3
      : 0;
    paramsF32[24] = this.#sceneBuffers.spotLightPosition[0];
    paramsF32[25] = this.#sceneBuffers.spotLightPosition[1];
    paramsF32[26] = this.#sceneBuffers.spotLightPosition[2];
    paramsF32[27] = this.#sceneBuffers.hasSpotLight ? 1 : 0;
    paramsF32[28] = this.#sceneBuffers.spotLightDirection[0];
    paramsF32[29] = this.#sceneBuffers.spotLightDirection[1];
    paramsF32[30] = this.#sceneBuffers.spotLightDirection[2];
    paramsF32[31] = this.#sceneBuffers.spotLightCosAngle;
    paramsF32[32] = this.#sceneBuffers.spotLightRadiance[0];
    paramsF32[33] = this.#sceneBuffers.spotLightRadiance[1];
    paramsF32[34] = this.#sceneBuffers.spotLightRadiance[2];
    paramsF32[35] =
      this.#causticStrategy === 'manifold-nee'
        ? 1
        : this.#causticStrategy === 'photon-map'
          ? 2
          : 0;
    paramsF32[36] = this.#sceneBuffers.environmentTint[0];
    paramsF32[37] = this.#sceneBuffers.environmentTint[1];
    paramsF32[38] = this.#sceneBuffers.environmentTint[2];
    paramsF32[39] = this.#sceneBuffers.hasEnvironmentMap ? 1 : 0;
    paramsF32[40] = this.#sceneBuffers.environmentSunDirection[0];
    paramsF32[41] = this.#sceneBuffers.environmentSunDirection[1];
    paramsF32[42] = this.#sceneBuffers.environmentSunDirection[2];
    paramsF32[43] = this.#sceneBuffers.environmentSunStrength;
    paramsF32[44] = this.#sceneBuffers.rectAreaPosition[0];
    paramsF32[45] = this.#sceneBuffers.rectAreaPosition[1];
    paramsF32[46] = this.#sceneBuffers.rectAreaPosition[2];
    paramsF32[47] = this.#sceneBuffers.hasRectAreaLight ? 1 : 0;
    paramsF32[48] = this.#sceneBuffers.rectAreaUAxis[0];
    paramsF32[49] = this.#sceneBuffers.rectAreaUAxis[1];
    paramsF32[50] = this.#sceneBuffers.rectAreaUAxis[2];
    paramsF32[51] = this.#mneeMaxIterations;
    paramsF32[52] = this.#sceneBuffers.rectAreaVAxis[0];
    paramsF32[53] = this.#sceneBuffers.rectAreaVAxis[1];
    paramsF32[54] = this.#sceneBuffers.rectAreaVAxis[2];
    paramsF32[55] = this.#mneeMaxChainLength;
    paramsF32[56] = this.#sceneBuffers.rectAreaRadiance[0];
    paramsF32[57] = this.#sceneBuffers.rectAreaRadiance[1];
    paramsF32[58] = this.#sceneBuffers.rectAreaRadiance[2];
    paramsF32[59] = this.#sceneBuffers.hasRectAreaLight ? 1 : 0;
    paramsF32[60] = this.#sceneBuffers.meshAreaTriA[0];
    paramsF32[61] = this.#sceneBuffers.meshAreaTriA[1];
    paramsF32[62] = this.#sceneBuffers.meshAreaTriA[2];
    paramsF32[63] = this.#sceneBuffers.hasMeshAreaLight ? 1 : 0;
    paramsF32[64] = this.#sceneBuffers.meshAreaTriB[0];
    paramsF32[65] = this.#sceneBuffers.meshAreaTriB[1];
    paramsF32[66] = this.#sceneBuffers.meshAreaTriB[2];
    paramsF32[67] = this.#sceneBuffers.environmentMapWidth;
    paramsF32[68] = this.#sceneBuffers.meshAreaTriC[0];
    paramsF32[69] = this.#sceneBuffers.meshAreaTriC[1];
    paramsF32[70] = this.#sceneBuffers.meshAreaTriC[2];
    paramsF32[71] = this.#sceneBuffers.environmentMapHeight;
    paramsF32[72] = this.#sceneBuffers.meshAreaRadiance[0];
    paramsF32[73] = this.#sceneBuffers.meshAreaRadiance[1];
    paramsF32[74] = this.#sceneBuffers.meshAreaRadiance[2];
    paramsF32[75] = this.#sceneBuffers.hasMeshAreaLight ? 1 : 0;
    paramsF32.set(invVp, 76); // float index 76 = byte 304; WGSL FrameParams.invViewProj starts at byte 304 (after 8×u32 + 17×vec4f = 304 bytes)
    paramsF32.set(vp, 92); // float index 92 = byte 368; viewProj follows invViewProj (304 + 64 = 368)
    const prevVp = multiplyMat4(
      input.prevProjMatrix ?? input.projMatrix,
      input.prevViewMatrix ?? input.viewMatrix,
    );
    paramsF32.set(prevVp, 108); // float index 108 = byte 432; prevViewProj follows viewProj (368 + 64 = 432)
    this.#device.queue.writeBuffer(this.#paramsBuffer, 0, paramsArrayBuffer);

    const bindGroup = this.#device.createBindGroup({
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
      ],
    });

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
