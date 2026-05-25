import type {
  AnalyticShape,
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
import { asBackendTexture, asMat4 } from '@vitrum/core';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import { buildPackedScene, uploadPackedScene, PT_WEBGPU_ANALYTIC_SHAPES, type UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { FrameParamsSlot } from './scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from './scene/materialPacking.js';
import { defaultDirectionalIrradiance, defaultDirectionalLight, packEmitterArrays } from './scene/emitterPacking.js';
import { environmentParams } from './scene/environmentPacking.js';
import { PT_WEBGPU_TRACE_WGSL } from './wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export { summarizeScene };
export type { SceneSummary };
export { buildSceneTlas, type TlasInstance, type TlasData } from './scene/tlasBridge.js';

export interface PTEngineWebGPUOptions extends EngineOptions {
  readonly device: GPUDevice;
}

const PROTOTYPE_MAX_BOUNCES = 8;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;
// Bind group 0 currently consumes 23 storage-buffer bindings in the trace pass
// (positions/normals/indices/materials/BVH + variance moments + analytic +
// environment + light arrays + TLAS buffers). Keep this guard in lockstep so
// devices that cannot bind the full layout fail fast at construction time.
const REQUIRED_STORAGE_BUFFERS_PER_STAGE = 23;

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

  static readonly #SUPPORTED_ANALYTIC_SHAPES = new Set(
    PT_WEBGPU_ANALYTIC_SHAPES.slice(1),
  );

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
      // Material-only primitive patches are uploaded in-place; all other facets
      // currently rebuild through setScene until broader partial uploads land.
      supportsIncrementalScene: true,
      incrementalPatchSupport: {
        transform: false,
        positions: false,
        material: true,
        emitter: true,
        topology: false,
      },
      supportsAuxBuffers: true,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      // Slot 0 is the "unknown" sentinel; supported shapes start at index 1.
      // Slot 0 is 'unknown' — strip before advertising. The remaining entries
      // are AnalyticShape-typed in the source array; cast through unknown here
      // because TS narrows the literal-tuple type back to a generic readonly
      // string[] after slice(1).
      supportedAnalyticShapes: new Set(
        PT_WEBGPU_ANALYTIC_SHAPES.slice(1) as unknown as readonly AnalyticShape[],
      ),
      supportedEmitterKinds: new Set<SceneEmitter['kind']>(
        ['directional', 'point', 'spot', 'rect-area', 'disc-area', 'mesh-area'],
      ),
      supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>([
        'mesh', 'instanced-mesh', 'analytic', 'skinned-mesh',
      ]),
      supportedEnvironmentKinds: new Set<Scene['environment']['kind']>([
        'none', 'hdri', 'procedural-sky',
      ]),
      presentationMode: 'offscreen-texture',
      experimentalFeatures: new Set(['prototype-backend']),
      causticStrategy: this.#causticStrategy,
      // W3-D8 — this engine exposes `debug.estimatedGpuMemoryBytes()`.
      debugSurface: true,
    };
  }

  #materialIndexForPrimitive(scene: Scene, primitiveId: string): number | null {
    let materialIndex = 0;
    for (const primitive of scene.primitives) {
      let contributesMaterial = false;
      if (primitive.kind === 'analytic') {
        contributesMaterial = PTEngineWebGPU.#SUPPORTED_ANALYTIC_SHAPES.has(primitive.shape);
      } else {
        contributesMaterial = true;
      }
      if (primitive.id === primitiveId) {
        return contributesMaterial ? materialIndex : null;
      }
      if (contributesMaterial) materialIndex += 1;
    }
    return null;
  }

  #canFastPathMaterialPatch(
    patch: Partial<ScenePrimitive>,
  ): patch is Partial<ScenePrimitive> & { material: ScenePrimitive['material'] } {
    if (patch.material == null) return false;
    for (const key of Object.keys(patch)) {
      if (key !== 'material' && key !== 'id' && key !== 'kind') return false;
    }
    return true;
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
    const invVp = invertMat4(asMat4(vp));
    if (invVp == null) {
      throw new Error('renderFrame: non-invertible view-projection matrix');
    }

    const paramsArrayBuffer = new ArrayBuffer(512);
    const paramsU32 = new Uint32Array(paramsArrayBuffer);
    const paramsF32 = new Float32Array(paramsArrayBuffer);
    paramsU32[FrameParamsSlot.width] = width;
    paramsU32[FrameParamsSlot.height] = height;
    paramsU32[FrameParamsSlot.frameIndex] = input.frameIndex >>> 0;
    paramsU32[FrameParamsSlot.frameSeed] = input.frameSeed >>> 0;
    paramsU32[FrameParamsSlot.triangleCount] = sb.triangleCount >>> 0;
    paramsU32[FrameParamsSlot.maxBounces] = this.#activeBounces >>> 0;
    paramsU32[FrameParamsSlot.bvhNodeCount] = sb.bvhNodeCount >>> 0;
    paramsU32[FrameParamsSlot.analyticCount] = sb.analyticCount >>> 0;
    paramsU32[FrameParamsSlot.pointLightCount] = sb.pointLightCount >>> 0;
    paramsU32[FrameParamsSlot.spotLightCount] = sb.spotLightCount >>> 0;
    paramsU32[FrameParamsSlot.rectAreaLightCount] = sb.rectAreaLightCount >>> 0;
    paramsU32[FrameParamsSlot.meshAreaLightCount] = sb.meshAreaLightCount >>> 0;
    paramsU32[FrameParamsSlot.mneeMaxIterations] = this.#mneeMaxIterations >>> 0;
    paramsU32[FrameParamsSlot.mneeMaxChainLength] = this.#mneeMaxChainLength >>> 0;
    paramsU32[FrameParamsSlot.hasEnvironmentMap] = sb.hasEnvironmentMap ? 1 : 0;
    paramsU32[FrameParamsSlot.causticStrategy] =
      this.#causticStrategy === 'manifold-nee'
        ? 1
        : this.#causticStrategy === 'photon-map'
          ? 2
          : 0;
    paramsU32[FrameParamsSlot.environmentMapWidth] = sb.environmentMapWidth >>> 0;
    paramsU32[FrameParamsSlot.environmentMapHeight] = sb.environmentMapHeight >>> 0;
    paramsF32[FrameParamsSlot.triIntersectEpsilon] = 1e-5; // triIntersectEpsilon: default metre-scale (D12)
    paramsU32[FrameParamsSlot.tlasNodeCount] = sb.tlasNodeCount >>> 0;
    // Slot 19 (_pad1) is padding; zero-initialized by ArrayBuffer.
    paramsF32[FrameParamsSlot.cameraPos] = input.cameraPosition[0];
    paramsF32[FrameParamsSlot.cameraPos + 1] = input.cameraPosition[1];
    paramsF32[FrameParamsSlot.cameraPos + 2] = input.cameraPosition[2];
    paramsF32[FrameParamsSlot.cameraPos + 3] = 1;
    paramsF32[FrameParamsSlot.lightDir] = sb.directionalLight[0];
    paramsF32[FrameParamsSlot.lightDir + 1] = sb.directionalLight[1];
    paramsF32[FrameParamsSlot.lightDir + 2] = sb.directionalLight[2];
    paramsF32[FrameParamsSlot.lightDir + 3] =
      (sb.directionalIrradiance[0] +
        sb.directionalIrradiance[1] +
        sb.directionalIrradiance[2]) /
      3;
    paramsF32[FrameParamsSlot.environmentTint] = sb.environmentTint[0];
    paramsF32[FrameParamsSlot.environmentTint + 1] = sb.environmentTint[1];
    paramsF32[FrameParamsSlot.environmentTint + 2] = sb.environmentTint[2];
    paramsF32[FrameParamsSlot.environmentTint + 3] = 0;
    paramsF32[FrameParamsSlot.environmentSun] = sb.environmentSunDirection[0];
    paramsF32[FrameParamsSlot.environmentSun + 1] = sb.environmentSunDirection[1];
    paramsF32[FrameParamsSlot.environmentSun + 2] = sb.environmentSunDirection[2];
    paramsF32[FrameParamsSlot.environmentSun + 3] = sb.environmentSunStrength;
    paramsF32.set(invVp, FrameParamsSlot.invViewProj);
    paramsF32.set(vp, FrameParamsSlot.viewProj);
    const prevVp = multiplyMat4(
      input.prevProjMatrix ?? input.projMatrix,
      input.prevViewMatrix ?? input.viewMatrix,
    );
    paramsF32.set(prevVp, FrameParamsSlot.prevViewProj);
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
    const currentScene = this.#scene!;
    const nextScene = patchPrimitiveInScene(currentScene, id, patch);
    if (this.#canFastPathMaterialPatch(patch) && this.#sceneBuffers != null) {
      const materialIndex = this.#materialIndexForPrimitive(nextScene, id);
      const primitive = nextScene.primitives.find((p) => p.id === id);
      if (materialIndex != null && primitive != null) {
        const packed = materialToPackedVec4s(primitive.material);
        if (packed.length === MATERIAL_FLOAT_STRIDE) {
          const materialData = new Float32Array(packed);
          const floatOffset = materialIndex * MATERIAL_FLOAT_STRIDE;
          const byteOffset = floatOffset * Float32Array.BYTES_PER_ELEMENT;
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.materialsBuffer,
            byteOffset,
            materialData.buffer,
            materialData.byteOffset,
            materialData.byteLength,
          );
          this.#sceneBuffers.materials.set(materialData, floatOffset);
          this.#scene = nextScene;
          this.reset();
          return;
        }
      }
    }
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    const currentScene = this.#scene!;
    const nextScene = patchEmitterInScene(currentScene, id, patch);
    if (this.#sceneBuffers != null) {
      const packed = packEmitterArrays(nextScene);
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.pointLightsBuffer,
        0,
        packed.pointLightsData.buffer,
        packed.pointLightsData.byteOffset,
        packed.pointLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.spotLightsBuffer,
        0,
        packed.spotLightsData.buffer,
        packed.spotLightsData.byteOffset,
        packed.spotLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.rectAreaLightsBuffer,
        0,
        packed.rectAreaLightsData.buffer,
        packed.rectAreaLightsData.byteOffset,
        packed.rectAreaLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.meshAreaLightsBuffer,
        0,
        packed.meshAreaLightsData.buffer,
        packed.meshAreaLightsData.byteOffset,
        packed.meshAreaLightsData.byteLength,
      );
      this.#sceneBuffers.pointLightsData.set(packed.pointLightsData);
      this.#sceneBuffers.spotLightsData.set(packed.spotLightsData);
      this.#sceneBuffers.rectAreaLightsData.set(packed.rectAreaLightsData);
      this.#sceneBuffers.meshAreaLightsData.set(packed.meshAreaLightsData);
      const mutableSceneBuffers = this.#sceneBuffers as unknown as {
        pointLightCount: number;
        spotLightCount: number;
        rectAreaLightCount: number;
        meshAreaLightCount: number;
        directionalLight: readonly [number, number, number];
        directionalIrradiance: readonly [number, number, number];
      };
      mutableSceneBuffers.pointLightCount = packed.pointLightCount;
      mutableSceneBuffers.spotLightCount = packed.spotLightCount;
      mutableSceneBuffers.rectAreaLightCount = packed.rectAreaLightCount;
      mutableSceneBuffers.meshAreaLightCount = packed.meshAreaLightCount;
      mutableSceneBuffers.directionalLight = defaultDirectionalLight(nextScene);
      mutableSceneBuffers.directionalIrradiance = defaultDirectionalIrradiance(nextScene);
      this.#scene = nextScene;
      for (const warning of packed.warnings) {
        console.warn(`[vitrum/pt-webgpu] ${warning}`);
      }
      this.reset();
      return;
    }
    this.setScene(nextScene);
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    this.#assertLive('updateEnvironment');
    const currentScene = this.#scene!;
    const nextScene: Scene = {
      ...currentScene,
      environment: env ?? { kind: 'none' },
    };
    if (this.#sceneBuffers != null) {
      const packed = environmentParams(nextScene);
      const texelLenMatches = packed.hdriTexels.length === this.#sceneBuffers.environmentMapTexels.length;
      const cdfLenMatches = packed.hdriCdf.length === this.#sceneBuffers.environmentMapCdf.length;
      if (texelLenMatches && cdfLenMatches) {
        if (packed.hdriTexels.byteLength > 0) {
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.environmentMapTexelsBuffer,
            0,
            packed.hdriTexels.buffer,
            packed.hdriTexels.byteOffset,
            packed.hdriTexels.byteLength,
          );
        }
        if (packed.hdriCdf.byteLength > 0) {
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.environmentMapCdfBuffer,
            0,
            packed.hdriCdf.buffer,
            packed.hdriCdf.byteOffset,
            packed.hdriCdf.byteLength,
          );
        }
        const mutableSceneBuffers = this.#sceneBuffers as unknown as {
          environmentTint: readonly [number, number, number];
          environmentSunDirection: readonly [number, number, number];
          environmentSunStrength: number;
          environmentMapWidth: number;
          environmentMapHeight: number;
          hasEnvironmentMap: boolean;
        };
        mutableSceneBuffers.environmentTint = packed.tint;
        mutableSceneBuffers.environmentSunDirection = packed.sunDirection;
        mutableSceneBuffers.environmentSunStrength = packed.sunStrength;
        mutableSceneBuffers.environmentMapWidth = packed.hdriWidth;
        mutableSceneBuffers.environmentMapHeight = packed.hdriHeight;
        mutableSceneBuffers.hasEnvironmentMap = packed.hasHdri;
        this.#sceneBuffers.environmentMapTexels.set(packed.hdriTexels);
        this.#sceneBuffers.environmentMapCdf.set(packed.hdriCdf);
        this.#scene = nextScene;
        for (const warning of packed.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
    }
    this.setScene(nextScene);
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');

    if (this.#slot.get() === 'paused') {
      const pq = input.quality ?? {};
      const targetSppPaused = Math.min(pq.samplesTarget ?? 16, this.#maxSamplesLimit);
      const accumTexture = this.#accumTexture;
      if (accumTexture == null) {
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      return {
        kind: 'rendered',
        primaryRadiance: asBackendTexture<'webgpu', GPUTexture>(accumTexture),
        ...(this.#normalDepthTexture != null
          ? { normalDepth: asBackendTexture<'webgpu', GPUTexture>(this.#normalDepthTexture) }
          : {}),
        ...(this.#albedoTexture != null
          ? { albedo: asBackendTexture<'webgpu', GPUTexture>(this.#albedoTexture) }
          : {}),
        ...(this.#varianceTexture != null
          ? { variance: asBackendTexture<'webgpu', GPUTexture>(this.#varianceTexture) }
          : {}),
        ...(this.#motionVectorsTexture != null
          ? { motionVectors: asBackendTexture<'webgpu', GPUTexture>(this.#motionVectorsTexture) }
          : {}),
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
          { binding: 24, resource: { buffer: this.#sceneBuffers.tlasNodesBuffer } },
          { binding: 25, resource: { buffer: this.#sceneBuffers.tlasInstanceIndicesBuffer } },
          { binding: 26, resource: { buffer: this.#sceneBuffers.tlasBlasRootsBuffer } },
          { binding: 27, resource: { buffer: this.#sceneBuffers.tlasInstanceWorldToLocalBuffer } },
          { binding: 28, resource: { buffer: this.#sceneBuffers.tlasInstanceLocalToWorldBuffer } },
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
    const accumTexture = this.#accumTexture;
    if (accumTexture == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }
    return {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'webgpu', GPUTexture>(accumTexture),
      ...(this.#normalDepthTexture != null
        ? { normalDepth: asBackendTexture<'webgpu', GPUTexture>(this.#normalDepthTexture) }
        : {}),
      ...(this.#albedoTexture != null
        ? { albedo: asBackendTexture<'webgpu', GPUTexture>(this.#albedoTexture) }
        : {}),
      ...(this.#varianceTexture != null
        ? { variance: asBackendTexture<'webgpu', GPUTexture>(this.#varianceTexture) }
        : {}),
      ...(this.#motionVectorsTexture != null
        ? { motionVectors: asBackendTexture<'webgpu', GPUTexture>(this.#motionVectorsTexture) }
        : {}),
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
  if (opts.device == null || typeof (opts.device).createCommandEncoder !== 'function') {
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
  const maxStorageBuffers = opts.device.limits?.maxStorageBuffersPerShaderStage;
  if (
    typeof maxStorageBuffers === 'number' &&
    Number.isFinite(maxStorageBuffers) &&
    maxStorageBuffers < REQUIRED_STORAGE_BUFFERS_PER_STAGE
  ) {
    throw new Error(
      `createPTEngine_WebGPU: adapter limit maxStorageBuffersPerShaderStage=${maxStorageBuffers} is below required ${REQUIRED_STORAGE_BUFFERS_PER_STAGE}`,
    );
  }
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(opts, slot);
  slot.set('ready');
  return engine;
};
