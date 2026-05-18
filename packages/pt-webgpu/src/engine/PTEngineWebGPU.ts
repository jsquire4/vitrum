import type {
  Engine,
  EngineCapabilities,
  EngineOptions,
  EngineState,
  FrameInput,
  FrameOutput,
  Scene,
  SceneEmitter,
  ScenePrimitive,
} from '@vitrum/core';
import { patchEmitterInScene, patchPrimitiveInScene } from '../scene/patchScene.js';
import { summarizeScene } from '../scene/flattenScene.js';
import {
  buildPackedScene,
  uploadPackedScene,
  PT_WEBGPU_ANALYTIC_SHAPES,
  type UploadedSceneBuffers,
} from '../scene/uploadSceneBuffers.js';
import { PTEngineWebGPUResources } from './PTEngineWebGPUResources.js';
import { PTEngineWebGPUPipeline } from './PTEngineWebGPUPipeline.js';
import { PTEngineWebGPUParamsPacker } from './PTEngineWebGPUParamsPacker.js';

export interface PTEngineWebGPUOptions extends EngineOptions {
  readonly device: GPUDevice;
}

export const PROTOTYPE_MAX_BOUNCES = 8;
export const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;

export interface StateSlot {
  readonly get: () => EngineState;
  readonly set: (s: EngineState) => void;
}

export function makeStateSlot(initial: EngineState = 'initializing'): StateSlot {
  let s: EngineState = initial;
  return {
    get: () => s,
    set: (v) => {
      s = v;
    },
  };
}

/**
 * Slim orchestrator class that wires together the four W4-A9 modules:
 *
 *   - `PTEngineWebGPUResources`   — accum + aux textures + storage buffers
 *   - `PTEngineWebGPUPipeline`    — compute pipeline, params UBO, bind group
 *   - `PTEngineWebGPUParamsPacker` — 512-byte FrameParams packer
 *   - `UploadedSceneBuffers`      — the per-scene buffer set (lives in scene/)
 *
 * The class itself owns lifecycle state (slot, scene, samplesAccumulated,
 * activeBounces), config (caustic strategy, mnee knobs, sample/bounce caps),
 * and the cached bind group. Everything else delegates.
 */
export class PTEngineWebGPU implements Engine {
  readonly #slot: StateSlot;
  readonly #device: GPUDevice;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;

  readonly #resources: PTEngineWebGPUResources;
  readonly #pipeline: PTEngineWebGPUPipeline;
  readonly #paramsPacker: PTEngineWebGPUParamsPacker;

  #scene: Scene | null = null;
  #sceneBuffers: UploadedSceneBuffers | null = null;
  #samplesAccumulated = 0;
  #activeBounces = 1;

  /** Reused bind group until scene buffers or accum views are recreated. */
  #pathTraceBindGroup: GPUBindGroup | null = null;

  constructor(opts: PTEngineWebGPUOptions, slot: StateSlot) {
    this.#slot = slot;
    this.#device = opts.device;
    this.#maxBouncesLimit = Math.max(1, Math.min(opts.maxBounces ?? 3, PROTOTYPE_MAX_BOUNCES));
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    const mneeIter =
      typeof causticOpts.mneeMaxIterations === 'number' ? causticOpts.mneeMaxIterations : 8;
    const mneeChain =
      typeof causticOpts.mneeMaxChainLength === 'number' ? causticOpts.mneeMaxChainLength : 3;
    this.#mneeMaxIterations = Math.max(1, mneeIter);
    this.#mneeMaxChainLength = Math.max(1, mneeChain);

    this.#resources = new PTEngineWebGPUResources(this.#device);
    this.#pipeline = new PTEngineWebGPUPipeline(this.#device);
    this.#paramsPacker = new PTEngineWebGPUParamsPacker();
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  get capabilities(): EngineCapabilities {
    return {
      // Honest reporting — updatePrimitive/updateEmitter currently delegate
      // to setScene; flip to true when real incremental patching lands.
      supportsIncrementalScene: false,
      supportsMotionBlur: false,
      supportsAuxBuffers: true,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      // Slot 0 is the "unknown" sentinel; supported shapes start at index 1.
      supportedAnalyticShapes: new Set<string>(PT_WEBGPU_ANALYTIC_SHAPES.slice(1)),
      supportedEmitterKinds: new Set<string>([
        'directional',
        'point',
        'spot',
        'rect-area',
        'disc-area',
        'mesh-area',
      ]),
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
        primaryRadiance: this.#resources.accumTexture,
        normalDepth: this.#resources.normalDepthTexture ?? undefined,
        albedo: this.#resources.albedoTexture ?? undefined,
        variance: this.#resources.varianceTexture ?? undefined,
        motionVectors: this.#resources.motionVectorsTexture ?? undefined,
        samplesAccumulated: this.#samplesAccumulated,
        isConverged: this.#samplesAccumulated >= targetSppPaused,
      };
    }

    const q = input.quality ?? {};
    this.#activeBounces = Math.max(
      1,
      Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit),
    );
    const resolution = q.resolutionFactor ?? 1;
    const width = Math.max(1, Math.floor(input.viewport.width * resolution));
    const height = Math.max(1, Math.floor(input.viewport.height * resolution));

    this.#resources.ensure(width, height);
    if (this.#resources.resizedSinceLastEnsure()) {
      this.#samplesAccumulated = 0;
      this.#pathTraceBindGroup = null;
    }
    this.#pipeline.ensure();
    if (this.#sceneBuffers == null) {
      throw new Error('renderFrame: failed to initialize WebGPU pipeline resources');
    }
    const accum = this.#resources.snapshot();
    const pipeline = this.#pipeline.snapshot();

    const paramsArrayBuffer = this.#paramsPacker.pack({
      input,
      width,
      height,
      activeBounces: this.#activeBounces,
      sceneBuffers: this.#sceneBuffers,
      mneeMaxIterations: this.#mneeMaxIterations,
      mneeMaxChainLength: this.#mneeMaxChainLength,
      causticStrategy: this.#causticStrategy,
    });
    this.#device.queue.writeBuffer(pipeline.paramsBuffer, 0, paramsArrayBuffer);

    let bindGroup = this.#pathTraceBindGroup;
    if (bindGroup == null) {
      bindGroup = this.#pipeline.buildPathTraceBindGroup(pipeline, accum, this.#sceneBuffers);
      this.#pathTraceBindGroup = bindGroup;
    }

    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.pathTrace.encoder',
    });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.pathTrace.pass' });
    pass.setPipeline(pipeline.computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP_SIZE), Math.ceil(height / WORKGROUP_SIZE), 1);
    pass.end();
    this.#device.queue.submit([encoder.finish()]);

    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    return {
      primaryRadiance: this.#resources.accumTexture,
      normalDepth: this.#resources.normalDepthTexture ?? undefined,
      albedo: this.#resources.albedoTexture ?? undefined,
      variance: this.#resources.varianceTexture ?? undefined,
      motionVectors: this.#resources.motionVectorsTexture ?? undefined,
      samplesAccumulated: this.#samplesAccumulated,
      isConverged: this.#samplesAccumulated >= targetSpp,
    };
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#samplesAccumulated = 0;
    this.#resources.clear();
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
    this.#resources.destroy();
    this.#pathTraceBindGroup = null;
    this.#pipeline.destroy();
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = null;
    this.#scene = null;
    this.#slot.set('disposed');
  }
}
