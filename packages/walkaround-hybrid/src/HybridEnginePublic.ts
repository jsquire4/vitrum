import type {
  BackendTexture,
  Engine,
  EngineDebugSurface,
  FrameStats,
  SceneEmitter,
  ScenePrimitivePatch,
} from '@vitrum/core';
import type { FrameBudgetControllerConfig, FrameBudgetDecision } from './FrameBudgetController.js';
import type { GIStateSnapshot } from './giStateSnapshot.js';
import type { NrcDiagnostics } from './neural/nrc/nrcDiagnostics.js';

/** Runtime-switchable render layers implemented by walkaround-hybrid. */
export type HybridRenderLayer = 'ddgi';

/** Live interpretation of the walkaround renderer's bounce control. */
export interface HybridBounceSemanticsStatus {
  readonly kind: 'ddgi-feedback';
  readonly configuredMaxBounces: 1 | 2;
  readonly active:
    | 'disabled'
    | 'direct-only'
    | 'multi-bounce-equilibrium';
}

/**
 * Stable public surface for the walkaround-hybrid backend beyond the generic
 * {@link Engine} contract.
 *
 * Kept in a small module so package-root type imports do not pull the concrete
 * engine implementation into host-agnostic consumers.
 */
export interface HybridEngineGISurface {
  /** Export the converged DDGI GI state ("cached light field") for host persistence. */
  exportGIState(): Promise<GIStateSnapshot | null>;
  /** Restore a previously exported GI-state snapshot. */
  importGIState(snapshot: GIStateSnapshot): boolean;
}

/**
 * One renderer-produced neural-denoiser training sample.
 *
 * All three arrays are interleaved RGB with exactly `width * height * 3`
 * float32 values in row-major, top-left-origin order:
 *
 * - `radiance` is the live pre-denoise linear-HDR texture consumed by the
 *   neural runtime (`hdrColorTexture`).
 * - `albedo` is the live demodulated diffuse-albedo auxiliary.
 * - `worldNormal` is decoded from the live normal/depth G-buffer into signed
 *   world-space components in `[-1, 1]`.
 *
 * The arrays are CPU-owned snapshots. They remain valid after the next frame,
 * resize, scene replacement, or engine disposal.
 */
export interface HybridDenoiserTrainingCapture {
  readonly width: number;
  readonly height: number;
  readonly radiance: Float32Array;
  readonly albedo: Float32Array;
  readonly worldNormal: Float32Array;
}

/**
 * Public, structural HybridEngine type exposed from the package root.
 *
 * The concrete class implements the same callable engine surface; this
 * interface lets facade packages type against it without importing internals.
 */
export interface HybridEngine extends Engine, HybridEngineGISurface {
  readonly debug?: EngineDebugSurface;
  readonly debugTimings: ReadonlyArray<{ readonly t: number; readonly ms: number }>;
  readonly lastGpuTimings: Record<string, number>;
  readonly lastGpuTimingsFrame: number;

  readGpuTimingsOnce(): Promise<{
    readonly perPass: Record<string, number>;
    readonly rawBigints: readonly string[];
  }>;
  /** Runtime NRC counters, or `null` while disabled/not ready. */
  getNrcDiagnostics(): NrcDiagnostics | null;

  setDdgiUpdateDivisor(divisor: number): void;
  enableFrameBudget(config?: Partial<FrameBudgetControllerConfig>): void;
  disableFrameBudget(): void;
  readonly frameBudgetEnabled: boolean;
  tickFrameBudget(measuredMs: number): FrameBudgetDecision | null;
  /** Update PPG training cadence; requires construction with `ppgEnabled: true`. */
  setPpgDispatchInterval(interval: number): void;
  /**
   * Retry a sealed PPG training epoch after bounded GPU-readback retries enter
   * the durable failed state.
   *
   * Returns `false` when PPG is disabled, no failed epoch is awaiting recovery,
   * the pipeline has not been published yet, or the engine was disposed.
   */
  requestPpgTrainingRecovery(): boolean;
  /** Poll the PPG epoch state; `'failed'` means recovery can be requested. */
  getPpgTrainingStatus():
    | 'unavailable'
    | 'disabled'
    | 'collecting'
    | 'readback'
    | 'retry-pending'
    | 'failed'
    | 'disposed';

  /**
   * Report the live DDGI feedback regime. Unlike the construction-time
   * capability detail, this reflects `setLayerEnabled('ddgi', false)`.
   */
  getBounceSemantics(): HybridBounceSemanticsStatus;

  updatePrimitive(id: string, patch: ScenePrimitivePatch): void;
  updateEmitter(id: string, patch: Partial<SceneEmitter>): void;
  applyGpuSkinnedRefit(
    id: string,
    localPositions: Float32Array,
    localNormals?: Float32Array,
  ): void;

  getGpuSkinningBvhBuffer(): GPUBuffer | null;
  getGpuSkinningBvhBinding(): GPUBufferBinding | null;
  getGpuSkinningNormalBuffer(): GPUBuffer | null;
  getGpuSkinningNormalBinding(): GPUBufferBinding | null;
  getMeshVertexRanges(): readonly unknown[] | null;
  getBvhMode(): string | null;
  getPrimitiveTlasBindings(): readonly unknown[] | null;

  getProgressiveSeedTexture(): {
    readonly texture: BackendTexture;
    readonly width: number;
    readonly height: number;
  } | null;

  /**
   * Read the exact renderer textures consumed by the neural denoiser.
   *
   * This is an explicit offline-capture/debug stall: the implementation copies
   * all three rgba16float textures into independent 256-byte-row-aligned staging
   * buffers in one queue submission, waits for every map operation, copies the
   * decoded RGB values into CPU-owned arrays, and destroys every staging buffer
   * before resolving. Returns `null` before the first renderable pipeline
   * generation is published.
   */
  captureDenoiserTrainingInputs(): Promise<HybridDenoiserTrainingCapture | null>;

  setLayerEnabled(layer: HybridRenderLayer, enabled: boolean): void;
  onFrame(cb: (stats: FrameStats) => void): () => void;
}
