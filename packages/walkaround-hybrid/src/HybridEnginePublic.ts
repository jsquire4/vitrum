import type {
  BackendTexture,
  Engine,
  EngineDebugSurface,
  FrameStats,
  SceneEmitter,
  ScenePrimitive,
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

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void;
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

  setLayerEnabled(layer: HybridRenderLayer, enabled: boolean): void;
  onFrame(cb: (stats: FrameStats) => void): () => void;
}
