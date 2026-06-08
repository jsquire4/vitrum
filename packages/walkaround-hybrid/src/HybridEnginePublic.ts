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

/**
 * Stable public surface for the walkaround-hybrid backend beyond the generic
 * {@link Engine} contract.
 *
 * Kept in a Three-free module so package-root type imports do not pull the
 * concrete legacy host adapter into host-agnostic consumers.
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
 * The concrete class still lives behind `@vitrum/walkaround-hybrid/three`,
 * where importing it is allowed to resolve Three.js. Root consumers get the
 * same callable engine surface without a root-level Three dependency.
 */
export interface HybridEngine extends Engine, HybridEngineGISurface {
  readonly debug: EngineDebugSurface;
  readonly debugTimings: ReadonlyArray<{ readonly t: number; readonly ms: number }>;
  readonly lastGpuTimings: Record<string, number>;
  readonly lastGpuTimingsFrame: number;

  readGpuTimingsOnce(): Promise<{
    readonly perPass: Record<string, number>;
    readonly rawBigints: readonly string[];
  }>;

  setDdgiUpdateDivisor(divisor: number): void;
  enableFrameBudget(config?: Partial<FrameBudgetControllerConfig>): void;
  disableFrameBudget(): void;
  readonly frameBudgetEnabled: boolean;
  tickFrameBudget(measuredMs: number): FrameBudgetDecision | null;
  setPpgDispatchInterval(interval: number): void;

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void;
  updateEmitter(id: string, patch: Partial<SceneEmitter>): void;
  applyGpuSkinnedRefit(
    id: string,
    localPositions: Float32Array,
    localNormals?: Float32Array,
  ): void;

  getGpuSkinningBvhBuffer(): GPUBuffer | null;
  getGpuSkinningNormalBuffer(): GPUBuffer | null;
  getMeshVertexRanges(): readonly unknown[] | null;
  getBvhMode(): string | null;
  getPrimitiveTlasBindings(): readonly unknown[] | null;
  refreshDdgiLightsFromThreeScene(): void;

  getProgressiveSeedTexture(): {
    readonly texture: BackendTexture;
    readonly width: number;
    readonly height: number;
  } | null;

  setLayerEnabled(layer: string, enabled: boolean): void;
  onFrame(cb: (stats: FrameStats) => void): () => void;
}
