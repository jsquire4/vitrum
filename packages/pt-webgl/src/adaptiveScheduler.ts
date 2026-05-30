// Adaptive sample/tile scheduler + render-size planner for PTEngineWebGL2.
//
// Extracted verbatim from `ptEngineWebGL2.ts` (Task 4.4 Theme A god-class
// decomposition). The engine holds ONE `AdaptiveScheduler` instance and
// delegates the per-frame scheduling decision (`update`), the render-size
// plan (`planRenderSize`), and the WebGL-context-loss reaction
// (`noteContextLost`) to it. The class owns the mutable trio
// (`samplesPerFrame` / `tileSize` / `contextLost`) plus the immutable
// `SchedulerOptions` config and the `DeviceLimits` it needs to clamp render
// dimensions — nothing else.
//
// Behavior is IDENTICAL to the pre-extraction inline `#updateScheduler` /
// `#planRenderSize` / context-lost-handler bodies: the same control flow, the
// same constants, the same clamps. The class is pure state-machine — no THREE,
// no fork, no GPU handle.

import type { PTEngineWebGL2QualityMode } from './ptEngineWebGL2QualityMode.js';

export interface SchedulerOptions {
  readonly qualityMode: PTEngineWebGL2QualityMode;
  readonly adaptive: boolean;
  readonly targetBatchMs: number;
  readonly minSamplesPerFrame: number;
  readonly maxSamplesPerFrame: number;
  readonly initialSamplesPerFrame: number;
  readonly initialTileSize: number;
  readonly maxTileSize: number;
  readonly renderTargetBudgetBytes: number;
}

/** Device-capability limits the render-size planner clamps against. The
 *  engine detects these once at construction and hands them to the scheduler;
 *  it keeps its own copy for the BDPT software-GL checks. */
export interface SchedulerDeviceLimits {
  readonly maxTextureSize: number;
  readonly maxRenderbufferSize: number;
}

export interface RenderSizePlan {
  readonly width: number;
  readonly height: number;
  readonly estimatedBytes: number;
  readonly guardrail: string | null;
}

/** Default tile-grid dimension for capture/safe modes and the context-loss
 *  fallback. (Was `DEFAULT_TILE_SIZE` in ptEngineWebGL2.) */
export const DEFAULT_TILE_SIZE = 3;

/** RGBA16F texel size: 4 channels × 2 bytes per channel. */
const BYTES_PER_RGBA16F_PIXEL = 8;
/** Number of full-resolution render targets the WebGL path tracer allocates:
 *  primary accumulation, depth, normal, motion vector. */
const ESTIMATED_RENDER_TARGET_COUNT = 4;
/** Per-renderer overhead in driver metadata + mip alignment + GL state, used
 *  to budget memory when computing the host's adaptive render-size plan. */
const DEFAULT_RENDER_TARGET_OVERHEAD_BYTES = 64 * 1024 * 1024;

/**
 * Adaptive sample/tile scheduler state machine. One per engine instance.
 *
 * The mutable trio (`samplesPerFrame` / `tileSize` / `contextLost`) is the
 * scheduler's entire state. `update(batchMs)` is the per-frame closed-loop
 * controller that nudges `samplesPerFrame` / `tileSize` toward the target
 * batch time; `planRenderSize` is a pure function of `(width, height)` and the
 * immutable options + limits; `noteContextLost` is the WebGL-context-loss
 * reaction (the body the engine's `webglcontextlost` listener used to run
 * inline).
 */
export class AdaptiveScheduler {
  readonly #options: SchedulerOptions;
  readonly #limits: SchedulerDeviceLimits;

  #samplesPerFrame: number;
  #tileSize: number;
  #contextLost = false;

  constructor(options: SchedulerOptions, limits: SchedulerDeviceLimits) {
    this.#options = options;
    this.#limits = limits;
    this.#samplesPerFrame = options.initialSamplesPerFrame;
    this.#tileSize = options.initialTileSize;
  }

  /** Immutable scheduler config (qualityMode / budget / clamps). */
  get options(): SchedulerOptions {
    return this.#options;
  }

  /** Current per-frame sample-batch size the engine should request. */
  get samplesPerFrame(): number {
    return this.#samplesPerFrame;
  }

  /** Current tile-grid dimension (NxN). */
  get tileSize(): number {
    return this.#tileSize;
  }

  /** Whether a WebGL context-loss has been observed (drives the telemetry
   *  guardrail string + keeps the scheduler clamped to minimal work). */
  get contextLost(): boolean {
    return this.#contextLost;
  }

  /**
   * React to a `webglcontextlost` event: clamp to a single sample per frame
   * and bump the tile grid to at least {@link DEFAULT_TILE_SIZE}. Verbatim the
   * body the engine's inline `#contextLostHandler` ran.
   */
  noteContextLost(): void {
    this.#contextLost = true;
    this.#samplesPerFrame = 1;
    this.#tileSize = Math.max(this.#tileSize, DEFAULT_TILE_SIZE);
  }

  /**
   * Per-frame closed-loop controller. Given the wall-clock time the last sample
   * batch took (`batchMs`), nudge `samplesPerFrame` / `tileSize` toward the
   * configured `targetBatchMs`. No-op when adaptive scheduling is disabled or
   * `targetBatchMs <= 0`. Identical decisions to the pre-extraction
   * `#updateScheduler`.
   */
  update(batchMs: number): void {
    if (!this.#options.adaptive || this.#options.targetBatchMs <= 0) return;
    if (this.#contextLost) {
      this.#samplesPerFrame = 1;
      this.#tileSize = Math.min(this.#options.maxTileSize, Math.max(this.#tileSize, DEFAULT_TILE_SIZE));
      return;
    }
    const target = this.#options.targetBatchMs;
    if (batchMs > target * 1.35) {
      this.#samplesPerFrame = Math.max(
        this.#options.minSamplesPerFrame,
        Math.floor(this.#samplesPerFrame * 0.5),
      );
      if (batchMs > target * 2 && this.#tileSize < this.#options.maxTileSize) {
        this.#tileSize += 1;
      }
      return;
    }
    if (batchMs < target * 0.55 && this.#samplesPerFrame < this.#options.maxSamplesPerFrame) {
      this.#samplesPerFrame = Math.min(
        this.#options.maxSamplesPerFrame,
        Math.max(this.#samplesPerFrame + 1, Math.ceil(this.#samplesPerFrame * 1.2)),
      );
      if (batchMs < target * 0.25 && this.#tileSize > this.#options.initialTileSize) {
        this.#tileSize -= 1;
      }
    }
  }

  #estimateRenderTargetBytes(width: number, height: number): number {
    return (
      width *
      height *
      BYTES_PER_RGBA16F_PIXEL *
      ESTIMATED_RENDER_TARGET_COUNT +
      DEFAULT_RENDER_TARGET_OVERHEAD_BYTES
    );
  }

  /**
   * Plan the actual render-target dimensions for a requested `(width, height)`:
   * clamp to the device's max texture/renderbuffer dimension, then downscale to
   * fit the `renderTargetBudgetBytes` memory budget. Pure function of its args +
   * the immutable options/limits. Identical output to the pre-extraction
   * `#planRenderSize`.
   */
  planRenderSize(width: number, height: number): RenderSizePlan {
    const requestedWidth = Math.max(1, Math.floor(width));
    const requestedHeight = Math.max(1, Math.floor(height));
    const maxDimension = Math.max(1, Math.min(this.#limits.maxTextureSize, this.#limits.maxRenderbufferSize));
    let scale = Math.min(1, maxDimension / requestedWidth, maxDimension / requestedHeight);
    let guardrail: string | null = scale < 1
      ? `capped to WebGL max render dimension ${maxDimension}`
      : null;
    let plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
    let plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
    let estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
    if (estimatedBytes > this.#options.renderTargetBudgetBytes) {
      const targetBytes = Math.max(
        1,
        this.#options.renderTargetBudgetBytes - DEFAULT_RENDER_TARGET_OVERHEAD_BYTES,
      );
      const pixelBytes = Math.max(1, plannedWidth * plannedHeight * BYTES_PER_RGBA16F_PIXEL * ESTIMATED_RENDER_TARGET_COUNT);
      const memoryScale = Math.min(1, Math.sqrt(targetBytes / pixelBytes));
      scale *= memoryScale;
      plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
      plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
      estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
      guardrail = guardrail == null
        ? `downscaled to fit ${Math.round(this.#options.renderTargetBudgetBytes / 1024 / 1024)} MiB render-target budget`
        : `${guardrail}; downscaled to fit render-target budget`;
    }
    return {
      width: plannedWidth,
      height: plannedHeight,
      estimatedBytes,
      guardrail,
    };
  }
}
