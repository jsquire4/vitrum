/**
 * Denoiser registry — single source of truth for the {@link Denoiser}
 * implementations selectable via `EngineOptions.denoiser`.
 *
 * Before this abstraction, the denoiser was a 4-string union switched
 * across 5 files (HybridEngine, WalkaroundGPUPipeline, pipelineCompiler,
 * timestampQueries, resourceManager) with ~16 string-compare sites
 * (complexity sweep 2026-05-17 Theme B / Integration I3). With this
 * registry, adding a denoiser is a single new file + one registry
 * entry; the orchestrator looks the denoiser up by ID and dispatches
 * polymorphically.
 *
 *   denoisers/none.ts            (W1-R3 — pass-through no-op)
 *   denoisers/atrous.ts          (W1-R3 — legacy 3-iter à-trous)
 *   denoisers/atrousVariance.ts  (W1-R3 — Welford + variance + 3 × atrous, default)
 *   denoisers/svgfReal.ts        (W1-R3 — real Schied 2017 SVGF)
 *   denoisers/bmfr.ts            (Koskela 2019 — blockwise multi-order feature regression)
 *   denoisers/neural.ts          (W1-R3, enabled when InferenceGraph supplied; disabled placeholder otherwise)
 *   denoisers/oidnFinal.ts       (W1-R3, enabled when modelUrl supplied; disabled placeholder otherwise)
 *
 * Concrete entries register themselves via {@link registerBuiltinDenoisers}
 * (see `denoisers/registerBuiltinDenoisers.ts`).
 */

import type { BGLCache } from '../bindGroupLayouts.js';
import type { FrameResources } from '../resourceManager.js';
import type { PassLabel } from '../timestampQueries.js';

/** Identifier union for built-in denoisers. Both `'neural'` and `'oidn-final'`
 *  have real implementations; they register as disabled only when no
 *  InferenceGraph / modelUrl is supplied. */
export type DenoiserId =
  | 'none'
  | 'atrous'
  | 'atrous-variance'
  | 'svgf-real'
  | 'bmfr'
  | 'neural'
  | 'oidn-final';

/**
 * Static per-denoiser timestamp-query label table.
 *
 * Each concrete {@link Denoiser} class references its entry here for its
 * `passLabels` field, and `buildPassLayout` reads from the same map so
 * the GPU timing slots stay in lockstep with the actual dispatch order.
 *
 * 'neural' declares exactly the 2 passes it dispatches: `neural-pack`
 * (input texture → tensor buffers) + `neural-unpack` (denoised tensor →
 * output texture). The InferenceGraph itself is self-managing and does not
 * participate in the timestamp-query pass layout. When disabled (no
 * InferenceGraph supplied) the registry rejects it at `lookup`, so the label
 * list is consulted only by `buildPassLayout` for querySet sizing.
 *
 * 'oidn-final' is `[]` because OIDN dispatch does not use timestamp-query
 * pass labels.
 */
export const DENOISER_PASS_LABELS: Readonly<Record<DenoiserId, readonly PassLabel[]>> = Object.freeze({
  'none': Object.freeze([]),
  'atrous': Object.freeze(['atrous-0', 'atrous-1', 'atrous-2']),
  'atrous-variance': Object.freeze([
    'welford-temporal',
    'atrous-variance-variance',
    'atrous-variance-atrous-0',
    'atrous-variance-atrous-1',
    'atrous-variance-atrous-2',
  ]),
  'svgf-real': Object.freeze([
    'svgf-real-reproj',
    'svgf-real-moments',
    'svgf-real-7x7',
    'svgf-real-atrous-0',
    'svgf-real-atrous-1',
    'svgf-real-atrous-2',
    'svgf-real-atrous-3',
    'svgf-real-atrous-4',
  ]),
  'bmfr': Object.freeze(['bmfr']),
  'neural': Object.freeze(['neural-pack', 'neural-unpack']),
  'oidn-final': Object.freeze([]),
} as Record<DenoiserId, readonly PassLabel[]>);

/**
 * Initialization context handed to {@link Denoiser.initialize}.
 *
 * The denoiser may read `bglCache` to look up shared (universal) BGLs
 * — frame/scene/ubo/composite — but must not register denoiser-specific
 * BGLs there: a denoiser-private BGL is private to the denoiser entry.
 *
 * `frameResources` lets the denoiser pre-bind persistent resources it
 * does not own (e.g. SVGF's persistent history textures live in
 * {@link FrameResources.svgf}).
 */
export interface DenoiserInitContext {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  readonly bglCache: BGLCache;
  readonly frameResources: FrameResources;
}

/**
 * Per-frame dispatch context. Concrete denoisers narrow `resources` to
 * the {@link FrameResources} sub-struct they need; the interface keeps
 * the field typed loosely so this module does not have to know what
 * each denoiser reads.
 *
 * `sharedAtrousPipeline` is the compiled à-trous compute pipeline used
 * by the legacy 3-iter denoiser AND by the always-on indirect-channel
 * chain that runs outside any denoiser (see
 * `WalkaroundGPUPipeline._dispatchAtrousIndirect`). It stays in shared
 * compilation so both consumers can reuse the same module + BGL; the
 * legacy `AtrousDenoiser.dispatch` reads it from this context.
 *
 * `computeDesc` is a callback that builds the optional `timestampWrites`
 * field for a {@link GPUComputePassDescriptor}. Sharing the closure
 * means denoisers report their timings under the same per-frame
 * {@link PassLayout} as the rest of the pipeline.
 */
export interface DenoiserDispatchContext {
  readonly device: GPUDevice;
  readonly encoder: GPUCommandEncoder;
  readonly width: number;
  readonly height: number;
  readonly frameIndex: number;
  /** Concrete denoisers cast this to {@link FrameResources}. */
  readonly resources: FrameResources;
  /** Shared atrous pipeline; see field doc above. */
  readonly sharedAtrousPipeline: GPUComputePipeline;
  /** Cache of universal BGLs (frame/scene/ubo/composite/accum/atrous). */
  readonly bglCache: BGLCache;
  /** Pre-resolved G-buffer normal+depth view used as edge-stop input. */
  readonly gNormalDepthView: GPUTextureView;
  /** B3a (2026-05-19) — atrous DIRECT-channel sigmas the active denoiser
   *  may use for its edge-stop filter. Cornell default `[128, 5, 0.05]`. */
  readonly atrousDirectSigmas: readonly [number, number, number];
  /** Current temporal-accumulator read texture (the AtrousVariance
   *  denoiser needs this for the welford alpha-reset path). */
  readonly readAccum: GPUTexture;
  /** Whether the camera moved this frame (drives accumulator reset). */
  readonly isMoving: boolean;
  /** Pre-computed 16×16-workgroup dispatch counts. */
  readonly wgX16: number;
  readonly wgY16: number;
  /** Build a `GPUComputePassDescriptor` with optional timestampWrites. */
  readonly computeDesc: (label: PassLabel) => GPUComputePassDescriptor;
}

export interface Denoiser {
  readonly id: DenoiserId;

  /** Timestamp-query labels this denoiser emits in dispatch order. Consumed
   *  by `buildPassLayout` so the GPU querySet is sized to the active
   *  denoiser. Pass-through denoisers return `[]`. The order MUST match
   *  the order the labels appear inside {@link Denoiser.dispatch}. */
  readonly passLabels: readonly import('../timestampQueries.js').PassLabel[];

  /** Compile pipelines, create BGLs, allocate persistent textures/buffers.
   *  Awaited once at engine boot. */
  initialize(ctx: DenoiserInitContext): Promise<void>;

  /** Per-frame dispatch. Returns the resolved-radiance texture downstream
   *  composition should sample, or `null` if this denoiser is a no-op
   *  pass-through (in which case the engine sources radiance from the
   *  raw HDR texture directly). */
  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null;

  /** Optional cleanup hook called by the pipeline AFTER its
   *  `device.queue.submit()` has been called for the frame the denoiser
   *  just dispatched. Allows a denoiser to release per-frame transient
   *  GPU buffers (e.g. the per-à-trous-iter UBOs SVGF-real allocates)
   *  once the GPU queue has taken ownership of the encoded command
   *  buffer. Default implementations are no-ops. */
  cleanupAfterSubmit?(): void;

  /** Resize callback — denoiser may reallocate persistent resources. */
  resize(width: number, height: number): void;

  /** Release all GPU resources owned by this denoiser. */
  dispose(): void;

  /** When `true`, the denoiser is registered as a placeholder for an
   *  in-progress implementation; {@link DenoiserRegistry.lookup}
   *  rejects disabled IDs at validation time. Reserved for W10
   *  (neural) and W11 (oidn-final) before those workstreams ship. */
  readonly disabled?: boolean;
}

export class DenoiserRegistry {
  private readonly _denoisers = new Map<DenoiserId, Denoiser>();

  register(denoiser: Denoiser): void {
    if (this._denoisers.has(denoiser.id)) {
      throw new Error(`DenoiserRegistry: duplicate id "${denoiser.id}"`);
    }
    this._denoisers.set(denoiser.id, denoiser);
  }

  /** Resolve a denoiser by ID. Throws on unknown ID or on
   *  {@link Denoiser.disabled} placeholders. */
  lookup(id: DenoiserId): Denoiser {
    const d = this._denoisers.get(id);
    if (d === undefined) {
      const known = [...this._denoisers.keys()].join(', ');
      throw new Error(
        `DenoiserRegistry: unknown denoiser "${id}"; known: [${known}]`,
      );
    }
    if (d.disabled === true) {
      throw new Error(
        `DenoiserRegistry: denoiser "${id}" is registered but disabled ` +
          `(its required host config was not supplied — an InferenceGraph for ` +
          `'neural', or an OIDN modelUrl for 'oidn-final')`,
      );
    }
    return d;
  }

  /** All registered IDs, including disabled. For diagnostics + capability
   *  enumeration. */
  ids(): readonly DenoiserId[] {
    return Array.from(this._denoisers.keys());
  }

  has(id: DenoiserId): boolean {
    return this._denoisers.has(id);
  }

  size(): number {
    return this._denoisers.size;
  }
}
