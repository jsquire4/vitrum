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
 * Concrete implementations are added in W1-R3 (the next round of the
 * W1 workstream); this module ships the interface + registry only.
 *
 *   denoisers/none.ts            (W1-R3)
 *   denoisers/atrous.ts          (W1-R3)
 *   denoisers/atrousVariance.ts  (W1-R3)
 *   denoisers/svgfReal.ts        (W1-R3)
 *   denoisers/neural.ts          (W1-R3, gated `disabled` until W10)
 *   denoisers/oidnFinal.ts       (W1-R3, gated `disabled` until W11)
 */

/** Identifier union for built-in denoisers. The premium-grade plan
 *  reserves the slots `'neural'` (W10) and `'oidn-final'` (W11) which
 *  ship as registered-but-disabled placeholders until those workstreams
 *  land the real implementations. */
export type DenoiserId =
  | 'none'
  | 'atrous'
  | 'atrous-variance'
  | 'svgf-real'
  | 'neural'
  | 'oidn-final';

export interface DenoiserInitContext {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
}

export interface DenoiserDispatchContext {
  readonly device: GPUDevice;
  readonly encoder: GPUCommandEncoder;
  readonly width: number;
  readonly height: number;
  readonly frameIndex: number;
  /** Loose typing so this module does not depend on FrameResources. */
  readonly resources: unknown;
}

export interface Denoiser {
  readonly id: DenoiserId;

  /** Compile pipelines, create BGLs, allocate persistent textures/buffers.
   *  Awaited once at engine boot. */
  initialize(ctx: DenoiserInitContext): Promise<void>;

  /** Per-frame dispatch. Returns the resolved-radiance texture downstream
   *  composition should sample, or `null` if this denoiser is a no-op
   *  pass-through (in which case the engine sources radiance from the
   *  raw HDR texture directly). */
  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null;

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
        `DenoiserRegistry: denoiser "${id}" is registered but disabled; ` +
          `it will be enabled by a future refactor workstream`,
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
