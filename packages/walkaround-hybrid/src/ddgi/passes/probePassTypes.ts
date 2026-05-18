/**
 * Shared types for the DDGI probe-update Pass family.
 *
 * The DDGI subsystem mirrors the W1-R5 Pass interface shape but is NOT
 * registered into the main pipeline's {@link PassRegistry}. DDGI runs at a
 * different cadence (`DDGI.updateFrame()` — round-robin 1/STRIDE probes per
 * frame, 60 FPS capped) than the per-frame ReSTIR pipeline, so it keeps its
 * own self-contained dispatch chain.
 *
 * Each per-phase Pass:
 *   - Owns its own compiled compute pipeline (created in `compile()` so
 *     `ProbeUpdatePass.init` can await all five in parallel).
 *   - Constructs its bind group(s) per dispatch from the shared resources
 *     supplied via {@link ProbePassContext}.
 *   - Has no per-frame mutable state of its own.
 *
 * The shared GPU state (BVH buffers, UBOs, scratch textures, ray-results
 * SSBO, atlas ping-pong) lives on the orchestrator ({@link ProbeUpdatePass})
 * and is threaded through {@link ProbePassContext}. This mirrors the
 * orchestrator-owned-resources pattern used by `WalkaroundGPUPipeline` for
 * the per-frame pipeline.
 */

/**
 * Per-dispatch context handed to each probe-update Pass.
 *
 * `irrReadTex` / `irrWriteTex` / `visReadTex` / `visWriteTex` are the
 * irradiance + visibility atlas ping-pong textures resolved for THIS frame.
 * The border passes additionally take `irrScratchTex` / `visScratchTex`
 * (atlas-sized copies allocated lazily by the orchestrator).
 */
export interface ProbePassContext {
  readonly device: GPUDevice;
  /** Number of probes active this round-robin slice (RaysPass + BlendPasses). */
  readonly activeCount: number;
  /** Total probe count in the grid (BorderPasses dispatch one workgroup per probe). */
  readonly probeCount: number;
  // ── Atlas ping-pong (re-resolved each frame by the orchestrator) ─────
  readonly irrReadTex:  GPUTexture;
  readonly irrWriteTex: GPUTexture;
  readonly visReadTex:  GPUTexture;
  readonly visWriteTex: GPUTexture;
  // ── Border scratch (border passes only — null for rays/blend dispatches) ─
  readonly irrScratchTex: GPUTexture | null;
  readonly visScratchTex: GPUTexture | null;
  // ── Shared GPU buffers owned by the orchestrator ─────────────────────
  readonly bvhBuf:        GPUBuffer;
  readonly posBuf:        GPUBuffer;
  readonly idxBuf:        GPUBuffer;
  readonly normBuf:       GPUBuffer;
  readonly matIdBuf:      GPUBuffer;
  readonly materialsBuf:  GPUBuffer;
  readonly lightsBuf:     GPUBuffer;
  readonly gridParamsBuf: GPUBuffer;
  readonly frameParamsBuf:GPUBuffer;
  readonly blendParamsBuf:GPUBuffer;
  readonly borderIrrUboBuf: GPUBuffer;
  readonly borderVisUboBuf: GPUBuffer;
  readonly rayResultsBuf:   GPUBuffer;
  readonly activeProbesBuf: GPUBuffer;
  readonly linearSampler:   GPUSampler;
}

/**
 * Common shape mirrored by every per-phase probe-update Pass.
 *
 * `dispatch` is encoder-only and reads its dependencies from `ctx` —
 * matching the same orchestrator-supplies-context pattern the W1-R5 Pass
 * interface uses for the main pipeline (see `pipeline/Pass.ts`).
 */
export interface ProbePass {
  /** Stable identifier (e.g. `ddgi-probe-rays`). */
  readonly id: string;
  /** Encode the per-frame compute dispatch into `encoder`. */
  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void;
  /** Release any pass-owned GPU resources (e.g. the compiled pipeline). */
  dispose(): void;
}
