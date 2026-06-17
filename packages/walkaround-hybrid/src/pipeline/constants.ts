/**
 * Shared numeric constants for the walkaround pipeline.
 *
 * Zero-dependency module — safe to import from anywhere in the package without
 * risk of circular imports.
 */

/**
 * Size of the WalkaroundUBO in bytes (432 = 27 × 16-byte aligned vec4 slots).
 *
 * Layout details: see `uboUpdater.ts` header comment (the full field-by-field
 * breakdown lives there alongside `updateUBO`).
 *
 * The value must stay in sync with:
 *   - `createBuffer({ size: … })` in `createCommonFrameResources.ts`
 *   - `WALKAROUND_UBO_SIZE_BYTES` in `uboUpdater.ts`
 *   - The WGSL `WalkaroundUBO` struct in `shaders/walkaroundUbo.wgsl.ts`
 */
export const WALKAROUND_UBO_SIZE_BYTES = 432;

/** Legacy real-sun angular radius used when a scene directional emitter does
 *  not author `angularDiameter`. 0.5° angular diameter => 0.25° radius. */
export const WALKAROUND_DEFAULT_SUN_ANGULAR_RADIUS = 0.00436;

// ─────────────────────────────────────────────────────────────────────────────
// À-trous denoiser sigma bundles (moved from bindGroupBuilders.ts, I3.1)
// ─────────────────────────────────────────────────────────────────────────────

/** Per-axis sigma settings for the à-trous spatial denoiser pass. */
export interface AtrousSigmas {
  sigmaN: number;
  sigmaZ: number;
  sigmaC: number;
}

/** Direct-channel defaults — tight stops, preserves shadow / caustic edges.
 *  Per-frame overrides flow from `HybridEngineOptions.atrousDirectSigmas`
 *  through `PipelineFrameInputs` (B3a, 2026-05-19). */
export const ATROUS_DIRECT_SIGMAS: Readonly<AtrousSigmas> = Object.freeze({
  sigmaN: 128.0,
  sigmaZ: 5.0,
  sigmaC: 0.05,
});

/**
 * Indirect-channel sigmas. Broader on every axis because ReSTIR-GI already
 * smooths the indirect signal temporally + spatially; the remaining 2×2 quad
 * variance (from half-res GI reservoir reads) just needs a wide low-pass.
 *   σn=32  → still rejects perpendicular surfaces but blurs through mild curvature.
 *   σz=20  → ~4× the direct depth tolerance — fine for indirect, no hard-shadow edges.
 *   σc=0.5 → ~10× direct's color tolerance — allows blur across color-bleed transitions.
 *
 * Per-frame overrides flow from `HybridEngineOptions.atrousIndirectSigmas`
 * through `PipelineFrameInputs` (B3a, 2026-05-19).
 */
export const ATROUS_INDIRECT_SIGMAS: Readonly<AtrousSigmas> = Object.freeze({
  sigmaN: 32.0,
  sigmaZ: 20.0,
  sigmaC: 0.5,
});
