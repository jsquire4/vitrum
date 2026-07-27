/**
 * Neutral bind-group layout type definitions shared across the pipeline layer
 * and subsystem modules.
 *
 * Mirrors the {@link ./wgslTypes.ts} precedent: types that multiple modules
 * need (both `pipeline/bindGroupLayouts.ts` and `neural/nrc/nrcBindGroupLayout.ts`)
 * live here so neither layer imports from the other.
 *
 * `pipeline/bindGroupLayouts.ts` re-exports these for back-compatibility.
 * Shape-D smell resolved (I5.2, 2026-06-11 E sweep).
 */

/**
 * Universe of all bind-group layout cache keys (D3.16). Adding a new pass BGL
 * = one string here + one `getBGL*` getter in bindGroupLayouts.ts. No other
 * edits required.
 */
export type BGLKey =
  | 'frame' | 'scene' | 'ubo' | 'atrous' | 'composite' | 'accum'
  | 'hybridLayers' | 'hybridLayersNrc' | 'sampleBudget' | 'resolve' | 'motionVectors'
  | 'shadeHybridLayers' | 'risGiFrame'
  | 'gtao' | 'gtaoUpsample' | 'temporalGi' | 'spatialGi'
  | 'indirectCombine' | 'indirectTemporalAccum' | 'transparentOit'
  | 'lightTree' | 'regirBuild' | 'nrc' | 'cbPrefill';

/**
 * Memoization cache for all bind-group layouts. `Partial<Record<…>>` is
 * structurally identical to the prior 20-field optional interface (same keys,
 * same `GPUBindGroupLayout | undefined` values) — no callers need to change.
 */
export type BGLCache = Partial<Record<BGLKey, GPUBindGroupLayout>>;
