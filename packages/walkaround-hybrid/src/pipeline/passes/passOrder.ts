/**
 * Static pass-order metadata — the single source of truth shared between
 * the {@link Pass} implementations under this directory and the
 * timestamp-query layout in `../timestampQueries.ts`.
 *
 * Each entry declares the {@link PassLabel}s a pass class emits in dispatch
 * order. Concrete Pass classes import their entry's `labels` field for
 * their `passLabels` declaration; `buildPassLayout` flattens the
 * non-denoiser entries (in topological order, gated by `opts`) and
 * splices in the active denoiser's `passLabels` at the documented
 * insertion point.
 *
 * Adding a pass:
 *   1. Add an entry to {@link NON_DENOISER_PASS_ORDER} in the topological
 *      position it occupies in the frame.
 *   2. Reference the entry's `labels` from the new Pass class.
 *   3. No edits to `buildPassLayout` required.
 *
 * Why this lives separate from `passes/index.ts`: the Pass classes
 * themselves require compiled GPU pipelines to construct, so they cannot
 * be instantiated from a unit test that lacks a GPU. The label metadata
 * is plain data and can be imported by tests + the orchestrator alike
 * without a GPU dependency.
 */

import type { PassLabel } from '../timestampQueries.js';

/** Discriminator for the denoiser-injection point in the static order. */
interface NonDenoiserPassEntry {
  /** Stable id of the corresponding Pass class. */
  readonly id: string;
  /** Labels emitted by the pass in dispatch order. */
  readonly labels: readonly PassLabel[];
}

/**
 * Non-denoiser passes that participate in the GPU-timing layout, in
 * dispatch order. The active denoiser's labels are spliced in at the
 * `denoiser-adapter` entry (see {@link composePassLabels}).
 *
 * The `denoiser-adapter` entry is the timestamp-layout reflection of the
 * virtual {@link DenoiserAdapterPass}: its own `labels` are `[]` because
 * the labels actually emitted at this slot depend on which Denoiser is
 * active. Keeping the entry in the static order — rather than splicing
 * on `gtao-upsample + 1` — means `composePassLabels` has a single source
 * of truth (this table) instead of an implicit constant pointing into it.
 *
 * PPG update is included so timestamp-query layouts stay aligned when
 * `ppgEnabled` is true (see `PPGUpdatePass`).
 */
// File-local. The matching JSDoc + comment references in
// timestampQueries.ts (lines 127 + 137) cite the name by spelling, not by
// import. 2026-05-18 dead-code sweep verified zero non-self consumers.
const NON_DENOISER_PASS_ORDER: readonly NonDenoiserPassEntry[] = Object.freeze([
  { id: 'sample-budget', labels: ['sample-budget'] },
  { id: 'ris', labels: ['ris'] },
  { id: 'temporal', labels: ['temporal'] },
  // spatial-2 / gi-spatial-2 label sets are Phase-0 config-driven (1 vs 2
  // ping-pong passes). The static table carries the FULL (2-pass) labels;
  // `composePassLabels` slices them to the active pass count so the timestamp
  // layout matches the dispatched labels exactly (Risk R2).
  { id: 'spatial-2', labels: ['spatial-1', 'spatial-2'] },
  { id: 'gi-ris', labels: ['gi-ris'] },
  { id: 'gi-temporal', labels: ['gi-temporal'] },
  { id: 'gi-spatial-2', labels: ['gi-spatial-1', 'gi-spatial-2'] },
  { id: 'shade', labels: ['shade'] },
  { id: 'motion-vectors', labels: ['motion-vectors'] },
  { id: 'gtao', labels: ['gtao'] },
  { id: 'gtao-upsample', labels: ['gtao-upsample'] },
  { id: 'ppg-update', labels: ['ppg-update'] },
  // Virtual denoiser-adapter slot — labels come from the active Denoiser
  // and are spliced here by `composePassLabels`.
  { id: 'denoiser-adapter', labels: [] },
  { id: 'indirect-temporal-accum', labels: ['indirect-temporal-accum'] },
  {
    id: 'atrous-indirect-3',
    labels: [
      'atrous-indirect-0',
      'atrous-indirect-1',
      'atrous-indirect-2',
      'atrous-indirect-3',
    ],
  },
  { id: 'indirect-combine', labels: ['indirect-combine'] },
  { id: 'temporalAccum', labels: ['temporalAccum'] },
  { id: 'resolve', labels: ['resolve'] },
  { id: 'composite', labels: ['composite'] },
  // ReGIR grid-build (Boksansky 2021) — opt-in. Placed LAST in the static
  // timestamp order so its slot is a TRAILING addition that does not shift any
  // existing slot index (the slot layout is decoupled from dispatch order —
  // the registry runs `regir-build` FIRST via topo-sort, since it has no deps
  // and `regir-build` < `sample-budget` lexicographically, so the grid is
  // filled before RIS reads it). Reserved like the PPG passes: always in the
  // order; the runtime registry only registers the pass when ReGIR is live.
  { id: 'regir-build', labels: ['regir-build'] },
] satisfies NonDenoiserPassEntry[]);

/** Position in {@link NON_DENOISER_PASS_ORDER} at which the active
 *  denoiser's pass-labels are spliced — the `denoiser-adapter` virtual
 *  entry. */
const DENOISER_INSERTION_INDEX = (() => {
  const i = NON_DENOISER_PASS_ORDER.findIndex((e) => e.id === 'denoiser-adapter');
  if (i < 0) throw new Error('NON_DENOISER_PASS_ORDER must contain denoiser-adapter');
  return i;
})();

/**
 * Labels added by the DDGI border-fill pass, owned by `ddgi/probeUpdatePass.ts`
 * — a separate encoder runs these inside `HybridLayeredStage`, not the
 * walkaround pipeline. They appear in the timestamp layout (these passes
 * write into the same querySet) but no Pass entry under `passes/` owns
 * them. Kept here so the static order stays the single source of truth.
 *
 * Insertion point: between `indirect-combine` and `temporalAccum` —
 * matches the historical layout in `buildPassLayout`.
 */
// File-local — consumed only inside `buildPassLayout`. 2026-05-18
// dead-code sweep verified zero non-self consumers.
const DDGI_BORDER_LABELS: readonly PassLabel[] = Object.freeze([
  'ddgi-border-irr',
  'ddgi-border-vis',
]);

/**
 * Phase-0 productization — the per-config knobs that change which pass labels
 * are emitted. Threaded identically into BOTH the runtime registry (which Pass
 * instances dispatch) and `buildPassLayout` (timestamp slot layout), so the
 * two never desync (Risk R2). Optional fields default to the full-fidelity
 * (ultra) config, so callers that omit them get today's layout unchanged.
 */
export interface PassLayoutConfig {
  /** ReSTIR-DI spatial-reuse ping-pong pass count (1 or 2). Default 2. */
  readonly diSpatialPasses?: 1 | 2;
  /** ReSTIR-GI spatial-reuse ping-pong pass count (1 or 2). Default 2. */
  readonly giSpatialPasses?: 1 | 2;
  /** Whether GTAO + its bilateral upsample run. Default true. */
  readonly gtaoEnabled?: boolean;
}

/** Labels the {@link SpatialReservoirPass} emits for a given DI pass count.
 *  1-pass keeps only the terminal `spatial-2` label (shade depends on it). */
export function diSpatialPassLabels(passCount: 1 | 2): readonly PassLabel[] {
  return passCount === 1 ? ['spatial-2'] : ['spatial-1', 'spatial-2'];
}

/** Labels the {@link SpatialGIReservoirPass} emits for a given GI pass count.
 *  1-pass keeps only the terminal `gi-spatial-2` label. */
export function giSpatialPassLabels(passCount: 1 | 2): readonly PassLabel[] {
  return passCount === 1 ? ['gi-spatial-2'] : ['gi-spatial-1', 'gi-spatial-2'];
}

/** Build the ordered label array for a given denoiser + Phase-0 quality
 *  config. Used by `buildPassLayout` AND by the orchestrator if it wants to
 *  verify its runtime registry matches the static layout.
 *
 *  The `config` slices the spatial label sets to the active pass count and
 *  omits the gtao / gtao-upsample labels when GTAO is gated off, so the
 *  timestamp slot layout matches exactly what the gated runtime loop
 *  dispatches. */
export function composePassLabels(
  denoiserLabels: readonly PassLabel[],
  config: PassLayoutConfig = {},
): readonly PassLabel[] {
  const diPasses = config.diSpatialPasses ?? 2;
  const giPasses = config.giSpatialPasses ?? 2;
  const gtaoOn = config.gtaoEnabled ?? true;

  const result: PassLabel[] = [];
  for (let i = 0; i < NON_DENOISER_PASS_ORDER.length; i++) {
    if (i === DENOISER_INSERTION_INDEX) {
      result.push(...denoiserLabels);
    }
    const entry = NON_DENOISER_PASS_ORDER[i]!;
    // Config-driven label sets (the gated/variable passes).
    if (entry.id === 'spatial-2') {
      result.push(...diSpatialPassLabels(diPasses));
    } else if (entry.id === 'gi-spatial-2') {
      result.push(...giSpatialPassLabels(giPasses));
    } else if (entry.id === 'gtao' || entry.id === 'gtao-upsample') {
      if (gtaoOn) result.push(...entry.labels);
    } else {
      result.push(...entry.labels);
    }
    // Splice the DDGI border-fill labels (owned by a separate encoder
    // outside this pipeline) right after `indirect-combine`.
    if (entry.id === 'indirect-combine') {
      result.push(...DDGI_BORDER_LABELS);
    }
  }
  return result;
}
