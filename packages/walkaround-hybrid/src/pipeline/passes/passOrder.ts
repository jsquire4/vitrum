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
export interface NonDenoiserPassEntry {
  /** Stable id of the corresponding Pass class. */
  readonly id: string;
  /** Labels emitted by the pass in dispatch order. */
  readonly labels: readonly PassLabel[];
}

/**
 * Non-denoiser passes that participate in the GPU-timing layout, in
 * dispatch order. The denoiser is spliced in between `gtao-upsample` and
 * `indirect-temporal-accum` (see {@link composePassLabels}).
 *
 * PPGGuidePass + PPGUpdatePass are deliberately omitted — they have been
 * out of the timestamp layout since the D7 sweep removed PPG-only slots,
 * and `buildPassLayout` never appended their labels. The Pass classes
 * still dispatch via `computeDesc('ppg-guide')` / `'ppg-update'` when
 * ppgEnabled is true; that call returns `{ label }` only because no
 * adapter currently has both timestamp-query AND ppgEnabled together
 * (in that combo `layout.index('ppg-guide')` would throw — a pre-R5
 * latent bug deliberately preserved here for bit-identical layout
 * output. A future PPG enable workstream will add explicit slots.)
 */
export const NON_DENOISER_PASS_ORDER: readonly NonDenoiserPassEntry[] = Object.freeze([
  { id: 'sample-budget', labels: ['sample-budget'] },
  { id: 'ris', labels: ['ris'] },
  { id: 'temporal', labels: ['temporal'] },
  { id: 'spatial-2', labels: ['spatial-1', 'spatial-2'] },
  { id: 'gi-ris', labels: ['gi-ris'] },
  { id: 'gi-temporal', labels: ['gi-temporal'] },
  { id: 'gi-spatial-2', labels: ['gi-spatial-1', 'gi-spatial-2'] },
  { id: 'shade', labels: ['shade'] },
  { id: 'gtao', labels: ['gtao'] },
  { id: 'gtao-upsample', labels: ['gtao-upsample'] },
  // <denoiser passLabels splice here>
  { id: 'indirect-temporal-accum', labels: ['indirect-temporal-accum'] },
  {
    id: 'atrous-indirect-3',
    labels: ['atrous-indirect-0', 'atrous-indirect-1', 'atrous-indirect-2', 'atrous-indirect-3'],
  },
  { id: 'indirect-combine', labels: ['indirect-combine'] },
  { id: 'temporalAccum', labels: ['temporalAccum'] },
  { id: 'resolve', labels: ['resolve'] },
  { id: 'composite', labels: ['composite'] },
] satisfies NonDenoiserPassEntry[]);

/** Position in {@link NON_DENOISER_PASS_ORDER} after which the denoiser
 *  pass-labels are spliced — the slot AFTER `gtao-upsample`. */
const DENOISER_INSERTION_INDEX = (() => {
  const i = NON_DENOISER_PASS_ORDER.findIndex((e) => e.id === 'gtao-upsample');
  if (i < 0) throw new Error('NON_DENOISER_PASS_ORDER must contain gtao-upsample');
  return i + 1;
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
export const DDGI_BORDER_LABELS: readonly PassLabel[] = Object.freeze([
  'ddgi-border-irr',
  'ddgi-border-vis',
]);

/** Build the ordered label array for a given denoiser config. Used by
 *  `buildPassLayout` AND by the orchestrator if it wants to verify its
 *  runtime registry matches the static layout. */
export function composePassLabels(denoiserLabels: readonly PassLabel[]): readonly PassLabel[] {
  const result: PassLabel[] = [];
  for (let i = 0; i < NON_DENOISER_PASS_ORDER.length; i++) {
    if (i === DENOISER_INSERTION_INDEX) {
      result.push(...denoiserLabels);
    }
    const entry = NON_DENOISER_PASS_ORDER[i]!;
    result.push(...entry.labels);
    // Splice the DDGI border-fill labels (owned by a separate encoder
    // outside this pipeline) right after `indirect-combine`.
    if (entry.id === 'indirect-combine') {
      result.push(...DDGI_BORDER_LABELS);
    }
  }
  return result;
}
