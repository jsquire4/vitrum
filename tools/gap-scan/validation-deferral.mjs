/**
 * GPU / rendering validation deferred until after code-first remediation.
 * Policy: plan/VALIDATION-DEFERRED.md
 *
 * Deferred tasks are excluded from the parallel schedule (like SKIP_TASK_IDS).
 */

/** @typedef {{ id: string; disposition?: string; problem?: string; steps?: string[]; tests?: string[]; files?: string[] }} TaskLike */

export const VALIDATION_DEFERRAL_POLICY = `Code-first campaign: all GPU A/B, golden PNG,
reference-render, behavioral-gate render passes, material-furnace radiometric proof,
and wsl-gpu capture tasks are DEFERRED until implementation waves complete.
Track in plan/VALIDATION-DEFERRED.md.`;

const V28_BUCKETS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'B1', 'B2', 'B4', 'B8', 'B15', 'B16'];

/** Explicit task IDs — always deferred in code-first mode. */
export const DEFERRED_VALIDATION_TASK_IDS = new Set([
  // RT100 Phase 8 — V28-B + GRIS GPU proof
  'RT100-V28-000',
  'RT100-V19-GRIS',
  ...V28_BUCKETS.map((b) => `RT100-V28-${b}`),
  // RT100 render/promotion proof
  'RT100-WA-3E',
  'RT100-A9-BDPT',
  'RT100-PTWG-FURNACE',
  'RT100-5A-GOLDEN',
  'RT100-5C-GPU-MUT',
  'RT100-GATE-FULL',
  // RT100 decisions gated on GPU quality A/B
  'RT100-A6-DECIDE',
  'RT100-A10-WEIGHTS',
  // Code-gap render-validation tails
  'PTWG-017',
  'PTWG-080',
  'TOOL-004',
]);

const RENDER_VALIDATION_RE =
  /wsl-gpu|reference-render|golden\s*png|meanLum|V28-B|GPU\s*A\/B|radiometric-ab|material[- ]furnace|behavioral-gate\/gate\.mjs.*(?:render|boot|readback)|capture.*render|equal-time\s*RMSE/i;

const RENDER_VALIDATION_TEST_RE =
  /behavioral-gate\/gate\.mjs(?!\s*--filter\s*ptgl)|wsl-gpu|radiometric-ab|reference-render/i;

/**
 * @param {TaskLike} t
 * @returns {boolean}
 */
export function isDeferredValidationTask(t) {
  if (t.disposition === 'SKIP') return true;
  if (DEFERRED_VALIDATION_TASK_IDS.has(t.id)) return true;

  const blob = [t.problem ?? '', ...(t.steps ?? []), ...(t.tests ?? [])].join('\n');
  if (RENDER_VALIDATION_RE.test(blob)) return true;

  // VERIFY tasks whose only proof is a render harness
  if (t.disposition === 'VERIFY' && (t.tests ?? []).some((x) => RENDER_VALIDATION_TEST_RE.test(x))) {
    return true;
  }

  return false;
}

/**
 * @param {TaskLike} t
 * @returns {string}
 */
export function deferredValidationReason(t) {
  if (DEFERRED_VALIDATION_TASK_IDS.has(t.id)) {
    return 'Explicit code-first deferral (plan/VALIDATION-DEFERRED.md)';
  }
  if (t.disposition === 'SKIP') {
    return t.problem ?? 'Marked SKIP';
  }
  return 'Matched GPU/rendering validation pattern — deferred post-code';
}

/**
 * @param {TaskLike[]} tasks
 * @returns {TaskLike[]}
 */
export function listDeferredValidationTasks(tasks) {
  return tasks.filter(isDeferredValidationTask);
}
