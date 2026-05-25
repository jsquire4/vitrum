/**
 * GPU timestamp query helpers — DEV-only, feature-gated.
 *
 * The number and ordering of timestamp slots depends on frame configuration
 * (which denoiser is active). `buildPassLayout` returns a deterministic
 * `PassLabel` → slot-index map per frame, and the querySet is sized to the
 * worst-case slot count so a single allocation survives every configuration.
 *
 * Configurations (slot count in parentheses, current as of Sprint 18
 * follow-up — indirect-temporal-accum + 4-iter atrous-indirect chain):
 *   • legacy atrous (26): sample-budget, ris, temporal, spatial-1,
 *       spatial-2, gi-ris, gi-temporal, gi-spatial-1, gi-spatial-2, shade,
 *       gtao, gtao-upsample, atrous-0..2, indirect-temporal-accum,
 *       atrous-indirect-0..3, indirect-combine, ddgi-border-irr,
 *       ddgi-border-vis, temporalAccum, resolve, composite
 *   • atrous-variance (28): sample-budget, …shade, gtao+upsample,
 *       welford-temporal, atrous-variance-variance, atrous-variance-atrous-0..2,
 *       indirect-temporal-accum, atrous-indirect-0..3, indirect-combine,
 *       ddgi-border-irr, ddgi-border-vis, temporalAccum, resolve,
 *       composite — matches MAX_PASS_COUNT
 *
 * Sprint 9 adaptive-sampling wire-in adds `sample-budget` (prepended) and
 * `resolve` (inserted between temporalAccum and composite). Both passes
 * are part of the standard pipeline now — there is no opt-out path.
 *
 * Uses a ping-pong pair of readback buffers so one can be in-flight
 * (mapped/mapping) while the next frame writes into the other, avoiding
 * the "buffer in use" stall a single readback buffer would cause.
 */

import { DENOISER_PASS_LABELS } from './denoisers/index.js';
import { composePassLabels } from './passes/passOrder.js';

export type PassLabel =
  | 'sample-budget'
  | 'ris'
  | 'temporal'
  | 'spatial-1'
  | 'spatial-2'
  | 'gi-ris'
  | 'gi-temporal'
  | 'gi-spatial-1'
  | 'gi-spatial-2'
  | 'shade'
  | 'motion-vectors'
  | 'gtao'
  | 'gtao-upsample'
  | 'welford-temporal'
  | 'atrous-variance-variance'
  | 'atrous-variance-atrous-0'
  | 'atrous-variance-atrous-1'
  | 'atrous-variance-atrous-2'
  | 'atrous-0'
  | 'atrous-1'
  | 'atrous-2'
  // T2.H1 — svgf-real pass labels (5 passes: reproj → moments → 7x7 → 5 × atrous)
  | 'svgf-real-reproj'
  | 'svgf-real-moments'
  | 'svgf-real-7x7'
  | 'svgf-real-atrous-0'
  | 'svgf-real-atrous-1'
  | 'svgf-real-atrous-2'
  | 'svgf-real-atrous-3'
  | 'svgf-real-atrous-4'
  | 'indirect-temporal-accum'
  | 'atrous-indirect-0'
  | 'atrous-indirect-1'
  | 'atrous-indirect-2'
  | 'atrous-indirect-3'
  | 'indirect-combine'
  | 'ddgi-border-irr'
  | 'ddgi-border-vis'
  // T2.H3 — PPG paper-faithful (Müller 2017) opt-in passes
  | 'ppg-update'
  | 'ppg-guide'
  | 'temporalAccum'
  | 'resolve'
  | 'composite';

/**
 * Maximum slot count across all supported configurations. Used to size the
 * GPU querySet + resolve/readback buffers so allocation survives every
 * runtime layout.
 *
 * History: 15 (base) → 17 (Sprint 9: sample-budget + resolve) →
 *          19 (Sprint 15: gtao + gtao-upsample) → 20 (Sprint 16: gi-ris) →
 *          23 (Sprint 17: gi-temporal + gi-spatial-1 + gi-spatial-2) →
 *          24 (Sprint 18: indirect-combine) →
 *          22 (Original #7: trim 2 dead atrous-variance-atrous slots — iter count
 *          dropped from 5 to 3 in shared-denoisers but layout was stale) →
 *          26 (Sprint 18 follow-up: per-channel indirect atrous — replace the
 *          embedded bilateral in indirect-combine with a real 4-iter
 *          atrous chain (atrous-indirect-0..3) on the indirect channel) →
 *          27 (Sprint 18 follow-up: indirect-temporal-accum — pre-atrous
 *          temporal accumulator with TCBB clip to kill firefly admit
 *          + smooth shadow-region blotches before spatial filter) →
 *          26 (D7 sweep: PPG deleted — max is now atrous-variance without ppg-update) →
 *          28 (Item 3: DDGI atlas border fill — ddgi-border-irr + ddgi-border-vis
 *          appended after indirect-combine and before temporalAccum).
 *          31 (T2.H1: svgf-real replaces the 5 atrous-variance passes with 8
 *          svgf-real passes — reproj, moments, 7×7, 5 × atrous iters).
 *          33 (T2.H3: PPG paper-faithful opt-in — adds ppg-update + ppg-guide
 *          when ppgEnabled=true; layout returns 31 when ppgEnabled=false) →
 *          34 (W2 motion-vectors pass between shade and gtao).
 */
export const MAX_PASS_COUNT = 34;

interface PassLayoutOptions {
  /** T2.H2: 'neural' falls through to 'atrous-variance' pass layout (InferenceGraph is
   *  self-managing and doesn't participate in the timestamp-query pass layout).
   *  W1-R3: widened to {@link DenoiserId} so 'none' / 'oidn-final' compile;
   *  'oidn-final' is registered as `disabled` and never reaches this layout
   *  builder at runtime — the registry rejects it at `lookup()` time. */
  readonly denoiserMode: import('./denoisers/index.js').DenoiserId;
}

interface PassLayout {
  /** Slot index for the given label. Throws if the label is not active in this layout. */
  readonly index: (label: PassLabel) => number;
  /** Total slots used by this layout. Always ≤ MAX_PASS_COUNT. */
  readonly slotCount: number;
  /** Ordered slot labels (length === slotCount), used to label readback timings. */
  readonly labels: readonly PassLabel[];
}

/**
 * Build the per-frame timestamp-query slot layout.
 *
 * Single source of truth: consumes the static {@link NON_DENOISER_PASS_ORDER}
 * table from `passes/passOrder.ts` and splices in the active denoiser's
 * labels via {@link DENOISER_PASS_LABELS}. Adding a new non-denoiser pass
 * is one edit to the order table (no edits here); adding a new denoiser
 * is one entry in the labels map.
 */
export function buildPassLayout(opts: PassLayoutOptions): PassLayout {
  // Lazy imports break the module-init cycle (passOrder → timestampQueries
  // → denoisers/index → timestampQueries). At call time both target
  // modules have fully initialised their `*_PASS_LABELS` and
  // `NON_DENOISER_PASS_ORDER` exports.
  const denoiserLabels = DENOISER_PASS_LABELS[opts.denoiserMode];
  const labels = composePassLabels(denoiserLabels);

  const indexMap = new Map<PassLabel, number>();
  labels.forEach((label, i) => indexMap.set(label, i));

  return {
    index: (label) => {
      const i = indexMap.get(label);
      if (i === undefined) {
        throw new Error(
          `pass label "${label}" is not active in this layout (denoiser=${opts.denoiserMode})`,
        );
      }
      return i;
    },
    slotCount: labels.length,
    labels,
  };
}

/**
 * Build the optional timestampWrites struct for a pass. Returns undefined
 * when timestamp queries aren't enabled, so the spread trick used in
 * renderFrame doesn't degrade pass descriptors on adapters without the
 * feature.
 */
export function tsWrites(
  querySet: GPUQuerySet | null,
  layout: PassLayout,
  label: PassLabel,
): GPUComputePassTimestampWrites | undefined {
  if (!querySet) return undefined;
  const i = layout.index(label);
  return {
    querySet,
    beginningOfPassWriteIndex: i * 2,
    endOfPassWriteIndex: i * 2 + 1,
  };
}

export interface TimestampState {
  querySet: GPUQuerySet | null;
  resolveBuffer: GPUBuffer | null;
  readbackA: GPUBuffer | null;
  readbackB: GPUBuffer | null;
  readbackInFlight: 'A' | 'B' | null;
  periodNs: number;
  lastGpuTimings: Record<string, number>;
  lastGpuTimingsFrame: number;
}

export function makeTimestampState(): TimestampState {
  return {
    querySet: null,
    resolveBuffer: null,
    readbackA: null,
    readbackB: null,
    readbackInFlight: null,
    periodNs: 1.0,
    lastGpuTimings: {},
    lastGpuTimingsFrame: -1,
  };
}

/**
 * Initialize timestamp query infrastructure if the adapter supports the
 * feature and we are in DEV mode. Mutates `state` in place.
 */
export function initTimestampQueries(
  device: GPUDevice,
  state: TimestampState,
): void {
  // DEV guard: Vite injects `import.meta.env.DEV`; in non-Vite toolchains
  // the property is absent and we default to enabling queries (safe — the
  // feature-gate below only activates on adapters that expose timestamp-query).
  // The double unknown cast avoids TS2352 since ImportMeta has no index sig.
  const meta = import.meta as unknown as { env?: { DEV?: boolean } };
  const isDev = meta.env?.['DEV'] ?? true;
  if (!isDev) return;

  if (device.features.has('timestamp-query')) {
    const N = MAX_PASS_COUNT;
    state.querySet = device.createQuerySet({ type: 'timestamp', count: N * 2 });
    state.resolveBuffer = device.createBuffer({
      size: N * 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    state.readbackA = device.createBuffer({
      size: N * 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    state.readbackB = device.createBuffer({
      size: N * 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // WebGPU exposes timestampPeriod via adapter info (ns per tick).
    // Some browsers normalize to 1ns by spec.
    const adapterInfo = (device as unknown as { adapterInfo?: { timestampPeriod?: number } }).adapterInfo;
    state.periodNs = adapterInfo?.timestampPeriod ?? 1.0;
    console.log('[hybrid:debug] timestamp queries enabled',
      { maxPasses: N, periodNs: state.periodNs });
  } else {
    console.log('[hybrid:debug] timestamp queries unavailable on this adapter; falling back to JS-submit timing only');
  }
}

/**
 * Read back the most recent timestamp results into `state.lastGpuTimings`.
 * Async, fire-and-forget — uses ping-pong to avoid stalling the next frame.
 * `labels` is captured in the closure so async resolution always uses the
 * label set that was active when this readback was kicked, even if the
 * pipeline reconfigures between the kick and the mapAsync resolution.
 */
export function kickTimestampReadback(
  state: TimestampState,
  frameCount: number,
  labels: readonly PassLabel[],
): void {
  if (!state.resolveBuffer || !state.readbackA || !state.readbackB) return;
  if (state.readbackInFlight) return; // prior readback still pending

  const target = frameCount % 2 === 0 ? state.readbackA : state.readbackB;
  const slot: 'A' | 'B' = frameCount % 2 === 0 ? 'A' : 'B';
  state.readbackInFlight = slot;
  const periodNs = state.periodNs;
  const N = labels.length;

  target.mapAsync(GPUMapMode.READ).then(() => {
    try {
      const range = target.getMappedRange();
      const view = new BigInt64Array(range.slice(0));
      target.unmap();
      const next: Record<string, number> = {};
      let total = 0;
      for (let i = 0; i < N; i++) {
        const begin = view[i * 2] ?? 0n;
        const end   = view[i * 2 + 1] ?? 0n;
        // Monotonic clocks should never decrement, but at boot the first
        // frame's begin/end can be 0n — skip those.
        if (end <= begin) continue;
        const ms = Number(end - begin) * periodNs / 1_000_000;
        const label = labels[i];
        if (label !== undefined) next[label] = +ms.toFixed(3);
        total += ms;
      }
      next['total'] = +total.toFixed(3);
      state.lastGpuTimings = next;
      state.lastGpuTimingsFrame = frameCount;
    } catch {
      // ignore — buffer was likely unmapped during a dispose race
    } finally {
      state.readbackInFlight = null;
    }
  }).catch(() => {
    state.readbackInFlight = null;
  });
}

/**
 * Resolve the timestamp query set into the resolve buffer, then copy into
 * the inactive readback buffer (the ping-pong slot NOT currently in flight).
 * Must be called from within the renderFrame encoder submission sequence.
 */
export function resolveTimestamps(
  encoder: GPUCommandEncoder,
  state: TimestampState,
  frameCount: number,
  slotCount: number,
): void {
  if (!state.querySet || !state.resolveBuffer) return;

  const target = frameCount % 2 === 0 ? state.readbackA : state.readbackB;
  const slot: 'A' | 'B' = frameCount % 2 === 0 ? 'A' : 'B';

  // Skip resolve+copy if the target buffer is still mapped from a prior
  // readback (i.e. its slot matches `readbackInFlight`).
  if (target && state.readbackInFlight !== slot) {
    encoder.resolveQuerySet(state.querySet, 0, slotCount * 2, state.resolveBuffer, 0);
    encoder.copyBufferToBuffer(state.resolveBuffer, 0, target, 0, slotCount * 2 * 8);
  }
}

/**
 * Synchronous diagnostic readback: creates a fresh staging buffer, copies
 * the current resolveBuffer into it, awaits the mapAsync, returns the
 * decoded per-pass timings. Bypasses the ping-pong infrastructure entirely.
 *
 * Used by the validation harness to confirm whether timestamps are landing
 * in the queryset at all (vs the production fire-and-forget path's
 * readback that may have a state-flow gap).
 */
export async function readTimestampsOnce(
  device: GPUDevice,
  state: TimestampState,
  layout: PassLayout,
): Promise<{ perPass: Record<string, number>; rawBigints: string[] }> {
  if (!state.querySet || !state.resolveBuffer) {
    return { perPass: {}, rawBigints: [] };
  }
  const size = layout.slotCount * 2 * 8;
  const readback = device.createBuffer({
    label: 'timestamp-readback-debug',
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'timestamp-readback-debug' });
  encoder.resolveQuerySet(state.querySet, 0, layout.slotCount * 2, state.resolveBuffer, 0);
  encoder.copyBufferToBuffer(state.resolveBuffer, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const view = new BigInt64Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();

  const perPass: Record<string, number> = {};
  const rawBigints: string[] = [];
  let total = 0;
  for (let i = 0; i < layout.slotCount; i++) {
    const begin = view[i * 2] ?? 0n;
    const end = view[i * 2 + 1] ?? 0n;
    rawBigints.push(`${String(begin)}/${String(end)}`);
    if (end <= begin) continue;
    const ms = Number(end - begin) * state.periodNs / 1_000_000;
    const label = layout.labels[i];
    if (label) perPass[label] = +ms.toFixed(3);
    total += ms;
  }
  perPass['total'] = +total.toFixed(3);
  return { perPass, rawBigints };
}

/**
 * Destroy all timestamp query GPU resources. Skips mapped buffers
 * (mapped buffers can't be destroy()'d; the GC reclaims them when
 * their mapped range goes out of scope).
 */
export function disposeTimestampState(state: TimestampState): void {
  state.querySet?.destroy();
  state.resolveBuffer?.destroy();
  if (state.readbackInFlight !== 'A') state.readbackA?.destroy();
  if (state.readbackInFlight !== 'B') state.readbackB?.destroy();
}
