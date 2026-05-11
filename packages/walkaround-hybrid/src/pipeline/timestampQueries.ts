/**
 * GPU timestamp query helpers — DEV-only, feature-gated.
 *
 * The number and ordering of timestamp slots depends on frame configuration
 * (whether PPG is enabled and which denoiser is active). `buildPassLayout`
 * returns a deterministic `PassLabel` → slot-index map per frame, and the
 * querySet is sized to the worst-case slot count so a single allocation
 * survives every configuration.
 *
 * Configurations (slot count in parentheses):
 *   • PPG off, legacy atrous (10): ris…shade, atrous-0..2, temporalAccum, composite
 *   • PPG on,  legacy atrous (11): + ppg-update
 *   • PPG off, SVGF          (14): ris…shade, welford-temporal, svgf-variance, svgf-atrous-0..4, temporalAccum, composite
 *   • PPG on,  SVGF          (15): + ppg-update
 *
 * Uses a ping-pong pair of readback buffers so one can be in-flight
 * (mapped/mapping) while the next frame writes into the other, avoiding
 * the "buffer in use" stall a single readback buffer would cause.
 */

export type PassLabel =
  | 'ris'
  | 'temporal'
  | 'spatial-1'
  | 'spatial-2'
  | 'shade'
  | 'ppg-update'
  | 'welford-temporal'
  | 'svgf-variance'
  | 'svgf-atrous-0'
  | 'svgf-atrous-1'
  | 'svgf-atrous-2'
  | 'svgf-atrous-3'
  | 'svgf-atrous-4'
  | 'atrous-0'
  | 'atrous-1'
  | 'atrous-2'
  | 'temporalAccum'
  | 'composite';

/**
 * Maximum slot count across all supported configurations. Used to size the
 * GPU querySet + resolve/readback buffers so allocation survives every
 * runtime layout.
 */
export const MAX_PASS_COUNT = 15;

export interface PassLayoutOptions {
  readonly ppgEnabled: boolean;
  readonly denoiserMode: 'svgf' | 'atrous';
}

export interface PassLayout {
  /** Slot index for the given label. Throws if the label is not active in this layout. */
  readonly index: (label: PassLabel) => number;
  /** Total slots used by this layout. Always ≤ MAX_PASS_COUNT. */
  readonly slotCount: number;
  /** Ordered slot labels (length === slotCount), used to label readback timings. */
  readonly labels: readonly PassLabel[];
}

export function buildPassLayout(opts: PassLayoutOptions): PassLayout {
  const labels: PassLabel[] = ['ris', 'temporal', 'spatial-1', 'spatial-2', 'shade'];
  if (opts.ppgEnabled) labels.push('ppg-update');
  if (opts.denoiserMode === 'svgf') {
    labels.push(
      'welford-temporal',
      'svgf-variance',
      'svgf-atrous-0',
      'svgf-atrous-1',
      'svgf-atrous-2',
      'svgf-atrous-3',
      'svgf-atrous-4',
    );
  } else {
    labels.push('atrous-0', 'atrous-1', 'atrous-2');
  }
  labels.push('temporalAccum', 'composite');

  const indexMap = new Map<PassLabel, number>();
  labels.forEach((label, i) => indexMap.set(label, i));

  return {
    index: (label) => {
      const i = indexMap.get(label);
      if (i === undefined) {
        throw new Error(
          `pass label "${label}" is not active in this layout (ppg=${opts.ppgEnabled}, denoiser=${opts.denoiserMode})`,
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
