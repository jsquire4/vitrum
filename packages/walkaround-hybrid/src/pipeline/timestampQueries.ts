/**
 * GPU timestamp query helpers — DEV-only, feature-gated.
 *
 * 10 passes per frame: ris, temporal, spatial-1, spatial-2, shade,
 * atrous-0..2 (3 iterations, stepWidths 1/2/4), temporalAccum, composite.
 *
 * Uses a ping-pong pair of readback buffers so one can be in-flight
 * (mapped/mapping) while the next frame writes into the other, avoiding
 * the "buffer in use" stall a single readback buffer would cause.
 */

export const PASS_LABELS = [
  'ris', 'temporal', 'spatial-1', 'spatial-2', 'shade',
  'atrous-0', 'atrous-1', 'atrous-2',
  'temporalAccum', 'composite',
] as const;

export type PassLabel = typeof PASS_LABELS[number];
export const PASS_COUNT = PASS_LABELS.length;

/**
 * Build the optional timestampWrites struct for a pass at the given
 * pipeline-level pass index. Returns undefined when timestamp queries
 * aren't enabled, so the spread trick used in renderFrame doesn't
 * degrade pass descriptors on adapters without the feature.
 */
export function tsWrites(
  querySet: GPUQuerySet | null,
  passIndex: number,
): GPUComputePassTimestampWrites | undefined {
  if (!querySet) return undefined;
  return {
    querySet,
    beginningOfPassWriteIndex: passIndex * 2,
    endOfPassWriteIndex: passIndex * 2 + 1,
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
    const N = PASS_COUNT;
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
      { passes: N, periodNs: state.periodNs });
  } else {
    console.log('[hybrid:debug] timestamp queries unavailable on this adapter; falling back to JS-submit timing only');
  }
}

/**
 * Read back the most recent timestamp results into `state.lastGpuTimings`.
 * Async, fire-and-forget — uses ping-pong to avoid stalling the next frame.
 */
export function kickTimestampReadback(
  state: TimestampState,
  frameCount: number,
): void {
  if (!state.resolveBuffer || !state.readbackA || !state.readbackB) return;
  if (state.readbackInFlight) return; // prior readback still pending

  const target = frameCount % 2 === 0 ? state.readbackA : state.readbackB;
  const slot: 'A' | 'B' = frameCount % 2 === 0 ? 'A' : 'B';
  state.readbackInFlight = slot;
  const periodNs = state.periodNs;
  const labels = PASS_LABELS;
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
): void {
  if (!state.querySet || !state.resolveBuffer) return;

  const N = PASS_COUNT;
  const target = frameCount % 2 === 0 ? state.readbackA : state.readbackB;
  const slot: 'A' | 'B' = frameCount % 2 === 0 ? 'A' : 'B';

  // Skip resolve+copy if the target buffer is still mapped from a prior
  // readback (i.e. its slot matches `readbackInFlight`).
  if (target && state.readbackInFlight !== slot) {
    encoder.resolveQuerySet(state.querySet, 0, N * 2, state.resolveBuffer, 0);
    encoder.copyBufferToBuffer(state.resolveBuffer, 0, target, 0, N * 2 * 8);
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
