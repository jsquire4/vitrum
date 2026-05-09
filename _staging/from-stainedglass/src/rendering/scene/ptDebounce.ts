// Slice 3.4 — rate-aware debounce window for PathtracerSceneSync.
//
// Each Redux mutation that touches the lit scene re-fires the sync
// effect; the previous fixed 50ms debounce coalesced bursts well in
// the steady state but compounded the device-hang risk during rapid
// SceneControl edits (every slider tick on intensity / colour / time
// becomes a setScene call). When the recent edit rate exceeds the
// threshold, we extend the debounce so the BVH rebuild waits for the
// burst to settle.

export const PT_DEBOUNCE_MS_NORMAL = 50;
export const PT_DEBOUNCE_MS_BURST = 250;

/**
 * @param recentEditTimestamps - ring buffer of edit timestamps in ms
 *                               (DOMHighResTimeStamp from performance.now()).
 * @param now                  - current timestamp.
 * @param windowMs             - rolling window width (default 1s).
 * @param threshold            - edits within `windowMs` to switch to burst.
 * @returns the debounce window in ms.
 */
export function debounceMsForEditRate(
  recentEditTimestamps: readonly number[],
  now: number,
  windowMs: number = 1000,
  threshold: number = 5,
): number {
  const cutoff = now - windowMs;
  let count = 0;
  for (const t of recentEditTimestamps) {
    if (t >= cutoff) count++;
  }
  return count >= threshold ? PT_DEBOUNCE_MS_BURST : PT_DEBOUNCE_MS_NORMAL;
}
