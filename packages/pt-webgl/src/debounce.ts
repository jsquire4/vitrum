// Rate-aware debounce window for path-tracer scene sync.
//
// When the recent edit rate exceeds the threshold (rapid slider edits on
// intensity / colour / time-of-day), extend the debounce so the BVH rebuild
// waits for the burst to settle — reducing the device-hang risk during rapid
// SceneControl edits.

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
