/**
 * Shared denoiser-history reset policy.
 *
 * Walkaround mutations call `WalkaroundGPUPipeline.requestAccumReset()`, which
 * schedules the next frame with `frameIndex === 0`. Treat that same frame-zero
 * signal as a reset for denoisers with private temporal history; otherwise
 * material/light/geometry edits would drop the final accumulator but leave
 * Welford/BMFR/SVGF history warm with stale pre-mutation samples.
 */
export function shouldResetDenoiserHistory(frameIndex: number, isMoving: boolean): boolean {
  return isMoving || frameIndex <= 0;
}
