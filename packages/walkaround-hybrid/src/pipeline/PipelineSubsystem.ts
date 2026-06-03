/**
 * PipelineSubsystem — minimal common contract for pipeline-owned subsystem
 * coordinators in {@link WalkaroundGPUPipeline}.
 *
 * Five classes participate in this pattern:
 *   • {@link PPGCoordinator}   — implements cleanly (dispose + enabled)
 *   • {@link ReGIRCoordinator} — implements cleanly (dispose + enabled via config)
 *   • {@link DDGIBindingState} — implements dispose; no `enabled` (DDGI is
 *     always constructed; null inputs = placeholder mode, not "disabled")
 *   • {@link RCSubsystem}      — implements dispose; no `enabled` (nullability
 *     of _dispatcher / _bvhBuffers is the live gate; never has a boolean flag)
 *   • {@link NrcSubsystem}     — implements dispose; no `enabled` (constructed
 *     only when nrcEnabled; existence IS the gate)
 *
 * The ONLY invariant shared across all five is `dispose(): void`. The `enabled`
 * accessor is exposed as optional because:
 *   - PPGCoordinator  has  `get enabled(): boolean` (tracks ppgEnabled at init)
 *   - ReGIRCoordinator exposes `config.enabled` (the static host opt-in) and
 *     `live` (the runtime gate after `initialize()`); neither uses a
 *     `get enabled()` accessor directly — callers read `config.enabled` or
 *     `live`.
 *   - DDGIBindingState, RCSubsystem, NrcSubsystem have no `enabled` property.
 *
 * Forcing a uniform `enabled: boolean` onto all five would require non-trivial
 * semantic changes to three classes that use different gate patterns. The goal
 * here is a documented dispose contract + IDE discoverability, not forced
 * lifecycle uniformity.
 *
 * `PipelineSubsystem` is intentionally NOT used as a constructor parameter type
 * anywhere — each subsystem field retains its concrete type so callers keep
 * access to the full surface without a cast.
 */

/** Minimal shared contract for pipeline-owned subsystem coordinators. */
export interface PipelineSubsystem {
  /**
   * Release all GPU resources owned by this subsystem. Safe to call multiple
   * times (all implementations are idempotent). Called from
   * {@link WalkaroundGPUPipeline.dispose} in the pipeline teardown sequence.
   */
  dispose(): void;
}
