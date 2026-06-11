/**
 * PipelineSubsystem — minimal common contract for pipeline-owned subsystem
 * coordinators in {@link WalkaroundGPUPipeline}.
 *
 * Five classes participate in this pattern:
 *   • {@link PPGCoordinator}                  — Shape A (coordinator, in-pipeline)
 *   • {@link ReGIRCoordinator}                — Shape A (coordinator, in-pipeline)
 *   • {@link OptionalSubsystemBindingState}   — Shape B (thin binding holder)
 *   • {@link RCSubsystem}                     — Shape C (top-level, engine-driven)
 *   • {@link NrcSubsystem}                    — Shape D (inverted layout dep)
 *
 * ## The four integration shapes
 *
 * When adding a new GI subsystem, pick the shape that fits its integration
 * needs:
 *
 * ### Shape A — coordinator in-pipeline
 * _Example: PPGCoordinator, ReGIRCoordinator_
 *
 * The subsystem is self-contained: it owns its GPU resources, drives its own
 * passes, and exposes an `enabled` accessor that the pipeline orchestrator
 * reads to decide whether to schedule the subsystem's passes. Construction,
 * initialization, per-frame update, and disposal all happen inside the
 * pipeline. The host has no direct reference.
 *
 * Use this shape when: the subsystem is purely a pipeline concern, its passes
 * do not produce outputs that the host engine needs to route elsewhere, and its
 * enabled/disabled state is driven by a single flag or config option.
 *
 * ### Shape B — thin binding-state holder with heavyweight owner outside
 * _Example: OptionalSubsystemBindingState (DDGI + RC + PPG placeholder slots)_
 *
 * The subsystem's heavyweight compute (probe updates, cascade dispatch, tree
 * splits) lives in a separate object (DDGISubsystem, RCSubsystem, PPGCoordinator)
 * that is either engine-level or pipeline-level. `OptionalSubsystemBindingState`
 * is the single pipeline-side object that translates outputs from those objects
 * into concrete GPUBuffer / GPUTexture bindings for the shared `hybridLayers`
 * bind group. It holds placeholder buffers/textures for when a subsystem is
 * disabled; the heavyweight owner is responsible for calling `setInputs` /
 * `setRCInputs` each frame with real resources.
 *
 * Use this shape when: multiple optional subsystems feed a single bind group,
 * and the per-subsystem placeholders need to be co-managed to keep the BGL
 * layout stable. Do NOT add per-subsystem binding state classes for new
 * subsystems that fit this pattern — extend `OptionalSubsystemBindingState`.
 *
 * ### Shape C — top-level subsystem driven by the engine
 * _Example: RCSubsystem_
 *
 * The subsystem is created and driven by the engine layer
 * (`HybridEngineRC`) rather than by the pipeline. The pipeline only sees its
 * outputs (a cascade-0 GPUBuffer + packed params) via `setRCInputs` on
 * `OptionalSubsystemBindingState`. Disposal is the pipeline's responsibility
 * (the subsystem is still registered in the pipeline's dispose sequence) but
 * the engine owns the lifecycle for the pass scheduling.
 *
 * Use this shape when: the subsystem's outputs need to be routed by the engine
 * to multiple destinations (e.g., both the pipeline and external host APIs),
 * or when it needs access to engine-level state (device context, scene BVH)
 * that the pipeline does not expose.
 *
 * ### Shape D — subsystem created by the pipeline with an inverted layout dep
 * _Example: NrcSubsystem_
 *
 * The pipeline constructs the subsystem (conditioned on a config flag), but
 * the subsystem itself imports types / helpers from the pipeline layer rather
 * than the other way around. This "inside-out" dependency is a known code
 * smell: the subsystem should not depend on its host. The correct fix is to
 * extract the shared types into a neutral module that both the subsystem and
 * the pipeline can import.
 *
 * **Shape-D smell resolved (I5.2, 2026-06-11):** `BGLKey` and `BGLCache` have
 * been extracted to `src/bglTypes.ts` (mirroring the `wgslTypes.ts` precedent).
 * `neural/nrc/nrcBindGroupLayout.ts` now imports from `bglTypes.ts` directly;
 * `pipeline/bindGroupLayouts.ts` re-exports them for back-compat. The
 * `NrcSubsystem` instance IS the enabled gate (it is only constructed when
 * `nrcEnabled` is true); there is no separate `enabled` property.
 *
 * Use this shape only as a temporary arrangement when a new subsystem has a
 * genuine circular dep that cannot be resolved without a larger restructure.
 *
 * ---
 *
 * The ONLY invariant shared across all five is `dispose(): void`. The `enabled`
 * accessor is exposed as optional because:
 *   - PPGCoordinator  has  `get enabled(): boolean` (tracks ppgEnabled at init)
 *   - ReGIRCoordinator exposes `config.enabled` (the static host opt-in) and
 *     `live` (the runtime gate after `initialize()`); neither uses a
 *     `get enabled()` accessor directly — callers read `config.enabled` or
 *     `live`.
 *   - OptionalSubsystemBindingState, RCSubsystem, NrcSubsystem have no
 *     `enabled` property.
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
