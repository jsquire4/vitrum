# RFE-09 — Runtime Lighting Updates Without Pipeline Rebuild

**Date:** 2026-05-11
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Priority:** Medium
**Status:** Proposed
**Affects:** `@vitrum/walkaround-hybrid`

---

## 1. Motivation

`HybridEngine` consumes four lighting inputs as **creation-time** options:

- `primaryLightDir: [number, number, number]`
- `primaryLightIntensity: number`
- `skyTint: [number, number, number]`
- `skyIrradiance: number`

These are stamped into the BVH-build emitter list (for self-emission Le) AND into per-frame ReSTIR shading. Updating any of them requires the host to flip `pipelineRebuildKey` — which triggers `engine.reset()` and a full pipeline + BVH rebuild.

The host-side workaround is to quantize continuous inputs (time-of-day) into discrete slots so rebuilds happen only at slot transitions. This works (and `useVitrumWalkaroundEngine.ts` in the consumer already implements 8-slot quantization), but it caps the user-perceived smoothness of the time-of-day slider at ~3 hours of simulated time per slot. A continuous sun-position slider, a continuous fixture-intensity slider, and any animated lighting sequence all currently force a hitch on every value commit.

Equivalent concern for the DDGI light list: `HybridEngineOptions.lights?: DDGILight[]` is creation-time. Toggling a fixture's `on` flag or moving a fixture forces a full rebuild via `pipelineRebuildKey`.

The original in-tree `useHybridLayeredGI` stub at `vitrum-bridge/useVitrumWalkaroundEngine.ts` notes this as a known limitation:

> "HybridEngine stores primaryLightDir / skyTint etc. as creation-time options (not per-frame). Sprint N: extend HybridEngine with an updateLighting() method."

This RFE is that Sprint N entry.

---

## 2. Proposed API Surface

Add four runtime-update methods on `HybridEngine` (and matching capability flag in `@vitrum/core`'s `EngineCapabilities`):

```typescript
// in @vitrum/walkaround-hybrid/src/HybridEngine.ts

class HybridEngine implements Engine {
  /** Update the primary directional light's world-space direction.
   *  Patches the BVH emitter list AND the ReSTIR shading UBO without
   *  rebuilding the BVH. Re-bake of the in-place Le table on the BVH
   *  triangles is permitted if it is O(emitterCount) and does not
   *  invalidate the spatial structure. Engines that cannot patch
   *  without invalidating may fall back to a full reset(). */
  updatePrimaryLight(dir: [number, number, number], intensity: number): void;

  /** Update sky-dome RGB tint + irradiance scalar consumed by the
   *  sky-aperture probe and second-bounce sky-miss paths. UBO-only;
   *  no BVH touch. */
  updateSkyDome(tint: [number, number, number], irradiance: number): void;

  /** Replace the DDGI light list. Adding/removing/moving lights
   *  refreshes the probe-update pass uniform buffer. If the new list
   *  changes the *count* by more than a backend-defined threshold, the
   *  engine may internally fall back to a full pipeline rebuild.
   *  Engines should document the threshold in their capabilities. */
  updateLights(lights: DDGILight[]): void;

  /** Optional: combined update path for hosts that change all four
   *  inputs together (typical for time-of-day scrub). Atomic — engine
   *  either applies all four or rejects the call (state is unchanged
   *  on rejection). */
  updateLighting(patch: {
    primaryLight?: { dir: [number, number, number]; intensity: number };
    skyDome?: { tint: [number, number, number]; irradiance: number };
    lights?: DDGILight[];
  }): void;
}
```

Capability addition in `@vitrum/core`:

```typescript
// EngineCapabilities (in @vitrum/core/src/engine.ts)
interface EngineCapabilities {
  // ... existing fields ...

  /** Engine supports `updatePrimaryLight`, `updateSkyDome`, `updateLights`
   *  without a full pipeline rebuild. When false, hosts must trigger
   *  pipelineRebuildKey for any lighting change. */
  readonly supportsIncrementalLighting: boolean;
}
```

---

## 3. Backend semantics

For `@vitrum/walkaround-hybrid`'s HybridEngine the implementation paths are:

| Method                               | What patches                                                                                                                | Cost                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `updatePrimaryLight(dir, intensity)` | Re-bake `EmitterTri.Le` for all emitters tagged primary; rewrite `ReSTIR` shading UBO field; rewrite probe-update UBO field | O(primaryEmitterCount) + 2 buffer writes          |
| `updateSkyDome(tint, irradiance)`    | Rewrite ReSTIR shading UBO field + probe-update UBO field                                                                   | 2 buffer writes                                   |
| `updateLights(lights)`               | Rewrite probe-update lights UBO; if count change exceeds capacity, fall back to reset()                                     | 1 buffer write if count stable; else full rebuild |

`@vitrum/pt-webgl` reports `supportsIncrementalLighting: false` initially. (PT engines rebuild the BVH on `setScene` anyway, so the host can use `setScene` to push lighting changes; an incremental path is a future optimization.)

---

## 4. Why this is a contract addition, not a capability flag elaboration

The current contract has `setScene` (heavy, full rebuild) and `renderFrame` (per-frame). There is no middle tier. Adding `updateLighting` opens that middle tier — patches that are heavier than a frame uniform write but lighter than a full scene rebuild.

The pattern mirrors `Engine.updatePrimitive?` / `Engine.updateEmitter?` already in the contract — those are scene-level incremental patches gated by `supportsIncrementalScene`. The lighting variant is gated by `supportsIncrementalLighting` and has the same fallback-to-setScene-or-reset semantics for engines that don't support it.

---

## 5. Consumer-side benefit

stainedGlass's `useVitrumWalkaroundEngine` would drop its 8-slot time-of-day quantization workaround. The slider would update sun position smoothly at 60 FPS (4-byte UBO write + Le re-bake of ~1-10 emitter triangles) instead of triggering a ~500ms pipeline rebuild every 3 simulated hours.

Fixture-toggle latency would drop from ~500ms (pipeline rebuild) to single-frame.

---

## 6. Open questions for vitrum design review

1. Should `updateLighting` be a single batch method (option D in §2), separate methods (options A–C), or both? Recommendation: both — atomic batch for the common "time-of-day scrub" case, plus single methods for finer-grained updates.

2. What's the upper bound on lights-count change before fallback? Should it be exposed as a capability number, or just documented? Recommendation: a numeric capability — `EngineCapabilities.incrementalLightsCountTolerance: number`.

3. How does this interact with the existing `pipelineRebuildKey` signal? Recommendation: `pipelineRebuildKey` becomes a host-level "full reset" lever (e.g., on Canvas remount); incremental updates use these new methods. No semantic conflict.

---

## 7. Acceptance criteria

- `HybridEngine` implements the four new methods; capabilities reports `supportsIncrementalLighting: true`.
- Unit test: a contrived host loop calls `updatePrimaryLight` 60 times in 1 second; engine state stays `ready` throughout; frame rate stays within 10% of baseline.
- Unit test: `updateLights` with same-count list → no `engine.reset()` call. With +1 light beyond capacity → falls back to `reset()`.
- Documentation: `HybridEngine.ts` JSDoc on each new method links back to this RFE.
- Consumer-side: `useVitrumWalkaroundEngine.ts` in stainedGlass removes its time-of-day slot quantization in a follow-up PR.

---

## 8. References

- `~/projects/stainedGlass/packages/app/src/rendering/vitrum-bridge/useVitrumWalkaroundEngine.ts` (lines 33-39): the consumer-side documented workaround that this RFE retires.
- `~/projects/vitrum/packages/walkaround-hybrid/src/HybridEngine.ts` (lines 90-103, 1018-1059): current creation-time lighting wiring.
- `~/projects/vitrum/packages/core/src/engine.ts` (lines 95-110): existing capability-flag pattern for `supportsIncrementalScene`.
