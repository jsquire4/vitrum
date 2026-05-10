# Sprint 1 host-app checklist

> Items in this file are HOST-APP concerns. They require changes in the host
> application's React glue, NOT in the `@vitrum/*` library packages.
> The vitrum-side Sprint 1 deliverable (PT_PREVIEW_BOUNCES = 3) is complete.

---

## H1 — Fix outdoor HDRI preset 404s

**File**: `outdoorScenePresets.ts` (or equivalent file that declares HDRI URLs)

**Change**: Correct the HDRI URL paths for the four outdoor presets so the browser
network panel shows 200, not 404. Likely a base-path or public-folder mismatch.

**Verify**: Open DevTools → Network tab → filter by `.hdr` or `.exr` → confirm 200.

---

## H2 — PT_PREVIEW renders at 0.5× DPR

**File**: `PathTracingLayer.tsx` (or equivalent component that creates/configures the pathtracer)

**Change**: Pass `resolutionFactor: 0.5` (or equivalent) to the pathtracer constructor so
preview renders at half the device pixel ratio.

**Verify**: `pathtracer._pathTracer.target.width` should be `Math.floor(canvas.width / 2)`.

---

## H3 — Skip EffectComposer for first 8 samples after each reset

**File**: `PTPostProcessing.tsx` (or equivalent post-process integration)

**Change**: Gate EffectComposer rendering on `pathtracer.samples > 8`. When `samples <= 8`,
output the raw path-traced buffer directly without post-processing.

**Verify**: Frame-time profile shows no EffectComposer cost for the first ~8 frames after
a camera move. Confirmed via `performance.mark()` or DevTools Rendering tab.

---

## H4 — OrbitControls damping factor = 0.15

**File**: `StageOrbitControls.tsx` (or wherever `OrbitControls` is instantiated)

**Change**: Set `controls.dampingFactor = 0.15` (was probably 0.05 or default 0.25).
Also confirm `controls.enableDamping = true`.

**Verify**: After a drag release, the camera should glide briefly then settle within
~5–8 frames, not oscillate or snap instantly.

---

## Notes for the host integrator

- Items H2 and H3 (DPR halving + EffectComposer skip) together are the largest
  interactive framerate wins: expect ~2–3× faster preview at the cost of slightly
  softer image during scrubbing. PT_FINAL is unaffected (DPR = 1.0, full post).
- `PT_PREVIEW_BOUNCES = 3` is now in `@vitrum/pt-webgl/src/constants.ts`. If the
  host app previously hard-coded `bounces: 5` inline, remove that and import
  `PT_PREVIEW_OPTIONS` from `@vitrum/pt-webgl` instead.
- The host-side `PT_BOUNCES` import still works (backward-compat alias for
  `PT_PREVIEW_BOUNCES`), but the deprecated JSDoc comment will appear in IDEs.
  Migrate to `PT_PREVIEW_OPTIONS.bounces` for clarity.
