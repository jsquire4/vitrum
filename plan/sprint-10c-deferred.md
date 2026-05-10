# Sprint 10c — BDPT for Caustics (DEFERRED)

**Status**: Deferred — trigger criterion not yet evaluable.
**Created**: 2026-05-09
**Source**: `plan/phase-6-roadmap.md` §5, Sprint 10c

---

## Goal

Bidirectional path tracing for true caustic convergence in PT_FINAL mode.
Light-side rays trace from each emitter up to N=3 bounces and store vertices
in a per-frame buffer; eye-subpath connection attempts connections to every
stored light vertex with joint PDF + MIS weighting.

**Mode scope**: PT final only.

---

## Trigger criterion

> "Ship ONLY IF Sprint 7 (volume + equi-angular) doesn't visually close the
> caustic-convergence gap. Re-evaluate after Sprint 7 with a hero-render
> side-by-side comparison."

Per the open questions section (§7, item 2) of the roadmap:

> Suggested operational definition: hero render of a panel-floor caustic at
> 256 samples PT_FINAL; if floor caustic noise SD > X, BDPT triggers. Define X
> at re-evaluation time.

---

## Why it is deferred

Sprint 10c requires GPU verification to evaluate its trigger criterion.
Specifically:

1. Sprint 7 (volume + equi-angular PDF) must be **applied to the fork** and
   **rendered** before BDPT's trigger can be assessed.
2. The threshold value X for "floor caustic noise SD" cannot be established
   without capturing a reference render and measuring per-pixel standard
   deviation in the caustic region.
3. The autonomous-mode session that completed vitrum-side library work through
   Sprint 11 had no WebGPU or WebGL rendering environment available. The trigger
   condition ("does Sprint 7 visually close the caustic gap?") is structurally
   unanswerable without rendering.

Decision 3 from the roadmap also fixes the BDPT variant: **vanilla BDPT**,
NOT ReSTIR BDPT. ReSTIR BDPT (Hedstrom 2025) was evaluated and rejected on
2026-05-09 — it is hardwired to DirectX 12 + DXR + Falcor + RTX hardware.
See Decision 8 in the decision log for the full citation and porting assessment.

---

## How to un-defer

1. Acquire GPU verification capacity (browser with WebGL2 PT pipeline running).
2. Apply Sprint 2 through Sprint 7 fork patches in order (see
   `plan/sprint-N-pt-fork-patch.md` docs for each sprint).
3. Run `npm run build` in `~/projects/three-gpu-pathtracer/`.
4. Load the stained-glass reference scene; render to 256 samples in PT_FINAL.
5. Capture a screenshot. Measure per-pixel noise standard deviation in the
   floor caustic region (any spectral analysis tool that can read a PNG/EXR).
6. If the floor caustic noise SD is visually prominent (the specific threshold
   value X is to be defined at this re-evaluation step — the roadmap intentionally
   deferred pinning it until a real render is in hand), **schedule Sprint 10c**.
7. If noise SD is acceptable (floor caustic reads as converged at 256 samples),
   **close 10c permanently** — Sprint 7 was sufficient.

---

## Implementation shape (if triggered)

From the roadmap Definition of Done (verbatim):

> - Light subpath kernel: traces from each emitter up to N=3 bounces, stores
>   vertices in MRT/SSBO
> - Eye-subpath connection routine: at each eye vertex, attempt connections to
>   every stored light vertex, evaluate joint PDF + MIS weight
> - Visual A/B: floor caustic from sun-through-panel converges at ~256 samples
>   in PT_FINAL vs. ~1024+ samples for pure NEE

**Fork files affected**:
- `src/shader/shaders/pathtracing/path_tracer.glsl.js` — light subpath kernel
  inline or via new `light_path_tracer.glsl.js`
- `src/shader/shaders/pathtracing/direct_lighting.glsl.js` — eye-subpath
  connection routine + MIS weight evaluation
- `PhysicalPathTracingMaterial.js` — new uniforms for light-path vertex storage
  (MRT or texture ping-pong)

**WebGL2 constraint**: WebGL2 lacks compute shaders. Light-path vertex storage
must use MRT (multiple render targets written from a fragment shader "trace"
pass), or a texture ping-pong approach where the CPU alternates draw calls.
This is the primary implementation risk.

---

## Risk callouts

From the roadmap:

> Highest of any frontier item. WebGL2 lacks compute shaders → vertex storage
> requires creative MRT or texture ping-pong. Budget 2 weeks minimum. Defer
> until Sprint 7 ships and gap is visually re-assessed.

Minimum effort estimate: **8–10 days** after trigger criterion is met.
Realistic budget including WebGL2-MRT vertex storage complexity: **2 weeks**.
