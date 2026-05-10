# Sprint 13 — Custom WebGPU Neural Denoiser (DEFERRED)

**Status**: Deferred — trigger conditions not evaluable and likely not met.
**Created**: 2026-05-09
**Source**: `plan/phase-6-roadmap.md` §5, Sprint 13

---

## Goal

Production-grade real-time neural denoising for the walkaround engine. Custom
UNet-style CNN, trained on offline reference data, running entirely on WebGPU
compute shaders. Sprint 10b's OIDN-via-ONNX handles the PT_FINAL denoising use
case; Sprint 13 targets the walkaround real-time path.

**Mode scope**: walkaround only.

---

## Trigger criteria (all three must hold)

From Decision 14 and §5 of the roadmap — Sprint 13 ships ONLY IF:

1. **SVGF (Sprint 10a) leaves a visible quality gap** — i.e., SVGF-denoised
   walkaround output at 8 SPP is noticeably noisier than OIDN-class denoising
   in subjective A/B comparison.
2. **WebNN API is still 12+ months from production-ready** — WebNN is the
   browser standard for near-native ML inference. If it ships on all three
   major browsers (Chrome, Firefox, Safari) before Sprint 13 kicks off, the
   correct move is to route through WebNN rather than implement a custom
   training + inference pipeline.
3. **PPG (Sprint 11) shipped and did NOT close the noise gap on its own** —
   PPG path guiding targets ~2–3× sample efficiency improvement; if that
   improvement combined with SVGF is sufficient, Sprint 13 is unnecessary.

---

## Why it is deferred

Multiple independent blockers:

1. **GPU verification unavailable**: Trigger criterion (1) requires seeing
   SVGF walkaround output in a live browser and judging whether a gap remains.
   The autonomous-mode session that completed vitrum-side library work had no
   WebGPU rendering environment.

2. **WebNN production-readiness is a 2026+ tracking question**: As of
   2026-05-09, WebNN is behind a flag in Chrome/Edge and not shipping in
   Firefox or Safari. The correct trigger-evaluation window is "check again
   in 6–12 months." This assessment cannot be made once; it requires tracking
   the W3C WebNN spec milestone timeline.

3. **PPG is scaffolded but not GPU-dispatched**: Sprint 11 PPG shaders and
   buffers are authored and allocated, but the dispatch path is deferred pending
   GPU verification (see `plan/sprint-11-ppg-integration.md`). Trigger
   criterion (3) cannot be evaluated until PPG runs end-to-end and its noise
   reduction is measured.

4. **Sprint 10a SVGF is similarly deferred**: SVGF is authored but not
   dispatched in walkaround (see `plan/sprint-10a-walkaround-integration.md`).
   Criterion (1) therefore cannot be evaluated.

---

## How to un-defer

Assess all three trigger criteria in order:

1. Apply Sprint 10a SVGF walkaround wiring per
   `plan/sprint-10a-walkaround-integration.md`. Render at 8 SPP.
   If SVGF output is already acceptable (no visible grain in static
   diffuse surfaces), **close Sprint 13 permanently**.

2. Apply Sprint 11 PPG dispatch wiring per `plan/sprint-11-ppg-integration.md`.
   After PPG warm-up (~500 frames), re-evaluate the 8 SPP noise level.
   If PPG + SVGF together close the gap, **close Sprint 13 permanently**.

3. Check WebNN production-readiness at that time. If WebNN is shipping across
   browsers without a flag, the inference path via ORT-Web with the `webnn`
   execution provider (already configured in the Sprint 10b OIDN bridge) may
   extend to real-time use — evaluate before committing to a custom training
   pipeline.

4. If AND ONLY IF all three criteria still hold after the above, schedule the
   6–8 week Sprint 13 implementation.

---

## Implementation shape (if triggered)

From the roadmap Definition of Done (verbatim):

> - Training pipeline (Python/PyTorch) generates reference noisy/clean pairs
>   from offline path-tracer + walkaround output
> - Model architecture: UNet, ~1–3 MB weights, trained on ~10K image pairs
> - Inference graph: WebGPU compute shaders for conv2d, transposed conv2d,
>   ReLU, skip connections
> - ~10–50 ms inference time per frame on typical GPU
> - Visual A/B: real-time-denoised output indistinguishable from offline OIDN
>   at 4 samples-per-pixel

**Infrastructure required**:
- Python training environment (PyTorch, not part of the vitrum repo)
- Reference render pairs from the stained-glass scene (walkaround noisy + PT_FINAL clean)
- WebGPU compute shader kernels for conv2d, transposed conv2d, ReLU, skip
  connections — these do not exist anywhere in the vitrum codebase today

---

## Bail-out criterion

From the roadmap:

> "Bail-out criterion: if month-1 inference benchmarks don't hit <50 ms,
> abort and wait for WebNN."

The <50 ms threshold is at typical GPU (desktop discrete). On integrated GPU or
mobile, expect 2–5× higher inference time. The bail-out check should use the
reference GPU that the walkaround engine targets.

The question of whether to tighten the <50 ms threshold (roadmap §7, item 5)
should be answered at trigger time, not now.

---

## Risk callouts

From the roadmap:

> Research project, not feature work. Budget 8 weeks minimum.

This is the longest and most uncertain item in the entire Phase 6 plan.
Both the training pipeline and the WebGPU inference implementation are novel
engineering work with no prior vitrum code to build on. The bail-out criterion
exists precisely because the risk is high — use it.
