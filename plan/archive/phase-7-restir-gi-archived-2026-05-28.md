> ARCHIVED 2026-05-28 — SHIPPED (Sprints 15–18). See CLAUDE.md 'what's done'.

# Phase 7 — ReSTIR-GI + Short-Range Screen-Space AO

**Status**: Shipped (Sprints 15–18 on main, 2026-05-17+)
**Replaces**: The currently-disabled `Lo_ddgi * 0.0` term in `shade.wgsl`
**Created**: 2026-05-11
**Last Sprint**: Sprint 14 (layered-bsdf fork patch — PT-side; walkaround's last was Sprint 13 neural). This phase introduces Sprints 15–18.

## 0. Strategic frame

ReSTIR-GI treats each DDGI probe-octahedral-direction pair as a virtual-light sample for RIS. Per-pixel reservoir picks ONE such sample, then temporal + spatial reuse amortise across frames + neighbours. The reconstruction moves from probe-grid space (cells visible on smooth walls) to screen space (per-pixel, no grid).

DDGI atlas update stays unchanged; it becomes a sample-distribution prior, not a per-pixel basis. Existing ReSTIR-DI machinery (RIS / temporal / spatial passes + reservoir ping-pong) is reused — adds a second reservoir buffer for indirect, not a parallel pipeline.

## Sprint 15 — Short-range screen-space GTAO (1.5–2 days)

Half-res GTAO horizon-based AO, bilateral upsample, multiplied into the diffuse term before SVGF welford.

**New WGSL**: `gtao.wgsl.ts`, `gtaoUpsample.wgsl.ts`
**Pipeline**: `_dispatchGTAO` inserted between shade and welford-temporal
**Resources**: `aoHalfTexture` (W/2 × H/2 r16float), `aoFullTexture` (W × H r16float)
**Tests**: `__tests__/sprint15-gtao.test.ts`
**Reference renders**: Cornell + glass studio before/after
**Success**: contact shadows 4–10% darker, no ringing, <2 ms GPU
**Independent of GI work**; can ship first.

## Sprint 16 — ReSTIR-GI RIS (3–4 days)

Per-pixel RIS over M=8 probe-direction candidates. Half-res reservoir buffer. Single-frame; noisy but verifies the path.

**New WGSL**: `risGi.wgsl.ts`
**Pipeline**: `_dispatchGiRis` inserted between spatial-2 (DI) and shade. Workgroup 8×8, half-res dispatch.
**Resources**: `reservoirGiCurrent`, `reservoirGiPrevious`, `reservoirGiSpatial` (W/2 × H/2 × 64 B each = ~58 MB each, ~173 MB total)
**Shade change**: replace `Lo_ddgi * 0.0` with `Lo_indirect` read from `reservoirGiCurrent` × albedo × INV_PI × W
**Tests**: `__tests__/sprint16-restirGi-ris.test.ts` (WGSL content + ReservoirGI byte-equiv + pass layout)
**Reference renders**: Cornell + glass studio before/after
**Success**: cell-grid GONE, color bleed visible (noisy), <6 ms GPU

## Sprint 17 — ReSTIR-GI temporal + spatial reuse (4–5 days)

Two passes per frame: reproject prev reservoir, M-clamp at 20; two spatial passes with reconnection-shift jacobian correction.

**New WGSL**: `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`
**Pipeline**: 3 new dispatches after `gi-ris`. End-of-frame copy `reservoirGiSpatial → reservoirGiPrevious`
**Shade change**: read from `reservoirGiSpatial` not `reservoirGiCurrent`
**Tests**: `__tests__/sprint17-restirGi-temporal-spatial.test.ts` + integration test
**Reference renders**: Cornell + glass studio + (optional) interior
**Success**: stddev drop ≥ 5× vs Sprint 16; no cell-grid; <15 ms total GI cost

## Sprint 18 — Per-channel SVGF tuning (2–3 days)

Split SVGF: run twice per frame, once each on direct and indirect with different sigmas. Indirect blurs more aggressively (broader kernel, already temporally smooth from ReSTIR-GI).

**New WGSL**: `indirectAccumulator.wgsl.ts`, combine pass
**Pipeline**: doubled SVGF chain (welford → variance → atrous ×5) for each channel, combine before temporalAccum
**Sigmas**: direct (1.0/128/1.0), indirect (4.0/64/2.0)
**Tests**: `__tests__/sprint18-svgf-perchannel.test.ts`
**Reference renders**: all three scenes
**Success**: direct shadows stay sharp ±5%, indirect smooth, <2 ms additional

## Sequencing

```
Sprint 15 (AO, 1.5–2 d) ┐ independent
Sprint 16 (RIS-GI, 3–4 d) → Sprint 17 (temporal+spatial, 4–5 d) → Sprint 18 (SVGF split, 2–3 d)
```

Total wall-clock: **11–14 days** one developer.

## Cross-cutting

- **Backwards compat**: `restirGiWired` UBO bit defaults 0 → today's render byte-for-byte. New `HybridEngine.setReSTIRGiEnabled(on)` for opt-in.
- **DDGI new role**: producer unchanged; consumer is only the RIS-GI pass (reads `ddgiSampleFromBindings` at reconnection vertices, where 24-in cells are invisible due to perspective).
- **Memory**: +173 MB GI reservoirs at 2688×1344.
- **Testing**: per CLAUDE.md, every sprint captures before/after reference renders against `tools/reference-renders/baseline/`.

## Definition of done

- [ ] DDGI cell-grid GONE on smooth Cornell walls
- [ ] Color bleed visible + stable in Cornell
- [ ] No regression in outdoor scene caustics
- [ ] All existing tests pass
- [ ] 4 new sprint test files green
- [ ] Frame time <40 ms at 2688×1344
- [ ] `setReSTIRGiEnabled(false)` reverts byte-for-byte

## Critical files

- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` — ReservoirGI struct, RIS-GI helpers
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts` — `Lo_indirect` term, channel separation
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` — all `_dispatchGi*`, AO dispatch, ping-pong
- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts` — three GI reservoirs + AO + indirect textures
- `packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts` — 7 new pipelines (gtao, gtao-upsample, gi-ris, gi-temporal, gi-spatial, indirect-accum, combine)
