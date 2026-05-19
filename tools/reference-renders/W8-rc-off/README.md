# W8 — Radiance Cascades reference renders (rcEnabled: false baseline)

This directory holds Cornell-box captures from a HybridEngine with
`rcEnabled: false` (i.e., the pre-W8 indirect path: ReSTIR-GI only).

Pairs with `tools/reference-renders/W8-rc-on/` which captures the same
scene + same frame seeds with `rcEnabled: true, rcWeight: 1.0` (pure RC).

## Capture protocol

When `tools/benchmark-runner/` grows an `rc-acceptance` mode (Phase 4
follow-up), it should produce two files:

- `cornell-box-1080p-32spp.png` — 1080p Cornell with the standard
  directional sun + interior emitters, 32 frames accumulated.
- `cornell-box-indirect-1080p-32spp.png` — same render with the indirect
  texture isolated (no `Lo_direct`/`Lo_emit` contribution) so the RC vs
  ReSTIR-GI difference is visible on the floor/walls.

## A/B procedure

The `rcAcceptance.gpu.test.ts` acceptance test (gated by
`VITRUM_RC_ACCEPTANCE=1`) is the authoritative numerical check. The
visual diff between this directory and `W8-rc-on/` is the human-eye
sanity check.

## Status

- 2026-05-18: directory seeded. Captures pending the benchmark-runner
  harness wire (see `plan/w8-rc-mis-composition.md` Phase 4).
