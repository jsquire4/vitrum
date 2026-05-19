# W8 — Radiance Cascades reference renders (rcEnabled: true, rcWeight: 1.0)

This directory holds Cornell-box captures from a HybridEngine with RC at
full MIS weight (pure RC contribution, no ReSTIR-GI in `Lo_indirect`).

Pairs with `tools/reference-renders/W8-rc-off/` (the baseline). Visual
A/B between the two directories is the human-eye sanity check that the
W8 Phase 3 wiring lifted `sampleCascadeC0` into the shade pass.

## Capture protocol

See sibling `W8-rc-off/README.md` for the shared procedure. The only
config delta is:

```ts
new HybridEngine({
  …,
  rcEnabled: true,
  rcWeight:  1.0,
});
```

`rcWeight: 1.0` zeroes ReSTIR-GI in `Lo_indirect = wRestirGi · Lo_restirGi
+ wRc · Lo_rc` so the indirect channel is RC-only — the cleanest test of
whether the cascade-0 buffer is producing meaningful radiance.

`rcWeight: 0.5` would be the production tuning (equal-weight mix); the
`W8-rc-on` capture intentionally cranks RC to maximum to make the diff
unambiguous.

## Status

- 2026-05-18: directory seeded. Captures pending the benchmark-runner
  harness wire (see `plan/w8-rc-mis-composition.md` Phase 4).
