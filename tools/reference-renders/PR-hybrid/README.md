# PR-hybrid reference captures

Primary-release hybrid benchmarks (`plan/primary-release-and-webgpu-pt-parity-2026-05-26.md`).

**Status: archive/provenance set.** The old PR-hybrid benchmark scripts were
removed during the reference-render tooling cleanup. Keep these PNGs and
`manifest.json` as historical WSL-GPU/dzn captures; use `npm run validate:gpu:smoke`
for the current hybrid GPU smoke and the `benchmark:gap-closure` capture-adapter
flow for new visual captures.

## Scenarios

| Directory | Source | Capture |
|-----------|--------|---------|
| `tlas-on/` | Cornell 2-mesh TLAS vs merged A/B | `walkaround.html?scene=tlas10inst&bvhMode=tlas` |
| `material-edit/` | Material churn still frame | `PR-hybrid-material-churn` bench |
| `200k-static/` | ~200k tri bench frame | `PR-hybrid-200k-static` bench |

## Adapter requirements

`walkaround-hybrid` needs **≥16** storage buffers and **≥8** storage textures per shader stage.
After a successful GPU bench, perf JSON is copied to `perf/latest.json` and merged into `manifest.json`.

## Current validation path

- GPU smoke / traversal oracles: `npm run validate:gpu:smoke`.
- New capture sessions: follow `tools/reference-renders/README.md` with a running
  capture-capable example and `capture-adapter-playwright.mjs`.

Manual review is still required before treating new PNGs as golden references.
