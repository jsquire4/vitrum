# PR-hybrid reference captures

Primary-release hybrid benchmarks (`plan/primary-release-and-webgpu-pt-parity-2026-05-26.md`).

## Scenarios

| Directory | Source | Capture |
|-----------|--------|---------|
| `tlas-on/` | Cornell 2-mesh TLAS vs merged A/B | `walkaround.html?scene=tlas10inst&bvhMode=tlas` |
| `material-edit/` | Material churn still frame | `PR-hybrid-material-churn` bench |
| `200k-static/` | ~200k tri bench frame | `PR-hybrid-200k-static` bench |

## Adapter requirements

`walkaround-hybrid` needs **≥16** storage buffers and **≥8** storage textures per shader stage.
SwiftShader (10 / 4) cannot run hybrid; `npm run benchmark:pr-hybrid` **skips** GPU scenarios on
such hosts unless `VITRUM_PR_REQUIRE_GPU=1` (then the run fails). Use `npm run benchmark:pr-mechanical` on CPU-only CI.

## Capture workflow (GPU host)

```bash
# Terminal A
npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host 127.0.0.1 --port 5175

# Terminal B — records JSON + optional canvas PNG from bench harness
VITRUM_PR_START_SERVER=1 VITRUM_PR_SCENARIO=PR-hybrid-200k-static npm run benchmark:pr-hybrid
```

Automated smoke capture:

```bash
npm run benchmark:pr-hybrid-refs --workspace @vitrum/benchmark-runner
```

Writes `tlas-on/`, `material-edit/`, `200k-static/` PNGs + `manifest.json` (SHA-256 per file).
Manual review still recommended before treating as golden references.
