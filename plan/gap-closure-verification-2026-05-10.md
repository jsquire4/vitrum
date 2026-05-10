# Gap Closure Verification Report (2026-05-10)

## Mechanical Checks (Executed)

- `npm test --workspace @vitrum/pt-webgl`: PASS.
- `npm test --workspace @vitrum/pt-webgpu`: PASS.
- `npm run typecheck` (workspace): PASS.
- `node ./scripts/shader-smoke-check.js` (`three-gpu-pathtracer` fork): PASS.
- `npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner`: PASS (artifact generated, fail-closed `BLOCKED` statuses without GPU harness).

## Code Closure Summary (This Execution Wave)

- `pt-webgl` now reports selected `causticStrategy` in capabilities and forwards strategy controls through `forkUniformBridge`.
- Fork shader now implements deterministic mode-distinct caustic branches with bounded loops:
  - `manifold-nee`: refractive-chain walk with iteration + chain caps.
  - `photon-map`: deterministic refracted-cone density estimate.
- Fork regression guard added:
  - `scripts/check-caustic-strategy-regression.js`
  - script ensures required uniforms/branches exist and legacy ad-hoc gains are absent.
- `pt-webgpu` now uses a bounded rich material payload contract (20 vec4 per material):
  - thin-film: up to 8 per-layer `(ior, thicknessNm)` pairs,
  - spectral attenuation: fixed 32-sample grid + metadata,
  - layered/scattering fields preserved.
- WGSL now consumes per-layer/per-sample payload directly and re-enables strategy-specific caustic paths.
- New/updated tests cover payload contract + strategy capability signaling:
  - `scenePack.test.ts`, `factoryCapabilities.test.ts`, `wgslContract.test.ts`.

## Runtime / GPU Artifact Matrix

The required deterministic scenario set is defined in `plan/gap-closure-acceptance-matrix.md`.

Runtime execution attempts performed in this wave:

- Fork dependencies installed (`npm install`) to enable browser capture scripts.
- Fork strategy regression check executed (`npm run check-caustic-strategy`): PASS.
- Fork screenshot harness attempted:
  - `node ./scripts/update-screenshots.js -o ./screenshots/current -s khronos-DragonAttenuation -h true`
  - result: `ERR_CONNECTION_REFUSED` due harness expecting `localhost:1234` while current Vite server is `localhost:5173`.
- No vitrum-native GPU capture runner currently maps acceptance-matrix scenario IDs to deterministic render jobs in this workspace.

Deterministic artifact manifest for the acceptance matrix:

- `plan/gap-closure-artifacts-2026-05-10.json`
- Status: `blocked` for all scenarios in this environment (null hash/perf fields, blocker recorded).

| Scenario ID | Status | Evidence collected now | Remaining evidence |
|---|---|---|---|
| `rfe03-layered-front-back` | Blocked in this environment | Build/test/typecheck pass | Before/after image hashes + delta/perf |
| `rfe07-11-sss-mixed-panels` | Blocked in this environment | Build/test/typecheck pass | Before/after image hashes + delta/perf |
| `rfe08-13-spectral-payload` | Blocked in this environment | Build/test/typecheck pass | Before/after image hashes + delta/perf |
| `rfe14-thinfilm-angle-shift` | Blocked in this environment | Build/test/typecheck pass | Before/after image hashes + delta/perf |
| `rfe09-bridge-global-cmf` | Blocked in this environment | Unit tests + build pass | Uniform runtime snapshots + A/B |
| `rfe05-caustic-strategy` | Blocked in this environment | Strategy wiring + mode branches + tests present | 3-mode captures + perf |
| `ptwgpu-parity-material-fields` | Blocked in this environment | Rich payload + shader consumption + tests | A/B hashes + perf |

## Artifact Bundle Format (to be populated during GPU run)

For each scenario:

1. `scenarioId`
2. `seed`
3. `settings` (`resolution`, `bounces`, `spp`)
4. `beforeImageHash`
5. `afterImageHash`
6. `deltaSummary`
7. `perfBaselineMsPerSample`
8. `perfCandidateMsPerSample`
9. `passFail`
