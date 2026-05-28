> ARCHIVED 2026-05-28 — signoff complete; living maturity record is plan/backend-maturity-matrix-2026-05-26.md.

# Backend Maturity Sweep — Signoff (2026-05-26)

Technical sweep completion record for the single-program backend maturity work.
Distribution and release governance are explicitly out of scope.

## Wave completion

| Wave | Status | Evidence |
|------|--------|----------|
| W0 — Baseline harness | **Complete** | `npm run baseline:wave0`; reports under `tools/benchmark-runner/results/wave0/` |
| W1 — Contract + shared foundation | **Complete** | Branded `FrameOutput` / `Mat4`; `ForkAccess`; shared WGSL (`pcg`, `luminance`, octahedral); E1 `reuseSharedWebGpuDevice` opt-in |
| W2 — TLAS runtime (pt-webgpu) | **Complete** | GPU bindings 24–28; `traceTlasClosest` / `traceTlasAny`; transform refit fast path; tests in `wgslContract.test.ts`, `updatePrimitiveIncremental.test.ts`, `scenePack.test.ts` |
| W3A — pt-webgpu parity | **Complete** | Audit findings closed; telemetry (`onFrame`/`onProgress`); convergence short-circuit; experimental terminology |
| W3B — Walkaround stabilization | **Complete** | PPG uses kd-tree traversal (no linear cell scan); canonical oct/luminance in RC + hybrid common |
| W4 — Reliability hardening | **Complete** | `run-lifecycle-soak.mjs`, `run-wave4-hardening.mjs`, `npm run hardening:wave4` |
| W5 — Consolidation | **Complete** | `plan/backend-maturity-matrix-2026-05-26.md`, README/AGENTS/CLAUDE/library-architecture alignment |

## Verification evidence (HEAD `5312e00`, 2026-05-26)

| Gate | Result | Artifact |
|------|--------|----------|
| `npm run verify:mechanical` | **pass** | ~566s; all workspace tests + shader-compile-ci (7 scenarios) |
| `npm run baseline:wave0` | **pass** | `tools/benchmark-runner/results/wave0/wave0-baseline-2026-05-26T09-37-47-689Z.json` |
| `VITRUM_WAVE4_SKIP_MECHANICAL=1 npm run hardening:wave4` | **pass** | `tools/benchmark-runner/results/wave4/wave4-hardening-2026-05-26T09-50-38-707Z.json` (lifecycle soak: 4 iterations, 0 failures) |

Re-run commands:

```bash
npm run verify:mechanical
npm run typecheck --workspace @vitrum/benchmark-runner
npm test --workspace @vitrum/benchmark-runner
npm run baseline:wave0
VITRUM_WAVE4_SKIP_MECHANICAL=1 npm run hardening:wave4
```

Strict lifecycle soak (with auto-started dev server):

```bash
VITRUM_LIFECYCLE_SOAK_START_SERVER=1 VITRUM_LIFECYCLE_SOAK_STRICT=1 \
  VITRUM_LIFECYCLE_SOAK_ITERATIONS=4 npm run benchmark:lifecycle-soak --workspace @vitrum/benchmark-runner
```

## Residual follow-ups (not sweep blockers)

These are multi-week or host-workflow items, not regressions in the shipped sweep scope:

- **C2 TLAS in all hosts** — hybrid ReSTIR/DDGI/RC shipped (`main` 2026-05-27); pt-webgl remains merged-BVH
- **GPU reference captures** — formal A/B for BDPT/layered BSDF acceptance (host workflow)
- **pt-webgpu** — spectral / hero-MIS / denoiser parity vs pt-webgl (feature roadmap)

## Audit gate

Final `/audit` on this revision: **no blocking findings** (benchmark-runner env parsing, docs, and gate scripts verified).
