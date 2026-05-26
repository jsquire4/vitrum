# Backend Maturity Matrix

Date: 2026-05-26 (sweep signoff: [backend-maturity-sweep-signoff-2026-05-26.md](./backend-maturity-sweep-signoff-2026-05-26.md))

This matrix tracks technical maturity only (code, pipeline wiring, runtime reliability).
It intentionally excludes release governance and distribution posture.

## Capability Areas

- Contract conformance (`@vitrum/core` types/capabilities honored)
- Incremental scene updates (transform/material/emitter paths)
- Deep pipeline integration (TLAS/BLAS, GI/denoiser composition)
- Lifecycle reliability (pause/resume/reset/resize/dispose churn)
- Regression harness coverage (mechanical + render/perf evidence)

## `@vitrum/pt-webgl`

- Contract conformance: **strong**
- Incremental updates: **strong**
- Deep pipeline integration: **strong** (production renderer path)
- Lifecycle reliability: **strong**
- Regression harness coverage: **strong**
- Current focus: hold stability while shared-contract and hardening work lands.

## `@vitrum/walkaround-hybrid`

- Contract conformance: **strong**
- Incremental updates: **strong** (including primitive/emitter patch APIs)
- Deep pipeline integration: **strong** (DDGI/ReSTIR/GTAO/SVGF/RC composition shipped)
- Lifecycle reliability: **medium-strong** (continue soak evidence expansion)
- Regression harness coverage: **strong**
- Current focus: maintain pass-chain integrity and performance boundedness under stress.

## `@vitrum/pt-webgpu`

- Contract conformance: **medium-strong** (capability and telemetry parity substantially improved)
- Incremental updates: **medium-strong** (transform/material/emitter fast paths with TLAS-aware updates)
- Deep pipeline integration: **medium** (experimental backend, TLAS critical path still being hardened)
- Lifecycle reliability: **medium-strong** (Wave 4 strict soak harness; host cadence still owns pause/resize policy)
- Regression harness coverage: **medium-strong** (expanded WGSL and contract tests; GPU reference captures remain host workflow)
- Current focus: incremental feature parity vs pt-webgl (spectral, denoisers) without destabilizing experimental boundary.

## Shared Layers (`core`, `engine`, `shared-*`)

- Contract conformance: **strong**
- Canonical shared primitives: **strong** (ongoing WGSL dedup enforcement)
- Lifecycle ownership semantics: **strong**
- Regression harness coverage: **strong**
- Current focus: prevent drift as backend-specific hardening continues.

## Program-Level Exit Signal (for this sweep)

**Status: met on 2026-05-26** — see signoff doc for command log.

- Wave 4 hardening gate: `npm run hardening:wave4` (auto-starts dev server for lifecycle soak)
- Wave 0 baseline: `npm run baseline:wave0`
- No unresolved high/medium audit parity items for `@vitrum/pt-webgpu` in `plan/pt-webgpu-deep-audit.md`
- `/audit`: no blocking findings on sweep revision
