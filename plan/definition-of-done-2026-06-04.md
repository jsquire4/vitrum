# Definition of Done — vitrum 95% / 100% program (G0)

Date: **2026-06-04**
Source: the master program handed off 2026-06-04 (5 axes A–E, ~85 work items, 4–6 waves,
dependency graph + suggested team split). This file is **G0** from that program — the
global anchor every other item hangs off. Per-item IDs (A1…E11, G0…G4, S1…S5, T1…T8,
D1…D15, B1…B16, C1…C9) live in the source program; this matrix is the axis-level
contract + the cross-cutting reconciliation.

## Definitions

- **95%** = shippable professional product *inside the monorepo* + a credible external
  API. Honest capabilities, default pass graphs work, the headline algorithms are wired
  into the beauty render (not debug-only), docs match code.
- **100%** = no known *structural* gaps; parity where promised; every opt-in mode has an
  acceptance test; every public union member is implemented or removed.

## Axis-level DoD

| Axis | 95% done when | 100% done when | Primary test command(s) | Owner |
|------|---------------|----------------|-------------------------|-------|
| **A — Realtime GI** (`walkaround-hybrid`) | Default pass graph + denoisers + incremental patches (incl. wholesale→`setScene`) work **without THREE on ingest**; frame budget auto-applies when enabled; GRIS validated on real Dawn (not just lavapipe) | T1–T6 THREE-decouple complete; A5–A13 opt-in modes each have acceptance; no stale comments | `npm test -w @vitrum/walkaround-hybrid`; `wsl-gpu` GRIS hardware smoke | Decouple / Realtime |
| **B — Converged PT** (`pt-webgpu` full tier) | ReSTIR-PT **composited into beauty** + unbiased-enough vs brute-force; `experimental-backend` label dropped; README honest; camera-visible emitters; MNEE directional policy explicit | B4 spatial reuse, B10 incremental add/remove, B14 MNEE multi-reflector/curved-slab, B16 BDPT variance-win demo | `npm test -w @vitrum/pt-webgpu`; `wsl-gpu` A/B harnesses; `plan/renderer-fidelity-matrix.md` | Integrator |
| **C — WebGL PT** (`pt-webgl` + fork) | OIDN 3-buffer (MRT albedo/normal); `supportsAuxBuffers:true`; capabilities honest; fidelity rows promoted **where a GL capture path exists** | C8 instancing parity; C9 publish story executed | `npm test -w @vitrum/pt-webgl`; `npm run fork-shader-smoke` | WebGL |
| **D — Library product** (`core` + `engine`) | Publishable packages (build→`dist`, un-private/scoped); split `@vitrum/engine` entrypoints; docs + examples in CI; `BACKEND_PROMISE_LEDGER` complete | D12–D15 (semver, migration guide, `@vitrum/dev` vanilla stubs, supply-chain notes); D7 default alignment | `npm run typecheck`; `npm test`; `npm pack` (post-D1); headless example smoke | Platform |
| **E — Frontier** (inverse / NRC hero) | Path-replay adjoint for baseColor/roughness/emissive + FD fallback documented; `createInverseSession` end-to-end example | E1–E7 (spot+mesh-area NEE, ≥2-bounce closed-scene, TLAS, ior, emitter params, texture/ssim union members), E9–E10 (NRC terminator, neural weights pipeline) | `wsl-gpu` adjoint/MNEE self-validating harnesses; `npm test -w @vitrum/core` | Research |

## Global items (G0–G4)

| ID | Item | State (2026-06-04) |
|----|------|--------------------|
| G0 | This DoD matrix | **DONE** (this file) |
| G1 | CI gates: typecheck + test + **naga shader-compile on composed WGSL** | OPEN — the naga gate is the load-bearing add (string goldens miss `ptr<storage>` + leaked-UBO-field regressions; cf. the 2026-06-02 `reservoirGi` regression caught only by the GPU smoke). Today it runs in `wsl-gpu` pre-push, not in repo CI. |
| G2 | Extend `BACKEND_PROMISE_LEDGER` — add `supportsProgressiveSeedSource`, `supportsAccumulatorSeed`; document hybrid wholesale-topology = full `setScene` | OPEN — capabilities exist (capabilities.ts:63-76), ledger entries pending |
| G3 | Capability-honesty audit — every `capabilities.*` field pinned; no JSDoc contradicts code | **IN PROGRESS** — B7 pt-webgpu README stale-claim fixed this commit (see Reconciliation); HybridEngine THROW-comment audit pending |
| G4 | Harness discipline — radiometric change needs a deterministic oracle OR byte-identical compose; scene-noise A/B is not a merge gate alone | Effectively in force (the MNEE / adjoint / progressive harnesses are all deterministic-oracle); needs writing into the agent brief |

## Reconciliation — where the code has already outrun the program's premises

Grounded by file reads on 2026-06-04 (not assumed — the scan says where to look, the read says what's true):

- **A6 (skinned-mesh picking) → DONE on the "document" branch.** `pickPrimitive.ts:185` handles
  `skinned-mesh` with an explicit honest comment: *"rest-pose positions; deformation ignored
  (approximate, debug-only)."* The program's "fix OR document" is satisfied.
- **B7 (pt-webgpu README "experimental / no GPU-verified visual reference") → STALE; FIXED this commit.**
  The README claimed *"No GPU-verified visual reference… no end-to-end visual parity check exists,"*
  but `plan/renderer-fidelity-matrix.md` now lists **9 `supported` rendering rows** with committed,
  sha-pinned dzn (RTX 4090) baseline PNGs (promoted `3c58f39`). Updated to reflect that; the genuinely-
  true *"No texture maps"* limitation (README:114) is kept.
- **B6 (drop `experimental-backend`) → blocked on B1, NOT on fidelity.** The 9 *rendering* rows are
  `supported`; the README's remaining "experimental" framing is about *productionisation / API
  stability*, which is legitimate pre-publish. The label drops once ReSTIR-PT composites into beauty
  (B1) and the API stabilises — not a fidelity gap.
- **C6 (promote pt-webgl fidelity rows) → correctly OPEN; premise unmet.** The pt-webgl fork rows
  stay `experimental` because **no WebGL2 capture path exists** (lavapipe is a WebGPU device; V9
  Jakob-Hanika spectral has 59–72% out-of-gamut negatives at software-GL spp). C6's 95% bar is
  really "build the GL capture path *first*" — it cannot fire on evidence that does not yet exist.
- **D8 (createProgressiveEngine options surface) → DONE.** Verified by read: limit-union is correct
  (per-key `Math.max` of both backends' advertised floors), device is leak-safe (every post-acquire
  throw hits the `catch`→`device.destroy()`), capability preflight asserts seed source+sink, `dispose`
  idempotent. The program's own note ("already good; document") holds.
- **D8-adjacent LATENT (logged, low-priority, pt-webgpu-scoped — NOT a facade bug):**
  `computeProgressiveLimitUnion()` maps pt-webgpu's **per-group** peak (`…_PER_GROUP = 10`) onto the
  **per-stage** device key `maxStorageBuffersPerShaderStage`, while the true per-stage count is ~26
  (webgpuLimits.ts:16-17). This is *identical* to what standalone `ptWebgpuRequiredLimitsForAdapter`
  requests (also 10), and both work on every real discrete GPU because Dawn exposes the adapter max
  (≥64), not the requested floor — that's why the 9 fidelity rows AND the progressive e2e (0 errors)
  all pass. It would only bite if a browser ever returns *exactly-requested* limits. Hardening fix
  (request the true ~26 per-stage) belongs to pt-webgpu, affects standalone use equally, and is not a
  Track A regression.
- **B2 (ReSTIR-PT reuse unbiasedness) → IN FLIGHT.** Background agent resolving the ~15% suffix
  deficit (leading hypothesis: `Lo` suffix truncation, not the "legitimate tail" the build agent
  claimed — the numbers show ~40% of the multi-bounce tail missing). This is **Wave-1's gate**: B1
  (composite into beauty) is correctly blocked on it.
- **E-axis partial (per CLAUDE.md commits, not re-read this turn — labelled as such):** V24 GPU
  path-replay adjoint for point+rect-area NEE is DONE (`3d022f9`/`5a79307`); spot + multi-bounce were
  attempted + **reverted** (subtle dead-ends); MNEE caustic integration is COMPLETE (single-interface
  reflect+refract + 2-vertex glass chain, all GPU-validated). So E1/E2 (spot, ≥2-bounce closed-scene)
  are the *known-hard* re-attempts, not greenfield.

## Wave structure & dependency graph (from the source program)

```
G0–G4 (global) ── G0 DONE; G1 (naga CI) is the high-leverage next global
   │
Wave 1  B1 → B2 → B3 → B6–B7     [B2 IN FLIGHT; B7 done; A1 hybrid-GRIS-on-Dawn parallel]
   │
Wave 2  D1–D4 (packaging)        [parallel to Wave 1; gated on a publish-governance decision]
   │
Wave 3  T1 → T2 → T3 → T4        [THREE decouple; A=95% cannot be claimed until T1–T4 land]
   │
Wave 4  A3–A13, C1–C9, B15       [polish; needs T3 for true A 95%]
   │
Wave 5  E*, remaining 100%
```

## Decisions needed (user-owned — surfaced, not pre-decided)

- **Publish governance (D2 / C9 / D1).** Wave 2 wants packages un-private + a scoped-publish plan,
  but `CLAUDE.md` conventions are *"No npm publish yet"* and the `three-gpu-pathtracer` scope question
  is explicitly the user's call (vitrum-owned scope vs keep `pt-webgl` private). **This is a genuine
  fork I will NOT pre-decide.** It does not block the fidelity-first Wave-1 work, so I'll surface it
  as a single question when Wave 2 becomes the active front — not before (one-question-at-a-time).

## Active now

- **B2 suffix-bias agent** — Wave-1 gate, running. I verify its suffix-depth sweep + fix on dzn myself
  before it composites (B1).
- **G0 (this file) + B7 README** — landing this commit.
