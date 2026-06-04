# Vitrum — Comprehensive Integration-Depth Plan (2026-06-03)

> **Purpose.** Replace the ad-hoc groupings with ONE coherent, sequenced execution
> plan. Grounded in (a) an external code-review of the current tree, (b) my own
> re-verification of its load-bearing claims, and (c) the verified state of the
> P0–P8 campaign (`memory/campaign-p0-p8-2026-06-02.md`). This supersedes the
> piecemeal "do the next grouping" cadence; it is the single source of truth for
> *what we build, in what order, and how each piece is validated.*

## 0. The one-sentence diagnosis

The breadth is real and tested (~85–90% of a browser-SOTA stack on both stacks);
the gap is **integration depth + honesty + decoupling + perf**, not missing
algorithms. The recurring failure mode is *"the algorithm is validated in
isolation but the beauty pass runs the old approximation."* The headline instance:
**MNEE math is complete and GPU-validated in a side module, but the renderer still
runs cone-search.**

## 1. Verified current state (what is actually wired, not planned)

- **P0–P5 of the campaign largely shipped + on `origin/main`:** docs reconciliation
  (P0), the `TextureRef`/`UvTransform`/anim contract layer (P1), **pt-webgpu
  material textures end-to-end** (P2 — baseColor/emissive/ORM, GPU-validated
  `df9ae53…cf9b713`; the `overnight-campaign-status` doc calling P2 "blocked" is
  itself stale), gltf import + animation sampling (P3), tonemap operators (P4),
  contract-honesty + aux-buffer emission + instance-count rebuild + spotlight-cone
  GI (P5).
- **Real-time stack:** a true PassRegistry frame graph (ReSTIR-DI/GI, shade, GTAO,
  SVGF-family, temporal accum) with opt-in-but-real ReGIR / GRIS / PPG / NRC / RC.
- **Hero stack (pt-webgpu):** composed kernel with TLAS, analytic shapes, HDRI,
  multi-emitter light tree, GPU BDPT, spectral/thin-film/SSS, P2 textures.
- **This session:** the **entire MNEE math** (single-vertex solve reflect+refract,
  analytic step Jacobian, manifold Jacobian, connection-PDF determinant; **plus
  the 2-vertex glass chain** solve + chain connection-PDF) — all GPU-validated via
  self-validating harnesses. Facade GI-state forwarding closed (`6169426`).

## 2. The gap inventory (verified, organized by *kind*)

| ID | Gap | Status / evidence | Kind |
|----|-----|-------------------|------|
| **A1** | **MNEE not in the beauty pass** | math complete; `mneeNewton` is imported by NOTHING in the trace composer — renderer runs `manifoldNeeContribution` cone-search. **Headline.** | Integration |
| A2 | Inverse/adjoint is narrow | real GPU path-replay, but 1 bounce, `baseColor`/`roughness` only, no textures (`throws` on `kind:'texture'`), no indirect | Integration |
| A3 | GI persistence is DDGI-only | `exportGIState` snapshots probe atlases only; no ReSTIR/RC temporal history | Integration |
| B1 | THREE load-bearing for hybrid BVH | ~24 THREE imports in walkaround + shared-bvh *ingest* THREE geometry to build the BVH | Decoupling |
| B2 | `updateEnvironment` unimplemented on hybrid | ledger:false; not defined on `HybridEngine` | Decoupling |
| C1 | Fidelity-matrix rows `experimental` | a promotion/evidence gate, not missing WGSL (spectral/thin-film/SSS/BDPT are in the composed kernel) | Honesty |
| C2 | Residual stale capability comments | e.g. spot lights described "point-like" though the shader now has cone falloff | Honesty |
| D1 | No measured-ms perf loop | presets + `resolutionFactor` + DDGI stride are wired, but no closed-loop adapt-from-frame-time | Performance |
| E1 | No `@vitrum/progressive-engine` | walkaround↔PT progressive convergence — 0 files in tree | Frontier |
| E2 | No full path-space ReSTIR-PT hero integrator | walkaround has GI reuse/shift; pt-webgpu has no path reservoir | Frontier |
| E3 | NRC is walkaround-only | not a hero-stack terminator | Frontier |

**Out of scope (explicitly):** WebGPU RT extension, DLSS, wavefront PT (not
feasible/portable today); npm publish + release governance (separate program).

## 3. Validation discipline (applies to every item below)

The proven pattern, stated plainly: **self-validating math harnesses
(analytic==FD / residual→0) and engineering items (readback/serialize/wire) land
cleanly even on deep context; scene-dependent *noisy* radiometric A/Bs dead-end.**
Therefore, for every radiometric item, **design a DETERMINISTIC reference offline
first** — analytic where one exists (a point-light specular caustic is zero-measure
for ordinary sampling, so the *mirror-image irradiance* is exact, non-noisy ground
truth), or a converged-and-frozen reference where it does not. Commit only
validated work; revert the render-path change cleanly if the A/B can't be made
clean, keeping any isolated math that did validate.

## 4. Execution plan — phases, sequenced

### Phase I — Close the headline integration gaps (highest honesty value)

**I.1 — MNEE into the beauty pass (A1).** *The headline.*

> **STATUS (2026-06-03): COMPLETE + GPU-validated.** Sub-steps 1–4 all shipped,
> AND the glass-chain promotion (the old step-4 follow) landed. Reflection caustic `8cd52cf` (analytic mirror-image oracle, ratio
> 0.885). Transmissive single-flat-interface refraction caustic `893ef00`
> (`pointLightRefractionCaustic`, `E = I·T·|dω_L/dA_recv|` = Fresnel transmittance ×
> the refraction focusing Jacobian; forward-traced floor-flux oracle, ratio 0.987 /
> slope 0.984). Shared `causticTransmissiveLegBlocked` self-skip visibility.
> **DONE:** the 2-vertex GLASS-SLAB (chain) caustic `f38e881` — `pointLightGlassSlabCaustic`
> wires `mneeNewtonSolveChain2` into the beauty pass; `E = I·T1·T2·|dω_L/dA_recv|`
> (per-interface Fresnel transmittance product × chain focusing Jacobian). Forward-
> traced glass-slab A/B (re-verified on lavapipe): ratio 0.9962 / slope 0.9898 /
> 99.5% firing; refraction + reflection unregressed. A double-count bug (single-
> interface kernel firing on the slab bottom) was caught by the A/B + guarded.
> **MNEE caustic integration is COMPLETE** — single-interface + glass chain, all
> GPU-validated. The cone-search remains only as the directional-light fallback.

- **Sub-steps (each independently committable):**
  1. Offline analytic reference: point light + flat mirror + diffuse floor →
     mirror-image floor irradiance (deterministic oracle). *[clean]*
  2. Kernel-ready contribution core `mneeReflectionIrradiance(recv, mirror, light)`
     validated against (1). *[clean, math-harness]*
  3. Kernel wiring: at a diffuse hit, seed-trace toward the light to find the
     specular surface → Newton-solve the exact vertex on its plane → visibility →
     contribution; gated as `causticStrategy: 'manifold-nee'` mode 1 (photon map
     stays mode 2). *[render-path]*
  4. Render A/B vs the analytic reference; promote transmissive (glass chain) once
     the reflection path is proven.
- **Validation:** render-vs-analytic (deterministic). **Risk:** medium — the math
  is pre-validated, so a mismatch isolates to plumbing.
- **Fallback (per discipline):** if the render A/B resists, keep (1)+(2) committed,
  leave the cone-search in place, mark the wiring remaining. *No unvalidated render
  path ships.*

**I.2 — Honesty pass (C1, C2).** Fidelity-matrix promotion per
`plan/fidelity-promotion-playbook.md` (run the gap-closure scenes, promote rows
that pass); sweep the residual stale capability comments. **Validation:** the
existing fidelity gate + doc read. **Risk:** low.

### Phase II — Inverse-rendering depth (A2)
- **II.1** Widen `ADJOINT_ELIGIBLE_FIELDS` to `emissive` + `ior` — analytic BSDF
  partials, math-harness-validatable (the GPU==FD oracle pattern already exists).
  *[clean]*
- **II.2** Multi-bounce adjoint — **HARD, radiometric.** Needs ≥2 bounces + NEE-IS
  in a CLOSED scene designed offline (the open-Cornell 1-bounce attempt dead-ended
  this campaign). Gate II.2 behind a designed reference; do not start blind.

### Phase III — Persistence + decoupling
- **III.1** GI-persistence completeness (A3): extend `GIStateSnapshot` with the
  ReSTIR/RC temporal reservoirs + invalidation metadata. **Engineering** (the
  cached-field lesson: readback/serialize/upload lands clean). *[clean]*
- **III.2** THREE decoupling (B1 / campaign P6): a THREE-free geometry-ingestion
  path that builds the BVH from the `@vitrum/core` `Scene` contract instead of
  `THREE.Mesh`/`BufferAttribute`. **Large, cross-package**; typecheck+test gated;
  best as a dedicated refactor, not a blind slice. *[large, low-radiometric-risk]*
- **III.3** `updateEnvironment` on hybrid (B2). *[small engineering]*

### Phase IV — Performance (D1)
- **IV.1** Close the loop: `readGpuTimingsOnce()` already exists → a frame-budget
  controller that adapts `resolutionFactor` / DDGI stride / quality tier from
  measured ms. Turns "runs at 30–60fps on Class B" from an external claim into a
  measured, adaptive property. *[engineering; validate by a deterministic
  ms-injection test, not live perf]*

### Phase V — Frontier (campaign P8)
- **V.1** ReSTIR-PT — start with the **math-harness-testable** reconnection/shift
  Jacobian + GRIS contribution-weight algebra (the GI `jacobianReconnectionShift`
  exists as a special case). *[math-harness first; integrator later]*
- **V.2** `@vitrum/progressive-engine` (E1) — the walkaround↔PT freeze/resume
  temporal-MIS handoff (roadmap §8.1; rated "fun trick", below V.1/V.3).
- **V.3** NRC as a hero-stack terminator (E3).

## 5. Sequencing rationale

1. **Phase I first** because it is the largest *honesty* gap (the review's core
   critique) and the MNEE math is already paid for — Phase I is mostly wiring +
   one deterministic A/B, the best value/effort ratio left.
2. **Phase II.1 + III.1 + III.3** are clean engineering/math-harness items —
   schedule them as reliable wins between/around the riskier work.
3. **III.2 (THREE decouple)** is the biggest single lift; it is independent of the
   radiometric work, so it can run in parallel (different files) once Phase I lands.
4. **II.2 (multi-bounce adjoint)** and **Phase V (frontier)** are the deep
   radiometric/research items — gate them behind designed references; do not let
   them block the clean wins.

## 6. Scope + execution order (EVERYTHING is in scope)

Maintainer decision (2026-06-03): **all of it ships — no either/or scope cuts.**
The only variable is execution ORDER, which is mine to sequence by
(value × inverse-risk × dependency) and adjust as dependencies surface. Nothing
below is "deferred-as-cut"; "later" means "later in the queue," not "dropped."

1. **MNEE — BOTH reflection AND transmissive**, fully wired into the beauty pass
   (Phase I.1). Reflection lands *first on the shared plumbing* only because it has
   a deterministic analytic reference (so it de-risks the wiring both cases share);
   transmissive/glass follows immediately on that same wiring. Both ship.
2. **Honesty pass** (I.2) — runs alongside; low-risk.
3. **Inverse depth — emissive/ior AND multi-bounce** (Phase II). Multi-bounce is
   gated behind a *closed designed reference scene* — that is the validation
   discipline (§3), not a scope cut; it ships.
4. **GI persistence — full ReSTIR/RC history** (III.1); **full THREE decouple**
   (III.2, done incrementally); **hybrid `updateEnvironment`** (III.3).
5. **Measured-ms frame-budget controller** (Phase IV).
6. **All frontier** (Phase V): ReSTIR-PT (math-harness portion first, then the
   integrator), `@vitrum/progressive-engine`, NRC hero-terminator.

The sequencing rationale (§5) governs the *order*; §6 governs the *scope* (= all).

## 7. What "done" looks like

Every advertised technique is the *real algorithm in the beauty pass* (no
cone-search-while-Newton-sits-in-a-harness), the fidelity matrix's promoted rows
are evidence-backed, hybrid no longer hard-depends on THREE for ingestion, and the
library *measures and adapts to* its own frame time. That is the line between
"late release-candidate" (today) and "fully professional SOTA browser renderer."
