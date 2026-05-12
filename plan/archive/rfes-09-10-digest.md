# RFEs 09 + 10 Digest — 2026-05-10

## RFE-09: pt-webgl Material → Fork Uniform Bridge

**Summary**: Wire the `vitrum.Material` field values through to the `PhysicalPathTracingMaterial` uniforms inside `@vitrum/pt-webgl`, closing the last gap that prevents RFE-06 (dispersion) and RFE-07 (volume scattering) fork patches from producing any visual effect.

**Motivation**: The fork's Sprint 7 and Sprint 8 shader patches are already applied (commits `260c432`, `7ffd15d`), and `sceneToThree.ts` already stamps `userData.vitrum*` on THREE materials. But no code ever reads those stamps and calls `pathTracer.material.u_volumeDensity = ...` or the equivalent. All fork uniforms sit at zero default. The applied patches are effectively dead until this bridge exists.

**Scope**:
- `@vitrum/pt-webgl` (`packages/pt-webgl/src/sceneToThree.ts`, `packages/pt-webgl/src/index.ts`) — primary implementation site
- Fork (`PhysicalPathTracingMaterial.js`) — no changes needed; uniforms are already declared and wired in the shader
- `@vitrum/shared-samplers` — read-only consumer: `jakobHanika.ts` and `cauchyIor.ts` supply coefficient conversion math
- `@vitrum/core` types — no changes needed
- Host app — no changes needed
- Touches fork? No new fork patches required.

**Dependencies**:
- RFE-06 (Sprint 8 dispersion fork patch) — APPLIED (`7ffd15d`)
- RFE-07 (Sprint 7 volume scattering fork patch) — APPLIED (`260c432`)
- RFE-08 / Sprint 12 spectral — PARTIAL; the CMF upload portion of this RFE can land now; Beer-Lambert per-λ path is gated on RFE-13 (ray-payload restructure)
- RFE-13 — prerequisite for the full spectral Beer-Lambert uniform upload only; does not block dispersion or scattering bridge work

**Overlap with prior commits**: None of the uniform bridge is present. Verified: grep of `packages/pt-webgl/src/` for `u_volumeDensity`, `u_dispersionStrength`, `u_sssSigmaT`, `iorCauchyA`, `uCmfX` returns zero hits. The `userData.vitrum*` stamp side (`sceneToThree.ts` lines 106–122) is confirmed present and correct. The bridge is fully missing.

**Effort**: M — the math specs are fully written in `plan/sprint-8-pt-fork-patch.md §1` and the helpers exist in `@vitrum/shared-samplers`. The work is wiring + unit tests + a visual A/B render against a scattering scene and a dispersion scene. No new algorithms needed.

**Risk**:
- Uniform upload timing: must happen after `WebGLPathTracer` constructs its internal `PhysicalPathTracingMaterial` instance. If upload happens before the material is initialized the values will be clobbered. Needs a post-scene-set hook or deferred upload.
- CMF array upload (`uCmfX[81]` etc.) is a one-time engine-init cost; must not be called per-frame.
- Spectral Beer-Lambert half (RFE-08 full) is gated on RFE-13; implementing it prematurely will produce dead code or a broken contract.
- Acceptance criteria require GPU-verified visual output (SSS glow + prismatic fan), not just passing `npm test`. A reference render A/B is mandatory per the project testing protocol.

**Suggested action**: implement-now — all prerequisites are in place for the dispersion and scattering sub-paths. Implement those two; stub the spectral CMF upload and leave the Beer-Lambert per-λ path behind a `TODO: gated on RFE-13` comment.

---

## RFE-10: three-bindings userData Propagation

**Summary**: Status-update RFE confirming that the `THREE.MeshPhysicalMaterial userData.vitrum* → vitrum.Material` converter in `@vitrum/three-bindings` is fully implemented; retained as a closure record and convention reference.

**Motivation**: External consumers and future contributors need to know the work is done and the `vitrum*` prefix convention is canonical.

**Scope**:
- `@vitrum/three-bindings` (`packages/three-bindings/src/material.ts`) — already implemented at lines 93–157; no action needed
- `@vitrum/pt-webgl` (`sceneToThree.ts`) — stamping side already implemented at lines 100–136; no action needed
- Fork, host app, `@vitrum/core` — no changes needed

**Dependencies**:
- RFE-03 `frontLayer`/`backLayer` — propagation is done; BSDF evaluation side deferred on RFE-12
- RFE-06, 07, 08 — all userData keys are read and forwarded; confirmed at `material.ts` lines 93–157

**Overlap with prior commits**: Entirely closed by the round-trip work in `1036a8c`. Read-verified: `packages/three-bindings/src/material.ts` lines 93–157 contain all listed `userData.vitrum*` reads with proper type guards. Nothing remains to implement in this RFE.

**Effort**: trivial — zero implementation work; the only open items redirect to RFE-09 and RFE-12.

**Risk**: None for this RFE. The one residual risk is that stainedGlass still maintains its own `vitrumMaterialAdapter.ts` as a parallel path; this is acknowledged as optional migration work and explicitly out of scope here.

**Suggested action**: close — mark as DONE in `IMPLEMENTATION-STATUS.md`. No dispatch needed.

---

## Cross-RFE analysis

**Are they two halves of the same wire?** Yes, but the split is already resolved. RFE-10 covered the left half (`THREE userData → vitrum.Material`); RFE-09 covers the right half (`vitrum.Material → fork uniforms`). The middle segment — `vitrum.Material → userData stamp` in `sceneToThree.ts` — landed in `1036a8c`. So the chain is: THREE userData → vitrum.Material (done, RFE-10) → sceneToThree stamp (done, `1036a8c`) → fork uniforms (open, RFE-09).

**Do they conflict?** No. They operate on different ends of the pipeline with no overlapping code sites.

**Should they be implemented together or separately?** Separately. RFE-10 requires no implementation; RFE-09 is a self-contained `@vitrum/pt-webgl` task with well-specified math and clear acceptance criteria. Implementing RFE-09 does not require touching `three-bindings` at all.

---

## Recommendation to orchestrator

Close RFE-10 immediately by updating `IMPLEMENTATION-STATUS.md` (one-line edit, no code). Dispatch RFE-09 as a focused `@vitrum/pt-webgl` implementation task: wire `scatteringCoefficient/Anisotropy → u_volumeDensity/u_anisotropyG`, `dispersionAbbeNumber → u_dispersionStrength + Cauchy coefficients`, and CMF array upload on engine init. Leave the spectral Beer-Lambert path stubbed behind a `TODO: gated on RFE-13` comment. The task is medium effort (math is fully specified, helpers exist) and should include GPU-verified visual A/B renders against a scattering scene and a dispersion bevel scene as the acceptance gate — not just a unit-test pass.
