# External Requests — Implementation Status

**Date**: 2026-05-09
**Branch**: main
**Agent**: Claude Sonnet 4.6 (external-requests run)

---

## Summary

5 RFEs processed from `external_requests/`. All 5 implemented as API-contract
additions to `@vitrum/core`. All are contract-complete (types + capabilities
wiring); backend shader implementations remain deferred to their respective
sprint fork-patch documents.

4 audit findings from `plan/sprints-1-11-audit.md` remediated in the same
pass (H-1, H-2, M-3, L-1, L-3).

---

## RFE-01 — Spectral Attenuation and Dispersion

**Status**: Implemented (contract layer)

**What was requested**: `SpectralCurve` type + `spectralAttenuation` and
`dispersionAbbeNumber` fields on `Material` for per-wavelength Beer-Lambert
and hero-wavelength spectral sampling.

**What was implemented**:
- `SpectralCurve` interface added to `packages/core/src/scene.ts`.
- `Material.spectralAttenuation?: SpectralCurve` field added with JSDoc
  citing Wilkie et al. EGSR 2014.
- `Material.dispersionAbbeNumber?: number` field added with JSDoc citing
  OpenPBR v1.1.1.
- Both new types are exported from `@vitrum/core` automatically via
  `export * from './scene.js'`.

**Deferred**: Backend shader implementation (fork-side kernel rewrite) is
already specified in `plan/sprint-12-pt-fork-patch.md` and gated on the
trigger criterion defined there. No new deferral introduced.

**Tests added**: None needed — pure types with no runtime behavior.

**Best-judgment call**: The `SpectralCurve.values` field is `Float32Array`
(not `readonly Float32Array`) to match how consumers construct it (new Float32Array).
The interface itself is `readonly`. This is consistent with the spec's example
code.

---

## RFE-02 — Volume Scattering (Henyey-Greenstein)

**Status**: Implemented (contract layer)

**What was requested**: `scatteringCoefficient`, `scatteringAnisotropy`, and
`scatteringCoefficientRGB` fields on `Material` for volumetric delta tracking.

**What was implemented**:
- Three fields added to `Material` in `packages/core/src/scene.ts`.
- JSDoc cites Henyey & Greenstein 1941 (phase function) and Novák et al. 2018
  (delta tracking / null-collision framework).
- `scatteringCoefficientRGB` typed as `Vec3` (existing math primitive).

**Deferred**: Backend shader implementation specified in
`plan/sprint-7-pt-fork-patch.md`. Volume march loop already scaffolded
(Sprint 7 complete) — the `scatteringCoefficient` field connects to that
existing infrastructure.

**Tests added**: None needed — pure types.

---

## RFE-03 — Layered BSDF (Front/Back Asymmetric)

**Status**: Implemented (contract layer)

**What was requested**: `SurfaceAbsorptionLayer` type + `frontLayer` /
`backLayer` fields on `Material` for per-face BSDF asymmetry.

**What was implemented**:
- `SurfaceAbsorptionLayer` interface added to `packages/core/src/scene.ts`
  with `transmission: Vec3`, `roughness?: number`, `normalMap?: TextureRef`,
  `normalScale?: number`.
- `Material.frontLayer?: SurfaceAbsorptionLayer` and
  `Material.backLayer?: SurfaceAbsorptionLayer` fields added.
- JSDoc cites Belcour 2018 (atomic decomposition) and notes the simplified
  single-bounce approximation.
- Exported automatically via `export * from './scene.js'`.

**Deferred**: Backend shader implementation (~20–30 GLSL lines at shade time).
No sprint doc exists yet — backend work is straightforward once fork patches
for earlier sprints are applied.

**Tests added**: None needed — pure types.

---

## RFE-04 — Multi-Layer Thin Film (TMM)

**Status**: Implemented (contract layer)

**What was requested**: `ThinFilmLayer` + `ThinFilmStack` types + `thinFilmStack`
field on `Material` for Transfer Matrix Method evaluation of multi-layer optical
coatings (Bragg reflectors, dichroic glass, structural color).

**What was implemented**:
- `ThinFilmLayer` interface: `ior: number`, `extinctionCoefficient?: number`,
  `thicknessNm: number`.
- `ThinFilmStack` interface: `layers: ReadonlyArray<ThinFilmLayer>`,
  `incidentIor?: number`, `angleDependent?: boolean`.
- `Material.thinFilmStack?: ThinFilmStack` field with JSDoc noting it overrides
  the single-layer iridescence model.
- JSDoc cites Born & Wolf "Principles of Optics" (Abeles TMM) and Belcour &
  Barla SIGGRAPH 2017.
- Both types exported automatically.

**Deferred**: Backend shader implementation (TMM loop in GLSL, ~100 lines
including complex 2×2 matrix multiply per layer per wavelength sample). Spec
in external_requests/04-multilayer-thinfilm.md §3.

**Tests added**: None needed — pure types.

---

## RFE-05 — Manifold NEE / Specular Caustics Strategy

**Status**: Implemented (contract layer + capabilities wiring)

**What was requested**: `causticStrategy`, `mneeMaxIterations`,
`mneeMaxChainLength` on `EngineOptions`; `causticStrategy` on
`EngineCapabilities`.

**What was implemented**:
- `EngineOptions.causticStrategy?: 'none' | 'manifold-nee' | 'photon-map'`
  added with full JSDoc citing Hanika et al. 2015.
- `EngineOptions.mneeMaxIterations?: number` (default 8) added.
- `EngineOptions.mneeMaxChainLength?: number` (default 3) added.
- `EngineCapabilities.causticStrategy: 'none' | 'manifold-nee' | 'photon-map'`
  added (required field — all backends must report their strategy).
- **`@vitrum/pt-webgl`**: `PTEngineWebGL2` stores `causticStrategy` from opts
  and reflects it in `capabilities.causticStrategy`. Factory comment documents
  that MNEE/photon-map are API-complete but fork-side implementation is deferred.
- **`@vitrum/walkaround-hybrid`**: `HybridEngine.capabilities.causticStrategy`
  hardcoded to `'none'` with comment explaining why real-time caustic strategies
  are incompatible with the walkaround engine's frame cadence.

**Files modified**:
- `packages/core/src/engine.ts`
- `packages/pt-webgl/src/index.ts`
- `packages/walkaround-hybrid/src/HybridEngine.ts`

**Deferred**: MNEE Newton iteration solver and photon-map forward-trace pass
are ~300–500 lines each of GLSL/WGSL. Spec in external_requests/05-manifold-nee.md.

**Tests added**: None needed — the new field is reflected in capabilities.
Existing pt-webgl and walkaround tests confirm the engine still constructs
without error (the capabilities object shape is now correct for tsc).

---

## Audit Findings Remediated

### H-1 — PPG leaf stride mismatch (HIGH, fixed)

`ppgUpdate.wgsl` computed `leafIdx * 32u` (128-byte offsets) while
`ppgSample.wgsl` expects `leafIdx * 64u` (256-byte offsets per
`PPGDirectionalLeaf` layout). Fixed to `64u` with a comment linking the
derivation to `PPG_LEAF_BYTE_STRIDE` in `types.ts`.

**File**: `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts`

### H-2 — PPG fixed-point radiance clamp saturates at 256 nits (HIGH, fixed)

`f32(0xFFFFFFu)` (16,777,215) capped representable luminance at ~256 nits
(with `PPG_RADIANCE_SCALE = 65536`), destroying directional discrimination in
HDR scenes (sun-through-glass easily exceeds 1000 nits). Changed to
`f32(0xFFFFFFFFu)` (4,294,967,295) → max representable ≈ 65,536 nits.
Comment updated to reflect the correct maximum.

**File**: `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts`

### M-3 — HWC↔NCHW helpers untested directly (MEDIUM, fixed)

`_hwcToNchw` and `_nchwToHwc` exported with underscore prefix from
`oidnBridge.ts`. 6 direct round-trip tests added to `oidnBridge.test.ts`:
- 1×1×3 single pixel round-trip
- 2×2×3 identity round-trip
- Explicit NCHW channel layout verification for 2×2×3
- Element count preservation for 4×8×3
- Float-precision round-trip for 4×4×3 with `sin()` values
- 1-channel (grayscale) HWC=NCHW identity check

**Files modified**: `packages/shared-denoisers/src/oidnBridge.ts`,
`packages/shared-denoisers/__tests__/oidnBridge.test.ts`

### L-1 — jakobHanika.ts TODO not tracked in plan docs (LOW, fixed)

Sprint 12 section of `plan/phase-6-roadmap.md` extended with a "Jakob+Hanika
precomputed table" tracking item. Notes the license investigation required
before integrating the Mitsuba 3 table, and the Cauchy-fallback path if the
license is non-commercial only.

**File**: `plan/phase-6-roadmap.md`

### L-3 — BDPT exports appear in public API before integration testing (LOW, fixed)

Deferred-status comment block added immediately before the BDPT export block
in `packages/shared-samplers/src/index.ts`. Comment states the trigger criterion,
points to the sprint plan doc, and documents the audit finding.

**File**: `packages/shared-samplers/src/index.ts`

---

## Audit Findings Requiring User Judgment (not remediated)

Per the audit's own classification ("Requires user judgment"):

- **M-1**: Light tree CDF rename/rebuild — affects public API; GPU unaffected.
- **M-2**: SVGF sigmaColor=10 provenance claim — may be intentional re-tuning.
- **M-4**: PPG brute-force O(N) scan — blocking condition is a sprint-ordering decision.
- **M-5**: mixturePdf all-zero documentation — decide throw vs. document.
- **M-6**: SVGF iteration count as dispatch vs. shader — current design valid.
- **L-2**: Clustered centroid warning in light tree — low-priority quality-of-life.

---

## Verification Results

- **tsc --noEmit**: clean (exit 0, no diagnostics)
- **npm test --workspaces --if-present**: 538 tests passing across 6 packages
  (532 pre-existing + 6 new HWC↔NCHW round-trip tests)
  - pt-webgl: 18
  - shared-bvh: 11
  - shared-denoisers: 75 (69 pre-existing + 6 new)
  - shared-samplers: 139
  - three-bindings: 1
  - walkaround-hybrid: 294
