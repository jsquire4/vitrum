// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// ── Public surface (production consumers) ────────────────────────────────────
// The TS modules in this package fall into two buckets:
//
//   1. PRODUCTION  — actually imported by another package (non-test). These
//      are re-exported below from this index and are the package public API.
//
//   2. TEST/SPEC ORACLE — CPU-side implementations whose primary role is to
//      verify the WGSL/GLSL canonical implementations (which live in shaders
//      in `pt-webgl`, `pt-webgpu`, the PT fork, or `walkaround-hybrid`).
//      These modules are kept in `src/` and unit-tested via deep imports
//      (`__tests__/foo.test.ts` → `import ... from '../src/foo.js'`), but
//      they are NOT re-exported from this index. Each oracle file is marked
//      `@internal` in its top-of-file docstring. The spec value is preserved
//      (do NOT delete — the JSDoc + math is the canonical written-down spec).
//
// If you are a consumer and need an oracle symbol in production, first review
// whether the canonical WGSL/GLSL belongs in this package (W2-C6 will move
// shared MIS WGSL here). Don't deep-import from `../src/foo.js` from outside
// the package.
//
// ── Phase notes (historical) ─────────────────────────────────────────────────
// Phase 1: Hammersley QMC + sphere sampling WGSL fragment.
// Sprint 3 (Phase 6): light tree CDF (CPU build + GPU pack), mixture PDF / MIS.
// Sprint 7 (Phase 6): HG phase function + equi-angular volume distance sampling.
// Sprint 8b (Phase 6): Jakob+Hanika spectral upsampling.
// Sprint 10c (Phase 6): BDPT vertex layout + connection-PMF MIS weights.
// Sprint 12 (Phase 6): CIE CMF tables, hero-wavelength sampling, Cauchy IOR.
//
// Future: Sobol QMC (Sprint 3 fork-side), Welford variance struct (Sprint 9).

// ── PRODUCTION exports ───────────────────────────────────────────────────────

// Hammersley QMC WGSL — consumed by pt-webgpu, walkaround-hybrid.
export * from './wgsl/hammersley.wgsl.js';

// Canonical octahedral encode/decode WGSL — consumed by shared-bvh, pt-webgpu.
export { OCTAHEDRAL_CORE_WGSL } from './wgsl/octahedralCore.wgsl.js';

// CIE 1931 standard observer color-matching tables — consumed by
// pt-webgl/forkUniformBridge.ts to drive the fork's GLSL hero-wavelength
// sampler uniforms. The integrals/CDFs derived from these tables are computed
// in the consumer (forkUniformBridge does its own buildCmfCdf + trapezoidal
// integration), so this package only ships the raw spectral data.
export {
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
} from './cieCmf.js';

// ── TEST/SPEC ORACLE modules — NOT re-exported here ──────────────────────────
//
// The following modules live in `src/` and are tested via deep imports, but
// are NOT part of the public package surface. Each file's top docstring is
// marked `@internal`:
//
//   lightTree.ts       — buildLightTree, packLightTreeForGPU, LightTreeNode,
//                        LightTreeBuildInput
//   mixturePdf.ts      — balanceHeuristic, powerHeuristic, mixturePdf
//   hgPhase.ts         — evaluateHG, sampleHG, pdfHG
//   equiAngular.ts     — sampleEquiAngular, EquiAngularSample, EquiAngularOptions
//   jakobHanika.ts     — rgbToApproxSpectralCoefficients,
//                        rgbToSpectralCoefficients, evaluateSpectrum,
//                        VISIBLE_LAMBDA_MIN, VISIBLE_LAMBDA_MAX
//   bdptVertex.ts      — BDPT_KIND_*, BDPT_VERTEX_FLOATS/BYTES,
//                        BDPT_MAX_LIGHT_BOUNCES, BDPT_MAX_EYE_BOUNCES,
//                        packBDPTVertex, unpackBDPTVertex, BDPTVertex
//                        (Sprint 10c gated for future BDPT dispatch)
//   bdptMIS.ts         — bdptConnectionMIS_partial/full,
//                        buildBDPTStrategyPDFs_partial/full,
//                        geometricTermG, BDPTFullVertex
//                        (Sprint 10c gated for future BDPT dispatch)
//   cieCmf.ts (rest)   — CIE_D65_TABLE, CIE_LAMBDA_MIN/MAX/STEP,
//                        CIE_TABLE_LENGTH, sampleCMF, xyzToLinearSRGB
//   wavelengthSampling.ts — sampleHeroWavelength, sampleHeroWavelengthMIS,
//                        wavelengthToRGB, Y_CMF_INTEGRAL, X_CMF_INTEGRAL,
//                        Z_CMF_INTEGRAL, X_CMF_CDF, Y_CMF_CDF, Z_CMF_CDF,
//                        HERO_LAMBDA_MIN, HERO_LAMBDA_MAX
//   cauchyIor.ts       — cauchyIOR, abbeNumber, CAUCHY_CROWN_GLASS,
//                        CAUCHY_FLINT_GLASS, CAUCHY_LEAD_CRYSTAL,
//                        FRAUNHOFER_D_NM / _F_NM / _C_NM
