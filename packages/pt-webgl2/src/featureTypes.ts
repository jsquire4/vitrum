// Shared feature/compile-flag contract — imported by the GL framework (src/gl/)
// and the GLSL composer (src/glsl/). Single source so the
// parallel modules can't disagree on the #define schema.
//
// Mirrors the fork's PhysicalPathTracingMaterial defines (plan/three-removal/
// 04-glsl-kernels.md §1a), verified at PhysicalPathTracingMaterial.js:55-90.

type RandomType = 0 | 1 | 2; // 0=PCG, 1=Sobol, 2=Stratified
type CameraType = 0 | 1 | 2; // 0=Perspective, 1=Orthographic, 2=Equirectangular

export interface TraceFeatures {
  // ── Host-controllable (an engine option drives these) ──────────────────────
  readonly mis: boolean;            // FEATURE_MIS (always on — the MIS integrator)
  readonly russianRoulette: boolean; // FEATURE_RUSSIAN_ROULETTE (always on)
  readonly bdpt: boolean;           // FEATURE_BDPT — opts.bdpt (A5, host-driven)
  readonly dof: boolean;            // FEATURE_DOF — opts.dof (thin-lens depth of field)
  readonly cameraType: CameraType;  // CAMERA_TYPE — opts.cameraType (persp/ortho/equirect)
  readonly stainedGlassPerturbation: boolean; // FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION

  // ── Internal, FIXED at compile defaults (NOT host-controllable) ────────────
  // Flag-plumbing audit (2026-06-10): these GLSL gates exist (verbatim from the fork)
  // but have no real host pathway, so they are deliberately pinned and NOT exposed as
  // options — leaving them switchable would be a silent dead claim. Each kept here
  // only so featureDefines() can emit the GLSL macro at its safe default:
  //   • fog            — FEATURE_FOG: the retained fog-volume material path needs a
  //                      core fog-volume primitive; @vitrum/core carries no such node,
  //                      so there is nothing honest to drive. The old global
  //                      homogeneous-medium uniforms/branch were removed from the
  //                      active shader because they had no host setter. Pinned false
  //                      until core gains a fog-volume node.
  //   • backgroundMap  — FEATURE_BACKGROUND_MAP: a SEPARATE background texture distinct
  //                      from the environment map; the core contract has no such field
  //                      (env IS the background). Pinned false.
  //   • randomType     — RANDOM_TYPE: PCG(0) is the default; Sobol(1) is host-
  //                      controllable via opts.sampling and uploads a real Sobol
  //                      texture. Stratified(2) still consumes dummy textures and
  //                      remains deliberately unexposed until its tables ship.
  //   • debugMode      — DEBUG_MODE: g-buffer/AOV debug visualisations, not a production
  //                      render path. Pinned 0.
  readonly fog: boolean;            // FEATURE_FOG (pinned false)
  readonly backgroundMap: boolean;  // FEATURE_BACKGROUND_MAP (pinned false)
  readonly randomType: RandomType;  // RANDOM_TYPE (PCG default, Sobol opt-in)
  readonly debugMode: number;       // DEBUG_MODE (pinned 0)
}

export const DEFAULT_TRACE_FEATURES: TraceFeatures = {
  mis: true,
  russianRoulette: true,
  bdpt: false,
  dof: false,
  cameraType: 0,
  stainedGlassPerturbation: false,
  // Internal fixed defaults (see TraceFeatures for why each is pinned, not optional).
  fog: false,
  backgroundMap: false,
  randomType: 0,
  debugMode: 0,
};

/** The fixed attribute-slot defines the GLSL `texelFetch1D(attributesArray, ATTR_*, ...)` reads.
 *  ATTR_UV1 (layer 4) carries the second UV channel (TextureRef.texCoord 1); filled from
 *  primitive.uv1, falling back to uv0 when absent so the layer is always valid. */
const ATTR_DEFINES = { ATTR_NORMAL: 0, ATTR_TANGENT: 1, ATTR_UV: 2, ATTR_COLOR: 3, ATTR_UV1: 4 } as const;

/** Build the `#define NAME VALUE` map the GlProgram preamble injects. */
export function featureDefines(f: TraceFeatures): Record<string, number> {
  return {
    FEATURE_MIS: f.mis ? 1 : 0,
    FEATURE_RUSSIAN_ROULETTE: f.russianRoulette ? 1 : 0,
    FEATURE_DOF: f.dof ? 1 : 0,
    FEATURE_BACKGROUND_MAP: f.backgroundMap ? 1 : 0,
    FEATURE_FOG: f.fog ? 1 : 0,
    FEATURE_BDPT: f.bdpt ? 1 : 0,
    FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION: f.stainedGlassPerturbation ? 1 : 0,
    RANDOM_TYPE: f.randomType,
    CAMERA_TYPE: f.cameraType,
    DEBUG_MODE: f.debugMode,
    ...ATTR_DEFINES,
  };
}

/**
 * The accumulation blend regime (plan 02-gl-framework §3):
 *  - 'alpha-composite' — PT NoBlend → BlendMaterial ping-pong (no EXT_float_blend, or bgAlpha≠1).
 *  - 'normal'          — running average via SRC_ALPHA / ONE_MINUS_SRC_ALPHA; opacity = 1/(samples+1).
 *
 * D3 (2026-06-09): 'additive' was removed. It was a dead union member — the engine
 * never assigned it (index.ts only used 'normal' or 'alpha-composite'). 'normal' IS
 * the EXT_float_blend running-average path that 'additive' was originally intended to
 * be; enabling 'additive' would have changed default accumulation on float-blend
 * devices without any host opt-in.
 */
export type AccumRegime = 'alpha-composite' | 'normal';
