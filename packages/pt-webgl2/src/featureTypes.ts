// Shared feature/compile-flag contract — imported by the GL framework (src/gl/),
// the GLSL composer (src/glsl/), and the frame-params packer. Single source so the
// parallel modules can't disagree on the #define schema.
//
// Mirrors the fork's PhysicalPathTracingMaterial defines (plan/three-removal/
// 04-glsl-kernels.md §1a), verified at PhysicalPathTracingMaterial.js:55-90.

export type RandomType = 0 | 1 | 2; // 0=PCG, 1=Sobol, 2=Stratified
export type CameraType = 0 | 1 | 2; // 0=Perspective, 1=Orthographic, 2=Equirectangular

export interface TraceFeatures {
  readonly mis: boolean;            // FEATURE_MIS
  readonly russianRoulette: boolean; // FEATURE_RUSSIAN_ROULETTE
  readonly dof: boolean;            // FEATURE_DOF
  readonly backgroundMap: boolean;  // FEATURE_BACKGROUND_MAP
  readonly fog: boolean;            // FEATURE_FOG
  readonly bdpt: boolean;           // FEATURE_BDPT
  readonly stainedGlassPerturbation: boolean; // FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION
  // D3 investigation (2026-06-09): FEATURE_ADDITIVE_ACCUM / 'additive' regime was
  // dead code — the engine never assigned AccumRegime='additive' (only 'normal' or
  // 'alpha-composite'). 'normal' IS the float-blend running-average path. The
  // 'additive' union member, blend branch, and shader blocks have been deleted;
  // behavior is byte-identical for 'normal' and 'alpha-composite'.
  readonly randomType: RandomType;  // RANDOM_TYPE
  readonly cameraType: CameraType;  // CAMERA_TYPE
  readonly debugMode: number;       // DEBUG_MODE
}

export const DEFAULT_TRACE_FEATURES: TraceFeatures = {
  mis: true,
  russianRoulette: true,
  dof: false,
  backgroundMap: false,
  fog: false,
  bdpt: false,
  stainedGlassPerturbation: false,
  // PCG (0) is the default RNG: pure-compute, no sobol/stratified texture packers
  // needed (the fork's stratified path is RANDOM_TYPE 2; we can add it later).
  randomType: 0,
  cameraType: 0,
  debugMode: 0,
};

/** The fixed attribute-slot defines the GLSL `texelFetch1D(attributesArray, ATTR_*, ...)` reads. */
export const ATTR_DEFINES = { ATTR_NORMAL: 0, ATTR_TANGENT: 1, ATTR_UV: 2, ATTR_COLOR: 3 } as const;

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
    // FEATURE_ADDITIVE_ACCUM is always 0: the 'additive' regime was dead code (D3).
    // The GLSL #if blocks remain for readability but always compile the else-branch.
    FEATURE_ADDITIVE_ACCUM: 0,
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
 * devices without any host opt-in. The FEATURE_ADDITIVE_ACCUM GLSL blocks remain in
 * the shader source (always-false) for provenance; they will be pruned in a follow-up
 * GLSL cleanup.
 */
export type AccumRegime = 'alpha-composite' | 'normal';
