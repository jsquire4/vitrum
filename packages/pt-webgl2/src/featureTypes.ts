// Shared feature/compile-flag contract — imported by the GL framework (src/gl/)
// and the GLSL composer (src/glsl/). Single source so the
// parallel modules can't disagree on the #define schema.
//
// Mirrors the fork's PhysicalPathTracingMaterial defines (plan/three-removal/
// 04-glsl-kernels.md §1a), verified at PhysicalPathTracingMaterial.js:55-90.

import { WEBGL2_MAX_PATH_STEPS } from './limits.js';

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
  //   • fog            — FEATURE_FOG is scene-derived from transmitted materials
  //                      with positive scattering coefficients. It is false only
  //                      before a medium-bearing scene is installed.
  //   • backgroundMap  — FEATURE_BACKGROUND_MAP: a SEPARATE background texture distinct
  //                      from the environment map; the core contract has no such field
  //                      (env IS the background). Pinned false.
  //   • randomType     — RANDOM_TYPE: PCG(0) is the default; Sobol(1) is host-
  //                      controllable via opts.sampling and uploads a real Sobol
  //                      texture. Stratified(2) still consumes dummy textures and
  //                      remains deliberately unexposed until its tables ship.
  //   • debugMode      — DEBUG_MODE: g-buffer/AOV debug visualisations, not a production
  //                      render path. Pinned 0.
  readonly fog: boolean;            // FEATURE_FOG (scene-derived)
  readonly backgroundMap: boolean;  // FEATURE_BACKGROUND_MAP (pinned false)
  readonly randomType: RandomType;  // RANDOM_TYPE (PCG default, Sobol opt-in)
  readonly debugMode: number;       // DEBUG_MODE (pinned 0)
  /** Static compiler-visible loop ceiling: two physical steps per configured bounce. */
  readonly pathStepLimit: number;
  /** Scene-proven opaque base-PBR subset; unknown/authored optional fields force false. */
  readonly basicMaterials: boolean;
  /** Scene-proven texture-free full transport subset; unknown/map fields force false. */
  readonly scalarRichMaterials: boolean;
  /** Scene-proven texture-capable opaque base-PBR subset. */
  readonly mappedPbrMaterials: boolean;
  /** Scene-proven complete public MaterialSpec graph, including mixed maps and transport. */
  readonly mappedRichMaterials: boolean;
  readonly analyticLights: boolean;
  readonly meshLights: boolean;
  readonly environmentLight: boolean;
}

export const DEFAULT_TRACE_FEATURES: TraceFeatures = {
  mis: true,
  russianRoulette: true,
  bdpt: false,
  dof: false,
  cameraType: 0,
  stainedGlassPerturbation: false,
  // Internal defaults. Scene-derived values (including fog and material tiers)
  // override these in PTEngineWebGL2.#traceFeatures.
  fog: false,
  backgroundMap: false,
  randomType: 0,
  debugMode: 0,
  pathStepLimit: WEBGL2_MAX_PATH_STEPS,
  basicMaterials: false,
  scalarRichMaterials: false,
  mappedPbrMaterials: false,
  mappedRichMaterials: true,
  analyticLights: true,
  meshLights: true,
  environmentLight: true,
};

/** Fixed compatibility slots used by direct GLSL attribute reads. ATTR_UV1
 *  carries TextureRef.texCoord 1 and falls back to UV0 when absent. Arbitrary
 *  additional texCoord ids use scene-local dense layers selected from the
 *  material record rather than compile-time defines. */
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
