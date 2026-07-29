// Shared feature/compile-flag contract — imported by the GL framework (src/gl/)
// and the GLSL composer (src/glsl/). Single source so the
// parallel modules can't disagree on the #define schema.
//
// Mirrors the fork's PhysicalPathTracingMaterial defines (plan/three-removal/
// 04-glsl-kernels.md §1a), verified at PhysicalPathTracingMaterial.js:55-90.

type RandomType = 0 | 1; // 0=PCG, 1=Sobol
type CameraType = 0 | 1 | 2; // 0=Perspective, 1=Orthographic, 2=Equirectangular

export interface TraceFeatures {
  // ── Host-controllable (an engine option drives these) ──────────────────────
  readonly bdpt: boolean;           // FEATURE_BDPT — opts.bdpt (A5, host-driven)
  readonly dof: boolean;            // FEATURE_DOF — opts.dof (thin-lens depth of field)
  readonly cameraType: CameraType;  // CAMERA_TYPE — opts.cameraType (persp/ortho/equirect)

  // ── Scene-derived / internal compiler dimensions ───────────────────────────
  // MIS and Russian roulette are production invariants, emitted as constant-on
  // defines below rather than represented as changeable cache-key dimensions.
  readonly fog: boolean;            // FEATURE_FOG (scene-derived)
  readonly randomType: RandomType;  // composed RNG source (PCG default, Sobol opt-in)
  /** Scene-proven opaque base-PBR subset; unknown/authored optional fields force false. */
  readonly basicMaterials: boolean;
  /** Scene-proven texture-free full transport subset; unknown/map fields force false. */
  readonly scalarRichMaterials: boolean;
  /** Scene-proven texture-capable opaque base-PBR subset. */
  readonly mappedPbrMaterials: boolean;
  /** Scene-proven complete public MaterialSpec graph, including mixed maps and transport. */
  readonly mappedRichMaterials: boolean;
}

export const DEFAULT_TRACE_FEATURES: TraceFeatures = {
  bdpt: false,
  dof: false,
  cameraType: 0,
  // Internal defaults. Scene-derived values (including fog and material tiers)
  // override these in PTEngineWebGL2.#traceFeatures.
  fog: false,
  randomType: 0,
  basicMaterials: false,
  scalarRichMaterials: false,
  mappedPbrMaterials: false,
  mappedRichMaterials: true,
};

/** Fixed compatibility slots used by direct GLSL attribute reads. ATTR_UV1
 *  carries TextureRef.texCoord 1 and falls back to UV0 when absent. Arbitrary
 *  additional texCoord ids use scene-local dense layers selected from the
 *  material record rather than compile-time defines. */
const ATTR_DEFINES = { ATTR_NORMAL: 0, ATTR_TANGENT: 1, ATTR_UV: 2, ATTR_COLOR: 3, ATTR_UV1: 4 } as const;

/** Build the `#define NAME VALUE` map the GlProgram preamble injects. */
export function featureDefines(f: TraceFeatures): Record<string, number> {
  return {
    FEATURE_MIS: 1,
    FEATURE_RUSSIAN_ROULETTE: 1,
    FEATURE_DOF: f.dof ? 1 : 0,
    FEATURE_FOG: f.fog ? 1 : 0,
    FEATURE_BDPT: f.bdpt ? 1 : 0,
    CAMERA_TYPE: f.cameraType,
    ...ATTR_DEFINES,
  };
}
