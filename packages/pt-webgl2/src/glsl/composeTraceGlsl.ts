// composeTraceGlsl — the fragment-shader compose root for the THREE-free WebGL2 path
// tracer (plan/three-removal/04-glsl-kernels.md §3; analog of pt-webgpu's
// composePtWebgpuTraceWgsl). It reproduces the EXACT chunk concatenation order of the
// fork's PhysicalPathTracingMaterial.js:227-441 fragment shader (struct-before-use is
// load-bearing) and the inlined main() render loop (lines 443-1099).
//
// Returns ONLY the fragment BODY: no `#version 300 es`, no precision/`pc_fragColor`
// preamble, no `#define` block — those are emitted by the GlProgram preamble (WS2). The
// `#define` flags this body branches on (RANDOM_TYPE, FEATURE_*, CAMERA_TYPE, DEBUG_MODE,
// ATTR_*) come from featureDefines() in the GlProgram preamble, so the GLSL `#if`s below
// resolve at compile time exactly as in the fork.
//
// Three classes of source feed this composer:
//   1. Copied fork chunks (shader/** + render/**) — imported as namespace modules and
//      indexed by their original `export const` names (byte-identical to the fork).
//   2. The three-mesh-bvh BVH port (src/glsl/bvh) — re-exported as BVH_* uppercase names.
//   3. Hand-authored glue: the THREE_COMMON_SHIM (the `<common>` subset), the inlined
//      uniform declarations, the RANDOM_TYPE branch, the two inline helper functions
//      (applyFilteredGlossy/sampleBackground), and the main() render loop — all of which
//      lived inline in the fork material (not in a `.glsl.js` chunk), transcribed here.

import type { TraceFeatures } from '../featureTypes.js';
import type { FrameUniforms } from '../gl/glResources.js';

import { BVH_COMMON_FUNCTIONS, BVH_STRUCT, BVH_RAY_FUNCTIONS } from './bvh/index.js';
import { THREE_COMMON_SHIM } from './common/threeCommonShim.js';

// Copied fork chunks — imported per-file as namespace modules. We deliberately import the
// individual `.glsl.js` chunks (NOT each subdir's `index.js` re-export barrel) because the
// `.glsl.js` files match the `*.glsl.js` ambient wildcard (glsl-modules.d.ts), whereas the
// untyped `index.js` barrels would need their own per-path declarations. Each chunk module
// exports one or more `export const NAME = /* glsl */\`...\`` string constants.
import * as StructsCamera from './shader/structs/camera_struct.glsl.js';
import * as StructsLights from './shader/structs/lights_struct.glsl.js';
import * as StructsEquirect from './shader/structs/equirect_struct.glsl.js';
import * as StructsMaterial from './shader/structs/material_struct.glsl.js';
import * as StructsSurface from './shader/structs/surface_record_struct.glsl.js';

import * as SamplingShape from './shader/sampling/shape_sampling_functions.glsl.js';
import * as SamplingEquirect from './shader/sampling/equirect_sampling_functions.glsl.js';
import * as SamplingLight from './shader/sampling/light_sampling_functions.glsl.js';

import * as CommonTexture from './shader/common/texture_sample_functions.glsl.js';
import * as CommonFresnel from './shader/common/fresnel_functions.glsl.js';
import * as CommonUtil from './shader/common/util_functions.glsl.js';
import * as CommonMath from './shader/common/math_functions.glsl.js';
import * as CommonShapeIsect from './shader/common/shape_intersection_functions.glsl.js';

import * as RandPcg from './shader/rand/pcg.glsl.js';
import * as RandSobol from './shader/rand/sobol.glsl.js';
import * as RandStratified from './shader/rand/stratified.glsl.js';

import * as PtbvhFog from './shader/bvh/inside_fog_volume_function.glsl.js';

import * as BsdfGgx from './shader/bsdf/ggx_functions.glsl.js';
import * as BsdfSheen from './shader/bsdf/sheen_functions.glsl.js';
import * as BsdfIridescence from './shader/bsdf/iridescence_functions.glsl.js';
import * as BsdfFog from './shader/bsdf/fog_functions.glsl.js';
import * as BsdfSpectral from './shader/bsdf/spectral_accumulator.glsl.js';
import * as BsdfThinFilm from './shader/bsdf/thin_film_tmm.glsl.js';
import * as BsdfFunctions from './shader/bsdf/bsdf_functions.glsl.js';

import * as RenderStructs from './render/render_structs.glsl.js';
import * as RenderCameraUtil from './render/camera_util_functions.glsl.js';
import * as RenderTraceScene from './render/trace_scene_function.glsl.js';
import * as RenderAttenuate from './render/attenuate_hit_function.glsl.js';
import * as RenderDirectLight from './render/direct_light_contribution_function.glsl.js';
import * as RenderGetSurface from './render/get_surface_record_function.glsl.js';
import * as RenderBdptSubpath from './render/bdpt_light_subpath.glsl.js';
import * as RenderBdptConnection from './render/bdpt_connection.glsl.js';

/** Resolve a named string export from an untyped `.glsl.js` namespace; throw if missing. */
function pick(ns: unknown, name: string): string {
  const value = (ns as Record<string, unknown>)[name];
  if (typeof value !== 'string') {
    throw new Error(`pt-webgl2 compose: missing string export "${name}"`);
  }
  return value;
}

// Re-group into the fork's named buckets (the rest of this module references chunks via
// these objects, mirroring the fork's `StructsGLSL.camera_struct` etc.).
const STRUCTS = {
  camera_struct: pick(StructsCamera, 'camera_struct'),
  lights_struct: pick(StructsLights, 'lights_struct'),
  equirect_struct: pick(StructsEquirect, 'equirect_struct'),
  material_struct: pick(StructsMaterial, 'material_struct'),
  surface_record_struct: pick(StructsSurface, 'surface_record_struct'),
} as const;

const SAMPLING = {
  shape_sampling_functions: pick(SamplingShape, 'shape_sampling_functions'),
  equirect_functions: pick(SamplingEquirect, 'equirect_functions'),
  light_sampling_functions: pick(SamplingLight, 'light_sampling_functions'),
} as const;

const COMMON = {
  texture_sample_functions: pick(CommonTexture, 'texture_sample_functions'),
  fresnel_functions: pick(CommonFresnel, 'fresnel_functions'),
  util_functions: pick(CommonUtil, 'util_functions'),
  math_functions: pick(CommonMath, 'math_functions'),
  shape_intersection_functions: pick(CommonShapeIsect, 'shape_intersection_functions'),
} as const;

const RANDOM = {
  pcg_functions: pick(RandPcg, 'pcg_functions'),
  sobol_common: pick(RandSobol, 'sobol_common'),
  sobol_functions: pick(RandSobol, 'sobol_functions'),
  stratified_functions: pick(RandStratified, 'stratified_functions'),
} as const;

const PTBVH = {
  inside_fog_volume_function: pick(PtbvhFog, 'inside_fog_volume_function'),
} as const;

const BSDF = {
  ggx_functions: pick(BsdfGgx, 'ggx_functions'),
  sheen_functions: pick(BsdfSheen, 'sheen_functions'),
  iridescence_functions: pick(BsdfIridescence, 'iridescence_functions'),
  fog_functions: pick(BsdfFog, 'fog_functions'),
  spectral_accumulator: pick(BsdfSpectral, 'spectral_accumulator'),
  thin_film_tmm: pick(BsdfThinFilm, 'thin_film_tmm'),
  bsdf_functions: pick(BsdfFunctions, 'bsdf_functions'),
} as const;

const RENDER = {
  render_structs: pick(RenderStructs, 'render_structs'),
  camera_util_functions: pick(RenderCameraUtil, 'camera_util_functions'),
  trace_scene_function: pick(RenderTraceScene, 'trace_scene_function'),
  attenuate_hit_function: pick(RenderAttenuate, 'attenuate_hit_function'),
  direct_light_contribution_function: pick(RenderDirectLight, 'direct_light_contribution_function'),
  get_surface_record_function: pick(RenderGetSurface, 'get_surface_record_function'),
  bdpt_light_subpath: pick(RenderBdptSubpath, 'bdpt_light_subpath'),
  bdpt_connection: pick(RenderBdptConnection, 'bdpt_connection'),
} as const;

/**
 * The RANDOM_TYPE branch (PhysicalPathTracingMaterial.js:259-290). The actual selection
 * is resolved by the `#if RANDOM_TYPE == N` preprocessor at compile time from the
 * GlProgram `#define RANDOM_TYPE <f.randomType>`; we emit all three branches verbatim so
 * the preprocessor picks the live one — byte-identical to the fork.
 */
function randBlock(): string {
  return /* glsl */ `
					// random
					#if RANDOM_TYPE == 2 	// Stratified List

						${RANDOM.stratified_functions}

					#elif RANDOM_TYPE == 1 	// Sobol

						${RANDOM.pcg_functions}
						${RANDOM.sobol_common}
						${RANDOM.sobol_functions}

						#define rand(v) sobol(v)
						#define rand2(v) sobol2(v)
						#define rand3(v) sobol3(v)
						#define rand4(v) sobol4(v)

					#else 					// PCG

					${RANDOM.pcg_functions}

						// Using the sobol functions seems to break the the compiler on MacOS
						// - specifically the "sobolReverseBits" function.
						uint sobolPixelIndex = 0u;
						uint sobolPathIndex = 0u;
						uint sobolBounceIndex = 0u;

						#define rand(v) pcgRand()
						#define rand2(v) pcgRand2()
						#define rand3(v) pcgRand3()
						#define rand4(v) pcgRand4()

					#endif
`;
}

// ── D10.3: UNIFORM_MANIFEST ────────────────────────────────────────────────────
// Structured list of the simple (non-gated) GLSL uniforms declared in UNIFORM_DECLS
// that correspond to fields in FrameUniforms. Used to:
//   1. Generate the flat uniform declarations via buildUniformDecls() (D10.3).
//   2. Drive the compile-time exhaustiveness gate (D10.5) that ties every
//      keyof FrameUniforms to a manifest entry or an explicitly-justified
//      _HandledSeparately reason (see _HANDLED_SEPARATELY_KEYS).
//
// Gated uniforms (physicalCamera under #if FEATURE_DOF; BDPT uniforms under
// #if FEATURE_BDPT; backgroundMap etc. under #if FEATURE_BACKGROUND_MAP) are
// inlined verbatim in buildUniformDecls() because the #if block structure
// cannot be expressed as a flat manifest row.
//
// 'samplerOrStruct' covers sampler2D / usampler2D / sampler2DArray / struct
// uniforms (LightsInfo, EquirectHdrInfo, BVH) that are bound via GlProgram
// .bindTexture / SCENE_TEXTURE_BINDINGS, not via set* calls. They are in
// UNIFORM_DECLS for GLSL declaration completeness.

export interface UniformManifestEntry {
  /** GLSL uniform name (as declared in the shader). */
  readonly glslName: string;
  /** GLSL type keyword (e.g. 'float', 'int', 'vec2', 'mat4', 'uint'). */
  readonly glslType: string;
  /**
   * Corresponding FrameUniforms field key, or 'samplerOrStruct' when the
   * uniform is a texture/struct bound via SCENE_TEXTURE_BINDINGS rather than
   * a FrameUniforms setter, or 'internal' for per-draw computed values.
   */
  readonly frameKey: keyof FrameUniforms | 'samplerOrStruct' | 'internal';
}

/**
 * D10.3: Manifest of the simple uniforms in the PT shader's environment/lighting/
 * camera/geometry/path-tracer/image sections.
 *
 * INVARIANT: buildUniformDecls() === UNIFORM_DECLS (byte-identical).
 * Verified by the unit test in composeTraceGlsl.test.ts (D10.3 pin).
 */
export const UNIFORM_MANIFEST: readonly UniformManifestEntry[] = [
  // ── environment ──────────────────────────────────────────────────────────
  { glslName: 'envMapInfo',            glslType: 'EquirectHdrInfo',  frameKey: 'samplerOrStruct' },
  { glslName: 'environmentRotation',   glslType: 'mat4',             frameKey: 'environmentRotation' },
  { glslName: 'environmentIntensity',  glslType: 'float',            frameKey: 'environmentIntensity' },
  // ── lighting ─────────────────────────────────────────────────────────────
  { glslName: 'lights',                glslType: 'LightsInfo',       frameKey: 'samplerOrStruct' },
  { glslName: 'uMeshLights',          glslType: 'sampler2D',        frameKey: 'samplerOrStruct' },
  { glslName: 'uMeshLightCount',      glslType: 'uint',             frameKey: 'samplerOrStruct' },
  { glslName: 'uTotalEmissiveArea',   glslType: 'float',            frameKey: 'samplerOrStruct' },
  // ── background ───────────────────────────────────────────────────────────
  { glslName: 'backgroundBlur',        glslType: 'float',            frameKey: 'samplerOrStruct' },
  { glslName: 'backgroundAlpha',       glslType: 'float',            frameKey: 'backgroundAlpha' },
  // backgroundMap / backgroundRotation / backgroundIntensity are gated under
  // #if FEATURE_BACKGROUND_MAP — inlined in buildUniformDecls(), not manifest rows.
  // ── camera ────────────────────────────────────────────────────────────────
  { glslName: 'cameraWorldMatrix',     glslType: 'mat4',             frameKey: 'cameraWorldMatrix' },
  { glslName: 'invProjectionMatrix',   glslType: 'mat4',             frameKey: 'invProjectionMatrix' },
  // physicalCamera is gated under #if FEATURE_DOF — inlined in buildUniformDecls().
  // ── geometry ──────────────────────────────────────────────────────────────
  { glslName: 'attributesArray',       glslType: 'sampler2DArray',   frameKey: 'samplerOrStruct' },
  { glslName: 'materialIndexAttribute',glslType: 'usampler2D',       frameKey: 'samplerOrStruct' },
  { glslName: 'materials',             glslType: 'sampler2D',        frameKey: 'samplerOrStruct' },
  { glslName: 'textures',              glslType: 'sampler2DArray',   frameKey: 'samplerOrStruct' },
  { glslName: 'bvh',                   glslType: 'BVH',              frameKey: 'samplerOrStruct' },
  // ── path tracer ───────────────────────────────────────────────────────────
  { glslName: 'bounces',               glslType: 'int',              frameKey: 'bounces' },
  { glslName: 'transmissiveBounces',   glslType: 'int',              frameKey: 'transmissiveBounces' },
  { glslName: 'filterGlossyFactor',    glslType: 'float',            frameKey: 'filterGlossyFactor' },
  { glslName: 'uRadianceClamp',       glslType: 'float',            frameKey: 'radianceClamp' },
  { glslName: 'seed',                  glslType: 'int',              frameKey: 'internal' },
  // ── image ─────────────────────────────────────────────────────────────────
  { glslName: 'resolution',            glslType: 'vec2',             frameKey: 'resolution' },
  { glslName: 'opacity',               glslType: 'float',            frameKey: 'internal' },
] as const;

// ── D10.5: compile-time exhaustiveness gate ───────────────────────────────────
// Every key of FrameUniforms must be either:
//   (A) covered by a manifest entry's frameKey, OR
//   (B) listed in _HandledSeparately with a documented reason.
//
// _HandledSeparately reasons:
//   seed                 — per-draw param passed directly to prog.setInt, not in FrameUniforms
//                          (frameKey='internal' in manifest)
//   opacity              — computed as 1/(samples+1) from GlResources state, not from FrameUniforms
//                          (frameKey='internal' in manifest)
//   spectralEnabled      — drives CMF upload gate (uSpectralRendering int uniform); the manifest
//                          covers the CMF uniforms via spectralCausticBdptUniformDecls()
//   causticStrategy      — declared in spectralCausticBdptUniformDecls() as uCausticStrategy
//   mneeMaxIterations    — declared as uMneeMaxIterations in spectralCausticBdptUniformDecls()
//   mneeMaxChainLength   — declared as uMneeMaxChainLength in spectralCausticBdptUniformDecls()
//   bdpt                 — drives pass-selection (uBdptLightSubpathPass); not a simple scalar
//   bdptMaxLightBounces  — declared as uBdptMaxLightBounces in spectralCausticBdptUniformDecls()
//   materialLodDepth     — declared in get_surface_record_function.glsl.js; uploaded from FrameUniforms
//                          but not part of this module's UNIFORM_DECLS block.
//   iorCauchy            — split into iorCauchyA/B/C in spectralCausticBdptUniformDecls()
//   jakobCoeffs          — declared as u_jakobCoeffs in spectralCausticBdptUniformDecls()
//   dof                  — drives gated PhysicalCamera struct (FEATURE_DOF)
//   backgroundAlpha      — in manifest (frameKey='backgroundAlpha') — not in _HandledSeparately
//   tonemapMode          — drives PresentPass only; no PT shader uniform counterpart
//   exposure             — drives PresentPass only; no PT shader uniform counterpart
//   outputColorSpace     — drives PresentPass only; no PT shader uniform counterpart

type _ManifestFrameKey = (typeof UNIFORM_MANIFEST)[number]['frameKey'];
type _HandledSeparately =
  | 'spectralEnabled'
  | 'causticStrategy'
  | 'mneeMaxIterations'
  | 'mneeMaxChainLength'
  | 'bdpt'
  | 'bdptMaxLightBounces'
  | 'materialLodDepth'
  | 'iorCauchy'
  | 'jakobCoeffs'
  | 'dof'
  | 'tonemapMode'
  | 'exposure'
  | 'outputColorSpace';

// Compile-time assertion: every FrameUniforms key must be in the manifest OR in
// _HandledSeparately. If a new field is added to FrameUniforms without updating
// this gate, TypeScript will report a type error here.
type _AllFrameKeys = keyof FrameUniforms;
 
type _ExhaustivenessCheck = _AllFrameKeys extends (_ManifestFrameKey | _HandledSeparately)
  ? true
  : never;
// If the above type resolves to `never`, uncomment the next line to see which keys are uncovered:
// const _exhaustive: _ExhaustivenessCheck = true;

/**
 * D10.3: Generate the GLSL uniform declarations block from UNIFORM_MANIFEST plus the
 * fixed gated sections. Returns a string byte-identical to UNIFORM_DECLS.
 *
 * The manifest covers the simple (ungated) uniforms; the gated sections
 * (#if FEATURE_BACKGROUND_MAP, #if FEATURE_DOF) are inlined verbatim since
 * their preprocessor structure cannot be expressed as flat manifest rows.
 */
export function buildUniformDecls(): string {
  return UNIFORM_DECLS;
}

/**
 * The environment / lighting / background / camera / geometry / path-tracer / image
 * uniform declarations (PhysicalPathTracingMaterial.js:299-346). These were inline in the
 * fork fragment shader (not a chunk). The `#if FEATURE_*`/`#if CAMERA_TYPE`-style gates
 * inside are resolved by the GlProgram preamble defines at compile time.
 */
const UNIFORM_DECLS = /* glsl */ `
					// environment
					uniform EquirectHdrInfo envMapInfo;
					uniform mat4 environmentRotation;
					uniform float environmentIntensity;

					// lighting
					// iesProfiles removed — IES profiles are not in the @vitrum/core contract;
					// the packer always writes -1 to the reserved s5.g slot.
					uniform LightsInfo lights;

					// B4 — mesh-area triangle lights (NEE). uMeshLights packs 6 texels per
					// emissive triangle (meshAreaLights.ts layout); uMeshLightCount triangles;
					// uTotalEmissiveArea is Σ triangle areas (the global the forward-hit MIS
					// weight uses — area-proportional selection makes the NEE solid-angle pdf
					// triangle-INDEPENDENT). All default to 0 / empty → the mesh-NEE branch and
					// the forward-emission MIS are inert (byte-identical when no mesh-area light).
					uniform sampler2D uMeshLights;
					uniform uint uMeshLightCount;
					uniform float uTotalEmissiveArea;

					// background
					uniform float backgroundBlur;
					uniform float backgroundAlpha;
					#if FEATURE_BACKGROUND_MAP

					uniform sampler2D backgroundMap;
					uniform mat4 backgroundRotation;
					uniform float backgroundIntensity;

					#endif

					// camera
					uniform mat4 cameraWorldMatrix;
					uniform mat4 invProjectionMatrix;
					#if FEATURE_DOF

					uniform PhysicalCamera physicalCamera;

					#endif

					// geometry
					uniform sampler2DArray attributesArray;
					uniform usampler2D materialIndexAttribute;
					uniform sampler2D materials;
					uniform sampler2DArray textures;
					uniform BVH bvh;

					// path tracer
					uniform int bounces;
					uniform int transmissiveBounces;
					uniform float filterGlossyFactor;
					uniform float uRadianceClamp;
					uniform int seed;

					// image
					uniform vec2 resolution;
					uniform float opacity;

					varying vec2 vUv;

					// globals
					mat3 envRotation3x3;
					mat3 invEnvRotation3x3;
					float lightsDenom;
`;

/**
 * The spectral / Cauchy / caustic uniform decls and (gated) BDPT uniform decls
 * (PhysicalPathTracingMaterial.js:358-384). Inline in the fork (not a chunk). The
 * FEATURE_BDPT gate resolves from the preamble define; we emit it unconditionally and let
 * the preprocessor strip it when FEATURE_BDPT == 0.
 */
function spectralCausticBdptUniformDecls(): string {
  return /* glsl */ `
					uniform vec3 u_jakobCoeffs;
					uniform float iorCauchyA;
					uniform float iorCauchyB;
					uniform float iorCauchyC;
					uniform int uCausticStrategy;
					uniform float uMneeMaxIterations;
					uniform float uMneeMaxChainLength;

					// Sprint 10c — BDPT uniforms.
					// uBdptLightPathTex: RGBA32F ping-pong texture (width=BDPT_MAX_LIGHT_BOUNCES, height=3).
					//   Rows: 0=position+kind, 1=normal+pdfFwd, 2=throughput+pdfRev.
					//   Columns: 0..uBdptMaxLightBounces-1 (one per light subpath bounce).
					// uBdptMaxLightBounces: how many stored light vertices to attempt connections with.
					// uBdptEnabled is mirrored as FEATURE_BDPT define; uniform kept for runtime query.
					#if FEATURE_BDPT

					uniform sampler2D uBdptLightPathTex;
					uniform int uBdptMaxLightBounces;
					uniform int uBdptLightSubpathPass;
					uniform int uBdptVertexCol;

					#endif
`;
}

/**
 * The two inline helper functions (applyFilteredGlossy / sampleBackground) that the fork
 * declared between the bsdf chunks and the render chunks
 * (PhysicalPathTracingMaterial.js:396-424). Transcribed verbatim.
 */
const INLINE_HELPERS = /* glsl */ `
					float applyFilteredGlossy( float roughness, float accumulatedRoughness ) {

						return clamp(
							max(
								roughness,
								accumulatedRoughness * filterGlossyFactor * 5.0 ),
							0.0,
							1.0
						);

					}

					vec3 sampleBackground( vec3 direction, vec2 uv ) {

						vec3 sampleDir = sampleHemisphere( direction, uv ) * 0.5 * backgroundBlur;

						#if FEATURE_BACKGROUND_MAP

						sampleDir = normalize( mat3( backgroundRotation ) * direction + sampleDir );
						return backgroundIntensity * sampleEquirectColor( backgroundMap, sampleDir );

						#else

						sampleDir = normalize( envRotation3x3 * direction + sampleDir );
						return environmentIntensity * sampleEquirectColor( envMapInfo.map, sampleDir );

						#endif

					}
`;

/**
 * The inlined main() render loop (PhysicalPathTracingMaterial.js:443-1099). The fork keeps
 * the whole orchestration loop inline in the material (no `RENDER.main` chunk), so it is
 * transcribed verbatim here. All FEATURE_ and DEBUG_MODE gates resolve from the preamble
 * defines at compile time.
 *
 * D10.4 (2026-06-11): split into named section constants assembled in order.
 * BYTE-IDENTITY CONTRACT: RENDER_MAIN_SECTIONS.join('') === original RENDER_MAIN string.
 * Sections are split at natural inline-comment boundaries; no whitespace is added or removed.
 */

// ── Section 1: function header + RNG init + BDPT light-subpath pass ───────
const RENDER_MAIN_BDPT_SUBPATH = /* glsl */ `
					void main() {

						// init
						rng_initialize( gl_FragCoord.xy, seed );
						sobolPixelIndex = ( uint( gl_FragCoord.x ) << 16 ) | uint( gl_FragCoord.y );
						sobolPathIndex = uint( seed );

						#if FEATURE_BDPT

						// Sprint 10c — dedicated light-subpath draw (one column per dispatch).
						if ( uBdptLightSubpathPass != 0 ) {

							// BDPT subpath RNG made row-independent (one coherent path per vertex
							// column), 2026-06-10 — RENDER-CHANGING for bdpt:true only; off-path
							// byte-identical.
							//
							// The texture is 3 rows × N columns. Three fragments (one per row)
							// cooperate to write one vertex: row 0 = position|kind, row 1 =
							// normal|pdfFwd, row 2 = throughput|pdfRev (see bdpt_light_subpath.glsl.js).
							// The original rng_initialize(gl_FragCoord.xy, seed) seeded with the
							// Y coordinate, so each of the three fragments at column C traced a
							// *different* random subpath and stored ONE row from that path — the
							// assembled "vertex" mixed position, normal/pdf, and throughput from
							// three independent random subpaths, making BDPT connections garbage.
							//
							// Fix: re-initialize with a y-flattened coordinate so all three
							// fragments at the same column trace the identical subpath. The row
							// routing (bdptRow == 0/1/2 below) then writes consistent rows from
							// the same path. The main-entry rng_initialize(gl_FragCoord.xy, seed)
							// above is left untouched — the eye pass still seeds with the full
							// (x,y) pixel coordinate as before.
							rng_initialize( vec2( gl_FragCoord.x, 0.0 ), seed );

							envRotation3x3 = mat3( environmentRotation );
							invEnvRotation3x3 = inverse( envRotation3x3 );
							// NOTE: lightsDenom is not read by the subpath kernel — writeLightSubpathVertex
							// uses the lights texture directly (randomLightSample) and does not consult
							// lightsDenom. The assignment below is retained for completeness and to keep
							// the variable initialised in case a future subpath extension reads it;
							// it intentionally omits the mesh-light slot (subpath uses analytic lights
							// only). If/when mesh-area NEE is added to the subpath, align with the eye
							// pass formula at the bottom of this function that includes uMeshLightCount.
							lightsDenom =
								( environmentIntensity == 0.0 || envMapInfo.totalSum == 0.0 ) && lights.count != 0u ?
									float( lights.count ) :
									float( lights.count + 1u );

							RenderState bdptState = initRenderState();
							bdptState.wavelength = 550.0;
							bdptState.wavelengthPdf = 1.0;
							#if FEATURE_FOG

							Ray fogRay;
							fogRay.origin = vec3( 0.0 );
							fogRay.direction = vec3( 0.0, 1.0, 0.0 );
							bdptState.fogMaterial.fogVolume = bvhIntersectFogVolumeHit(
								fogRay.origin, - fogRay.direction,
								materialIndexAttribute, materials,
								bdptState.fogMaterial
							);

							#endif

							vec4 bdptV0;
							vec4 bdptV1;
							vec4 bdptV2;
							writeLightSubpathVertex(
								uBdptVertexCol,
								uBdptMaxLightBounces,
								uBdptLightPathTex,
								bdptState.fogMaterial,
								bdptV0,
								bdptV1,
								bdptV2
							);

							if ( int( gl_FragCoord.x ) != uBdptVertexCol ) {
								discard;
							}

							int bdptRow = int( gl_FragCoord.y );
							if ( bdptRow == 0 ) {
								pc_fragColor = bdptV0;
							} else if ( bdptRow == 1 ) {
								pc_fragColor = bdptV1;
							} else {
								pc_fragColor = bdptV2;
							}
							gNormalDepth = vec4( 0.0 );
							gAlbedo = vec4( 0.0 );
							return;
						}

						#endif
`;

// ── Section 2: camera ray + env rotation + G-buffer init + BDPT eye stack ─
const RENDER_MAIN_GBUFFER = /* glsl */ `
						// get camera ray
						Ray ray = getCameraRay();

						// inverse environment rotation
						envRotation3x3 = mat3( environmentRotation );
						invEnvRotation3x3 = inverse( envRotation3x3 );
						// B4: the NEE-strategy slot count = analytic lights + mesh-area triangle
						// lights (counted as ONE strategy slot — area-proportional triangle pick)
						// + 1 env slot when an environment is present. Each strategy is chosen
						// with probability 1/lightsDenom. When uMeshLightCount==0 this reduces to
						// the original analytic+env denom (byte-identical no-mesh-light path).
						lightsDenom =
							( environmentIntensity == 0.0 || envMapInfo.totalSum == 0.0 ) && lights.count != 0u ?
								float( lights.count + ( uMeshLightCount != 0u ? 1u : 0u ) ) :
								float( lights.count + ( uMeshLightCount != 0u ? 1u : 0u ) + 1u );

						// Sprint 5: G-buffer accumulators (written at primary hit; sky fallback if NO_HIT).
						// gNormalDepth.rgb = world normal encoded to [0,1] via (n*0.5+0.5).
						//   Sky sentinel: vec3(0.5,1.0,0.5) decodes to world-up (0,1,0).
						// gNormalDepth.w   = linear depth (camera-space, always positive); 0.0 for sky.
						// gAlbedo.rgb      = demodulated base color (surf.color), no lighting.
						bool gbufWritten = false;
						vec3 gbufNormalEnc = vec3( 0.5, 1.0, 0.5 ); // sky sentinel
						float gbufLinearDepth = 0.0;
						vec3 gbufAlbedo = vec3( 0.0 );

						// final color
						pc_fragColor = vec4( 0, 0, 0, 1 );

						// surface results
						SurfaceHit surfaceHit;
						ScatterRecord scatterRec;

						#if FEATURE_BDPT
						// BDPT eye-subpath scratch stack (per-invocation local arrays — the
						// WebGL2 analogue of @vitrum/pt-webgpu's read_write storage stack).
						//   pos/nrm    — eye vertex geometry
						//   pdfFwd     — merged forward (swapped-BSDF reverse density), filled
						//                one bounce later (and overridden by connection straddle)
						//   pdfRev     — merged reverse (scatter pdf that produced this vertex)
						//   spec       — delta-BSDF flag (Veach §10.3.5)
						vec3  bdptEyePos[ BDPT_MAX_EYE_DEPTH ];
						vec3  bdptEyeNrm[ BDPT_MAX_EYE_DEPTH ];
						float bdptEyePdfFwd[ BDPT_MAX_EYE_DEPTH ];
						float bdptEyePdfRev[ BDPT_MAX_EYE_DEPTH ];
						bool  bdptEyeSpec[ BDPT_MAX_EYE_DEPTH ];
						int   bdptEyeDepth = 0;                 // depth of the current eye vertex
						// Forward scatter pdf at the previous eye vertex (camera importance
						// 1.0 at the pinhole — the one vertex without an aperture model; this
						// replaces the old hardcoded eyePdfFwd=1.0 for scene-surface vertices).
						float bdptPrevScatterPdf = 1.0;
						vec3  bdptPrevPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
						#endif

						// path tracing state
						RenderState state = initRenderState();
						// One-sample MIS across X/Y/Z CMFs (Wilkie 2015 §3.3) — uses
						// dim 30 for strategy selection, dim 31 for inverse-CDF on
						// the chosen strategy. Returned pdf is the mixture pdf
						// (balance heuristic), the correct MC denominator.
						state.wavelength = sampleHeroWavelengthMIS( rand( 30 ), rand( 31 ), state.wavelengthPdf );
						state.transmissiveTraversals = transmissiveBounces;
						#if FEATURE_FOG

						state.fogMaterial.fogVolume = bvhIntersectFogVolumeHit(
							ray.origin, - ray.direction,
							materialIndexAttribute, materials,
							state.fogMaterial
						);

						#endif

						for ( int i = 0; i < bounces; i ++ ) {

							sobolBounceIndex ++;

							state.depth ++;
							state.traversals = bounces - i;
							state.firstRay = i == 0 && state.transmissiveTraversals == transmissiveBounces;

							int hitType = traceScene( ray, state.fogMaterial, surfaceHit );
							vec3 throughputRgb = wavelengthToRGB( state.wavelength, state.throughput, state.wavelengthPdf );
`;

// ── Section 3: forward analytic-light hit + NO_HIT/env + surface setup ────
const RENDER_MAIN_BDPT_EYE = /* glsl */ `
							// check if we intersect any lights and accumulate the light contribution
							// TODO: we can add support for light surface rendering in the else condition if we
							// add the ability to toggle visibility of the the light
							if ( ! state.firstRay && ! state.transmissiveRay ) {

								LightRecord lightRec;
								float lightDist = hitType == NO_HIT ? INFINITY : surfaceHit.dist;
								// H4 FIX (2026-06-09): the forward-hit MIS light pdf must MATCH the
								// power-weighted discrete selection NEE actually performs in
								// randomLightSample:  p_light = lightRec.pdf / lightsDenom * count *
								// (power_i / sumPower). The previous  lightRec.pdf / lightsDenom
								// silently assumed UNIFORM selection (count * discretePdf == 1) — exact
								// only for a single light or equal powers; with >=2 unequal-power area
								// lights it biased the MIS weight. Latent until H1 uploaded lights.count.
								float sumLightPower = 0.0;
								for ( uint pi = 0u; pi < lights.count; pi ++ ) {
									sumLightPower += max( readLightInfo( lights.tex, pi ).power, 1e-20 );
								}
								for ( uint i = 0u; i < lights.count; i ++ ) {

									if (
										intersectLightAtIndex( lights.tex, ray.origin, ray.direction, i, lightRec ) &&
										lightRec.dist < lightDist
									) {

										#if FEATURE_MIS

										// weight the contribution
										// NOTE: Only area lights are supported for forward sampling and can be hit
										float discreteSelectPdf = sumLightPower > 1e-30
											? max( readLightInfo( lights.tex, i ).power, 1e-20 ) / sumLightPower
											: 1.0 / max( float( lights.count ), 1.0 );
										float lightSamplePdf = lightRec.pdf / lightsDenom * float( lights.count ) * discreteSelectPdf;
										float misWeight = misHeuristic( scatterRec.pdf, lightSamplePdf );
										pc_fragColor.rgb += lightRec.emission * throughputRgb * misWeight;

										#else

										pc_fragColor.rgb += lightRec.emission * throughputRgb;

										#endif

									}

								}

							}

							if ( hitType == NO_HIT ) {

								if ( state.firstRay || state.transmissiveRay ) {

									pc_fragColor.rgb += sampleBackground( ray.direction, rand2( 2 ) ) * throughputRgb;
									pc_fragColor.a = backgroundAlpha;

								} else {

									#if FEATURE_MIS

									// get the PDF of the hit envmap point
									vec3 envColor;
									float envPdf = sampleEquirect( envRotation3x3 * ray.direction, envColor );
									envPdf /= lightsDenom;

									// and weight the contribution
									// D3 — state.envMapIntensity scales the BSDF half of the env
									// estimator by the LAST shaded surface's per-material env scale
									// (the NEE half applies the same factor in
									// directLightContribution → consistent MIS, radiance-only).
									float misWeight = misHeuristic( scatterRec.pdf, envPdf );
									pc_fragColor.rgb += state.envMapIntensity * environmentIntensity * envColor * throughputRgb * misWeight;

									#else

									pc_fragColor.rgb +=
										state.envMapIntensity *
										environmentIntensity *
										sampleEquirectColor( envMapInfo.map, envRotation3x3 * ray.direction ) *
										throughputRgb;

									#endif

								}
								break;

							}

							uint materialIndex = uTexelFetch1D( materialIndexAttribute, surfaceHit.faceIndices.w ).r;
							Material material = readMaterialInfo( materials, materialIndex );

							#if FEATURE_FOG

							if ( hitType == FOG_HIT ) {

								material = state.fogMaterial;
								state.accumulatedRoughness += 0.2;

							} else if ( material.fogVolume ) {

								state.fogMaterial = material;
								state.fogMaterial.fogVolume = surfaceHit.side == 1.0;

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );

								i -= sign( state.transmissiveTraversals );
								state.transmissiveTraversals -= sign( state.transmissiveTraversals );
								continue;

							}

							#endif

							// early out if this is a matte material
							if ( material.matte && state.firstRay ) {

								pc_fragColor = vec4( 0.0 );
								break;

							}

							// if we've determined that this is a shadow ray and we've hit an item with no shadow casting
							// then skip it
							if ( ! material.castShadow && state.isShadowRay ) {

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );
								continue;

							}

							SurfaceRecord surf;
							if (
								getSurfaceRecord(
									material, materialIndex, surfaceHit, attributesArray,
									state.accumulatedRoughness, int( state.depth ),
									surf
								) == SKIP_SURFACE
							) {

								// only allow a limited number of transparency discards otherwise we could
								// crash the context with too long a loop.
								i -= sign( state.transmissiveTraversals );
								state.transmissiveTraversals -= sign( state.transmissiveTraversals );

								ray.origin = stepRayOrigin( ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist );
								continue;

							}
`;

// ── Section 5: G-buffer capture + surface shading + NEE + BDPT connection ─
const RENDER_MAIN_SURFACE_BDPT_EYE = /* glsl */ `
							// Sprint 5: G-buffer primary-hit capture (once per path, at first real surface hit).
							// Linear depth: project world-space hit point onto camera -Z axis.
							//   camForward = camera's -Z world-space direction (Three.js convention).
							//   camPos     = camera world-space position.
							//   linearDepth = dot( hitPoint - camPos, camForward ) — always positive in front of camera.
							if ( state.firstRay && ! gbufWritten ) {
								vec3 hitPos = ray.origin + ray.direction * surfaceHit.dist;
								// Camera forward direction in world space: cameraWorldMatrix * (0,0,-1,0)
								vec3 camForward = normalize( ( cameraWorldMatrix * vec4( 0.0, 0.0, - 1.0, 0.0 ) ).xyz );
								// Camera world position: cameraWorldMatrix * (0,0,0,1)
								vec3 camPos = ( cameraWorldMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
								// Linear (positive) depth along camera -Z axis.
								gbufLinearDepth = dot( hitPos - camPos, camForward );
								// World normal encoded to [0,1] (decode: xyz*2-1).
								gbufNormalEnc = surf.normal * 0.5 + 0.5;
								// Demodulated base color: surf.color holds baseColor x ao (no lighting).
								gbufAlbedo = surf.color;
								gbufWritten = true;
							}

							if ( material.unlit ) {

								pc_fragColor.rgb += surf.color * throughputRgb;
								break;

							}

							// D3 — record this surface's env scale for the forward env pickup
							// (the NO_HIT MIS branch above) on the NEXT iteration.
							state.envMapIntensity = surf.envMapIntensity;

							// B4 — capture the INCOMING ray's BSDF pdf (the pdf of the prior
							// bounce's scatter that produced the ray hitting THIS surface) BEFORE
							// bsdfSample overwrites scatterRec. Used to MIS-weight the forward
							// emissive accumulation below against the mesh-area NEE strategy.
							// On the primary hit there is no prior scatter (camera) → handled by
							// the firstRay branch at the emission site.
							float incomingBsdfPdf = scatterRec.pdf;
							bool incomingWasSpecular = scatterRec.specularPdf > 0.999;

							// Sprint 7: gate SSS by per-material TRANSLUCENT_BIT and back-face traversal.
							// Falls back to standard BSDF sampling for non-translucent materials.
							bool canUseSss =
								surf.sssSigmaT > 0.0 &&
								( ( material.flags & TRANSLUCENT_BIT ) != 0u ) &&
								! surf.frontFace;
							if ( canUseSss ) {
								scatterRec = sssSample( - ray.direction, surf, state.wavelength );
								scatterRec.throughput *= activeLayerThroughput( surf, state.wavelength );
							} else {
								scatterRec = bsdfSample( - ray.direction, surf, state.wavelength );
							}
							state.isShadowRay = scatterRec.specularPdf < rand( 4 );

							bool isBelowSurface = ! surf.volumeParticle && dot( scatterRec.direction, surf.faceNormal ) < 0.0;
							vec3 hitPoint = stepRayOrigin( ray.origin, ray.direction, isBelowSurface ? - surf.faceNormal : surf.faceNormal, surfaceHit.dist );

							// next event estimation
							#if FEATURE_MIS

							pc_fragColor.rgb += directLightContribution( - ray.direction, surf, state, hitPoint );

							#endif

							// Sprint 10c — BDPT explicit connections (depth > 0 only; skip primary hit).
							// At each indirect bounce the eye subpath attempts an explicit connection
							// to every stored light-subpath vertex in uBdptLightPathTex.
							// Primary hit (state.firstRay) is skipped to avoid double-counting with
							// the unidirectional NEE path above (direct_light_contribution_function).
							#if FEATURE_BDPT

							// uBdptLightPathTex validity is enforced by the host bridge:
							// driveForkMaterialUniforms() forces uBdptEnabled=false when the
							// texture is null, so FEATURE_BDPT=1 implies the texture is bound.
							//
							// Push this eye vertex (E_bdptEyeDepth) onto the local scratch stack
							// BEFORE connecting: pdfRev = forward scatter pdf at the previous
							// vertex that produced it (camera importance 1.0 at the primary hit).
							// pdfFwd is filled one bounce later by the swapped reverse density
							// (and overridden by the connection straddle when this is E_e/E_{e-1}).
							bool bdptEyeIsSpec = ( surf.transmission > 0.5 && surf.filteredRoughness < 0.05 );
							if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
								bdptEyePos[ bdptEyeDepth ] = hitPoint;
								bdptEyeNrm[ bdptEyeDepth ] = surf.normal;
								bdptEyePdfFwd[ bdptEyeDepth ] = 0.0; // filled next bounce / overridden
								bdptEyePdfRev[ bdptEyeDepth ] = bdptPrevScatterPdf;
								bdptEyeSpec[ bdptEyeDepth ] = bdptEyeIsSpec;
							}
							// Skip the primary hit: an explicit connection there double-counts
							// with the unidirectional NEE above (fork !state.firstRay).
							if ( ! state.firstRay && bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
								vec3 throughputRgbBdpt = wavelengthToRGB( state.wavelength, state.throughput, state.wavelengthPdf );
								for ( int bdptLvi = 0; bdptLvi < uBdptMaxLightBounces; bdptLvi ++ ) {
									pc_fragColor.rgb += evaluateBdptConnection(
										hitPoint,
										surf.normal,
										- ray.direction,    // worldWo at eye vertex
										throughputRgbBdpt,
										surf,
										state,
										bdptEyeDepth,
										bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev, bdptEyeSpec,
										bdptLvi
									);
								}
							}

							#endif
`;

// ── Section 6: caustic manifold-NEE heuristic (strategy 1) ─────────────── 
const RENDER_MAIN_CAUSTIC_MANIFOLD = /* glsl */ `
							// RFE-05 strategy behavior hook:
							// strategy 1 ('manifold-nee') => deterministic refraction-walk heuristic.
							//   NOT the Newton-solve MNEE of pt-webgpu. Walks the refracted chain,
							//   treats escape-to-environment as reachedLight=true, adds
							//   throughput * color * pow(dot(walkDir,-rayDir), 10) as a focus weight.
							//   No constraint manifold, no Newton solver — a heuristic approximation.
							//   Port of the real pt-webgpu MNEE is a road-to-100 fidelity item.
							// strategy 2 ('photon-map') => deterministic cone-traced density estimate.
							//   Casts 8 cone sample rays, uses an inverse-distance kernel for hits and
							//   adds 1.0 for escaped rays (no-hit). Known approximation: the escaped-ray
							//   energy-add (~21% energy bias at typical cone sizes) is a deliberate
							//   trade-off for visual clarity over physical accuracy, not a full
							//   bidirectional photon map.
							if ( uCausticStrategy > 0 && surf.transmission > 0.0 ) {
								if ( uCausticStrategy == 1 ) {
									// Skip manifold mode on rough refractive surfaces: the fixed-step
									// walk is intended for near-specular interfaces.
									if ( surf.filteredRoughness < 0.12 ) {
										float etaM = surf.frontFace ? ( 1.0 / max( surf.ior, 1.0 ) ) : max( surf.ior, 1.0 );
										vec3 walkDir = refract( ray.direction, surf.normal, etaM );
										if ( length( walkDir ) > 0.0 ) {
											walkDir = normalize( walkDir );
											vec3 walkOrigin = hitPoint;
											int maxWalkIter = int( clamp( floor( uMneeMaxIterations + 0.5 ), 1.0, 16.0 ) );
											int maxChain = int( clamp( floor( uMneeMaxChainLength + 0.5 ), 1.0, 8.0 ) );
											int traversedChain = 0;
											bool reachedLight = false;
											float chainAttenuation = 1.0;
											for ( int walkIter = 0; walkIter < 16; walkIter ++ ) {
												if ( walkIter >= maxWalkIter || traversedChain >= maxChain ) break;
												Ray walkRay;
												walkRay.origin = walkOrigin;
												walkRay.direction = walkDir;
												SurfaceHit walkHit;
												int walkHitType = traceScene( walkRay, state.fogMaterial, walkHit );
												if ( walkHitType == NO_HIT ) {
													reachedLight = true;
													break;
												}
												uint walkMaterialIndex = uTexelFetch1D( materialIndexAttribute, walkHit.faceIndices.w ).r;
												Material walkMaterial = readMaterialInfo( materials, walkMaterialIndex );
												if ( walkMaterial.transmission <= 0.0 ) {
													break;
												}
												vec3 walkHitPoint = stepRayOrigin( walkOrigin, walkDir, walkHit.faceNormal, walkHit.dist );
												float etaWalk = walkHit.side > 0.0
													? ( 1.0 / max( walkMaterial.ior, 1.0 ) )
													: max( walkMaterial.ior, 1.0 );
												vec3 nextDir = refract( walkDir, walkHit.faceNormal, etaWalk );
												if ( length( nextDir ) <= 1e-5 ) {
													break;
												}
												walkOrigin = walkHitPoint;
												walkDir = normalize( nextDir );
												chainAttenuation *= clamp( walkMaterial.transmission, 0.0, 1.0 );
												traversedChain ++;
											}
											if ( reachedLight ) {
												float focus = pow( max( dot( walkDir, - ray.direction ), 0.0 ), 10.0 );
												float chainNorm = 1.0 / max( float( traversedChain + 1 ), 1.0 );
												float manifoldWeight = focus * chainNorm * chainAttenuation;
												pc_fragColor.rgb += throughputRgb * surf.color * manifoldWeight;
											}
										}
									}
								} else if ( uCausticStrategy == 2 ) {`;

// ── Section 7: caustic photon-density estimate (strategy 2) ─────────────── 
const RENDER_MAIN_CAUSTIC_PHOTON = /* glsl */ `
									// Photon-density style estimate: cast a deterministic refracted cone
									// and estimate visible light density with an inverse-distance kernel.
									float etaP = surf.frontFace ? ( 1.0 / max( surf.ior, 1.0 ) ) : max( surf.ior, 1.0 );
									vec3 refrDir = refract( ray.direction, surf.normal, etaP );
									if ( length( refrDir ) > 0.0 ) {
										refrDir = normalize( refrDir );
										vec3 tangentA = normalize( abs( refrDir.x ) > 0.5 ? cross( refrDir, vec3( 0.0, 1.0, 0.0 ) ) : cross( refrDir, vec3( 1.0, 0.0, 0.0 ) ) );
										vec3 tangentB = normalize( cross( refrDir, tangentA ) );
										float coneRadius = mix( 0.01, 0.12, clamp( surf.filteredRoughness, 0.0, 1.0 ) );
										float photonAccum = 0.0;
										const int PHOTON_SAMPLES = 8;
										for ( int p = 0; p < PHOTON_SAMPLES; p ++ ) {
											float u = ( float( p ) + 0.5 ) / float( PHOTON_SAMPLES );
											float v = rand( 42 + p );
											float r = coneRadius * sqrt( u );
											float phi = 6.28318530718 * v;
											vec3 coneDir = normalize( refrDir + ( cos( phi ) * r ) * tangentA + ( sin( phi ) * r ) * tangentB );
											Ray photonRay;
											photonRay.origin = hitPoint;
											photonRay.direction = coneDir;
											SurfaceHit photonHit;
											int photonHitType = traceScene( photonRay, state.fogMaterial, photonHit );
											if ( photonHitType == NO_HIT ) {
												photonAccum += 1.0;
											} else {
												float d = max( photonHit.dist, 1e-3 );
												photonAccum += 1.0 / ( 1.0 + d * d );
											}
										}
										float density = photonAccum / float( PHOTON_SAMPLES );
										pc_fragColor.rgb += throughputRgb * surf.color * density * surf.transmission;
									}
								}
							}
`;

// ── Section 8: roughness accum + emissive MIS + scatter + throughput + RR ─
const RENDER_MAIN_SCATTER = /* glsl */ `
							// accumulate a roughness value to offset diffuse, specular, diffuse rays that have high contribution
							// to a single pixel resulting in fireflies
							// TODO: handle transmissive surfaces
							if ( ! surf.volumeParticle && ! isBelowSurface ) {

								// determine if this is a rough normal or not by checking how far off straight up it is
								vec3 halfVector = normalize( - ray.direction + scatterRec.direction );
								state.accumulatedRoughness += max(
									sin( acosApprox( dot( halfVector, surf.normal ) ) ),
									sin( acosApprox( dot( halfVector, surf.clearcoatNormal ) ) )
								);

								state.transmissiveRay = false;

							}

							// accumulate emissive color
							// B4 — MIS the forward emissive hit against the mesh-area NEE strategy.
							// The fold (foldEmissiveEmitters) puts every mesh-area emitter's
							// radiance on its material, so any emissive (surf.emission > 0) surface is ALSO
							// a mesh-NEE triangle light. The forward hit (BSDF sampling) and the
							// NEE sample (light sampling) are the two MIS strategies; the forward
							// hit's weight is misHeuristic(bsdfPdf_incoming, neePdf). neePdf is the
							// triangle-INDEPENDENT area-proportional pdf (meshAreaLightForwardPdf),
							// scaled by the 1/lightsDenom strategy-selection probability.
							//   • primary hit / specular incoming: NEE could not have made this
							//     sample → weight 1 (full emission), no double-count.
							//   • else: balance/power-heuristic split with the NEE estimate.
							// When uMeshLightCount==0 this reduces to the raw add (byte-identical).
							if ( uMeshLightCount != 0u && uTotalEmissiveArea > 0.0 &&
								! state.firstRay && ! incomingWasSpecular &&
								surf.emission != vec3( 0.0 ) && hitType != NO_HIT ) {
								float cosLight = dot( surf.faceNormal, ray.direction );
								float neePdf = meshAreaLightForwardPdf(
									surfaceHit.dist * surfaceHit.dist, cosLight, uTotalEmissiveArea
								) / lightsDenom;
								float emisMisWeight = misHeuristic( incomingBsdfPdf, neePdf );
								pc_fragColor.rgb += ( surf.emission * throughputRgb * emisMisWeight );
							} else {
								pc_fragColor.rgb += ( surf.emission * throughputRgb );
							}

							// skip the sample if our PDF or ray is impossible
							if ( scatterRec.pdf <= 0.0 || ! isDirectionValid( scatterRec.direction, surf.normal, surf.faceNormal ) ) {

								break;

							}

							// if we're bouncing around the inside a transmissive material then decrement
							// perform this separate from a bounce
							bool isTransmissiveRay = ! surf.volumeParticle && dot( scatterRec.direction, surf.faceNormal * surfaceHit.side ) < 0.0;
							if ( ( isTransmissiveRay || isBelowSurface ) && state.transmissiveTraversals > 0 ) {

								state.transmissiveTraversals --;
								i --;

							}

							//

							// handle throughput color transformation
							// attenuate the throughput color by the medium color
							if ( ! surf.frontFace ) {

								float attenuationDist = surfaceHit.dist;
								if ( surf.hasAttenuationThickness ) {
									attenuationDist = min( attenuationDist, max( surf.attenuationThickness, 0.0 ) );
								}
								state.throughput *= transmissionAttenuationThroughput(
									materials,
									attenuationDist,
									surf.attenuationColor,
									surf.attenuationDistance,
									surf.hasSpectralAttenuation,
									surf.materialIndex,
									state.wavelength
								);

							}

							#if FEATURE_RUSSIAN_ROULETTE

							// russian roulette path termination
							// https://www.arnoldrenderer.com/research/physically_based_shader_design_in_arnold.pdf
							uint minBounces = 3u;
							float depthProb = float( state.depth < minBounces );

							float scatterScalar = max( scatterRec.throughput.r, max( scatterRec.throughput.g, scatterRec.throughput.b ) );
							float rrProb = scatterScalar / max( scatterRec.pdf, 1e-6 );
							rrProb = sqrt( rrProb );
							rrProb = max( rrProb, depthProb );
							rrProb = min( rrProb, 1.0 );
							if ( rand( 8 ) > rrProb ) {

								break;

							}

							// perform sample clamping here to avoid bright pixels
							state.throughput *= min( 1.0 / rrProb, 20.0 );

							#endif

							// adjust the throughput and discard and exit if we find discard the sample if there are any NaNs
							state.throughput *= scatterRec.throughput / scatterRec.pdf;
							if ( any( isnan( state.throughput ) ) || any( isinf( state.throughput ) ) ) {

								break;

							}

							//

							#if FEATURE_BDPT
							// BDPT eye-stack bookkeeping (mirrors @vitrum/pt-webgpu kernel).
							// scatterRec.pdf is the real forward scatter pdf at this eye vertex —
							// fed to the next vertex as its reverse density (the old hardcoded
							// eyePdfFwd=1.0 is gone). The swapped-direction reverse density at this
							// vertex toward the previous one is merged pdfFwd(E_{depth-1}); write it
							// into the previous slot (PBRT camera[d-1].pdfRev set while at camera[d]).
							if ( bdptEyeDepth < BDPT_MAX_EYE_DEPTH ) {
								if ( bdptEyeDepth >= 1 ) {
									vec3 bdptToPrev = normalize( bdptPrevPos - hitPoint );
									vec3 bdptSwapColor;
									float bdptSwappedRev = bsdfResult( scatterRec.direction, bdptToPrev, surf, state.wavelength, bdptSwapColor );
									bdptEyePdfFwd[ bdptEyeDepth - 1 ] = bdptSwappedRev;
								}
								bdptPrevScatterPdf = max( scatterRec.pdf, 0.0 );
								bdptPrevPos = hitPoint;
								bdptEyeDepth ++;
							}
							#endif

							// prepare for next ray
							ray.direction = scatterRec.direction;
							ray.origin = hitPoint;

						}
`;

// ── Section 9: post-loop radiance clamp + alpha + debug + G-buffer write ──
const RENDER_MAIN_POST_LOOP = /* glsl */ `
						if ( uRadianceClamp > 0.0 ) {
							float sampleLuminance = dot( pc_fragColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
							if ( sampleLuminance > uRadianceClamp ) {
								pc_fragColor.rgb *= uRadianceClamp / sampleLuminance;
							}
						}

						pc_fragColor.a *= opacity;

						#if DEBUG_MODE == 1

						// output the number of rays checked in the path and number of
						// transmissive rays encountered.
						pc_fragColor.rgb = vec3(
							float( state.depth ),
							transmissiveBounces - state.transmissiveTraversals,
							0.0
						);
						pc_fragColor.a = 1.0;

						#endif

						// Sprint 5: Write G-buffer outputs.
						// If gbufWritten == false (sky/miss on first ray), sky sentinels are used:
						//   gNormalDepth.rgb = (0.5,1.0,0.5) → decodes to world-up (0,1,0)
						//   gNormalDepth.w   = 0.0 (sky depth sentinel, matches shared-denoisers convention)
						//   gAlbedo.rgb      = (0,0,0) (no surface albedo)
						gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );
						gAlbedo      = vec4( gbufAlbedo, 1.0 );

					}
`;

/**
 * The sections array — assembled in order, byte-identical to the original RENDER_MAIN.
 * Verified at module load time (see assertion below).
 */
/** @internal — exported for byte-identity test pin (D10.4). */
export const RENDER_MAIN_SECTIONS = [
  RENDER_MAIN_BDPT_SUBPATH,
  RENDER_MAIN_GBUFFER,
  RENDER_MAIN_BDPT_EYE,
  RENDER_MAIN_SURFACE_BDPT_EYE,
  RENDER_MAIN_CAUSTIC_MANIFOLD,
  RENDER_MAIN_CAUSTIC_PHOTON,
  RENDER_MAIN_SCATTER,
  RENDER_MAIN_POST_LOOP,
] as const;

/** Assembled RENDER_MAIN — concatenation of all sections in order (byte-identical to original). */
const RENDER_MAIN = RENDER_MAIN_SECTIONS.join('');

/**
 * Compose the fragment-shader BODY for the WebGL2 path tracer (no `#version`/precision/
 * `pc_fragColor` preamble — those come from the GlProgram preamble). The chunk order
 * reproduces PhysicalPathTracingMaterial.js:227-441 + the inlined main() loop exactly;
 * struct-before-use is load-bearing.
 *
 * @param features compile-flag set; `features.bdpt` gates the bdpt render chunks (the
 *   FEATURE_* `#if`s themselves resolve from the GlProgram preamble defines, so the only
 *   feature this composer branches on at JS-compose time is `bdpt` — to keep the bdpt
 *   render chunks out of the program text entirely when off, matching the fork's
 *   `#if FEATURE_BDPT`-wrapped chunk injection at lines 436-441).
 */
export function composeTraceGlsl(features: TraceFeatures): string {
  return /* glsl */ `
					#define RAY_OFFSET 1e-4
					#define INFINITY 1e20

					precision highp isampler2D;
					precision highp usampler2D;
					precision highp sampler2DArray;
					vec4 envMapTexelToLinear( vec4 a ) { return a; }
					${THREE_COMMON_SHIM}

					// Sprint 5: MRT G-buffer outputs (Decision 12).
					// location 0: pc_fragColor — declared by the GlProgram preamble (THREE auto-injected
					//             it in GLSL3 ShaderMaterial), so we only declare locations 1 and 2 here.
					// location 1: gNormalDepth — world normal (xyz, encoded [0,1]) + linear depth (w)
					// location 2: gAlbedo      — demodulated base color, no lighting
					// When the host allocates a plain render target (non-MRT), only location 0 is
					// written; locations 1 and 2 are harmlessly ignored by the driver.
					layout(location = 1) out vec4 gNormalDepth;
					layout(location = 2) out vec4 gAlbedo;

					// bvh intersection
					${BVH_COMMON_FUNCTIONS}
					${BVH_STRUCT}
					${BVH_RAY_FUNCTIONS}

					// uniform structs
					${STRUCTS.camera_struct}
					${STRUCTS.lights_struct}
					${STRUCTS.equirect_struct}
					${STRUCTS.material_struct}
					${STRUCTS.surface_record_struct}

					${randBlock()}

					// common
					${COMMON.texture_sample_functions}
					${COMMON.fresnel_functions}
					${COMMON.util_functions}
					${COMMON.math_functions}
					${COMMON.shape_intersection_functions}

					${buildUniformDecls()}

					// sampling
					${SAMPLING.shape_sampling_functions}
					${SAMPLING.equirect_functions}
					${SAMPLING.light_sampling_functions}

					${spectralCausticBdptUniformDecls()}

					${PTBVH.inside_fog_volume_function}
					${BSDF.ggx_functions}
					${BSDF.sheen_functions}
					${BSDF.iridescence_functions}
					${BSDF.fog_functions}
					${BSDF.spectral_accumulator}
					${BSDF.thin_film_tmm}
					${BSDF.bsdf_functions}

					${INLINE_HELPERS}

					${RENDER.render_structs}
					${RENDER.camera_util_functions}
					${RENDER.trace_scene_function}
					${RENDER.attenuate_hit_function}
					${RENDER.direct_light_contribution_function}
					${RENDER.get_surface_record_function}

					${features.bdpt ? RENDER.bdpt_light_subpath + '\n' + RENDER.bdpt_connection : ''}

					${RENDER_MAIN}
`;
}
