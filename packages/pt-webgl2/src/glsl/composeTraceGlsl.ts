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

import { BVH_COMMON_FUNCTIONS, BVH_STRUCT, BVH_RAY_FUNCTIONS } from './bvh/index.js';
import { THREE_COMMON_SHIM } from './common/threeCommonShim.js';
// D11-3 (T3-D): the uniform manifest + declaration builder + exhaustiveness gate
// live in uniformManifest.ts; the RENDER_MAIN_* section constants live in
// renderMain.glsl.ts. This module is the assembly root that composes them.
import { buildUniformDecls, spectralCausticBdptUniformDecls } from './uniformManifest.js';
import { RENDER_MAIN } from './renderMain.glsl.js';

// Re-export the moved symbols so the public compose surface (consumed by
// composeTraceGlsl.test.ts + downstream) is unchanged after the split.
export { buildUniformDecls, UNIFORM_MANIFEST } from './uniformManifest.js';
export type { UniformManifestEntry } from './uniformManifest.js';
export { RENDER_MAIN_SECTIONS } from './renderMain.glsl.js';

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
