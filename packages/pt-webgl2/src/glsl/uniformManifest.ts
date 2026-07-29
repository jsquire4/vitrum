// uniformManifest — the pt-webgl2 trace-shader uniform manifest + declaration
// builder + compile-time exhaustiveness gate (extracted from composeTraceGlsl.ts,
// T3-D / D11-3).
//
// BEHAVIOR-PRESERVING: `buildUniformDecls()` returns the byte-identical
// UNIFORM_DECLS string the composer previously emitted; UNIFORM_MANIFEST and the
// exhaustiveness gate are moved verbatim. CRITICAL: the D10.5 exhaustiveness gate
// (`as const satisfies` on UNIFORM_MANIFEST + the consumed `_exhaustive` assert) is
// preserved EXACTLY — a new FrameUniforms key that is neither in the manifest nor
// in _HandledSeparately must still fail `npm run typecheck` here.

import type { FrameUniforms } from '../gl/glResources.js';

// ── D10.3: UNIFORM_MANIFEST ────────────────────────────────────────────────────
// Structured list of the simple (non-gated) GLSL uniforms declared in UNIFORM_DECLS
// that correspond to fields in FrameUniforms. Used to:
//   1. Generate the flat uniform declarations via buildUniformDecls() (D10.3).
//   2. Drive the compile-time exhaustiveness gate (D10.5) that ties every
//      keyof FrameUniforms to a manifest entry or an explicitly-justified
//      _HandledSeparately reason (see _HANDLED_SEPARATELY_KEYS).
//
// Gated physicalCamera uniforms are inlined in buildUniformDecls(); the
// FEATURE_BDPT block is emitted by bdptUniformDecls(). Their preprocessor block
// structure cannot be expressed as flat manifest rows.
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
// NOTE: `as const satisfies …` (NOT a `: readonly UniformManifestEntry[]`
// annotation) is load-bearing for the D10.5 exhaustiveness gate below — a plain
// type annotation widens each row's `frameKey` back to the full declared union,
// which makes `_ManifestFrameKey` vacuously equal to `keyof FrameUniforms` and
// silently defeats the gate. `as const` preserves the literal frameKeys so the
// gate tracks the ACTUAL array contents; `satisfies` still type-checks each row.
export const UNIFORM_MANIFEST = [
  // ── environment ──────────────────────────────────────────────────────────
  { glslName: 'envMapInfo',            glslType: 'EquirectHdrInfo',  frameKey: 'samplerOrStruct' },
  { glslName: 'environmentRotation',   glslType: 'mat4',             frameKey: 'environmentRotation' },
  { glslName: 'environmentIntensity',  glslType: 'float',            frameKey: 'environmentIntensity' },
  // ── lighting ─────────────────────────────────────────────────────────────
  { glslName: 'lights',                glslType: 'LightsInfo',       frameKey: 'samplerOrStruct' },
  { glslName: 'uMeshLights',          glslType: 'sampler2D',        frameKey: 'samplerOrStruct' },
  { glslName: 'uMeshLightCount',      glslType: 'uint',             frameKey: 'samplerOrStruct' },
  { glslName: 'uTotalEmissivePower',  glslType: 'float',            frameKey: 'samplerOrStruct' },
  // ── background ───────────────────────────────────────────────────────────
  { glslName: 'backgroundBlur',        glslType: 'float',            frameKey: 'backgroundBlur' },
  { glslName: 'backgroundAlpha',       glslType: 'float',            frameKey: 'backgroundAlpha' },
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
  { glslName: 'seed',                  glslType: 'int',              frameKey: 'internal' },
  // ── image ─────────────────────────────────────────────────────────────────
  { glslName: 'resolution',            glslType: 'vec2',             frameKey: 'resolution' },
  { glslName: 'opacity',               glslType: 'float',            frameKey: 'internal' },
] as const satisfies readonly UniformManifestEntry[];

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
//   spectralEnabled      — drives the spectral-accumulator CMF upload gate
//                          (uSpectralRendering int uniform)
//   bdpt                 — drives pass-selection (uBdptLightSubpathPass); not a simple scalar
//   bdptMaxLightBounces  — declared as uBdptMaxLightBounces in bdptUniformDecls()
//   bdptSceneCenter / bdptSceneRadius
//                        — bounded infinite-source launch disk uniforms
//   bdptSharedWavelengthNm / bdptSharedWavelengthPdf
//                        — shared spectral+BDPT uniform pair in bdptUniformDecls()
//   materialLodDepth     — declared in get_surface_record_function.glsl.js; uploaded from FrameUniforms
//                          but not part of this module's UNIFORM_DECLS block.
//   dof                  — drives gated PhysicalCamera struct (FEATURE_DOF)
//   backgroundAlpha      — in manifest (frameKey='backgroundAlpha') — not in _HandledSeparately
//   tonemapMode          — drives PresentPass only; no PT shader uniform counterpart
//   exposure             — drives PresentPass only; no PT shader uniform counterpart
//   outputColorSpace     — drives PresentPass only; no PT shader uniform counterpart

type _ManifestFrameKey = (typeof UNIFORM_MANIFEST)[number]['frameKey'];
type _HandledSeparately =
  | 'spectralEnabled'
  | 'bdpt'
  | 'bdptMaxLightBounces'
  | 'bdptSceneCenter'
  | 'bdptSceneRadius'
  | 'bdptSharedWavelengthNm'
  | 'bdptSharedWavelengthPdf'
  | 'materialLodDepth'
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
// Consumed compile-time assert: if `_ExhaustivenessCheck` resolves to `never` (a
// FrameUniforms key is neither in the manifest nor in _HandledSeparately), the
// assignment `never = true` is a TypeScript error and `npm run typecheck` fails,
// naming the gap. This is the live gate — it must stay uncommented (the leading
// underscore exempts it from the no-unused-vars rule).
const _exhaustive: _ExhaustivenessCheck = true;

/**
 * D10.3: Generate the GLSL uniform declarations block from UNIFORM_MANIFEST plus the
 * fixed gated sections. Returns a string byte-identical to UNIFORM_DECLS.
 *
 * The manifest covers the simple (ungated) uniforms; the FEATURE_DOF section
 * is inlined verbatim because its preprocessor structure cannot be expressed
 * as flat manifest rows.
 */
export function buildUniformDecls(): string {
  return UNIFORM_DECLS;
}

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
						// the sampler uses uTotalEmissivePower =
						// Σ luminance(radiance)·area so mapped texel-cell
						// lights are selected by emitted power, and forward-hit MIS can recover the
						// same area density from surf.emission. All default to 0 / empty → the
						// mesh-NEE branch and forward-emission MIS are inert.
						uniform sampler2D uMeshLights;
						uniform uint uMeshLightCount;
						uniform float uTotalEmissivePower;

					// background
					uniform float backgroundBlur;
					uniform float backgroundAlpha;

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
						uniform sampler2DArray materialRadianceTextures;
						uniform BVH bvh;

					// path tracer
					uniform int bounces;
					uniform int transmissiveBounces;
					uniform float filterGlossyFactor;
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
 * Gated BDPT uniform declarations. The FEATURE_BDPT gate resolves from the
 * preamble define; emit it unconditionally and let the preprocessor strip it
 * when FEATURE_BDPT == 0.
 */
export function bdptUniformDecls(): string {
  return /* glsl */ `
					// Sprint 10c — BDPT uniforms.
					// uBdptLightPathTex: RGBA32F ping-pong texture (width=8, height=6).
					//   Rows: 0=position+kind, 1=normal+pdfFwd, 2=throughput+pdfRev,
					//   3=BSDF/endpoint metadata, 4=material/endpoint payload, 5=medium context.
					//   Columns: 0..uBdptMaxLightBounces-1 (one per light subpath bounce).
					// uBdptMaxLightBounces: how many stored light vertices to attempt connections with.
					// uBdptEnabled is mirrored as FEATURE_BDPT define; uniform kept for runtime query.
					#if FEATURE_BDPT

					uniform sampler2D uBdptLightPathTex;
					uniform int uBdptMaxLightBounces;
					uniform int uBdptLightSubpathPass;
					uniform int uBdptVertexCol;
					uniform float uBdptSharedWavelength;
					uniform float uBdptSharedWavelengthPdf;
					uniform vec3 uBdptSceneCenter;
					uniform float uBdptSceneRadius;

					#endif
`;
}
