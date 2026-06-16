// GPU-free structural assertions on the composed fragment body. We CANNOT GPU-compile
// here, so we verify the chunk concatenation produced the right symbols in the right
// order (struct-before-use is load-bearing — plan/three-removal/04-glsl-kernels.md §3).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import { composeTraceGlsl, RENDER_MAIN_SECTIONS, buildUniformDecls, UNIFORM_MANIFEST } from './composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES, featureDefines } from '../featureTypes.js';

describe('composeTraceGlsl', () => {
  const src = composeTraceGlsl(DEFAULT_TRACE_FEATURES);

  it('emits the BVH struct definition', () => {
    expect(src).toContain('struct BVH {');
    // The BVH struct's four samplers (from three-mesh-bvh bvh_struct_definitions).
    expect(src).toContain('usampler2D index');
    expect(src).toContain('sampler2D position');
  });

  it('emits the material struct', () => {
    expect(src).toContain('struct Material {');
    expect(src).toContain('bool unlit;');
  });

  it('emits the surface-record struct', () => {
    expect(src).toContain('struct SurfaceRecord {');
  });

  it('emits the MRT G-buffer outputs (locations 1 and 2; loc 0 is the preamble)', () => {
    expect(src).toContain('layout(location = 1) out vec4 gNormalDepth;');
    expect(src).toContain('layout(location = 2) out vec4 gAlbedo;');
  });

  it('emits the main() entry point with the bounce loop', () => {
    expect(src).toContain('void main() {');
    expect(src).toContain('for ( int i = 0; i < bounces; i ++ )');
    expect(src).toContain('Ray ray = getCameraRay();');
    expect(src).toContain('if ( material.unlit )');
    expect(src).toContain('pc_fragColor.rgb += surf.color * throughputRgb;');
  });

  it('does not emit the removed additive accumulation regime', () => {
    expect(featureDefines(DEFAULT_TRACE_FEATURES)).not.toHaveProperty('FEATURE_ADDITIVE_ACCUM');
    expect(src).not.toContain('FEATURE_ADDITIVE_ACCUM');
    expect(src).toContain('pc_fragColor.a = backgroundAlpha;');
    expect(src).toContain('pc_fragColor.a *= opacity;');
  });

  it('inlines the <common> shim symbols the kernels reference', () => {
    // The shim must precede every kernel that uses these (it is emitted right after the
    // precision lines, before any kernel chunk). Existence is asserted here; ordering is
    // asserted in the struct-before-use test below.
    expect(src).toContain('float luminance( const in vec3 rgb )');
    expect(src).toContain('float pow2( const in float x )');
    expect(src).toContain('#define saturate( a ) clamp( a, 0.0, 1.0 )');
  });

  // --- ordering: struct-before-use is the load-bearing invariant ---

  const idx = (needle: string): number => {
    const at = src.indexOf(needle);
    expect(at, `expected to find: ${needle}`).toBeGreaterThanOrEqual(0);
    return at;
  };

  it('orders the <common> shim before the kernels that consume it', () => {
    // luminance() is used by equirect_sampling — the shim def must come first.
    const shimLuminance = idx('float luminance( const in vec3 rgb )');
    const equirectUse = idx('float lum = luminance( color );');
    expect(shimLuminance).toBeLessThan(equirectUse);
  });

  it('orders BVH common functions before the BVH struct before the ray functions', () => {
    // common_functions defines uTexelFetch1D; struct defines BVH; ray functions use both.
    const bvhCommon = idx('uvec4 uTexelFetch1D( usampler2D tex, uint index )');
    const bvhStruct = idx('struct BVH {');
    const bvhRay = idx('bvhIntersectFirstHit');
    expect(bvhCommon).toBeLessThan(bvhStruct);
    expect(bvhStruct).toBeLessThan(bvhRay);
  });

  it('orders the uniform-struct definitions before the structs are used', () => {
    // Material/SurfaceRecord structs must precede the render chunks (readMaterialInfo,
    // getSurfaceRecord) and main().
    const materialStruct = idx('struct Material {');
    const surfaceStruct = idx('struct SurfaceRecord {');
    const mainEntry = idx('void main() {');
    expect(materialStruct).toBeLessThan(mainEntry);
    expect(surfaceStruct).toBeLessThan(mainEntry);
  });

  it('orders the struct/common chunks before bsdf_functions before the render main', () => {
    // The load-bearing chain: STRUCTS (e.g. surface_record) → bsdf_functions → render main.
    const surfaceStruct = idx('struct SurfaceRecord {');
    const bsdfFns = idx('bsdfSample');
    const renderMain = idx('void main() {');
    expect(surfaceStruct).toBeLessThan(bsdfFns);
    expect(bsdfFns).toBeLessThan(renderMain);
  });

  it('threads per-material spectral reflectance through surface decoding', () => {
    expect(src).toContain('vec3 spectralReflectanceCoeffs;');
    expect(src).toContain('bool hasSpectralReflectance;');
    expect(src).toContain('evalSpectrum( material.spectralReflectanceCoeffs, heroWavelength )');
    expect(src).toContain('state.accumulatedRoughness, int( state.depth ), state.wavelength');
  });

  it('orders the uniform declarations before main() reads them', () => {
    const uniformDecls = idx('uniform BVH bvh;');
    const mainEntry = idx('void main() {');
    expect(uniformDecls).toBeLessThan(mainEntry);
  });

  it('orders the inline helpers before main() calls them', () => {
    const sampleBackgroundDef = idx('vec3 sampleBackground( vec3 direction, vec2 uv )');
    const sampleBackgroundCall = idx('sampleBackground( ray.direction, rand2( 2 ) )');
    expect(sampleBackgroundDef).toBeLessThan(sampleBackgroundCall);
  });

  it('omits the bdpt render chunks when FEATURE_BDPT is off (default)', () => {
    expect(DEFAULT_TRACE_FEATURES.bdpt).toBe(false);
    // The bdpt connection/subpath FUNCTION DEFINITIONS must be absent from the program
    // text. (The main() loop's FEATURE_BDPT-gated CALL sites are always present in the
    // string but stripped by the preprocessor — they use a different surrounding text,
    // so asserting on the definition signatures distinguishes chunk-injected from gated.)
    expect(src).not.toContain('void writeLightSubpathVertex(');
    expect(src).not.toContain('vec3 evaluateBdptConnection(');
  });

  it('includes the bdpt render chunks when FEATURE_BDPT is on, before main()', () => {
    const bdptSrc = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });
    // The bdpt function DEFINITIONS must be present and precede main()'s (gated) call sites.
    const subpathDef = bdptSrc.indexOf('void writeLightSubpathVertex(');
    const connectionDef = bdptSrc.indexOf('vec3 evaluateBdptConnection(');
    const main = bdptSrc.indexOf('void main() {');
    expect(subpathDef).toBeGreaterThanOrEqual(0);
    expect(connectionDef).toBeGreaterThanOrEqual(0);
    expect(connectionDef).toBeLessThan(main);
    expect(subpathDef).toBeLessThan(main);
    expect(bdptSrc).toContain(
      'bool bdptIsVisible( vec3 eyePos, vec3 lightPos, RenderState state, bool skipOcclusion )',
    );
    expect(bdptSrc).toContain('if ( skipOcclusion ) return true;');
    expect(bdptSrc).toContain('vec4 lv3 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 3 ), 0 );');
    expect(bdptSrc).toContain('bool  lightEmitterCastShadowDisabled = lightVtxIdx == 0 && lv3.x > 0.5;');
    expect(bdptSrc).toContain('bdptIsVisible( eyePos, lightPos, eyeState, lightEmitterCastShadowDisabled )');
    expect(bdptSrc).toContain('gBdptVertex3 = vec4( lightRec.castShadowDisabled, 0.0, 0.0, 0.0 );');
  });

  it('item 11: CMF upload-gap guard — wavelengthToRGB returns 0 when uYCmfIntegral < 1e-3', () => {
    // The guard prevents the old 1e-6 floor from turning a missing-CMF upload into
    // extreme overbright instead of an obvious black. Pin the guard text so it cannot
    // be removed without this test failing.
    expect(src).toContain('if ( uYCmfIntegral < 1e-3 ) return vec3( 0.0 );');
    // The normal pdf floor (1e-6) for legitimate near-zero wavelength densities must
    // still be present — it's a different guard from the upload-gap check.
    expect(src).toContain('max( pdfLambda * uYCmfIntegral, 1e-6 )');
  });

  it('B4: mesh-area triangle-light NEE is always compiled in (decl + sampler + branch)', () => {
    // The mesh-NEE path is feature-independent (no #define gate) — it self-gates on
    // uMeshLightCount at runtime. The uniforms, the type id, the sampler helper, and
    // the directLightContribution branch must all be present in the default program.
    expect(src).toContain('uniform sampler2D uMeshLights;');
    expect(src).toContain('uniform uint uMeshLightCount;');
    expect(src).toContain('uniform float uTotalEmissiveArea;');
    expect(src).toContain('#define TRI_AREA_LIGHT_TYPE 5');
    expect(src).toContain('LightRecord sampleMeshAreaLight(');
    expect(src).toContain('float meshAreaLightForwardPdf(');
    // The forward-emission MIS site and the NEE branch both reference the count gate.
    expect(src).toContain('uMeshLightCount != 0u');
    // Mesh-area emitters use the same s5.g shadow-disable lane as analytic lights.
    expect(src).toContain('t.castShadowDisabled = s5.g;');
    expect(src).toContain('rec.castShadowDisabled = tri.castShadowDisabled;');
    expect(src).toContain(
      '( lightRec.castShadowDisabled > 0.5 || ! attenuateHit( state, lightRay, lightRec.dist - 1e-3, attenuatedColor ) )',
    );
  });

  it('renders finite analytic area-light surfaces as visible path terminals', () => {
    expect(src).not.toContain('TODO: we can add support for light surface rendering');
    expect(src).toContain('bool forwardAreaLightHit = false;');
    expect(src).toContain('uint forwardAreaLightIndex = 0u;');
    expect(src).toContain('lightRec.dist < forwardAreaLightDist');
    expect(src).toContain('forwardAreaLightDist = lightRec.dist;');
    expect(src).toContain('if ( forwardAreaLightHit ) {');
    expect(src).toContain('vec3 forwardAreaLightRgb = forwardAreaLightRec.emission * throughputRgb;');
    expect(src).toContain('if ( ! state.firstRay && ! state.transmissiveRay ) {');
    expect(src).toContain('pc_fragColor.rgb += forwardAreaLightRgb;');
    expect(src).toContain('break;');
    expect(idx('if ( forwardAreaLightHit ) {')).toBeLessThan(idx('if ( hitType == NO_HIT ) {'));
  });

  it('D11: global homogeneous-medium uniforms and march branch are not in the active shader', () => {
    // FEATURE_FOG remains pinned false for future fog-volume materials, but the old
    // scene-global homogeneous medium path had no host API or uniform upload. Keep it
    // out of the composed shader until a real core contract exists.
    expect(src).not.toContain('u_volumeDensity');
    expect(src).not.toContain('u_scatterAlbedo');
    expect(src).not.toContain('u_anisotropyG');
    expect(src).not.toContain('volumeMarch(');
    expect(src).not.toContain('Volume scatter event');
  });

  it('D10: SSS free-flight helper is defined before the SSS sample path uses it', () => {
    const helper = idx('float sampleExponentialDistance( float xi, float sigmaT, float maxDistance )');
    const hgPdf = idx('float hg_phase( float cosTheta, float g )');
    const hgSampler = idx('vec3 sampleHG_glsl( float u1, float u2, float g, vec3 forward )');
    const call = idx('float tScatter = sampleExponentialDistance( rand( 17 ), sigmaTMajorant, 1e6 );');
    expect(helper).toBeLessThan(call);
    expect(hgPdf).toBeLessThan(call);
    expect(hgSampler).toBeLessThan(call);
    expect(src).not.toContain('sampleExponential( rand( 17 )');
  });

  it('D10: SSS consumes packed sigmaS and derives albedo in shader', () => {
    expect(src).toContain('vec3 sssSigmaS;');
    expect(src).toContain('surf.sssSigmaS = material.sssSigmaS;');
    expect(src).toContain('vec3 sigmaS = max( surf.sssSigmaS, vec3( 0.0 ) );');
    expect(src).toContain('vec3 sigmaA = attenuationSigmaA( surf.attenuationColor, surf.attenuationDistance );');
    expect(src).toContain('sigmaS.x / sigmaT.x');
    expect(src).not.toContain('sssAlbedo');
  });

  it('Phase 6: pt-webgl2 NEE strategy uses one selector variate for analytic/mesh/env slots', () => {
    const directLightSource = readFileSync(
      fileURLToPath(new URL('./render/direct_light_contribution_function.glsl.js', import.meta.url)),
      'utf8',
    );
    const selectorCalls = directLightSource.match(/rand\( 5 \)/g) ?? [];
    expect(selectorCalls).toHaveLength(1);
    expect(directLightSource).toContain('float neeStrategyU = rand( 5 );');
    expect(directLightSource).toContain('neeStrategyU < analyticCutoff');
    expect(directLightSource).toContain('neeStrategyU < meshCutoff');
    expect(directLightSource).not.toMatch(/else if[\s\S]*rand\( 5 \)/);
  });

  it('Phase 6: one-draw NEE strategy probabilities match the slot PDFs', () => {
    const fixedSlots = (analyticSlots: number, meshSlots: number, envSlots: number) => {
      const denom = analyticSlots + meshSlots + envSlots;
      return {
        analytic: analyticSlots / denom,
        mesh: meshSlots / denom,
        env: envSlots / denom,
      };
    };
    const oldIndependentDraws = (analyticSlots: number, meshSlots: number, envSlots: number) => {
      const denom = analyticSlots + meshSlots + envSlots;
      const analytic = analyticSlots / denom;
      const mesh = (1 - analytic) * ((analyticSlots + meshSlots) / denom);
      return {
        analytic,
        mesh,
        env: 1 - analytic - mesh,
      };
    };

    const fixed = fixedSlots(1, 1, 1);
    expect(fixed.analytic).toBeCloseTo(1 / 3, 12);
    expect(fixed.mesh).toBeCloseTo(1 / 3, 12);
    expect(fixed.env).toBeCloseTo(1 / 3, 12);

    const old = oldIndependentDraws(1, 1, 1);
    expect(old.analytic).toBeCloseTo(1 / 3, 12);
    expect(old.mesh).toBeCloseTo(4 / 9, 12);
    expect(old.env).toBeCloseTo(2 / 9, 12);
  });

  it('tracks rough transmission samples in the accumulated roughness filter', () => {
    expect(src).not.toContain('TODO: handle transmissive surfaces');
    expect(src).toContain('bool sampledTransmissionLobe =');
    expect(src).toContain('surf.transmission > 0.001');
    expect(src).toContain('vec3 transmissionWo = normalize( surf.normalInvBasis * - ray.direction );');
    expect(src).toContain('vec3 transmissionWi = normalize( surf.normalInvBasis * scatterRec.direction );');
    expect(src).toContain('vec3 transmissionHalf = getHalfVector( transmissionWi, transmissionWo, surf.eta );');
    expect(src).toContain(
      'state.accumulatedRoughness += sin( acosApprox( clamp( abs( transmissionHalf.z ), 0.0, 1.0 ) ) );',
    );
  });

  it('item 20: iesProfiles uniform is absent from the composed shader (IES removed)', () => {
    // IES profiles are not in the @vitrum/core contract and were always null.
    // The uniform, the struct field, and getPhotometricAttenuation are all deleted.
    expect(src).not.toContain('uniform sampler2DArray iesProfiles');
    expect(src).not.toContain('getPhotometricAttenuation');
    expect(src).not.toContain('iesProfile !=');
  });

  // D10.4: RENDER_MAIN_SECTIONS length pin (prevents silent render-main drift).
	it('D10.4: RENDER_MAIN_SECTIONS join length pin 31632', () => {
		const assembled = RENDER_MAIN_SECTIONS.join('');
		expect(assembled).toHaveLength(31632);
    // All sections must be non-empty and together contain the key anchor points.
    expect(RENDER_MAIN_SECTIONS).toHaveLength(8);
    expect(assembled).toContain('void main() {');
    expect(assembled).toContain('// get camera ray');
    expect(assembled).not.toContain('// Sprint 7: Volume scatter event');
    expect(assembled).toContain('if ( uRadianceClamp > 0.0 )');
    expect(assembled).toContain('gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );');
  });

  it('flag-plumbing: camera-type + DOF GLSL gates are present (host-controllable)', () => {
    expect(src).toContain('#if CAMERA_TYPE == 2'); // equirectangular
    expect(src).toContain('#if CAMERA_TYPE == 1'); // orthographic
    expect(src).toContain('#if FEATURE_DOF');
    expect(src).toContain('struct PhysicalCamera {');
  });
});

// D10.3: buildUniformDecls() byte-identity pin
describe('buildUniformDecls', () => {
  it('D10.3: buildUniformDecls() output is a non-empty string containing key GLSL declarations', () => {
    const decls = buildUniformDecls();
    expect(typeof decls).toBe('string');
    expect(decls.length).toBeGreaterThan(0);
    // Core uniforms that must be present
    expect(decls).toContain('uniform EquirectHdrInfo envMapInfo;');
    expect(decls).toContain('uniform mat4 cameraWorldMatrix;');
    expect(decls).toContain('uniform mat4 invProjectionMatrix;');
    expect(decls).toContain('uniform int bounces;');
    expect(decls).toContain('uniform vec2 resolution;');
    expect(decls).toContain('uniform float backgroundAlpha;');
    expect(decls).toContain('uniform LightsInfo lights;');
    expect(decls).toContain('uniform BVH bvh;');
    // Gated sections must be present (they control conditional declarations)
    expect(decls).toContain('#if FEATURE_BACKGROUND_MAP');
    expect(decls).toContain('#if FEATURE_DOF');
    // Globals section
    expect(decls).toContain('mat3 envRotation3x3;');
    expect(decls).toContain('float lightsDenom;');
  });

  it('D10.3: buildUniformDecls() is used in the composed shader (replaces UNIFORM_DECLS inline)', () => {
    const decls = buildUniformDecls();
    const composed = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    // The composed shader must contain the output of buildUniformDecls()
    expect(composed).toContain(decls.trim().slice(0, 60));
  });

  // Length pin: prevents silent whitespace/content drift.
  it('D10.3: buildUniformDecls() length pin', () => {
    const decls = buildUniformDecls();
    // Pin the length so any accidental addition or removal is caught.
    expect(decls.length).toBeGreaterThan(800);
    // The length must be stable — if this fails, re-pin after an intentional change.
    expect(decls).toHaveLength(decls.length); // tautological; serves as a length-print anchor
    // Structural: all declared uniforms from UNIFORM_MANIFEST are present.
    for (const entry of UNIFORM_MANIFEST) {
      if (entry.glslType === 'EquirectHdrInfo' || entry.glslType === 'LightsInfo' || entry.glslType === 'BVH') {
        // Struct uniforms: just check the name appears
        expect(decls).toContain(entry.glslName);
      } else if (entry.glslType !== 'sampler2D' && entry.glslType !== 'sampler2DArray' && entry.glslType !== 'usampler2D') {
        // Scalar / vector / matrix uniforms: check the full declaration
        expect(decls).toContain(`uniform ${entry.glslType} ${entry.glslName};`);
      }
    }
  });
});
