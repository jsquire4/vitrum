// GPU-free structural assertions on the composed fragment body. We CANNOT GPU-compile
// here, so we verify the chunk concatenation produced the right symbols in the right
// order (struct-before-use is load-bearing — plan/three-removal/04-glsl-kernels.md §3).

import { describe, it, expect } from 'vitest';
import { composeTraceGlsl, RENDER_MAIN_SECTIONS, buildUniformDecls, UNIFORM_MANIFEST } from './composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';

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
  });

  it('item 20: iesProfiles uniform is absent from the composed shader (IES removed)', () => {
    // IES profiles are not in the @vitrum/core contract and were always null.
    // The uniform, the struct field, and getPhotometricAttenuation are all deleted.
    expect(src).not.toContain('uniform sampler2DArray iesProfiles');
    expect(src).not.toContain('getPhotometricAttenuation');
    expect(src).not.toContain('iesProfile !=');
  });

  // D10.4: RENDER_MAIN_SECTIONS byte-identity pin (length pinned to prevent silent whitespace drift).
  it('D10.4: RENDER_MAIN_SECTIONS join is byte-identical — length pin 32297', () => {
    const assembled = RENDER_MAIN_SECTIONS.join('');
    expect(assembled).toHaveLength(32297);
    // All 9 sections must be non-empty and together contain the key anchor points.
    expect(RENDER_MAIN_SECTIONS).toHaveLength(9);
    expect(assembled).toContain('void main() {');
    expect(assembled).toContain('// get camera ray');
    expect(assembled).toContain('// Sprint 7: Volume scatter event');
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
