import { describe, expect, it } from 'vitest';
import * as PathTracerPkg from 'three-gpu-pathtracer';

describe('three-gpu-pathtracer uniform contract (integration)', () => {
  it('exposes BDPT and spectral bridge uniforms on PhysicalPathTracingMaterial', () => {
    const PhysicalPathTracingMaterial = (PathTracerPkg as unknown as { PhysicalPathTracingMaterial?: new () => unknown })
      .PhysicalPathTracingMaterial;
    expect(typeof PhysicalPathTracingMaterial).toBe('function');
    if (PhysicalPathTracingMaterial == null) return;
    const material = new PhysicalPathTracingMaterial();
    const uniforms = (material as unknown as { uniforms?: Record<string, unknown> }).uniforms ?? {};

    expect(Object.hasOwn(uniforms, 'uBdptEnabled')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uBdptLightPathTex')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uBdptMaxLightBounces')).toBe(true);

    expect(Object.hasOwn(uniforms, 'uXCmfCdf')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uZCmfCdf')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uXCmfIntegral')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uZCmfIntegral')).toBe(true);
  });

  // -- Jakob & Hanika 2019 RGB->spectrum upsampling is actually consumed -------
  // Previously the host-side u_jakobCoeffs upload had ZERO call sites in the
  // integrator: evalSpectrumAtHero existed but was never referenced, so spectral
  // upsampling died at the uniform. These assertions pin the wiring into the
  // assembled fragment shader so the feature cannot silently regress.
  it("consumes evalSpectrumAtHero in the integrator under a gate, not on the default path", () => {
    const PhysicalPathTracingMaterial = (
      PathTracerPkg as unknown as { PhysicalPathTracingMaterial?: new () => unknown }
    ).PhysicalPathTracingMaterial;
    expect(typeof PhysicalPathTracingMaterial).toBe('function');
    if (PhysicalPathTracingMaterial == null) return;

    const material = new PhysicalPathTracingMaterial();
    const fragmentRaw = (material as unknown as { fragmentShader?: string }).fragmentShader ?? '';
    expect(fragmentRaw.length).toBeGreaterThan(0);
    // Collapse runs of whitespace so the structural assertions below are robust
    // to formatting (tabs / newlines) in the assembled GLSL.
    const fragment = fragmentRaw.replace(/\s+/g, ' ');

    // (1) The integrator must define the gated medium-albedo helper.
    expect(fragment).toContain('float mediumAlbedoHero( vec3 rgb, float heroWavelength )');

    // (2) Under the spectral gate it routes through the paper-accurate sigmoid
    //     reflectance evalSpectrumAtHero -- the previously-dead consumer.
    const helperBody = fragment.slice(
      fragment.indexOf('float mediumAlbedoHero( vec3 rgb, float heroWavelength )'),
    );
    expect(helperBody).toContain('if ( spectralUpsamplingActive() )');
    expect(helperBody).toContain('return evalSpectrumAtHero( heroWavelength )');

    // (3) The gate requires BOTH spectral rendering on AND a non-flat coefficient
    //     set. The flat (0,0,0) default gives sigmoid(0) = 1/2 which would wash
    //     colour, so it must NOT route through evalSpectrum -- the default RGB
    //     path stays bit-identical to pre-wiring behaviour.
    expect(fragment).toContain(
      'bool spectralUpsamplingActive() { return uSpectralRendering == 1 && ' +
        '( u_jakobCoeffs.x != 0.0 || u_jakobCoeffs.y != 0.0 || u_jakobCoeffs.z != 0.0 )',
    );

    // (4) Off-gate fallback is the legacy smoothstep projection (heroScalarFromRgb).
    expect(helperBody).toContain('return heroScalarFromRgb( rgb, heroWavelength )');

    // (5) The helper is actually called through the vec3 throughput adapters at
    //     the medium single-scatter albedo sites
    //     (volume scatter + SSS), so the gated reflectance reaches the shaded result.
    expect(fragment).toContain(
      'state.throughput *= mediumAlbedoThroughput( u_scatterAlbedo, state.wavelength ) * transmittance',
    );
    expect(fragment).toContain(
      'sssRec.throughput = mediumAlbedoThroughput( surf.sssAlbedo, heroWavelength ) * beerLambert',
    );

    // (5a) Default rendering keeps RGB throughput; spectral mode collapses to a
    //      replicated hero scalar only behind the explicit spectral gate.
    expect(fragment).toContain('vec3 pathThroughputFromRgb( vec3 rgb, float heroWavelength )');
    expect(fragment).toContain('if ( uSpectralRendering == 0 ) return max( rgb, vec3( 0.0 ) )');

    // (5b) SSS reads the PER-MATERIAL SurfaceRecord fields (packed from the
    //      MaterialsTexture via material_struct → get_surface_record), NOT the
    //      never-set global u_sss* uniforms. Pins the mis-driven-SSS fix:
    //      u_sssSigmaT/u_sssAlbedo/u_sssAnisotropyG only ever held their
    //      constructor defaults (sigmaT=0), so per-material SSS was degenerate.
    //      (u_scatterAlbedo on the GLOBAL volume scatter above is correct — that
    //      IS a global-medium uniform, not per-material.)
    expect(fragment).toContain('sampleExponential( rand( 17 ), surf.sssSigmaT, 1e6 )');
    expect(fragment).toContain('sampleHG_glsl( rand( 18 ), rand( 19 ), surf.sssAnisotropyG, rd )');

    // (6) GLSL compile order: evalSpectrumAtHero declared before mediumAlbedoHero.
    const evalIdx = fragment.indexOf('float evalSpectrumAtHero( float lambdaNm )');
    const helperIdx = fragment.indexOf('float mediumAlbedoHero( vec3 rgb, float heroWavelength )');
    expect(evalIdx).toBeGreaterThanOrEqual(0);
    expect(helperIdx).toBeGreaterThan(evalIdx);
  });

  it('declares BDPT light sample point/normal fields consumed by the light-subpath pass', () => {
    const PhysicalPathTracingMaterial = (
      PathTracerPkg as unknown as { PhysicalPathTracingMaterial?: new () => unknown }
    ).PhysicalPathTracingMaterial;
    expect(typeof PhysicalPathTracingMaterial).toBe('function');
    if (PhysicalPathTracingMaterial == null) return;

    const material = new PhysicalPathTracingMaterial();
    const fragment = ((material as unknown as { fragmentShader?: string }).fragmentShader ?? '')
      .replace(/\s+/g, ' ');
    const structStart = fragment.indexOf('struct LightRecord');
    expect(structStart).toBeGreaterThanOrEqual(0);
    const structEnd = fragment.indexOf('};', structStart);
    const lightRecordStruct = fragment.slice(structStart, structEnd);
    expect(lightRecordStruct).toContain('vec3 point;');
    expect(lightRecordStruct).toContain('vec3 normal;');
    expect(fragment).toContain('vec3 emitPos = lightRec.point;');
    expect(fragment).toContain('vec3 emitNormal = normalize( lightRec.normal );');
  });

  it('keeps BDPT connection contribution dimensionally correct: BSDF*cos is not multiplied by pdf', () => {
    const PhysicalPathTracingMaterial = (
      PathTracerPkg as unknown as { PhysicalPathTracingMaterial?: new () => unknown }
    ).PhysicalPathTracingMaterial;
    expect(typeof PhysicalPathTracingMaterial).toBe('function');
    if (PhysicalPathTracingMaterial == null) return;

    const material = new PhysicalPathTracingMaterial();
    const fragment = ((material as unknown as { fragmentShader?: string }).fragmentShader ?? '')
      .replace(/\s+/g, ' ');
    expect(fragment).toContain('float eyeBsdfPdf = bsdfResult(');
    expect(fragment).toContain('vec3 eyeBsdfCosTheta = eyeBsdfColor;');
    expect(fragment).not.toContain('vec3 eyeBsdfCosTheta = eyeBsdfColor * eyeBsdfPdf;');
  });
});
