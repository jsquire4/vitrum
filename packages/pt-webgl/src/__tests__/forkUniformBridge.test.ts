import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshPhysicalMaterial, Scene } from 'three';
import { driveForkMaterialUniforms } from '../forkUniformBridge.js';

function makeStubPathTracer() {
  return {
    _pathTracer: {
      material: {
        uniforms: {
          u_volumeDensity: { value: 0 },
          u_sssSigmaT: { value: 0 },
          u_anisotropyG: { value: 0 },
          u_scatterAlbedo: { value: [0, 0, 0] as [number, number, number] },
          u_sssAlbedo: { value: [0, 0, 0] as [number, number, number] },
          u_sssAnisotropyG: { value: 0 },
          u_ior0: { value: 1.5 },
          u_dispersionStrength: { value: 0 },
          u_jakobCoeffs: { value: [0, 0, 0] as [number, number, number] },
          iorCauchyA: { value: 1.5 },
          iorCauchyB: { value: 0 },
          iorCauchyC: { value: 0 },
          uCmfX: { value: null as unknown },
          uCmfY: { value: null as unknown },
          uCmfZ: { value: null as unknown },
          uYCmfCdf: { value: null as unknown },
          uYCmfIntegral: { value: 0 },
          uThinFilmEnabled: { value: 0 },
          uThinFilmLayerCount: { value: 0 },
          uThinFilmLayerIors: { value: null as unknown },
          uThinFilmLayerThicknessNm: { value: null as unknown },
        },
      },
    },
  };
}

describe('driveForkMaterialUniforms', () => {
  it('drives scattering and dispersion uniforms from material userData', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0x99ccff, ior: 1.58 });
    mat.userData['vitrumScatteringCoefficient'] = 2.5;
    mat.userData['vitrumScatteringAnisotropy'] = 0.7;
    mat.userData['vitrumScatteringCoefficientRGB'] = [0.2, 0.3, 0.4];
    mat.userData['vitrumDispersionAbbeNumber'] = 32;
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.u_volumeDensity.value).toBeCloseTo(2.5);
    expect(uniforms.u_sssSigmaT.value).toBeCloseTo(2.5);
    expect(uniforms.u_anisotropyG.value).toBeCloseTo(0.7);
    expect(uniforms.u_scatterAlbedo.value).toEqual([0.2, 0.3, 0.4]);
    // Sprint 7 SSS-specific uniforms: u_sssAlbedo mirrors scatter albedo, u_sssAnisotropyG mirrors HG g.
    expect(uniforms.u_sssAlbedo.value).toEqual([0.2, 0.3, 0.4]);
    expect(uniforms.u_sssAnisotropyG.value).toBeCloseTo(0.7);
    expect(uniforms.u_ior0.value).toBeCloseTo(1.58);
    expect(uniforms.u_dispersionStrength.value).toBeGreaterThan(0);
    expect(uniforms.iorCauchyB.value).toBeGreaterThan(0);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
  });

  it('leaves scattering and dispersion at zero when no vitrum userData is stamped', () => {
    const scene = new Scene();
    // Plain material — no vitrumDispersionAbbeNumber, no vitrumScattering* stamps.
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    // When the dominant source has no meaningful signal (scatteringCoeff=0, abbe=0),
    // the bridge may return null and apply safe zero-defaults for all per-material uniforms.
    // CMF tables are still uploaded unconditionally.
    expect(uniforms.u_dispersionStrength.value).toBe(0);
    expect(uniforms.u_volumeDensity.value).toBe(0);
    expect(uniforms.u_sssSigmaT.value).toBe(0);
    expect(uniforms.u_sssAlbedo.value).toEqual([0.9, 0.9, 0.9]);
    expect(uniforms.u_sssAnisotropyG.value).toBe(0);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
  });

  it('drives thin-film layer uniforms from userData stack', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    mat.userData['vitrumThinFilmStack'] = {
      layers: [
        { ior: 2.35, thicknessNm: 57.0 },
        { ior: 1.46, thicknessNm: 91.0 },
      ],
    };
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uThinFilmEnabled.value).toBe(1);
    expect(uniforms.uThinFilmLayerCount.value).toBe(2);
    const iors = uniforms.uThinFilmLayerIors.value as Float32Array;
    const th = uniforms.uThinFilmLayerThicknessNm.value as Float32Array;
    expect(iors[0]).toBeCloseTo(2.35);
    expect(iors[1]).toBeCloseTo(1.46);
    expect(th[0]).toBeCloseTo(57);
    expect(th[1]).toBeCloseTo(91);
  });
});
