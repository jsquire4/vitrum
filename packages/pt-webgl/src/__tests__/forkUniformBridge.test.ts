import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshPhysicalMaterial, Scene } from 'three';
import { driveForkMaterialUniforms } from '../forkUniformBridge.js';

function makeStubPathTracer() {
  return {
    _pathTracer: {
      material: {
        uniforms: {
          u_volumeDensity: { value: 123 },
          uCausticStrategy: { value: -1 },
          uMneeMaxIterations: { value: 0 },
          uMneeMaxChainLength: { value: 0 },
          uCmfX: { value: null as unknown },
          uCmfY: { value: null as unknown },
          uCmfZ: { value: null as unknown },
          uYCmfCdf: { value: null as unknown },
          uYCmfIntegral: { value: 0 },
          uSpectralRendering: { value: -1 },
          uRadianceClamp: { value: -1 },
        },
      },
    },
  };
}

describe('driveForkMaterialUniforms', () => {
  it('uploads CMF and CDF tables when scene is set', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0x99ccff, ior: 1.58 });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene, {
      strategy: 'manifold-nee',
      mneeMaxIterations: 12,
      mneeMaxChainLength: 4,
    });
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uCmfX.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfY.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfZ.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfCdf.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
    expect(uniforms.uSpectralRendering.value).toBe(0);
    expect(uniforms.uRadianceClamp.value).toBe(0);
    expect(uniforms.uCausticStrategy.value).toBe(1);
    expect(uniforms.uMneeMaxIterations.value).toBe(12);
    expect(uniforms.uMneeMaxChainLength.value).toBe(4);
  });

  it('does not override per-material scalar uniforms', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.u_volumeDensity.value).toBe(123);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
  });

  it('maps strategy "none" to zero code', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
    });
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uCausticStrategy.value).toBe(0);
    expect(uniforms.uMneeMaxIterations.value).toBe(6);
    expect(uniforms.uMneeMaxChainLength.value).toBe(2);
  });

  it('enables experimental hero-wavelength reconstruction when requested', () => {
    const scene = new Scene();
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), new MeshPhysicalMaterial({ color: 0xffffff })));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: true,
      radianceClamp: 8,
    });

    expect(pathTracer._pathTracer.material.uniforms.uSpectralRendering.value).toBe(1);
    expect(pathTracer._pathTracer.material.uniforms.uRadianceClamp.value).toBe(8);
  });

  it('handles mixed-material scenes without clobbering scalar controls', () => {
    const scene = new Scene();
    const base = new MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, ior: 1.52 });
    base.userData['vitrumScatteringCoefficient'] = 0.2;
    base.userData['vitrumScatteringAnisotropy'] = 0.4;
    base.userData['vitrumDispersionAbbeNumber'] = 32;
    base.userData['vitrumFrontLayer'] = { transmission: [0.8, 0.85, 0.9], roughness: 0.25 };
    base.userData['vitrumBackLayer'] = { transmission: [0.95, 0.95, 0.95], roughness: 0.1 };
    base.userData['vitrumSpectralAttenuation'] = {
      samples: [
        [380, 0.2],
        [550, 0.5],
        [780, 0.8],
      ],
    };
    base.userData['vitrumThinFilmStack'] = {
      layers: [
        { ior: 1.33, thicknessNm: 120 },
        { ior: 1.5, thicknessNm: 250 },
      ],
    };
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), base));

    const second = new MeshPhysicalMaterial({ color: 0xccddff, roughness: 0.4, metalness: 0.1 });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), second));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, scene, {
      strategy: 'photon-map',
      mneeMaxIterations: 10,
      mneeMaxChainLength: 5,
    });
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.u_volumeDensity.value).toBe(123);
    expect(uniforms.uCausticStrategy.value).toBe(2);
    expect(uniforms.uMneeMaxIterations.value).toBe(10);
    expect(uniforms.uMneeMaxChainLength.value).toBe(5);
    expect(uniforms.uCmfX.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
  });
});
