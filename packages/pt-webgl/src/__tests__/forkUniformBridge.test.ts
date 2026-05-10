import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshPhysicalMaterial, Scene } from 'three';
import { driveForkMaterialUniforms } from '../forkUniformBridge.js';

function makeStubPathTracer() {
  return {
    _pathTracer: {
      material: {
        uniforms: {
          u_volumeDensity: { value: 123 },
          uCmfX: { value: null as unknown },
          uCmfY: { value: null as unknown },
          uCmfZ: { value: null as unknown },
          uYCmfCdf: { value: null as unknown },
          uYCmfIntegral: { value: 0 },
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
    driveForkMaterialUniforms(pathTracer, scene);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uCmfX.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfY.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfZ.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfCdf.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
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
});
