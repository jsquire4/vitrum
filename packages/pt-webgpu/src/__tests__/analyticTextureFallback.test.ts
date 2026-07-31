import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

const textureHandle = {
  width: 1,
  height: 1,
  data: new Uint8Array([255, 64, 32, 255]),
};

function mappedSphere(withFallback: boolean): Scene {
  return {
    primitives: [{
      kind: 'analytic',
      id: 'mapped-sphere',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: {
        baseColor: [1, 1, 1],
        roughness: 0.4,
        metallic: 0,
        baseColorMap: { handle: textureHandle },
      },
      ...(withFallback
        ? {
            fallbackMesh: {
              positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
              normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
              uvs: new Float32Array([0.125, 0.25, 0.75, 0.25, 0.125, 0.875]),
              indices: new Uint32Array([0, 1, 2]),
            },
          }
        : {}),
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu mapped analytic fallback classification', () => {
  it('routes a mapped native-supported analytic through its authored fallback mesh', () => {
    const packed = buildPackedScene(mappedSphere(true));

    expect(packed.analyticCount).toBe(0);
    expect(packed.triangleCount).toBe(1);
    expect(packed.materialTextureSources).toEqual([textureHandle]);
    expect(Array.from(packed.uvs.slice(0, 2))).toEqual([0.125, 0.25]);
    expect(packed.warnings.join('\n')).toMatch(
      /mapped-sphere.*rendering its fallbackMesh.*UV streams/,
    );
  });

  it('rejects a mapped analytic without fallback geometry before upload', () => {
    expect(() => buildPackedScene(mappedSphere(false))).toThrow(
      /mapped-sphere.*native analytic hits do not expose mesh UVs.*no fallbackMesh/,
    );
  });

  it('lowers an implicitly emissive analytic to one mesh shared by forward hits and NEE', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'analytic',
        id: 'glowing-sphere',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.4,
          metallic: 0,
          emissive: [3, 2, 1],
          emissiveIntensity: 2,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.analyticCount).toBe(0);
    expect(packed.triangleCount).toBeGreaterThan(0);
    expect(packed.meshAreaLightCount).toBe(packed.triangleCount);
    expect(packed.warnings.join('\n')).toMatch(
      /glowing-sphere.*emissive analytic.*mesh fallback.*light-sampling support/i,
    );
  });

});
