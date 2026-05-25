import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

function makeScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [0.25, 0.5, 0.75],
        roughness: 0.4,
        metallic: 0.2,
        emissive: [0.2, 0.1, 0],
        emissiveIntensity: 2,
      },
    }],
    emitters: [{
      kind: 'directional',
      id: 'sun',
      direction: [0, -1, 0],
      color: [1, 0.8, 0.6],
      intensity: 3,
    }],
    environment: { kind: 'none' },
  };
}

describe('buildPackedScene core packing', () => {
  it('packs one triangle and material payload', () => {
    const packed = buildPackedScene(makeScene());
    expect(packed.triangleCount).toBe(1);
    expect(Array.from(packed.indices)).toEqual([0, 1, 2, 0]);
    expect(packed.materials.length).toBe(88);
    expect(packed.materials[0]).toBeCloseTo(0.25);
    expect(packed.materials[4]).toBeCloseTo(0.4);
  });

  it('packs analytic primitive payload for shader intersections', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      primitives: [
        ...base.primitives,
        {
          kind: 'analytic',
          id: 'a-sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 0.5]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.5, metallic: 0.1 },
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.analyticCount).toBe(1);
    expect(packed.analyticHeaders.length).toBe(4);
    expect(packed.analyticParams.length).toBe(8);
  });

  it('builds and uploads TLAS metadata buffers', () => {
    const packed = buildPackedScene(makeScene());
    expect(packed.tlasNodes.length).toBeGreaterThan(0);
    expect(packed.tlasBlasRoots[0]).toBe(0);
    expect(packed.tlasInstanceTransforms.length).toBe(16);
  });
});
