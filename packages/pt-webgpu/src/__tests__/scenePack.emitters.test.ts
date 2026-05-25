import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

function baseScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('buildPackedScene emitter + environment packing', () => {
  it('packs point, spot, rect-area, and mesh-area lights', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        { kind: 'point', id: 'p', position: [2, 3, 4], color: [0.5, 1, 0.25], intensity: 8 },
        { kind: 'spot', id: 's', position: [5, 6, 7], direction: [0, -1, 0], angle: 0.5, color: [1, 0.5, 0.25], intensity: 4 },
        { kind: 'rect-area', id: 'r', position: [0, 1, 0], uAxis: [0.5, 0, 0], vAxis: [0, 0.5, 0], color: [1, 1, 1], intensity: 10 },
        { kind: 'mesh-area', id: 'm', meshId: 'tri', color: [0.5, 0.25, 1], intensity: 6 },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.pointLightCount).toBe(1);
    expect(packed.spotLightCount).toBe(1);
    expect(packed.rectAreaLightCount).toBe(1);
    expect(packed.meshAreaLightCount).toBe(1);
  });

  it('packs HDRI map payload and CDF', () => {
    const scene: Scene = {
      ...baseScene(),
      environment: {
        kind: 'hdri',
        hdri: { width: 2, height: 2, data: new Float32Array([4,1,1, 1,4,1, 1,1,4, 2,2,2]) },
      },
    };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(true);
    expect(packed.environmentMapTexels.length).toBe(16);
    expect(packed.environmentMapCdf.length).toBe(5);
  });

  it('warns and falls back when HDRI payload is opaque', () => {
    const scene: Scene = { ...baseScene(), environment: { kind: 'hdri', hdri: { mock: true } as never } };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(false);
    expect(packed.warnings.some((w) => w.includes('HDRI environment'))).toBe(true);
  });
});
