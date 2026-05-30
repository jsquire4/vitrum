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

  it('cameraVisibleEmitters re-attaches mesh-area emitter radiance onto the primitive material (color·intensity)', () => {
    // `sceneFromThreeJS` zeroes a converted emissive mesh's material emissive (it
    // becomes a sampled mesh-area emitter). With cameraVisibleEmitters the packer
    // re-attaches the emitter radiance so the primitive glows to the camera. The
    // re-attached emissive (packed floats 4..6 of material slot 0, pre-multiplied
    // by emissiveIntensity) must EXACTLY equal the mesh-area NEE radiance
    // color·intensity = [0.5,0.25,1]·6 — so camera glow matches the lit appearance.
    const scene: Scene = {
      ...baseScene(),
      emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'tri', color: [0.5, 0.25, 1], intensity: 6 }],
    };
    const off = buildPackedScene(scene);
    // Default (off): the primitive material emissive stays at its scene value (0).
    expect([off.materials[4], off.materials[5], off.materials[6]]).toEqual([0, 0, 0]);
    const on = buildPackedScene(scene, { cameraVisibleEmitters: true });
    expect(on.materials[4]).toBeCloseTo(0.5 * 6, 5);
    expect(on.materials[5]).toBeCloseTo(0.25 * 6, 5);
    expect(on.materials[6]).toBeCloseTo(1 * 6, 5);
    // Everything else byte-identical (only emissive changed).
    expect(on.materials[0]).toBe(off.materials[0]); // baseColor.r
    expect(on.materials.length).toBe(off.materials.length);
  });

  it('cameraVisibleEmitters does NOT re-attach when the emitter has no matching primitive (meshId mismatch)', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'no-such-mesh', color: [1, 1, 1], intensity: 9 }],
    };
    const on = buildPackedScene(scene, { cameraVisibleEmitters: true });
    expect([on.materials[4], on.materials[5], on.materials[6]]).toEqual([0, 0, 0]);
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
