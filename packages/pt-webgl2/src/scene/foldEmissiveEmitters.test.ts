import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, SceneEmitter } from '@vitrum/core';
import { foldMeshAreaEmittersIntoMaterials } from './foldEmissiveEmitters.js';

// foldMeshAreaEmittersIntoMaterials re-attaches `mesh-area` emitter radiance onto
// its referenced primitive's material (three-bindings strips it for NEE backends;
// the fork integrator lights by hitting the emissive surface). These pin the
// contract the Cornell black-render fix depends on.

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

function quad(id: string, material: MaterialSpec): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(12),
    normals: new Float32Array(12),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material,
  };
}

function meshAreaEmitter(meshId: string, color: [number, number, number], intensity: number): SceneEmitter {
  return { kind: 'mesh-area', id: `e-${meshId}`, meshId, color, intensity, castShadow: true };
}

describe('foldMeshAreaEmittersIntoMaterials', () => {
  it('folds a mesh-area emitter radiance onto its primitive material', () => {
    const scene: Scene = {
      primitives: [quad('wall', GREY), quad('light', { baseColor: [0, 0, 0], roughness: 1, metallic: 0 })],
      emitters: [meshAreaEmitter('light', [1, 1, 1], 12)],
      environment: { kind: 'none' },
    };

    const folded = foldMeshAreaEmittersIntoMaterials(scene);
    const light = folded.primitives.find((p) => p.id === 'light')!;
    const wall = folded.primitives.find((p) => p.id === 'wall')!;

    // light material now carries the emitter's radiance (emissiveIntensity * emissive = color * intensity)
    expect(light.material.emissive).toEqual([1, 1, 1]);
    expect(light.material.emissiveIntensity).toBe(12);
    // baseColor and other fields are preserved
    expect(light.material.baseColor).toEqual([0, 0, 0]);
    // non-referenced primitive is untouched (no emissive injected)
    expect(wall.material.emissive).toBeUndefined();
  });

  it('returns the same scene reference when there are no mesh-area emitters', () => {
    const scene: Scene = {
      primitives: [quad('wall', GREY)],
      emitters: [{ kind: 'point', id: 'p', position: [0, 0, 0], color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    expect(foldMeshAreaEmittersIntoMaterials(scene)).toBe(scene);
  });

  it('does not mutate the input scene (pure)', () => {
    const lightMat: MaterialSpec = { baseColor: [0, 0, 0], roughness: 1, metallic: 0 };
    const scene: Scene = {
      primitives: [quad('light', lightMat)],
      emitters: [meshAreaEmitter('light', [2, 3, 4], 5)],
      environment: { kind: 'none' },
    };
    foldMeshAreaEmittersIntoMaterials(scene);
    expect(lightMat.emissive).toBeUndefined();
    expect(scene.primitives[0]!.material).toBe(lightMat);
  });

  it('carries mesh-area emitter castShadow:false as a folded material flag', () => {
    const scene: Scene = {
      primitives: [quad('light', { baseColor: [0, 0, 0], roughness: 1, metallic: 0 })],
      emitters: [{ ...meshAreaEmitter('light', [1, 1, 1], 3), castShadow: false }],
      environment: { kind: 'none' },
    };

    const folded = foldMeshAreaEmittersIntoMaterials(scene);
    const light = folded.primitives[0]!.material as MaterialSpec & {
      meshEmitterCastShadowDisabled?: boolean;
    };

    expect(light.emissive).toEqual([1, 1, 1]);
    expect(light.emissiveIntensity).toBe(3);
    expect(light.meshEmitterCastShadowDisabled).toBe(true);
    expect(
      (scene.primitives[0]!.material as {
        meshEmitterCastShadowDisabled?: boolean;
      }).meshEmitterCastShadowDisabled,
    )
      .toBeUndefined();
  });

  it('makes an explicit emitter authoritative over the implicit-only skipEmitter hint', () => {
    const sourceMaterial: MaterialSpec = {
      baseColor: [0, 0, 0],
      roughness: 1,
      metallic: 0,
      extensions: { skipEmitter: true, preserved: 7 },
    };
    const scene: Scene = {
      primitives: [quad('light', sourceMaterial)],
      emitters: [meshAreaEmitter('light', [1, 1, 1], 2)],
      environment: { kind: 'none' },
    };

    const folded = foldMeshAreaEmittersIntoMaterials(scene);
    expect(folded.primitives[0]!.material.extensions).toEqual({ preserved: 7 });
    expect(sourceMaterial.extensions).toEqual({ skipEmitter: true, preserved: 7 });
  });
});
