import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { sceneFromThreeJS } from '../index.js';

describe('sceneFromThreeJS', () => {
  it('maps one MeshPhysical mesh to a mesh primitive', () => {
    const s = new THREE.Scene();
    const g = new THREE.BoxGeometry(1, 1, 1);
    const m = new THREE.MeshPhysicalMaterial({
      color: 0xff3322,
      roughness: 0.42,
      metalness: 0.08,
    });
    s.add(new THREE.Mesh(g, m));

    const v = sceneFromThreeJS(s);
    expect(v.primitives.length).toBe(1);
    expect(v.primitives[0]!.kind).toBe('mesh');
    expect(v.emitters.length).toBe(0);
    expect(v.environment.kind).toBe('none');
  });

  it('emissive MeshPhysicalMaterial emits mesh-area and strips duplicate emissive on primitive', () => {
    const s = new THREE.Scene();
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        emissive: new THREE.Color(0xff0000),
        emissiveIntensity: 2,
      }),
    );
    mesh.updateMatrixWorld(true);
    s.add(mesh);

    const v = sceneFromThreeJS(s);
    expect(v.emitters.length).toBe(1);
    expect(v.emitters[0]!.kind).toBe('mesh-area');
    const mp = v.primitives[0]!;
    expect(mp.kind).toBe('mesh');
    expect(mp.material.emissive).toEqual([0, 0, 0]);
    expect(mp.material.emissiveIntensity ?? -1).toBe(0);
    expect(v.emitters[0]).toMatchObject({ meshId: mesh.uuid });
  });
});
