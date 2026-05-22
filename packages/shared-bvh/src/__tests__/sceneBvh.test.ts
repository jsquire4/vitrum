import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { SceneBvh } from '../sceneBvh.js';

function makeMesh(): THREE.Mesh {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff });
  return new THREE.Mesh(geo, mat);
}

describe('SceneBvh', () => {
  it('clears cached buffers when scene has no visible meshes', () => {
    const bvh = new SceneBvh();
    const scene = new THREE.Scene();
    const mesh = makeMesh();
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    bvh.update(scene);
    expect(bvh.buffers).not.toBeNull();

    scene.remove(mesh);
    scene.updateMatrixWorld(true);
    bvh.update(scene);

    expect(bvh.buffers).toBeNull();
  });
});
