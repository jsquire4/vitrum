/**
 * animationImport.test.ts — P3: convertAnimations maps THREE.AnimationClip[]
 * (as GLTFLoader produces) into vitrum AnimationClip[], resolving each track's
 * target node to its uuid and mapping the property to a glTF target path.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertAnimations } from '../animationImport.js';

describe('convertAnimations — THREE clips → vitrum AnimationClip (P3)', () => {
  it('maps a position track to a translation channel targeting the node uuid', () => {
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
    mesh.name = 'Cube';
    const root = new THREE.Scene();
    root.add(mesh);
    const track = new THREE.VectorKeyframeTrack('Cube.position', [0, 1], [0, 0, 0, 1, 2, 3]);
    const clip = new THREE.AnimationClip('Move', 1, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.name).toBe('Move');
    expect(out?.duration).toBe(1);
    expect(out?.channels).toHaveLength(1);
    expect(out?.channels[0]?.target.node).toBe(mesh.uuid);
    expect(out?.channels[0]?.target.path).toBe('translation');
    expect(out?.channels[0]?.sampler.values).toEqual(new Float32Array([0, 0, 0, 1, 2, 3]));
  });

  it('maps quaternion → rotation and scale → scale; skips unknown properties', () => {
    const mesh = new THREE.Mesh();
    mesh.name = 'N';
    const root = new THREE.Scene();
    root.add(mesh);
    const q = new THREE.QuaternionKeyframeTrack('N.quaternion', [0], [0, 0, 0, 1]);
    const s = new THREE.VectorKeyframeTrack('N.scale', [0], [1, 1, 1]);
    const clip = new THREE.AnimationClip('c', 0, [q, s]);

    const [out] = convertAnimations([clip], root);
    const paths = (out?.channels ?? []).map((c) => c.target.path).sort();
    expect(paths).toEqual(['rotation', 'scale']);
  });

  it('skips tracks whose target node cannot be resolved', () => {
    const root = new THREE.Scene();
    const track = new THREE.VectorKeyframeTrack('Ghost.position', [0], [0, 0, 0]);
    const clip = new THREE.AnimationClip('c', 0, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.channels).toHaveLength(0);
  });
});
