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
    const q = new THREE.QuaternionKeyframeTrack(
      'N.quaternion',
      [0, 0.5],
      [0, 0, 0, 1, 0, 0.70710678, 0, 0.70710678],
    );
    const s = new THREE.VectorKeyframeTrack('N.scale', [0], [1, 1, 1]);
    const clip = new THREE.AnimationClip('c', 0, [q, s]);

    const [out] = convertAnimations([clip], root);
    const paths = (out?.channels ?? []).map((c) => c.target.path).sort();
    expect(paths).toEqual(['rotation', 'scale']);
    const rotation = out?.channels.find((c) => c.target.path === 'rotation');
    expect(rotation?.sampler.times).toEqual(new Float32Array([0, 0.5]));
    expect(rotation?.sampler.values).toEqual(
      new Float32Array([0, 0, 0, 1, 0, 0.70710678, 0, 0.70710678]),
    );
    expect(rotation?.sampler.times).toHaveLength(2);
    expect(rotation?.sampler.values).toHaveLength(8);
  });

  it('skips tracks whose target node cannot be resolved', () => {
    const root = new THREE.Scene();
    const track = new THREE.VectorKeyframeTrack('Ghost.position', [0], [0, 0, 0]);
    const clip = new THREE.AnimationClip('c', 0, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.channels).toHaveLength(0);
  });

  it('maps THREE InterpolateSmooth to LINEAR because glTF CUBICSPLINE needs tangent triples', () => {
    const mesh = new THREE.Mesh();
    mesh.name = 'N';
    const root = new THREE.Scene();
    root.add(mesh);
    const track = new THREE.VectorKeyframeTrack('N.position', [0, 1], [0, 0, 0, 1, 1, 1]);
    track.setInterpolation(THREE.InterpolateSmooth);
    const clip = new THREE.AnimationClip('smooth', 1, [track]);

    const [out] = convertAnimations([clip], root);
    const sampler = out?.channels[0]?.sampler;
    expect(sampler?.interpolation).toBe('LINEAR');
    expect(sampler?.times).toEqual(new Float32Array([0, 1]));
    expect(sampler?.values).toEqual(new Float32Array([0, 0, 0, 1, 1, 1]));
    expect(sampler?.values).toHaveLength(6);
  });

  // G-P0.4(e) / G-P2.7: discrete (STEP) interpolation was previously untested.
  // THREE.InterpolateDiscrete must map to glTF 'STEP' so a consumer evaluating
  // the emitted sampler holds each keyframe value (no interpolation) rather than
  // lerping — the audit flagged the absence of STEP/CUBICSPLINE import coverage.
  it('maps THREE InterpolateDiscrete to STEP', () => {
    const mesh = new THREE.Mesh();
    mesh.name = 'N';
    const root = new THREE.Scene();
    root.add(mesh);
    const track = new THREE.VectorKeyframeTrack('N.position', [0, 1], [0, 0, 0, 1, 1, 1]);
    track.setInterpolation(THREE.InterpolateDiscrete);
    const clip = new THREE.AnimationClip('step', 1, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.channels[0]?.sampler.interpolation).toBe('STEP');
  });

  // Default linear tracks (the GLTFLoader common case) keep LINEAR — guards the
  // STEP/SMOOTH special-cases from accidentally swallowing the default path.
  it('maps the default (InterpolateLinear) track to LINEAR', () => {
    const mesh = new THREE.Mesh();
    mesh.name = 'N';
    const root = new THREE.Scene();
    root.add(mesh);
    const track = new THREE.VectorKeyframeTrack('N.position', [0, 1], [0, 0, 0, 1, 1, 1]);
    track.setInterpolation(THREE.InterpolateLinear);
    const clip = new THREE.AnimationClip('linear', 1, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.channels[0]?.sampler.interpolation).toBe('LINEAR');
  });

  // A morphTargetInfluences track resolves to the 'weights' channel — pins the
  // 4th entry in PROPERTY_TO_PATH that no other test exercised.
  it('maps morphTargetInfluences to the weights channel', () => {
    const mesh = new THREE.Mesh();
    mesh.name = 'Morpher';
    const root = new THREE.Scene();
    root.add(mesh);
    const track = new THREE.NumberKeyframeTrack(
      'Morpher.morphTargetInfluences',
      [0, 0.25, 0.5],
      [0, 0.75, 1],
    );
    const clip = new THREE.AnimationClip('morph', 1, [track]);

    const [out] = convertAnimations([clip], root);
    expect(out?.channels).toHaveLength(1);
    expect(out?.channels[0]?.target.path).toBe('weights');
    expect(out?.channels[0]?.sampler.times).toEqual(new Float32Array([0, 0.25, 0.5]));
    expect(out?.channels[0]?.sampler.values).toEqual(new Float32Array([0, 0.75, 1]));
    expect(out?.channels[0]?.sampler.times).toHaveLength(3);
    expect(out?.channels[0]?.sampler.values).toHaveLength(3);
  });
});
