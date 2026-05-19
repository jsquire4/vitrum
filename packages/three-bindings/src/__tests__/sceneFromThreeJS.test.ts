import { describe, it, expect, vi } from 'vitest';
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

  // ── Unsupported types throw ───────────────────────────────────────────────
  it('throws on InstancedMesh', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshPhysicalMaterial(),
      1,
    );
    s.add(im);
    expect(() => sceneFromThreeJS(s)).toThrow(/InstancedMesh/);
  });

  it('converts SkinnedMesh into a skinned-mesh primitive (C1 — 2026-05-19)', () => {
    const s = new THREE.Scene();

    // Build a minimal SkinnedMesh: 1 bone, 1 inverse-bind, 8 box-cube
    // vertices each bound to bone 0 with weight 1.
    const geo = new THREE.BoxGeometry(1, 1, 1);
    // BoxGeometry doesn't ship skinIndex/skinWeight — add them.
    const vertCount = geo.attributes.position!.count;
    const skinIdx = new Uint16Array(vertCount * 4);     // all bone 0
    const skinW = new Float32Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) {
      skinW[i * 4 + 0] = 1.0;                            // 100% bone 0
    }
    geo.setAttribute('skinIndex', new THREE.BufferAttribute(skinIdx, 4));
    geo.setAttribute('skinWeight', new THREE.BufferAttribute(skinW, 4));

    const bone = new THREE.Bone();
    bone.position.set(0, 0, 0);
    const skeleton = new THREE.Skeleton([bone]);

    const sm = new THREE.SkinnedMesh(geo, new THREE.MeshPhysicalMaterial({ color: 0x99aabb }));
    sm.add(bone);
    sm.bind(skeleton);
    s.add(sm);

    const v = sceneFromThreeJS(s);
    expect(v.primitives.length).toBe(1);
    const prim = v.primitives[0]!;
    expect(prim.kind).toBe('skinned-mesh');
    if (prim.kind !== 'skinned-mesh') return;            // narrow for TS
    expect(prim.positions.length).toBe(vertCount * 3);
    expect(prim.normals.length).toBe(vertCount * 3);
    expect(prim.skinIndices.length).toBe(vertCount * 4);
    expect(prim.skinWeights.length).toBe(vertCount * 4);
    // Every vertex points at bone 0.
    for (let i = 0; i < vertCount; i++) {
      expect(prim.skinIndices[i * 4 + 0]).toBe(0);
    }
    // Skeleton: 1 bone, 1 inverse-bind, 16 floats each.
    expect(prim.bones.length).toBe(16);
    expect(prim.boneInverses.length).toBe(16);
  });

  it('throws on SkinnedMesh missing skinIndex attribute', () => {
    const s = new THREE.Scene();
    const geo = new THREE.BoxGeometry(1, 1, 1);
    // Intentionally do NOT add skinIndex/skinWeight.
    const bone = new THREE.Bone();
    const sm = new THREE.SkinnedMesh(geo, new THREE.MeshPhysicalMaterial());
    sm.add(bone);
    sm.bind(new THREE.Skeleton([bone]));
    s.add(sm);
    expect(() => sceneFromThreeJS(s)).toThrow(/skinIndex/);
  });

  it('throws on SkinnedMesh with empty skeleton', () => {
    const s = new THREE.Scene();
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const vertCount = geo.attributes.position!.count;
    geo.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array(vertCount * 4), 4));
    const w = new Float32Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) w[i * 4] = 1.0;
    geo.setAttribute('skinWeight', new THREE.BufferAttribute(w, 4));

    const sm = new THREE.SkinnedMesh(geo, new THREE.MeshPhysicalMaterial());
    sm.bind(new THREE.Skeleton([]));
    s.add(sm);
    expect(() => sceneFromThreeJS(s)).toThrow(/empty skeleton/);
  });

  it('throws on ShaderMaterial', () => {
    const s = new THREE.Scene();
    const m = new THREE.ShaderMaterial({ vertexShader: '', fragmentShader: '' });
    s.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), m));
    expect(() => sceneFromThreeJS(s)).toThrow(/ShaderMaterial/);
  });

  // ── Warn paths ─────────────────────────────────────────────────────────────
  it('warns once for AmbientLight (unsupported-skippable)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = new THREE.Scene();
    s.add(new THREE.AmbientLight(0xffffff, 1));
    s.add(new THREE.AmbientLight(0xffffff, 1));
    sceneFromThreeJS(s);
    // Dedup is per-call: even though we added two AmbientLights, we expect
    // exactly one warning for the type.
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toMatch(/AmbientLight/);
    warn.mockRestore();
  });

  it('warns again on a second sceneFromThreeJS call (per-call dedup, not module-global)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = new THREE.Scene();
    s.add(new THREE.HemisphereLight(0xffffff, 0x000000, 1));
    sceneFromThreeJS(s);
    sceneFromThreeJS(s);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
