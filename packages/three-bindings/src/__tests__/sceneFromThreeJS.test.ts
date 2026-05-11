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

  it('throws on SkinnedMesh', () => {
    const s = new THREE.Scene();
    const sm = new THREE.SkinnedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshPhysicalMaterial(),
    );
    s.add(sm);
    expect(() => sceneFromThreeJS(s)).toThrow(/SkinnedMesh/);
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
