import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { sceneFromThreeJS } from '../index.js';

function geometryFromPositions(positions: number[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  const normals = new Float32Array(positions.length);
  for (let i = 2; i < normals.length; i += 3) normals[i] = 1;
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return g;
}

function groupedQuadGeometry(): THREE.BufferGeometry {
  const g = geometryFromPositions([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 0,
  ]);
  g.setIndex([0, 1, 2, 2, 1, 3]);
  g.addGroup(0, 3, 1);
  g.addGroup(3, 3, 0);
  return g;
}

function expectMatrixClose(actual: ArrayLike<number>, expected: THREE.Matrix4): void {
  for (let k = 0; k < 16; k += 1) {
    expect(actual[k]).toBeCloseTo(expected.elements[k]!, 5);
  }
}

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

  it('splits grouped multi-material Mesh geometry into stable per-group primitives', () => {
    const s = new THREE.Scene();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 1, 0,
    ]), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3));
    g.setIndex([0, 1, 2, 2, 1, 3]);
    g.addGroup(0, 3, 1);
    g.addGroup(3, 3, 0);

    const mesh = new THREE.Mesh(g, [
      new THREE.MeshStandardMaterial({ color: new THREE.Color(0, 1, 0) }),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 0, 0) }),
    ]);
    s.add(mesh);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(2);

    const first = v.primitives[0]!;
    const second = v.primitives[1]!;
    expect(first.kind).toBe('mesh');
    expect(second.kind).toBe('mesh');
    if (first.kind !== 'mesh' || second.kind !== 'mesh') return;

    expect(first.id).toBe(`${mesh.uuid}:group:0:material:1`);
    expect(second.id).toBe(`${mesh.uuid}:group:1:material:0`);
    expect(first.indices).toBeUndefined();
    expect(second.indices).toBeUndefined();
    expect(first.positions).toEqual(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]));
    expect(second.positions).toEqual(new Float32Array([
      0, 1, 0,
      1, 0, 0,
      1, 1, 0,
    ]));
    expect(first.material.baseColor).toEqual([1, 0, 0]);
    expect(second.material.baseColor).toEqual([0, 1, 0]);
  });

  it('splits non-indexed grouped Mesh geometry and leaves group gaps unmaterialized', () => {
    const s = new THREE.Scene();
    const g = geometryFromPositions([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      10, 0, 0,
      11, 0, 0,
      10, 1, 0,
      20, 0, 0,
      21, 0, 0,
      20, 1, 0,
    ]);
    g.addGroup(0, 3, 0);
    g.addGroup(6, 3, 1);

    const mesh = new THREE.Mesh(g, [
      new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 0, 0) }),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(0, 0, 1) }),
    ]);
    s.add(mesh);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(2);

    const first = v.primitives[0]!;
    const second = v.primitives[1]!;
    expect(first.kind).toBe('mesh');
    expect(second.kind).toBe('mesh');
    if (first.kind !== 'mesh' || second.kind !== 'mesh') return;

    expect(first.positions).toEqual(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]));
    expect(second.positions).toEqual(new Float32Array([
      20, 0, 0,
      21, 0, 0,
      20, 1, 0,
    ]));
    expect(first.material.baseColor).toEqual([1, 0, 0]);
    expect(second.material.baseColor).toEqual([0, 0, 1]);
  });

  it('throws for grouped Mesh material indices outside the material array', () => {
    const s = new THREE.Scene();
    const g = geometryFromPositions([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    g.addGroup(0, 3, 2);
    const mesh = new THREE.Mesh(g, [
      new THREE.MeshStandardMaterial(),
      new THREE.MeshStandardMaterial(),
    ]);
    mesh.name = 'bad-material-index';
    s.add(mesh);

    expect(() => sceneFromThreeJS(s)).toThrow(
      /Mesh "bad-material-index" group 0 references material index 2, but the mesh has 2 materials/,
    );
  });

  it('throws for grouped Mesh ranges outside indexed and non-indexed geometry bounds', () => {
    const indexedScene = new THREE.Scene();
    const indexed = geometryFromPositions([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    indexed.setIndex([0, 1, 2]);
    indexed.addGroup(1, 3, 0);
    const indexedMesh = new THREE.Mesh(indexed, [
      new THREE.MeshStandardMaterial(),
      new THREE.MeshStandardMaterial(),
    ]);
    indexedMesh.name = 'bad-index-range';
    indexedScene.add(indexedMesh);
    expect(() => sceneFromThreeJS(indexedScene)).toThrow(
      /Mesh "bad-index-range" group 0 index range \[1, 4\) exceeds index count 3/,
    );

    const nonIndexedScene = new THREE.Scene();
    const nonIndexed = geometryFromPositions([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    nonIndexed.addGroup(1, 3, 0);
    const nonIndexedMesh = new THREE.Mesh(nonIndexed, [
      new THREE.MeshStandardMaterial(),
      new THREE.MeshStandardMaterial(),
    ]);
    nonIndexedMesh.name = 'bad-vertex-range';
    nonIndexedScene.add(nonIndexedMesh);
    expect(() => sceneFromThreeJS(nonIndexedScene)).toThrow(
      /Mesh "bad-vertex-range" group 0 vertex range \[1, 4\) exceeds vertex count 3/,
    );
  });

  it('targets split mesh-area emitters at the derived group primitive id', () => {
    const s = new THREE.Scene();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3));
    g.addGroup(0, 3, 1);

    const mesh = new THREE.Mesh(g, [
      new THREE.MeshStandardMaterial({ color: 0x333333 }),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: new THREE.Color(0.25, 0.5, 1),
        emissiveIntensity: 4,
      }),
    ]);
    s.add(mesh);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(1);
    expect(v.emitters).toHaveLength(1);
    expect(v.emitters[0]).toMatchObject({
      kind: 'mesh-area',
      meshId: `${mesh.uuid}:group:0:material:1`,
      color: [0.25, 0.5, 1],
      intensity: 4,
    });
    expect(v.primitives[0]!.kind).toBe('mesh');
    expect(v.primitives[0]!.material.emissive).toEqual([0, 0, 0]);
  });

  it('creates mesh-area emitters only for emissive multi-material groups', () => {
    const s = new THREE.Scene();
    const mesh = new THREE.Mesh(groupedQuadGeometry(), [
      new THREE.MeshStandardMaterial({ color: new THREE.Color(0.2, 0.8, 0.1) }),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: new THREE.Color(0.75, 0.25, 0.5),
        emissiveIntensity: 3,
      }),
    ]);
    s.add(mesh);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(2);
    expect(v.emitters).toHaveLength(1);
    expect(v.emitters[0]).toMatchObject({
      kind: 'mesh-area',
      meshId: `${mesh.uuid}:group:0:material:1`,
      color: [0.75, 0.25, 0.5],
      intensity: 3,
    });

    const emissivePrim = v.primitives.find((p) => p.id === `${mesh.uuid}:group:0:material:1`);
    const nonEmissivePrim = v.primitives.find((p) => p.id === `${mesh.uuid}:group:1:material:0`);
    expect(emissivePrim?.kind).toBe('mesh');
    expect(nonEmissivePrim?.kind).toBe('mesh');
    expect(emissivePrim?.material.emissive).toEqual([0, 0, 0]);
    expect(emissivePrim?.material.emissiveIntensity).toBe(0);
    expect(nonEmissivePrim?.material.emissive).toBeUndefined();
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
  it('converts InstancedMesh into instanced-mesh primitive (TLAS path)', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshPhysicalMaterial(),
      3,
    );
    const m = new THREE.Matrix4();
    for (let i = 0; i < 3; i += 1) {
      m.makeTranslation(i * 2, 0, 0);
      im.setMatrixAt(i, m);
    }
    im.instanceMatrix.needsUpdate = true;
    s.add(im);
    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(1);
    const prim = v.primitives[0]!;
    expect(prim.kind).toBe('instanced-mesh');
    if (prim.kind !== 'instanced-mesh') return;
    expect(prim.instances).toHaveLength(3);
    expect(prim.id).toBe(im.uuid);
  });

  it('splits grouped multi-material InstancedMesh geometry into per-group instanced primitives', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = new THREE.Scene();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 1, 0,
    ]), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]), 3));
    g.setIndex([0, 1, 2, 2, 1, 3]);
    g.addGroup(0, 3, 1);
    g.addGroup(3, 3, 0);

    const im = new THREE.InstancedMesh(g, [
      new THREE.MeshStandardMaterial({ color: new THREE.Color(0, 1, 0) }),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 0, 0) }),
    ], 3);
    const instanceMatrices = [
      new THREE.Matrix4().makeTranslation(2, 0, 0),
      new THREE.Matrix4().makeTranslation(0, 3, 0),
      new THREE.Matrix4().compose(
        new THREE.Vector3(-1, 2, 4),
        new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 3, Math.PI / 5)),
        new THREE.Vector3(2, 0.5, 1.25),
      ),
    ];
    for (let i = 0; i < instanceMatrices.length; i += 1) {
      im.setMatrixAt(i, instanceMatrices[i]!);
    }
    im.instanceMatrix.needsUpdate = true;
    s.add(im);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(2);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();

    const first = v.primitives[0]!;
    const second = v.primitives[1]!;
    expect(first.kind).toBe('instanced-mesh');
    expect(second.kind).toBe('instanced-mesh');
    if (first.kind !== 'instanced-mesh' || second.kind !== 'instanced-mesh') return;

    expect(first.id).toBe(`${im.uuid}:group:0:material:1`);
    expect(second.id).toBe(`${im.uuid}:group:1:material:0`);
    expect(first.indices).toBeUndefined();
    expect(second.indices).toBeUndefined();
    expect(first.positions).toEqual(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]));
    expect(second.positions).toEqual(new Float32Array([
      0, 1, 0,
      1, 0, 0,
      1, 1, 0,
    ]));
    expect(first.material.baseColor).toEqual([1, 0, 0]);
    expect(second.material.baseColor).toEqual([0, 1, 0]);
    expect(first.instances).toHaveLength(3);
    expect(second.instances).toHaveLength(3);
    for (let i = 0; i < instanceMatrices.length; i += 1) {
      expectMatrixClose(first.instances[i]!, instanceMatrices[i]!);
      expectMatrixClose(second.instances[i]!, instanceMatrices[i]!);
    }
  });

  it('emissive InstancedMesh emits mesh-area and strips duplicate emissive on primitive', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: new THREE.Color(0.1, 0.2, 0.4),
        emissiveIntensity: 5,
      }),
      2,
    );
    s.add(im);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(1);
    expect(v.emitters).toHaveLength(1);
    expect(v.emitters[0]).toMatchObject({
      kind: 'mesh-area',
      meshId: im.uuid,
      color: [0.1, 0.2, 0.4],
      intensity: 5,
    });
    const prim = v.primitives[0]!;
    expect(prim.kind).toBe('instanced-mesh');
    expect(prim.material.emissive).toEqual([0, 0, 0]);
    expect(prim.material.emissiveIntensity).toBe(0);
  });

  it('splits grouped multi-material SkinnedMesh geometry and preserves skin, morph, and skeleton payloads', () => {
    const s = new THREE.Scene();
    const g = geometryFromPositions([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 1, 0,
    ]);
    g.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array([
      0, 1, 0, 0,
      1, 0, 0, 0,
      1, 1, 0, 0,
      0, 0, 0, 0,
    ]), 4));
    g.setAttribute('skinWeight', new THREE.BufferAttribute(new Float32Array([
      0.75, 0.25, 0, 0,
      1, 0, 0, 0,
      0.5, 0.5, 0, 0,
      0.25, 0.75, 0, 0,
    ]), 4));
    const morphPositions = new Float32Array([
      0.1, 0.2, 0.3,
      1.1, 1.2, 1.3,
      2.1, 2.2, 2.3,
      3.1, 3.2, 3.3,
    ]);
    const morphNormals = new Float32Array([
      0.01, 0.02, 0.03,
      0.11, 0.12, 0.13,
      0.21, 0.22, 0.23,
      0.31, 0.32, 0.33,
    ]);
    g.morphTargetsRelative = true;
    g.morphAttributes.position = [new THREE.BufferAttribute(morphPositions, 3)];
    g.morphAttributes.normal = [new THREE.BufferAttribute(morphNormals, 3)];
    g.setIndex([0, 1, 2, 2, 1, 3]);
    g.addGroup(0, 3, 1);
    g.addGroup(3, 3, 0);

    const rootBone = new THREE.Bone();
    rootBone.position.set(0, 1, 0);
    const childBone = new THREE.Bone();
    childBone.position.set(2, 0, 0);
    rootBone.add(childBone);
    const skeleton = new THREE.Skeleton([rootBone, childBone]);
    const bindMatrix = new THREE.Matrix4().makeTranslation(4, -2, 0.5);
    const sm = new THREE.SkinnedMesh(g, [
      new THREE.MeshStandardMaterial({ color: new THREE.Color(0, 1, 0) }),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 0, 0) }),
    ]);
    sm.add(rootBone);
    sm.bind(skeleton, bindMatrix);
    sm.morphTargetInfluences = [0.65];
    s.add(sm);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(2);
    const first = v.primitives[0]!;
    const second = v.primitives[1]!;
    expect(first.kind).toBe('skinned-mesh');
    expect(second.kind).toBe('skinned-mesh');
    if (first.kind !== 'skinned-mesh' || second.kind !== 'skinned-mesh') return;

    expect(first.id).toBe(`${sm.uuid}:group:0:material:1`);
    expect(second.id).toBe(`${sm.uuid}:group:1:material:0`);
    expect(first.positions).toEqual(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]));
    expect(second.positions).toEqual(new Float32Array([
      0, 1, 0,
      1, 0, 0,
      1, 1, 0,
    ]));
    expect(first.skinIndices).toEqual(new Uint32Array([
      0, 1, 0, 0,
      1, 0, 0, 0,
      1, 1, 0, 0,
    ]));
    expect(first.skinWeights).toEqual(new Float32Array([
      0.75, 0.25, 0, 0,
      1, 0, 0, 0,
      0.5, 0.5, 0, 0,
    ]));
    expect(second.skinIndices).toEqual(new Uint32Array([
      1, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 0, 0,
    ]));
    expect(second.skinWeights).toEqual(new Float32Array([
      0.5, 0.5, 0, 0,
      1, 0, 0, 0,
      0.25, 0.75, 0, 0,
    ]));
    expect(first.morphTargets?.[0]).toEqual(new Float32Array([
      0.1, 0.2, 0.3,
      1.1, 1.2, 1.3,
      2.1, 2.2, 2.3,
    ]));
    expect(second.morphTargets?.[0]).toEqual(new Float32Array([
      2.1, 2.2, 2.3,
      1.1, 1.2, 1.3,
      3.1, 3.2, 3.3,
    ]));
    expect(first.morphTargetNormals?.[0]).toEqual(new Float32Array([
      0.01, 0.02, 0.03,
      0.11, 0.12, 0.13,
      0.21, 0.22, 0.23,
    ]));
    expect(second.morphTargetNormals?.[0]).toEqual(new Float32Array([
      0.21, 0.22, 0.23,
      0.11, 0.12, 0.13,
      0.31, 0.32, 0.33,
    ]));
    expect(first.morphWeights).toEqual(new Float32Array([0.65]));
    expect(second.morphWeights).toEqual(new Float32Array([0.65]));
    expect(first.bones).toHaveLength(32);
    expect(first.boneInverses).toHaveLength(32);
    expect(second.bones).toEqual(first.bones);
    expect(second.boneInverses).toEqual(first.boneInverses);
    expect(first.bindMatrix).toEqual(new Float32Array(bindMatrix.elements));
    expect(first.bindMatrixInverse).toEqual(new Float32Array(sm.bindMatrixInverse.elements));
    expect(second.bindMatrix).toEqual(first.bindMatrix);
    expect(second.bindMatrixInverse).toEqual(first.bindMatrixInverse);
    expect(first.material.baseColor).toEqual([1, 0, 0]);
    expect(second.material.baseColor).toEqual([0, 1, 0]);
  });

  it('accepts host-provided custom material conversion for ShaderMaterial', () => {
    const s = new THREE.Scene();
    const shader = new THREE.ShaderMaterial();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), shader);
    s.add(mesh);
    const converter = vi.fn(() => ({
      baseColor: [0.2, 0.4, 0.8] as [number, number, number],
      roughness: 0.35,
      metallic: 0,
    }));

    const v = sceneFromThreeJS(s, { materialConverter: converter });

    expect(converter).toHaveBeenCalledWith(shader, expect.objectContaining({
      label: mesh.uuid,
      meshTypeName: 'Mesh',
    }));
    expect(v.primitives).toHaveLength(1);
    expect(v.primitives[0]!.material.baseColor).toEqual([0.2, 0.4, 0.8]);
  });

  it('preserves rotation + scale + parent-world instance transforms (TLAS)', () => {
    // G-P0.4 / G-P2.7: all prior InstancedMesh fixtures used identity/translation
    // matrices, so the rotation/scale path through convertInstancedMesh
    // (getMatrixAt → premultiply(matrixWorld)) was unpinned. Build a non-trivial
    // per-instance matrix (rotate Z 30°, scale, translate) under a parent group
    // that itself carries a world transform, and assert the composed result
    // (parentWorld · instanceLocal) survives byte-for-byte into instances[i].
    const s = new THREE.Scene();

    const parent = new THREE.Object3D();
    parent.position.set(5, -2, 3);
    parent.rotation.set(0, Math.PI / 4, 0); // 45° about Y
    parent.scale.set(1, 2, 1);
    s.add(parent);

    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshPhysicalMaterial(),
      2,
    );
    parent.add(im);

    const local0 = new THREE.Matrix4().compose(
      new THREE.Vector3(1, 2, 3),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI / 6)), // 30° Z
      new THREE.Vector3(2, 0.5, 1.5),
    );
    const local1 = new THREE.Matrix4().compose(
      new THREE.Vector3(-4, 0, 1),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 3, 0, 0)), // 60° X
      new THREE.Vector3(0.25, 3, 1),
    );
    im.setMatrixAt(0, local0);
    im.setMatrixAt(1, local1);
    im.instanceMatrix.needsUpdate = true;

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(1);
    const prim = v.primitives[0]!;
    expect(prim.kind).toBe('instanced-mesh');
    if (prim.kind !== 'instanced-mesh') return;
    expect(prim.instances).toHaveLength(2);

    // Independent expected matrices: parentWorld · instanceLocal (column-major).
    s.updateMatrixWorld(true);
    const expected0 = new THREE.Matrix4().multiplyMatrices(im.matrixWorld, local0);
    const expected1 = new THREE.Matrix4().multiplyMatrices(im.matrixWorld, local1);

    for (let k = 0; k < 16; k += 1) {
      expect(prim.instances[0]![k]).toBeCloseTo(expected0.elements[k]!, 5);
      expect(prim.instances[1]![k]).toBeCloseTo(expected1.elements[k]!, 5);
    }

    // Sanity: the rotation/scale block is genuinely non-identity (would have been
    // missed by translation-only fixtures — the determinant of the upper 3×3 is
    // the product of the three scales × parent scale, not 1).
    const m0 = expected0.elements;
    const isIdentityRotScale =
      m0[0] === 1 && m0[5] === 1 && m0[10] === 1 &&
      m0[1] === 0 && m0[2] === 0 && m0[4] === 0 &&
      m0[6] === 0 && m0[8] === 0 && m0[9] === 0;
    expect(isIdentityRotScale).toBe(false);
  });

  it('skips transparent MeshBasicMaterial InstancedMesh overlays like plain meshes', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }),
      1,
    );
    im.setMatrixAt(0, new THREE.Matrix4());
    s.add(im);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(0);
  });

  it('throws for ShaderMaterial InstancedMesh the same way plain meshes do', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.ShaderMaterial(),
      1,
    );
    im.name = 'instanced-shader';
    s.add(im);

    expect(() => sceneFromThreeJS(s)).toThrow(/Unsupported THREE type at "instanced-shader"/);
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

  it('skips transparent MeshBasicMaterial SkinnedMesh overlays before skin attributes are required', () => {
    const s = new THREE.Scene();
    const sm = new THREE.SkinnedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }),
    );
    s.add(sm);

    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(0);
  });

  it('throws for ShaderMaterial SkinnedMesh before converter-specific work', () => {
    const s = new THREE.Scene();
    const sm = new THREE.SkinnedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.ShaderMaterial(),
    );
    sm.name = 'skinned-shader';
    s.add(sm);

    expect(() => sceneFromThreeJS(s)).toThrow(/Unsupported THREE type at "skinned-shader"/);
  });

  it('converts directional and spot light directions from world-space target positions', () => {
    const s = new THREE.Scene();

    const directionalGroup = new THREE.Object3D();
    directionalGroup.position.set(10, 0, 0);
    const dl = new THREE.DirectionalLight(0xffffff, 2);
    dl.name = 'sun';
    dl.position.set(0, 5, 0);
    dl.target.position.set(0, 0, 0);
    directionalGroup.add(dl);
    s.add(directionalGroup);

    const spotGroup = new THREE.Object3D();
    spotGroup.position.set(0, 4, 0);
    const spot = new THREE.SpotLight(0xffffff, 3);
    spot.name = 'spot';
    spot.position.set(0, 0, 3);
    spot.target.position.set(0, 0, 0);
    spotGroup.add(spot);
    s.add(spotGroup);

    const v = sceneFromThreeJS(s);
    const directional = v.emitters.find((e) => e.kind === 'directional');
    const spotEmitter = v.emitters.find((e) => e.kind === 'spot');
    expect(directional?.kind).toBe('directional');
    expect(spotEmitter?.kind).toBe('spot');
    if (directional?.kind !== 'directional' || spotEmitter?.kind !== 'spot') return;

    const sunLen = Math.hypot(10, 5, 0);
    expect(directional.direction[0]).toBeCloseTo(10 / sunLen);
    expect(directional.direction[1]).toBeCloseTo(5 / sunLen);
    expect(directional.direction[2]).toBeCloseTo(0);

    const spotLen = Math.hypot(0, 4, 3);
    expect(spotEmitter.direction[0]).toBeCloseTo(0);
    expect(spotEmitter.direction[1]).toBeCloseTo(4 / spotLen);
    expect(spotEmitter.direction[2]).toBeCloseTo(3 / spotLen);
  });

  it('preserves light castShadow and directional angular diameter metadata', () => {
    const s = new THREE.Scene();
    const dl = new THREE.DirectionalLight(0xffffff, 2);
    dl.castShadow = false;
    dl.userData['vitrumLightAngularDiameter'] = 0.0093;
    s.add(dl);

    const point = new THREE.PointLight(0xff0000, 1);
    point.castShadow = true;
    s.add(point);

    const v = sceneFromThreeJS(s);
    const directional = v.emitters.find((e) => e.kind === 'directional');
    const pointEmitter = v.emitters.find((e) => e.kind === 'point');
    expect(directional?.kind).toBe('directional');
    expect(pointEmitter?.kind).toBe('point');
    if (directional?.kind !== 'directional' || pointEmitter?.kind !== 'point') return;

    expect(directional.castShadow).toBe(false);
    expect(directional.angularDiameter).toBeCloseTo(0.0093);
    expect(pointEmitter.castShadow).toBe(true);
  });

  it('converts RectAreaLight axes from orientation and width/height without baking object scale', () => {
    const s = new THREE.Scene();
    const rect = new THREE.RectAreaLight(0xffffff, 1, 2, 4);
    rect.scale.set(3, 5, 1);
    s.add(rect);

    const v = sceneFromThreeJS(s);
    const emitter = v.emitters.find((e) => e.kind === 'rect-area');
    expect(emitter?.kind).toBe('rect-area');
    if (emitter?.kind !== 'rect-area') return;
    expect(Math.hypot(...emitter.uAxis)).toBeCloseTo(1);
    expect(Math.hypot(...emitter.vAxis)).toBeCloseTo(2);
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

  // ── convertFirstMaterial (T15 shared narrow) ─────────────────────────────
  it('accepts MeshBasicMaterial via the shared basic-material dispatch (flat emissive)', () => {
    const s = new THREE.Scene();
    const g = new THREE.BoxGeometry(1, 1, 1);
    const m = new THREE.MeshBasicMaterial({ color: 0x2244ff });
    s.add(new THREE.Mesh(g, m));
    const v = sceneFromThreeJS(s);
    expect(v.primitives).toHaveLength(1);
    expect(v.emitters).toHaveLength(0);
    const prim = v.primitives[0]!;
    expect(prim.kind).toBe('mesh');
    // convertBasicMaterial synthesizes a flat self-lit color via emissive.
    expect(prim.material.emissive).toBeDefined();
  });

  it('throws the unsupported-material error for an InstancedMesh with ShaderMaterial', () => {
    const s = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.ShaderMaterial({ vertexShader: '', fragmentShader: '' }),
      1,
    );
    s.add(im);
    expect(() => sceneFromThreeJS(s)).toThrow(/Unsupported THREE type at .*: ShaderMaterial/);
  });

  it('throws the unsupported-material error for a SkinnedMesh with ShaderMaterial', () => {
    const s = new THREE.Scene();
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const vc = geo.attributes.position!.count;
    geo.setAttribute('skinIndex', new THREE.BufferAttribute(new Uint16Array(vc * 4), 4));
    const w = new Float32Array(vc * 4);
    for (let i = 0; i < vc; i++) w[i * 4] = 1.0;
    geo.setAttribute('skinWeight', new THREE.BufferAttribute(w, 4));
    const bone = new THREE.Bone();
    const sm = new THREE.SkinnedMesh(
      geo,
      new THREE.ShaderMaterial({ vertexShader: '', fragmentShader: '' }),
    );
    sm.add(bone);
    sm.bind(new THREE.Skeleton([bone]));
    s.add(sm);
    expect(() => sceneFromThreeJS(s)).toThrow(/Unsupported THREE type at .*: ShaderMaterial/);
  });

  it('throws "Mesh ... has no normal attribute" via requireAttribute', () => {
    const s = new THREE.Scene();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
    // No normal attribute.
    s.add(new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial()));
    expect(() => sceneFromThreeJS(s)).toThrow(/Mesh ".*" has no normal attribute/);
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
