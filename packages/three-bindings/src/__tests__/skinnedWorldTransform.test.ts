// Regression — skinned-mesh world-transform fidelity through the converter.
//
// A SkinnedMesh bound at the origin and THEN positioned (the classic glTF /
// THREE layout: a root transform above both the armature and the mesh node)
// has world-space bone matrices that already carry the node transform, while
// THREE cancels the node transform back out via the attached-bindMode
// `bindMatrixInverse = inverse(matrixWorld)`. The converter must capture that
// inverse so `solveSkin` yields MESH-LOCAL positions and `prim.transform`
// applies exactly once. Before the fix the pair was only emitted when
// `bindMatrix` was non-identity, so solveSkin emitted world-space positions
// and consumers double-applied the world transform.
import { describe, expect, it } from 'vitest';
import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  Group,
  Matrix4,
  MeshStandardMaterial,
  Scene,
  Skeleton,
  SkinnedMesh,
  Vector3,
} from 'three';
import { solveSkin } from '@vitrum/core';
import { sceneFromThreeJS } from '../index.js';

function buildScene(): { scene: Scene; mesh: SkinnedMesh } {
  const geo = new BufferGeometry();
  geo.setAttribute(
    'position',
    new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
  );
  geo.setAttribute(
    'normal',
    new BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 3),
  );
  geo.setAttribute(
    'skinIndex',
    new BufferAttribute(new Uint16Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4),
  );
  geo.setAttribute(
    'skinWeight',
    new BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4),
  );
  geo.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2]), 1));

  const bone = new Bone();
  const mesh = new SkinnedMesh(geo, new MeshStandardMaterial());
  mesh.add(bone);
  // Bind at the origin (identity bindMatrix — the glTF-typical case).
  mesh.bind(new Skeleton([bone]));

  // THEN position the model root — both the mesh node and the bone chain
  // inherit this transform, reproducing the classic double-transform setup.
  const root = new Group();
  root.position.set(5, -2, 3);
  root.rotation.y = Math.PI / 4;
  root.add(mesh);

  const scene = new Scene();
  scene.add(root);
  scene.updateMatrixWorld(true);

  // Pose the bone after binding so skinning genuinely deforms.
  bone.rotation.z = Math.PI / 2;
  scene.updateMatrixWorld(true);
  return { scene, mesh };
}

describe('skinned-mesh world transform round trip', () => {
  it('solveSkin(prim) + prim.transform matches THREE applyBoneTransform world positions', () => {
    const { scene, mesh } = buildScene();
    const vitrumScene = sceneFromThreeJS(scene);
    const prim = vitrumScene.primitives.find((p) => p.kind === 'skinned-mesh');
    expect(prim).toBeDefined();
    if (prim == null || prim.kind !== 'skinned-mesh') return;

    // THREE only refreshes skeleton.boneMatrices inside the renderer; update
    // manually so applyBoneTransform sees the posed bones.
    mesh.skeleton.update();

    const { positions } = solveSkin(prim);
    const transform = new Matrix4().fromArray(prim.transform as unknown as number[]);

    const posAttr = mesh.geometry.getAttribute('position');
    for (let v = 0; v < 3; v += 1) {
      // Ground truth: THREE's own skinned local position, lifted to world.
      // applyBoneTransform skins the vector passed IN — seed it from the
      // rest-pose position attribute.
      const expected = new Vector3().fromBufferAttribute(posAttr, v);
      mesh.applyBoneTransform(v, expected);
      expected.applyMatrix4(mesh.matrixWorld);

      const got = new Vector3(
        positions[v * 3 + 0]!,
        positions[v * 3 + 1]!,
        positions[v * 3 + 2]!,
      ).applyMatrix4(transform);

      expect(got.x).toBeCloseTo(expected.x, 5);
      expect(got.y).toBeCloseTo(expected.y, 5);
      expect(got.z).toBeCloseTo(expected.z, 5);
    }
  });

  it('still omits the bind pair for an identity-world skinned mesh', () => {
    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(new Float32Array([0, 0, 0]), 3));
    geo.setAttribute('normal', new BufferAttribute(new Float32Array([0, 0, 1]), 3));
    geo.setAttribute('skinIndex', new BufferAttribute(new Uint16Array([0, 0, 0, 0]), 4));
    geo.setAttribute('skinWeight', new BufferAttribute(new Float32Array([1, 0, 0, 0]), 4));
    const bone = new Bone();
    const mesh = new SkinnedMesh(geo, new MeshStandardMaterial());
    mesh.add(bone);
    mesh.bind(new Skeleton([bone]));
    const scene = new Scene();
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const prim = sceneFromThreeJS(scene).primitives.find((p) => p.kind === 'skinned-mesh');
    expect(prim).toBeDefined();
    if (prim == null || prim.kind !== 'skinned-mesh') return;
    expect(prim.bindMatrix).toBeUndefined();
    expect(prim.bindMatrixInverse).toBeUndefined();
  });
});
