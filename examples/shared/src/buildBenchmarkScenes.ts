import * as THREE from 'three';

type Vec3 = readonly [number, number, number];

function countSceneTriangles(scene: THREE.Object3D): number {
  let tris = 0;
  scene.traverse((obj) => {
    if (obj instanceof THREE.InstancedMesh) {
      const g = obj.geometry;
      const index = g.index;
      const per = index != null ? index.count / 3 : g.attributes.position.count / 3;
      tris += per * obj.count;
      return;
    }
    if (!(obj instanceof THREE.Mesh)) return;
    const g = obj.geometry;
    const index = g.index;
    if (index != null) tris += index.count / 3;
    else tris += g.attributes.position.count / 3;
  });
  return Math.floor(tris);
}

/** Count triangles for benchmark assertions (CPU-side). */
export function countThreeSceneTriangles(scene: THREE.Scene): number {
  return countSceneTriangles(scene);
}

function addCornellShell(
  scene: THREE.Scene,
  white: THREE.MeshPhysicalMaterial,
  red: THREE.MeshPhysicalMaterial,
  green: THREE.MeshPhysicalMaterial,
): void {
  const t = 0.02;
  const mk = (geo: THREE.BufferGeometry, mat: THREE.Material, pos: Vec3) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    scene.add(mesh);
  };
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, -1, 0]);
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, 1, 0]);
  mk(new THREE.BoxGeometry(t, 2, 2), green, [1, 0, 0]);
  mk(new THREE.BoxGeometry(t, 2, 2), red, [-1, 0, 0]);
  mk(new THREE.BoxGeometry(2, 2, t), white, [0, 0, -1]);
}

/**
 * Static ~200k-triangle stress scene (PR-6 `PR-hybrid-200k-static`).
 * Dense subdivided floor inside a Cornell shell — no animation, no instancing.
 */
export function buildBenchmark200kThreeScene(targetTriangles = 200_000): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  const floorMat = new THREE.MeshPhysicalMaterial({
    color: 0xc8c8cc,
    roughness: 0.85,
    metalness: 0.05,
  });

  addCornellShell(scene, white, red, green);

  const shellTris = countSceneTriangles(scene);
  const budget = Math.max(0, targetTriangles - shellTris);
  const quadTarget = Math.max(1, Math.floor(budget / 2));
  const seg = Math.max(1, Math.round(Math.sqrt(quadTarget)));

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(1.9, 1.9, seg, seg),
    floorMat,
  );
  floor.name = 'bench200k-floor';
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.99;
  scene.add(floor);

  const areaLight = new THREE.RectAreaLight(0xffffff, 8, 0.7, 0.7);
  areaLight.position.set(0, 0.98, 0);
  areaLight.rotation.x = -Math.PI / 2;
  scene.add(areaLight);

  return scene;
}

/**
 * Ten instanced props in a Cornell box (PR-6 `PR-hybrid-tlas-10-inst`).
 */
export function buildTlas10InstThreeScene(): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  addCornellShell(scene, white, red, green);

  const geo = new THREE.BoxGeometry(0.18, 0.36, 0.18);
  const mat = new THREE.MeshPhysicalMaterial({ color: 0x6a8caf, roughness: 0.4, metalness: 0.2 });
  const inst = new THREE.InstancedMesh(geo, mat, 10);
  inst.name = 'bench-tlas-inst';
  const m = new THREE.Matrix4();
  for (let i = 0; i < 10; i += 1) {
    const col = i % 5;
    const row = Math.floor(i / 5);
    m.makeTranslation(col * 0.35 - 0.7, -0.72, row * 0.35 - 0.2);
    inst.setMatrixAt(i, m);
  }
  inst.instanceMatrix.needsUpdate = true;
  scene.add(inst);

  const areaLight = new THREE.RectAreaLight(0xffffff, 8, 0.7, 0.7);
  areaLight.position.set(0, 0.98, 0);
  areaLight.rotation.x = -Math.PI / 2;
  scene.add(areaLight);

  return scene;
}
