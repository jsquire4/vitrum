import * as THREE from 'three';

type Vec3 = readonly [number, number, number];

/** Minimal Cornell box (three.js) — same layout as the original cornell-box example. */
export function buildCornellBoxThreeScene(): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });

  const mk = (
    geo: THREE.BufferGeometry,
    mat: THREE.MeshPhysicalMaterial,
    pos: Vec3,
    scale: Vec3,
  ) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.scale.set(scale[0], scale[1], scale[2]);
    scene.add(mesh);
  };

  const t = 0.02;
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, -1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, 1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), green, [1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), red, [-1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, 2, t), white, [0, 0, -1], [1, 1, 1]);

  mk(new THREE.BoxGeometry(0.6, 0.6, 0.6), white, [-0.35, -0.65, 0.2], [1, 1, 1]);
  mk(new THREE.BoxGeometry(0.6, 1.2, 0.6), white, [0.3, -0.35, -0.3], [1, 1, 1]);

  const light = new THREE.RectAreaLight(0xffffff, 12, 1.0, 1.0);
  light.position.set(0, 0.98, 0);
  light.rotation.x = -Math.PI / 2;
  scene.add(light);

  return scene;
}
