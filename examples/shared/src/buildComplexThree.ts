import * as THREE from 'three';

type Vec3 = readonly [number, number, number];

/**
 * Stress-test / scale scene (three.js).
 *
 * Starts from the Cornell box geometry and adds:
 *   - a 5×5 grid of varied small props (mix of roughness + metalness)
 *   - 3 emissive panels (left ceiling, centre ceiling, right wall slit)
 *   - 4 metallic spheres around the room perimeter
 *   - a glass slab in the centre to exercise the caustic / refraction path
 *
 * Total: ≈ 50 meshes, 3 area lights — enough to stress BVH construction,
 * RIS light sampling, and the GI reservoir's reconnection-vertex coverage.
 *
 * Used by examples/two-engines-one-scene as `?scene=complex` to A/B against
 * the bare Cornell box (`?scene=cornell`).
 */
export function buildComplexThreeScene(): THREE.Scene {
  const scene = new THREE.Scene();

  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red   = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  const blue  = new THREE.MeshPhysicalMaterial({ color: 0x2f4eab, roughness: 0.6, metalness: 0 });
  const gold  = new THREE.MeshPhysicalMaterial({ color: 0xd4af37, roughness: 0.25, metalness: 1.0 });
  const chrome = new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, roughness: 0.08, metalness: 1.0 });
  const glass  = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.0,
    metalness: 0.0,
    transmission: 1.0,
    ior: 1.5,
    thickness: 0.3,
  });

  const mk = (geo: THREE.BufferGeometry, mat: THREE.MeshPhysicalMaterial, pos: Vec3, scale: Vec3 = [1, 1, 1]) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.scale.set(scale[0], scale[1], scale[2]);
    scene.add(mesh);
  };

  // ── Cornell box shell ────────────────────────────────────────────────────
  const t = 0.02;
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, -1, 0]);  // floor
  mk(new THREE.BoxGeometry(2, t, 2), white, [0,  1, 0]);  // ceiling
  mk(new THREE.BoxGeometry(t, 2, 2), green, [ 1, 0, 0]);  // right wall
  mk(new THREE.BoxGeometry(t, 2, 2), red,   [-1, 0, 0]);  // left wall
  mk(new THREE.BoxGeometry(2, 2, t), white, [0, 0, -1]);  // back wall

  // ── 5×5 grid of varied small props on the floor ──────────────────────────
  const propGeos = [
    new THREE.BoxGeometry(0.12, 0.12, 0.12),
    new THREE.SphereGeometry(0.075, 12, 8),
    new THREE.CylinderGeometry(0.07, 0.07, 0.18, 12),
  ];
  const propMats = [white, red, green, blue, gold];
  for (let gx = 0; gx < 5; gx++) {
    for (let gz = 0; gz < 5; gz++) {
      const x = -0.7 + gx * 0.35;
      const z = -0.7 + gz * 0.35;
      const geo = propGeos[(gx + gz) % propGeos.length]!;
      const mat = propMats[(gx * 5 + gz) % propMats.length]!;
      // Slight y-jitter so the props are not coplanar with the floor.
      const h = mat === gold ? 0.06 : 0.10;
      mk(geo, mat, [x, -0.9 + h, z]);
    }
  }

  // ── 4 metallic spheres perimeter ─────────────────────────────────────────
  const sphere = new THREE.SphereGeometry(0.18, 16, 12);
  mk(sphere, chrome, [-0.75, -0.82,  0.75]);
  mk(sphere, gold,   [ 0.75, -0.82,  0.75]);
  mk(sphere, chrome, [ 0.75, -0.82, -0.75]);
  mk(sphere, gold,   [-0.75, -0.82, -0.75]);

  // ── Glass slab in the centre for caustic stress ──────────────────────────
  mk(new THREE.BoxGeometry(0.35, 0.6, 0.05), glass, [0.0, -0.7, 0.0]);

  // ── 3 emissive panels (multi-light stress for RIS / light tree) ──────────
  const lightCeilCentre = new THREE.RectAreaLight(0xffffff, 10, 0.8, 0.8);
  lightCeilCentre.position.set(0, 0.98, 0);
  lightCeilCentre.rotation.x = -Math.PI / 2;
  scene.add(lightCeilCentre);

  const lightCeilLeft = new THREE.RectAreaLight(0xffd0a0, 6, 0.4, 0.4);
  lightCeilLeft.position.set(-0.6, 0.98, -0.4);
  lightCeilLeft.rotation.x = -Math.PI / 2;
  scene.add(lightCeilLeft);

  const lightWallRight = new THREE.RectAreaLight(0xa0c0ff, 4, 0.6, 0.3);
  lightWallRight.position.set(0.99, 0.3, 0.0);
  lightWallRight.rotation.y = -Math.PI / 2;
  scene.add(lightWallRight);

  return scene;
}
