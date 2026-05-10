/**
 * Builds a THREE.Scene consumable by three-gpu-pathtracer from a @vitrum/core Scene.
 *
 * Supports mesh primitives and directional / rect-area / point / spot emitters, plus
 * HDRI environments when `SceneEnvironment.hdri` is already a THREE texture.
 * Other primitive / emitter / env kinds are skipped with a console warning.
 *
 * Sprint 2 (Phase 6): each THREE light object created from a SceneEmitter has
 * `userData.cellPower` set to the emitter's total radiant flux:
 *   cellPower = luminance(color × intensity) × area
 *
 * Area conventions per emitter kind:
 *   directional — no surface; cellPower = 0 (sentinel, documented below).
 *   disc-area   — area = π × radius².
 *   rect-area   — area = 4 × |uAxis × vAxis| (uAxis/vAxis are half-extent vectors).
 *   point       — no surface; cellPower = luminance(color × intensity) (point flux).
 *   spot        — no surface; cellPower = luminance(color × intensity) (point flux).
 *   mesh-area   — cellPower is 0 at emitter-creation time; the BVH build computes
 *                 the true per-triangle power directly from mesh geometry.
 *
 * The three-gpu-pathtracer fork patch (plan/sprint-2-pt-fork-patch.md) will read
 * `light.userData.cellPower` when constructing the lights texture. Until that patch
 * lands, the field is carried silently on the userData object.
 */

import type {
  Scene as VitrumScene,
  ScenePrimitive,
  SceneEmitter,
  MeshPrimitive,
  Material as VitrumMaterial,
  NoneEnvironment,
  Vec3,
} from '@vitrum/core';
import {
  BufferGeometry,
  BufferAttribute,
  Color,
  DirectionalLight,
  DoubleSide,
  Matrix4,
  Mesh,
  MeshPhysicalMaterial,
  Object3D,
  PointLight,
  RectAreaLight,
  Scene,
  SpotLight,
  Vector3,
} from 'three';
import type { Texture } from 'three';

function isNoneEnv(env: VitrumScene['environment']): env is NoneEnvironment {
  return env.kind === 'none';
}

/**
 * Standard photometric luminance weights (Rec. 709).
 * Used to convert an RGB radiance to a scalar power value for cellPower.
 */
function luminance(color: Vec3, intensity: number): number {
  return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) * intensity;
}

function vitrumMaterialToThree(m: VitrumMaterial): MeshPhysicalMaterial {
  const color = new Color(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
  const mat = new MeshPhysicalMaterial({
    color,
    roughness: m.roughness,
    metalness: m.metallic,
    emissive: m.emissive
      ? new Color(m.emissive[0], m.emissive[1], m.emissive[2])
      : new Color(0, 0, 0),
    emissiveIntensity: m.emissiveIntensity ?? 1,
    side: DoubleSide,
  });
  if (m.transmission != null && m.transmission > 0) {
    mat.transmission = m.transmission;
    if (m.ior != null) mat.ior = m.ior;
    if (m.attenuationColor != null) {
      mat.attenuationColor.set(
        m.attenuationColor[0],
        m.attenuationColor[1],
        m.attenuationColor[2],
      );
    }
    if (m.attenuationDistance != null) mat.attenuationDistance = m.attenuationDistance;
    if (m.thickness != null) mat.thickness = m.thickness;
  }
  if (m.baseColorMap != null && isTexture(m.baseColorMap)) mat.map = m.baseColorMap;
  if (m.normalMap != null && isTexture(m.normalMap)) {
    mat.normalMap = m.normalMap;
    mat.normalScale.set(m.normalScale ?? 1, m.normalScale ?? 1);
  }
  if (m.roughnessMap != null && isTexture(m.roughnessMap)) mat.roughnessMap = m.roughnessMap;
  if (m.metallicMap != null && isTexture(m.metallicMap)) mat.metalnessMap = m.metallicMap;
  if (m.emissiveMap != null && isTexture(m.emissiveMap)) mat.emissiveMap = m.emissiveMap;
  if (m.transmissionMap != null && isTexture(m.transmissionMap)) mat.transmissionMap = m.transmissionMap;
  return mat;
}

function isTexture(x: unknown): x is Texture {
  return x != null && typeof x === 'object' && 'isTexture' in x && (x as Texture).isTexture === true;
}

function meshPrimitiveToThree(p: MeshPrimitive): Mesh {
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(p.positions, 3));
  geo.setAttribute('normal', new BufferAttribute(p.normals, 3));
  if (p.uvs) geo.setAttribute('uv', new BufferAttribute(p.uvs, 2));
  if (p.tangents) geo.setAttribute('tangent', new BufferAttribute(p.tangents, 4));
  if (p.indices) geo.setIndex(new BufferAttribute(p.indices, 1));
  const mat = vitrumMaterialToThree(p.material);
  const mesh = new Mesh(geo, mat);
  mesh.name = String(p.id);
  const m = new Matrix4();
  if (p.transform) m.fromArray(Array.from(p.transform));
  else m.identity();
  mesh.matrix.copy(m);
  mesh.matrixWorld.copy(m);
  mesh.matrixAutoUpdate = false;
  return mesh;
}

const _u = new Vector3();
const _v = new Vector3();
const _x = new Vector3();
const _y = new Vector3();
const _z = new Vector3();

function emitterToThree(e: SceneEmitter): Object3D | null {
  switch (e.kind) {
    case 'directional': {
      const L = new DirectionalLight(
        new Color(e.color[0], e.color[1], e.color[2]),
        e.intensity,
      );
      L.name = String(e.id);
      const d = e.direction;
      _u.set(d[0], d[1], d[2]).multiplyScalar(1000);
      L.position.copy(_u);
      L.target.position.set(0, 0, 0);
      L.add(L.target);
      // Directional has no surface area → cellPower = 0 (sentinel).
      // Sun irradiance is infinite-distance; power per unit area is not
      // meaningful in the same way as a finite-area emitter. The fork-side
      // light tree (Sprint 3 fork patch) should treat cellPower=0 as
      // "always-sample-via-env" or exclude from the light tree CDF entirely.
      L.userData['cellPower'] = 0;
      return L;
    }
    case 'rect-area': {
      const uVec = _u.set(e.uAxis[0], e.uAxis[1], e.uAxis[2]);
      const vVec = _v.set(e.vAxis[0], e.vAxis[1], e.vAxis[2]);
      const w = 2 * uVec.length();
      const h = 2 * vVec.length();
      const L = new RectAreaLight(
        new Color(e.color[0], e.color[1], e.color[2]),
        e.intensity,
        Math.max(w, 1e-6),
        Math.max(h, 1e-6),
      );
      _x.copy(uVec).normalize();
      _y.copy(vVec).normalize();
      _z.crossVectors(_x, _y);
      if (_z.lengthSq() < 1e-12) {
        console.warn(`@vitrum/pt-webgl: rect-area emitter "${e.id}" has degenerate u/v axes; skipping`);
        return null;
      }
      _z.normalize();
      const basis = new Matrix4().makeBasis(_x, _y, _z);
      L.position.set(e.position[0], e.position[1], e.position[2]);
      L.matrix.copy(basis);
      L.matrix.setPosition(L.position);
      L.matrixAutoUpdate = false;
      L.matrixWorld.copy(L.matrix);
      // Sprint 2: rect-area cellPower = luminance(color×intensity) × area.
      // uAxis and vAxis are HALF-extent vectors per @vitrum/core/scene.ts:
      //   "uAxis: Vec3 // half-width vector"
      // Full area = 4 × |uAxis × vAxis|  (2× width × 2× height = 4 × half-cross).
      const crossLen = _z.crossVectors(
        _u.set(e.uAxis[0], e.uAxis[1], e.uAxis[2]),
        _v.set(e.vAxis[0], e.vAxis[1], e.vAxis[2]),
      ).length();
      const rectArea = 4 * crossLen;
      L.userData['cellPower'] = luminance(e.color, e.intensity) * rectArea;
      return L;
    }
    case 'point': {
      const L = new PointLight(
        new Color(e.color[0], e.color[1], e.color[2]),
        e.intensity,
        e.distance ?? 0,
        e.decay ?? 2,
      );
      L.name = String(e.id);
      L.position.set(e.position[0], e.position[1], e.position[2]);
      // Sprint 2: point emitters have no surface area; treat luminous
      // intensity as the power value (point flux convention).
      L.userData['cellPower'] = luminance(e.color, e.intensity);
      return L;
    }
    case 'spot': {
      const L = new SpotLight(
        new Color(e.color[0], e.color[1], e.color[2]),
        e.intensity,
        e.distance ?? 0,
        e.angle,
        e.penumbra ?? 0,
        e.decay ?? 2,
      );
      L.name = String(e.id);
      L.position.set(e.position[0], e.position[1], e.position[2]);
      const dir = e.direction;
      L.target.position.set(
        e.position[0] - dir[0] * 10,
        e.position[1] - dir[1] * 10,
        e.position[2] - dir[2] * 10,
      );
      L.add(L.target);
      // Sprint 2: spot emitters have no surface area; treat luminous
      // intensity as the power value (point flux convention, same as point).
      L.userData['cellPower'] = luminance(e.color, e.intensity);
      return L;
    }
    default: {
      console.warn(
        `@vitrum/pt-webgl: emitter kind "${(e as SceneEmitter).kind}" not implemented for three-gpu-pathtracer path — skipped`,
      );
      return null;
    }
  }
}

function applyEnvironment(threeScene: Scene, env: VitrumScene['environment']): void {
  if (isNoneEnv(env)) {
    threeScene.background = new Color(0, 0, 0);
    threeScene.environment = null;
    return;
  }
  if (env.kind === 'hdri') {
    if (isTexture(env.hdri)) {
      threeScene.environment = env.hdri;
      threeScene.background = env.hdri;
      threeScene.environmentIntensity = env.intensity ?? 1;
    } else {
      console.warn(
        '@vitrum/pt-webgl: HDRI environment requires THREE.Texture handle; got opaque TextureRef — using black background',
      );
      threeScene.background = new Color(0, 0, 0);
      threeScene.environment = null;
    }
    return;
  }
  console.warn('@vitrum/pt-webgl: procedural-sky environment not wired in pt-webgl — use HDRI or none');
  threeScene.background = new Color(0.02, 0.02, 0.03);
  threeScene.environment = null;
}

/**
 * @param vitrumScene Source scene from @vitrum/core or @vitrum/three-bindings
 */
export function vitrumSceneToThree(vitrumScene: VitrumScene): Scene {
  const threeScene = new Scene();
  for (const p of vitrumScene.primitives) {
    if (p.kind === 'mesh') {
      threeScene.add(meshPrimitiveToThree(p));
    } else {
      console.warn(
        `@vitrum/pt-webgl: primitive kind "${(p as ScenePrimitive).kind}" skipped (mesh-only in this slice)`,
      );
    }
  }
  for (const e of vitrumScene.emitters) {
    const obj = emitterToThree(e);
    if (obj) threeScene.add(obj);
  }
  applyEnvironment(threeScene, vitrumScene.environment);
  return threeScene;
}
