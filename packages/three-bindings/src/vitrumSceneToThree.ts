/**
 * Build a `THREE.Scene` from a `@vitrum/core` `Scene` (mesh + principal emitters + HDRI).
 *
 * Used by `@vitrum/pt-webgl` (path-tracer sync) and `@vitrum/walkaround-hybrid`
 * (ReSTIR BVH when the host drives `setScene` via the core contract).
 *
 * Sprint 2: each THREE light has `userData.cellPower` (radiant-flux helper).
 */

import type {
  Scene as VitrumScene,
  SceneEmitter,
  ScenePrimitive,
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
import { VITRUM_USER_DATA_KEYS as K } from './userDataKeys.js';
import { luminance as rec709Luminance } from './math.js';

function isNoneEnv(env: VitrumScene['environment']): env is NoneEnvironment {
  return env.kind === 'none';
}

function luminance(color: Vec3, intensity: number): number {
  return rec709Luminance(color[0], color[1], color[2], intensity);
}

/** Apply all texture-map fields from a vitrum Material onto a Three material. */
function applyTextureMaps(mat: MeshPhysicalMaterial, m: VitrumMaterial): void {
  if (m.baseColorMap != null && isTexture(m.baseColorMap)) mat.map = m.baseColorMap;
  if (m.normalMap != null && isTexture(m.normalMap)) {
    mat.normalMap = m.normalMap;
    mat.normalScale.set(m.normalScale ?? 1, m.normalScale ?? 1);
  }
  if (m.roughnessMap != null && isTexture(m.roughnessMap)) mat.roughnessMap = m.roughnessMap;
  if (m.metallicMap != null && isTexture(m.metallicMap)) mat.metalnessMap = m.metallicMap;
  if (m.emissiveMap != null && isTexture(m.emissiveMap)) mat.emissiveMap = m.emissiveMap;
  if (m.transmissionMap != null && isTexture(m.transmissionMap)) mat.transmissionMap = m.transmissionMap;
}

/** Stamp vitrum-specific extension fields into `mat.userData` using the
 *  canonical keys from `userDataKeys.ts`. Each field is set only when the
 *  source field is defined, so callers without the new RFEs get a clean
 *  userData object (no phantom keys). */
function stampVitrumUserData(mat: MeshPhysicalMaterial, m: VitrumMaterial): void {
  const ud: Record<string, unknown> = mat.userData ?? {};

  // RFE-06 (Sprint 8 — chromatic dispersion)
  if (m.dispersionAbbeNumber !== undefined) {
    ud[K.DISPERSION_ABBE] = m.dispersionAbbeNumber;
  }

  // RFE-07 (Sprint 7 — volume scattering)
  if (m.scatteringCoefficient !== undefined) ud[K.SCATTERING_COEFF] = m.scatteringCoefficient;
  if (m.scatteringCoefficientRGB !== undefined) ud[K.SCATTERING_RGB] = m.scatteringCoefficientRGB;
  if (m.scatteringAnisotropy !== undefined) ud[K.SCATTERING_ANISO] = m.scatteringAnisotropy;

  // RFE-08 (Sprint 12 — spectral attenuation + thin-film stack)
  if (m.spectralAttenuation !== undefined) ud[K.SPECTRAL_ATTEN] = m.spectralAttenuation;
  if (m.thinFilmStack !== undefined) ud[K.THIN_FILM_STACK] = m.thinFilmStack;

  // RFE-03 (per-face surface absorption layers)
  if (m.frontLayer !== undefined) ud[K.FRONT_LAYER] = m.frontLayer;
  if (m.backLayer !== undefined) ud[K.BACK_LAYER] = m.backLayer;

  mat.userData = ud;
}

/** Additive diffuse emission from `mesh-area` emitters referencing this mesh (`color * intensity`). */
function vitrumMaterialToThree(m: VitrumMaterial, meshAreaRgb?: Vec3): MeshPhysicalMaterial {
  const color = new Color(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
  const baseIntensity = m.emissiveIntensity ?? 1;
  const ba = meshAreaRgb ?? [0, 0, 0];
  const ei = Math.max(baseIntensity, 1e-8);
  const er = (m.emissive?.[0] ?? 0) * ei + ba[0];
  const eg = (m.emissive?.[1] ?? 0) * ei + ba[1];
  const eb = (m.emissive?.[2] ?? 0) * ei + ba[2];
  const mat = new MeshPhysicalMaterial({
    color,
    roughness: m.roughness,
    metalness: m.metallic,
    emissive: new Color(er, eg, eb),
    emissiveIntensity: 1,
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
  applyTextureMaps(mat, m);
  stampVitrumUserData(mat, m);
  return mat;
}

function isTexture(x: unknown): x is Texture {
  return x != null && typeof x === 'object' && 'isTexture' in x && (x as Texture).isTexture === true;
}

function meshPrimitiveToThree(p: MeshPrimitive, meshAreaRadianceRgb?: Vec3): Mesh {
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(p.positions, 3));
  geo.setAttribute('normal', new BufferAttribute(p.normals, 3));
  if (p.uvs) geo.setAttribute('uv', new BufferAttribute(p.uvs, 2));
  if (p.tangents) geo.setAttribute('tangent', new BufferAttribute(p.tangents, 4));
  if (p.indices) geo.setIndex(new BufferAttribute(p.indices, 1));
  const mat = vitrumMaterialToThree(p.material, meshAreaRadianceRgb);
  const mesh = new Mesh(geo, mat);
  mesh.name = String(p.id);
  const m = new Matrix4();
  if (p.transform) m.fromArray(p.transform);
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

/** Accumulate diffuse radiance `(color * intensity)` from mesh-area emitters per mesh primitive id. */
function meshEmitterBoostByPrimitiveId(scene: VitrumScene): Map<string, [number, number, number]> {
  const map = new Map<string, [number, number, number]>();
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area') continue;
    const id = String(e.meshId);
    const add = [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ] as [number, number, number];
    const prev = map.get(id);
    if (!prev) {
      map.set(id, [...add]);
      continue;
    }
    prev[0] += add[0];
    prev[1] += add[1];
    prev[2] += add[2];
  }
  return map;
}

/**
 * Circular disc emitter → approximate axis-aligned-parallelogram rect in three.js.
 * Half-span along tangent/bitangent is √π·r/2 per axis so the rectangular patch has
 * the same π·radius² footprint as an ideal disc (different sampling density).
 */
function discAreaEmitterToRectThree(e: Extract<SceneEmitter, { kind: 'disc-area' }>): RectAreaLight | null {
  console.warn(
    `[vitrum/three-bindings] DiscAreaEmitter "${e.id}" converted to RectAreaLight ` +
      `(area-preserving rectangle approximation; Three has no native disc-area light). ` +
      `Round-tripping the scene through sceneFromThreeJS will yield a RectAreaEmitter, ` +
      `not a DiscAreaEmitter.`,
  );
  const n = _z.set(e.normal[0], e.normal[1], e.normal[2]);
  if (n.length() < 1e-8) {
    console.warn(
      `@vitrum/three-bindings: disc-area emitter "${e.id}" has degenerate normal; skipping`,
    );
    return null;
  }
  n.normalize();
  let up = _x.set(0, 1, 0);
  if (Math.abs(n.y) > 0.999) {
    up.set(1, 0, 0);
  }
  const t = _u.copy(up).cross(n);
  if (t.lengthSq() < 1e-12) return null;
  t.normalize();
  const b = _v.copy(n).cross(t).normalize();
  const s = (Math.sqrt(Math.PI) * e.radius) / 2;
  const uVec = t.multiplyScalar(s);
  const vVec = b.multiplyScalar(s);
  const rectArea = 4 * Math.max(_x.copy(uVec).cross(_y.copy(vVec)).length(), 1e-6);
  const w = Math.max(2 * uVec.length(), 1e-6);
  const h = Math.max(2 * vVec.length(), 1e-6);
  const L = new RectAreaLight(new Color(e.color[0], e.color[1], e.color[2]), e.intensity, w, h);
  _x.copy(uVec).normalize();
  _y.copy(vVec).normalize();
  _z.crossVectors(_x, _y);
  if (_z.lengthSq() < 1e-12) return null;
  _z.normalize();
  const basis = new Matrix4().makeBasis(_x, _y, _z);
  L.position.set(e.position[0], e.position[1], e.position[2]);
  L.matrix.copy(basis);
  L.matrix.setPosition(L.position);
  L.matrixAutoUpdate = false;
  L.matrixWorld.copy(L.matrix);
  L.name = String(e.id);
  L.userData['cellPower'] = luminance(e.color, e.intensity) * rectArea;
  return L;
}

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
        console.warn(
          `@vitrum/three-bindings: rect-area emitter "${e.id}" has degenerate u/v axes; skipping`,
        );
        return null;
      }
      _z.normalize();
      const basis = new Matrix4().makeBasis(_x, _y, _z);
      L.position.set(e.position[0], e.position[1], e.position[2]);
      L.matrix.copy(basis);
      L.matrix.setPosition(L.position);
      L.matrixAutoUpdate = false;
      L.matrixWorld.copy(L.matrix);
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
      L.userData['cellPower'] = luminance(e.color, e.intensity);
      return L;
    }
    case 'disc-area':
      return discAreaEmitterToRectThree(e);
    case 'mesh-area':
      // Emission is folded into the referenced mesh's material in vitrumSceneToThree().
      return null;
    default: {
      console.warn(
        `@vitrum/three-bindings: emitter kind "${(e as SceneEmitter).kind}" not implemented for vitrumSceneToThree — skipped`,
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
        '@vitrum/three-bindings: HDRI environment requires THREE.Texture handle; got opaque TextureRef — using black background',
      );
      threeScene.background = new Color(0, 0, 0);
      threeScene.environment = null;
    }
    return;
  }
  console.warn(
    '@vitrum/three-bindings: procedural-sky environment not wired — use HDRI or none',
  );
  threeScene.background = new Color(0.02, 0.02, 0.03);
  threeScene.environment = null;
}

/**
 * Convert a `@vitrum/core` scene graph into a throwaway or persistent `THREE.Scene`
 * (meshes + lights + environment handles).
 *
 * Only `mesh` primitives are supported. `InstancedMeshPrimitive` (kind `'instanced-mesh'`)
 * is not implemented and will throw — consistent with how `sceneFromThreeJS` handles
 * unsupported THREE types. Implement `InstancedMesh` conversion before passing
 * instanced primitives to this function.
 */
export function vitrumSceneToThree(vitrumScene: VitrumScene): Scene {
  const threeScene = new Scene();
  const meshBoost = meshEmitterBoostByPrimitiveId(vitrumScene);
  const meshPrimitiveIds = new Set(
    vitrumScene.primitives.filter((p) => p.kind === 'mesh').map((p) => String(p.id)),
  );
  for (const e of vitrumScene.emitters) {
    if (e.kind === 'mesh-area' && !meshPrimitiveIds.has(String(e.meshId))) {
      console.warn(
        `@vitrum/three-bindings: mesh-area emitter "${e.id}" references unknown mesh id "${String(e.meshId)}" — ignored`,
      );
    }
  }
  for (const p of vitrumScene.primitives) {
    if (p.kind === 'mesh') {
      const add = meshBoost.get(String(p.id));
      threeScene.add(meshPrimitiveToThree(p, add));
    } else {
      throw new Error(
        `Unsupported @vitrum/core primitive kind "${(p as ScenePrimitive).kind}" in vitrumSceneToThree. Supported types are added per Phase 6 sprint.`,
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

/** Dispose geometries/materials under a root built by {@link vitrumSceneToThree}. */
export function disposeVitrumThreeSceneRoot(root: Object3D): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    if (mesh.isMesh === true) {
      mesh.geometry?.dispose();
      const m = mesh.material;
      if (Array.isArray(m)) {
        for (const x of m) x?.dispose?.();
      } else {
        m?.dispose?.();
      }
    }
  });
}
