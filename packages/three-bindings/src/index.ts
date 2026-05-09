import type * as THREE from 'three';
import type {
  Scene,
  ScenePrimitive,
  SceneEmitter,
  SceneEnvironment,
  Material,
  Vec3,
  Mat4,
  MeshPrimitive,
  DirectionalEmitter,
  RectAreaEmitter,
  PointEmitter,
  SpotEmitter,
} from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function colorToVec3(c: THREE.Color): Vec3 {
  return [c.r, c.g, c.b];
}

function normalizeVec3(x: number, y: number, z: number): Vec3 {
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len === 0) return [0, 0, -1];
  return [x / len, y / len, z / len];
}

function extractAttribute(
  geo: THREE.BufferGeometry,
  name: string,
): Float32Array | undefined {
  const attr = geo.getAttribute(name);
  if (attr == null) return undefined;
  const arr = attr.array;
  if (arr instanceof Float32Array) return arr;
  return new Float32Array(arr);
}

function extractIndex(
  geo: THREE.BufferGeometry,
): Uint32Array | Uint16Array | undefined {
  const idx = geo.index;
  if (idx == null) return undefined;
  const arr = idx.array;
  if (arr instanceof Uint32Array || arr instanceof Uint16Array) return arr;
  return new Uint32Array(arr);
}

// ────────────────────────────────────────────────────────────────────────────
// Material conversion
// ────────────────────────────────────────────────────────────────────────────

type ThreeStdMat = THREE.MeshStandardMaterial;
type ThreePhysMat = THREE.MeshPhysicalMaterial;

function isPhysical(m: ThreeStdMat): m is ThreePhysMat {
  return (m as ThreePhysMat).isMeshPhysicalMaterial === true;
}

function convertMaterial(m: ThreeStdMat): Material {
  const base: Material = {
    baseColor: colorToVec3(m.color),
    roughness: m.roughness,
    metallic: m.metalness,
  };

  const emR = m.emissive.r * m.emissiveIntensity;
  const emG = m.emissive.g * m.emissiveIntensity;
  const emB = m.emissive.b * m.emissiveIntensity;
  if (emR !== 0 || emG !== 0 || emB !== 0) {
    base.emissive = [emR, emG, emB];
    base.emissiveIntensity = m.emissiveIntensity;
  }

  if (m.map != null) base.baseColorMap = m.map;
  if (m.normalMap != null) {
    base.normalMap = m.normalMap;
    base.normalScale = m.normalScale.x;
  }
  if (m.roughnessMap != null) base.roughnessMap = m.roughnessMap;
  if (m.metalnessMap != null) base.metallicMap = m.metalnessMap;
  if (m.emissiveMap != null) base.emissiveMap = m.emissiveMap;
  if (m.alphaMap != null) base.alphaMap = m.alphaMap;

  if (!isPhysical(m)) return base;

  const p = m;
  if (p.transmission !== 0) base.transmission = p.transmission;
  if (p.ior !== 1.5) base.ior = p.ior;

  if (p.attenuationDistance !== Infinity) {
    base.attenuationColor = colorToVec3(p.attenuationColor);
    base.attenuationDistance = p.attenuationDistance;
  }

  if (p.thickness !== 0) base.thickness = p.thickness;
  if (p.transmissionMap != null) base.transmissionMap = p.transmissionMap;

  if (p.sheen !== 0) {
    base.sheen = p.sheen;
    base.sheenColor = colorToVec3(p.sheenColor);
    base.sheenRoughness = p.sheenRoughness;
  }

  if (p.clearcoat !== 0) {
    base.clearcoat = p.clearcoat;
    base.clearcoatRoughness = p.clearcoatRoughness;
  }

  if (p.iridescence !== 0) {
    base.iridescence = p.iridescence;
    // THREE uses iridescenceIOR (caps); core uses iridescenceIor (camelCase).
    base.iridescenceIor = p.iridescenceIOR;
    base.iridescenceThicknessRange = p.iridescenceThicknessRange;
  }

  return base;
}

// ────────────────────────────────────────────────────────────────────────────
// Mesh conversion
// ────────────────────────────────────────────────────────────────────────────

function convertMesh(obj: THREE.Mesh): MeshPrimitive {
  const geo = obj.geometry as THREE.BufferGeometry;

  const positions = extractAttribute(geo, 'position');
  if (positions == null) {
    throw new Error(`Mesh "${obj.name || obj.uuid}" has no position attribute.`);
  }

  const normals = extractAttribute(geo, 'normal');
  if (normals == null) {
    throw new Error(`Mesh "${obj.name || obj.uuid}" has no normal attribute. Compute normals before calling sceneFromThreeJS.`);
  }

  const uvs = extractAttribute(geo, 'uv');
  const tangents = extractAttribute(geo, 'tangent');
  const indices = extractIndex(geo);

  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  const rawMat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  if (
    rawMat == null ||
    (!(rawMat as THREE.MeshStandardMaterial).isMeshStandardMaterial &&
      !(rawMat as THREE.MeshPhysicalMaterial).isMeshPhysicalMaterial)
  ) {
    const typeName = rawMat != null ? (rawMat as object).constructor.name : 'null';
    throw new Error(
      `Unsupported THREE type at "${obj.name || obj.uuid}": material ${typeName}. Supported types are added per Phase 6 sprint.`,
    );
  }

  const material = convertMaterial(rawMat as THREE.MeshStandardMaterial);

  return {
    kind: 'mesh',
    id: obj.uuid,
    positions,
    normals,
    transform,
    material,
    ...(uvs != null ? { uvs } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(indices != null ? { indices } : {}),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Unsupported-type warning (logged once per type name per call)
// ────────────────────────────────────────────────────────────────────────────

const warnedTypes = new Set<string>();
function warnOnce(typeName: string, label: string): void {
  if (warnedTypes.has(typeName)) return;
  warnedTypes.add(typeName);
  console.warn(
    `@vitrum/three-bindings: skipping unsupported light type "${typeName}" at "${label}". Supported types are added per Phase 6 sprint.`,
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Converts a THREE.Scene into a @vitrum/core Scene.
 *
 * Call this whenever scene topology changes (geometry added/removed, materials
 * swapped). For property-only edits (color sliders, intensity), prefer
 * engine.updatePrimitive / engine.updateEmitter if the backend supports
 * incremental updates.
 */
export function sceneFromThreeJS(threeScene: THREE.Scene): Scene {
  threeScene.updateMatrixWorld(true);

  const primitives: ScenePrimitive[] = [];
  const emitters: SceneEmitter[] = [];

  threeScene.traverse((obj: THREE.Object3D) => {
    const label = obj.name || obj.uuid;

    // ── Unsupported mesh sub-types ──────────────────────────────────────────
    if ((obj as THREE.InstancedMesh).isInstancedMesh === true) {
      throw new Error(
        `Unsupported THREE type at "${label}": InstancedMesh. Supported types are added per Phase 6 sprint.`,
      );
    }
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh === true) {
      throw new Error(
        `Unsupported THREE type at "${label}": SkinnedMesh. Supported types are added per Phase 6 sprint.`,
      );
    }

    // ── Meshes ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Mesh).isMesh === true) {
      const mesh = obj as THREE.Mesh;
      const rawMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (
        rawMat != null &&
        ((rawMat as THREE.ShaderMaterial).isShaderMaterial === true ||
          (rawMat as THREE.RawShaderMaterial).isRawShaderMaterial === true)
      ) {
        throw new Error(
          `Unsupported THREE type at "${label}": ${(rawMat as object).constructor.name}. Supported types are added per Phase 6 sprint.`,
        );
      }
      primitives.push(convertMesh(mesh));

      // Sprint 2/3 will detect emissive meshes as MeshAreaEmitters when the light-tree work begins.
      return;
    }

    // ── Lights ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Light).isLight !== true) return;

    const light = obj as THREE.Light;
    const color = colorToVec3(light.color);
    const id = light.uuid;

    if ((light as THREE.AmbientLight).isAmbientLight === true) {
      warnOnce('AmbientLight', label);
      return;
    }

    if ((light as THREE.HemisphereLight).isHemisphereLight === true) {
      warnOnce('HemisphereLight', label);
      return;
    }

    if ((light as THREE.DirectionalLight).isDirectionalLight === true) {
      const dl = light as THREE.DirectionalLight;
      const dx = dl.position.x - dl.target.position.x;
      const dy = dl.position.y - dl.target.position.y;
      const dz = dl.position.z - dl.target.position.z;
      const direction = normalizeVec3(dx, dy, dz);
      const emitter: DirectionalEmitter = {
        kind: 'directional',
        id,
        color,
        intensity: dl.intensity,
        direction,
      };
      emitters.push(emitter);
      return;
    }

    if ((light as THREE.RectAreaLight).isRectAreaLight === true) {
      const rl = light as THREE.RectAreaLight;
      // RectAreaLight faces -Z in local space; derive uAxis/vAxis from world matrix.
      const me = rl.matrixWorld.elements;
      // Column 0 = local X (half-width), column 1 = local Y (half-height).
      const hw = rl.width / 2;
      const hh = rl.height / 2;
      const uAxis: Vec3 = [me[0] * hw, me[1] * hw, me[2] * hw];
      const vAxis: Vec3 = [me[4] * hh, me[5] * hh, me[6] * hh];
      const position: Vec3 = [me[12], me[13], me[14]];
      const emitter: RectAreaEmitter = {
        kind: 'rect-area',
        id,
        color,
        intensity: rl.intensity,
        position,
        uAxis,
        vAxis,
      };
      emitters.push(emitter);
      return;
    }

    if ((light as THREE.PointLight).isPointLight === true) {
      const pl = light as THREE.PointLight;
      const position: Vec3 = [pl.position.x, pl.position.y, pl.position.z];
      const emitter: PointEmitter = {
        kind: 'point',
        id,
        color,
        intensity: pl.intensity,
        position,
        distance: pl.distance,
        decay: pl.decay,
      };
      emitters.push(emitter);
      return;
    }

    if ((light as THREE.SpotLight).isSpotLight === true) {
      const sl = light as THREE.SpotLight;
      const position: Vec3 = [sl.position.x, sl.position.y, sl.position.z];
      const dx = sl.position.x - sl.target.position.x;
      const dy = sl.position.y - sl.target.position.y;
      const dz = sl.position.z - sl.target.position.z;
      const direction = normalizeVec3(dx, dy, dz);
      const emitter: SpotEmitter = {
        kind: 'spot',
        id,
        color,
        intensity: sl.intensity,
        position,
        direction,
        angle: sl.angle,
        penumbra: sl.penumbra,
        distance: sl.distance,
        decay: sl.decay,
      };
      emitters.push(emitter);
      return;
    }

    throw new Error(
      `Unsupported THREE type at "${label}": ${light.constructor.name}. Supported types are added per Phase 6 sprint.`,
    );
  });

  const environment = resolveEnvironment(threeScene);

  return { primitives, emitters, environment };
}

// ────────────────────────────────────────────────────────────────────────────
// Environment resolution
// ────────────────────────────────────────────────────────────────────────────

function resolveEnvironment(threeScene: THREE.Scene): SceneEnvironment {
  if (threeScene.environment != null) {
    return { kind: 'hdri', hdri: threeScene.environment };
  }
  // A solid-color background is not an IBL source — treat as no environment.
  return { kind: 'none' };
}
