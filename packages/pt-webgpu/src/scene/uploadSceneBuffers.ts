import type { DiscAreaEmitter, Material, Scene, SceneEmitter } from '@vitrum/core';
import { transformDirection, transformPoint } from '../math/mat4.js';
import { invertMat4 } from '../math/mat4.js';
import { buildCpuBvh } from './buildCpuBvh.js';

export interface PackedSceneData {
  readonly positions: Float32Array; // vec4f packed
  readonly normals: Float32Array; // vec4f packed
  readonly indices: Uint32Array; // vec4u packed (xyz used)
  readonly triMaterialIds: Uint32Array;
  readonly materials: Float32Array; // MATERIAL_VEC4_STRIDE * vec4f per material
  readonly bvhNodes: Float32Array; // 8 floats (32 bytes) per node
  readonly analyticHeaders: Float32Array; // vec4f per analytic primitive: [shapeId, materialId, paramsOffset, 0]
  readonly analyticParams: Float32Array; // vec4f array, two vec4f per analytic primitive (8 floats)
  readonly analyticLocalToWorld: Float32Array; // 4 vec4f (mat4) per analytic primitive
  readonly analyticWorldToLocal: Float32Array; // 4 vec4f (mat4) per analytic primitive
  readonly triangleCount: number;
  readonly analyticCount: number;
  readonly warnings: readonly string[];
  readonly directionalLight: readonly [number, number, number];
  readonly directionalIrradiance: readonly [number, number, number];
  readonly pointLightPosition: readonly [number, number, number];
  readonly pointLightRadiance: readonly [number, number, number];
  readonly hasPointLight: boolean;
  readonly spotLightPosition: readonly [number, number, number];
  readonly spotLightDirection: readonly [number, number, number];
  readonly spotLightCosAngle: number;
  readonly spotLightRadiance: readonly [number, number, number];
  readonly hasSpotLight: boolean;
  readonly rectAreaPosition: readonly [number, number, number];
  readonly rectAreaUAxis: readonly [number, number, number];
  readonly rectAreaVAxis: readonly [number, number, number];
  readonly rectAreaRadiance: readonly [number, number, number];
  readonly hasRectAreaLight: boolean;
  readonly meshAreaTriA: readonly [number, number, number];
  readonly meshAreaTriB: readonly [number, number, number];
  readonly meshAreaTriC: readonly [number, number, number];
  readonly meshAreaRadiance: readonly [number, number, number];
  readonly hasMeshAreaLight: boolean;
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
  readonly environmentMapWidth: number;
  readonly environmentMapHeight: number;
  readonly hasEnvironmentMap: boolean;
  readonly environmentMapTexels: Float32Array; // rgba = radiance.rgb + pdfOmega
  readonly environmentMapCdf: Float32Array; // length N + 1
}

const THIN_FILM_LAYER_LIMIT = 8;
const SPECTRAL_SAMPLE_COUNT = 32;
/** Matches WGSL `MATERIAL_VEC4_STRIDE` (base layers + 8×(ior,thicknessNm,k) + spectral grid + tail). */
const MATERIAL_VEC4_STRIDE = 22;
const MATERIAL_FLOAT_STRIDE = MATERIAL_VEC4_STRIDE * 4;

export const MAX_POINT_LIGHTS = 16;
export const MAX_SPOT_LIGHTS = 8;
export const MAX_RECT_AREA_LIGHTS = 8;
export const MAX_MESH_AREA_LIGHTS = 8;

/** vec4 pairs per point light: position, radiance */
export const POINT_LIGHT_FLOAT_STRIDE = 8;
export const SPOT_LIGHT_FLOAT_STRIDE = 12;
export const RECT_AREA_LIGHT_FLOAT_STRIDE = 16;
export const MESH_AREA_LIGHT_FLOAT_STRIDE = 16;
const SPECTRAL_LAMBDA_MIN_NM = 380;
const SPECTRAL_LAMBDA_MAX_NM = 780;

/** Map disc emitter to rect-axis payload — half-span √(π)/2·radius on each orthogonal tangent so WGSL quad area (=4|u×v|) equals π·r². Sampling differs from a true disc. */
function discAreaPackedAsRect(e: DiscAreaEmitter): {
  readonly position: readonly [number, number, number];
  readonly uAxis: readonly [number, number, number];
  readonly vAxis: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
} {
  const nx = e.normal[0];
  const ny = e.normal[1];
  const nz = e.normal[2];
  const nLen = Math.hypot(nx, ny, nz);
  const ux = nx / nLen;
  const uy = ny / nLen;
  const uz = nz / nLen;
  let ax = 0;
  let ay = 1;
  let az = 0;
  if (Math.abs(uy) > 0.999) {
    ax = 1;
    ay = 0;
    az = 0;
  }
  const tx = ay * uz - az * uy;
  const ty = az * ux - ax * uz;
  const tz = ax * uy - ay * ux;
  const tLen = Math.hypot(tx, ty, tz);
  const tcx = tx / tLen;
  const tcy = ty / tLen;
  const tcz = tz / tLen;
  const bx = uy * tcz - uz * tcy;
  const by = uz * tcx - ux * tcz;
  const bz = ux * tcy - uy * tcx;
  const s = (Math.sqrt(Math.PI) * e.radius) / 2;
  const uAxis = [tcx * s, tcy * s, tcz * s] as const;
  const vAxis = [bx * s, by * s, bz * s] as const;
  return {
    position: [e.position[0], e.position[1], e.position[2]],
    uAxis,
    vAxis,
    radiance: [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ],
  };
}

export interface UploadedSceneBuffers {
  readonly triangleCount: number;
  readonly bvhNodeCount: number;
  readonly materialCount: number;
  readonly analyticCount: number;
  readonly directionalLight: readonly [number, number, number];
  readonly directionalIrradiance: readonly [number, number, number];
  readonly pointLightPosition: readonly [number, number, number];
  readonly pointLightRadiance: readonly [number, number, number];
  readonly hasPointLight: boolean;
  readonly spotLightPosition: readonly [number, number, number];
  readonly spotLightDirection: readonly [number, number, number];
  readonly spotLightCosAngle: number;
  readonly spotLightRadiance: readonly [number, number, number];
  readonly hasSpotLight: boolean;
  readonly rectAreaPosition: readonly [number, number, number];
  readonly rectAreaUAxis: readonly [number, number, number];
  readonly rectAreaVAxis: readonly [number, number, number];
  readonly rectAreaRadiance: readonly [number, number, number];
  readonly hasRectAreaLight: boolean;
  readonly meshAreaTriA: readonly [number, number, number];
  readonly meshAreaTriB: readonly [number, number, number];
  readonly meshAreaTriC: readonly [number, number, number];
  readonly meshAreaRadiance: readonly [number, number, number];
  readonly hasMeshAreaLight: boolean;
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
  readonly environmentMapWidth: number;
  readonly environmentMapHeight: number;
  readonly hasEnvironmentMap: boolean;
  readonly positionsBuffer: GPUBuffer;
  readonly normalsBuffer: GPUBuffer;
  readonly indicesBuffer: GPUBuffer;
  readonly triMaterialIdsBuffer: GPUBuffer;
  readonly materialsBuffer: GPUBuffer;
  readonly bvhNodesBuffer: GPUBuffer;
  readonly analyticHeadersBuffer: GPUBuffer;
  readonly analyticParamsBuffer: GPUBuffer;
  readonly analyticLocalToWorldBuffer: GPUBuffer;
  readonly analyticWorldToLocalBuffer: GPUBuffer;
  readonly environmentMapTexelsBuffer: GPUBuffer;
  readonly environmentMapCdfBuffer: GPUBuffer;
  readonly pointLightsBuffer: GPUBuffer;
  readonly spotLightsBuffer: GPUBuffer;
  readonly rectAreaLightsBuffer: GPUBuffer;
  readonly meshAreaLightsBuffer: GPUBuffer;
  readonly destroy: () => void;
}

function createStorageBuffer(device: GPUDevice, label: string, data: ArrayBufferView): GPUBuffer {
  const minSize = data.byteLength === 0 ? 16 : data.byteLength;
  const buffer = device.createBuffer({
    label,
    size: Math.ceil(minSize / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  } else {
    device.queue.writeBuffer(buffer, 0, new Uint32Array([0, 0, 0, 0]));
  }
  return buffer;
}

function sampleSpectralCurve(curve: Material['spectralAttenuation'], lambdaNm: number): number {
  if (curve == null || curve.values.length === 0) return 0;
  const start = curve.wavelengthStart;
  const end = curve.wavelengthEnd;
  const denom = Math.max(end - start, 1e-6);
  const t = Math.min(1, Math.max(0, (lambdaNm - start) / denom));
  const f = t * (curve.values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, curve.values.length - 1);
  const a = curve.values[i0] ?? 0;
  const b = curve.values[i1] ?? a;
  return a + (b - a) * (f - i0);
}

function materialToPackedVec4s(material: Material): number[] {
  const finite = (v: number, fallback = 0): number => (Number.isFinite(v) ? v : fallback);
  const clamp01 = (v: number): number => Math.min(1, Math.max(0, finite(v)));
  const base = material.baseColor;
  const emissive = material.emissive ?? [0, 0, 0];
  const emissiveIntensity = material.emissiveIntensity ?? 1;
  const roughness = material.roughness ?? 0.5;
  const metallic = material.metallic ?? 0;
  const transmission = material.transmission ?? 0;
  const ior = material.ior ?? 1.5;
  const scatteringCoeff = material.scatteringCoefficient ?? 0;
  const scatteringAnisotropy = material.scatteringAnisotropy ?? 0;
  const scatteringRgb = material.scatteringCoefficientRGB ?? [
    scatteringCoeff,
    scatteringCoeff,
    scatteringCoeff,
  ];
  const frontLayerRaw = material.frontLayer?.transmission ?? [1, 1, 1];
  const frontLayerTx: readonly [number, number, number] = [
    clamp01(frontLayerRaw[0] ?? 1),
    clamp01(frontLayerRaw[1] ?? 1),
    clamp01(frontLayerRaw[2] ?? 1),
  ];
  const frontLayerRoughness =
    material.frontLayer?.roughness == null ? -1 : clamp01(material.frontLayer.roughness);
  const backLayerRaw = material.backLayer?.transmission ?? [1, 1, 1];
  const backLayerTx: readonly [number, number, number] = [
    clamp01(backLayerRaw[0] ?? 1),
    clamp01(backLayerRaw[1] ?? 1),
    clamp01(backLayerRaw[2] ?? 1),
  ];
  const backLayerRoughness =
    material.backLayer?.roughness == null ? -1 : clamp01(material.backLayer.roughness);
  const thinFilmLayers = material.thinFilmStack?.layers ?? [];
  const thinFilmLayerCount = Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT);
  const thinFilmEnabled = thinFilmLayerCount > 0 ? 1 : 0;
  const incidentIor = Math.max(finite(material.thinFilmStack?.incidentIor ?? 1, 1), 1);
  const angleDependentFlag = material.thinFilmStack?.angleDependent === true ? 1 : 0;
  const spectralCurve = material.spectralAttenuation;
  let spectralSampleCount = 0;
  let spectralAvgMu = 0;
  let spectralMinMu = Number.POSITIVE_INFINITY;
  let spectralMaxMu = Number.NEGATIVE_INFINITY;
  const spectralSamples = new Array<number>(SPECTRAL_SAMPLE_COUNT).fill(0);
  if (spectralCurve != null && spectralCurve.values.length > 0) {
    spectralSampleCount = SPECTRAL_SAMPLE_COUNT;
    let sum = 0;
    for (let i = 0; i < SPECTRAL_SAMPLE_COUNT; i += 1) {
      const t = i / Math.max(SPECTRAL_SAMPLE_COUNT - 1, 1);
      const lambda = SPECTRAL_LAMBDA_MIN_NM + t * (SPECTRAL_LAMBDA_MAX_NM - SPECTRAL_LAMBDA_MIN_NM);
      const v = Math.max(sampleSpectralCurve(spectralCurve, lambda), 0);
      spectralSamples[i] = v;
      sum += v;
      spectralMinMu = Math.min(spectralMinMu, v);
      spectralMaxMu = Math.max(spectralMaxMu, v);
    }
    spectralAvgMu = sum / SPECTRAL_SAMPLE_COUNT;
    if (!Number.isFinite(spectralMinMu)) spectralMinMu = 0;
    if (!Number.isFinite(spectralMaxMu)) spectralMaxMu = 0;
  } else {
    spectralMinMu = 0;
    spectralMaxMu = 0;
  }
  const packed = [
    base[0],
    base[1],
    base[2],
    roughness,
    emissive[0] * emissiveIntensity,
    emissive[1] * emissiveIntensity,
    emissive[2] * emissiveIntensity,
    metallic,
    transmission,
    ior,
    scatteringCoeff,
    scatteringAnisotropy,
    scatteringRgb[0],
    scatteringRgb[1],
    scatteringRgb[2],
    spectralSampleCount > 0 ? 1 : 0,
    frontLayerTx[0],
    frontLayerTx[1],
    frontLayerTx[2],
    frontLayerRoughness,
    backLayerTx[0],
    backLayerTx[1],
    backLayerTx[2],
    backLayerRoughness,
    thinFilmEnabled,
    thinFilmLayerCount,
    incidentIor,
    angleDependentFlag,
  ];
  for (let i = 0; i < THIN_FILM_LAYER_LIMIT; i += 1) {
    if (i < thinFilmLayerCount) {
      const layer = thinFilmLayers[i];
      packed.push(
        Math.max(finite(layer?.ior ?? 1, 1), 1),
        Math.max(finite(layer?.thicknessNm ?? 0), 0),
        Math.max(finite(layer?.extinctionCoefficient ?? 0), 0),
      );
    } else {
      packed.push(0, 0, 0);
    }
  }
  packed.push(...spectralSamples);
  packed.push(spectralAvgMu, spectralMinMu, spectralMaxMu, spectralSampleCount);
  return packed;
}

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function analyticShapeId(shape: string): number {
  switch (shape) {
    case 'sphere':
      return 1;
    case 'box':
      return 2;
    case 'capsule':
      return 3;
    case 'cylinder':
      return 4;
    case 'h-channel-came':
      return 5;
    default:
      return 0;
  }
}

function defaultDirectionalLight(scene: Scene): readonly [number, number, number] {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return [0.4, 1.0, 0.2];
  const x = directional.direction[0];
  const y = directional.direction[1];
  const z = directional.direction[2];
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [0.4, 1.0, 0.2];
  // Core contract: direction points AT the light, so incoming light is -direction.
  return [-x / len, -y / len, -z / len];
}

function defaultDirectionalIrradiance(scene: Scene): readonly [number, number, number] {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return [1, 1, 1];
  const scale = directional.intensity;
  return [
    directional.color[0] * scale,
    directional.color[1] * scale,
    directional.color[2] * scale,
  ];
}

function firstPointLight(scene: Scene): {
  readonly position: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
  readonly hasPointLight: boolean;
} {
  const point = scene.emitters.find((e) => e.kind === 'point');
  if (point == null) {
    return {
      position: [0, 0, 0],
      radiance: [0, 0, 0],
      hasPointLight: false,
    };
  }
  return {
    position: [point.position[0], point.position[1], point.position[2]],
    radiance: [
      point.color[0] * point.intensity,
      point.color[1] * point.intensity,
      point.color[2] * point.intensity,
    ],
    hasPointLight: true,
  };
}

function firstSpotLight(scene: Scene): {
  readonly position: readonly [number, number, number];
  readonly direction: readonly [number, number, number];
  readonly cosAngle: number;
  readonly radiance: readonly [number, number, number];
  readonly hasSpotLight: boolean;
} {
  const spot = scene.emitters.find((e) => e.kind === 'spot');
  if (spot == null) {
    return {
      position: [0, 0, 0],
      direction: [0, -1, 0],
      cosAngle: 0,
      radiance: [0, 0, 0],
      hasSpotLight: false,
    };
  }
  const d = spot.direction;
  const len = Math.hypot(d[0], d[1], d[2]);
  const dir: readonly [number, number, number] =
    len < 1e-8 ? [0, -1, 0] : [d[0] / len, d[1] / len, d[2] / len];
  return {
    position: [spot.position[0], spot.position[1], spot.position[2]],
    direction: dir,
    cosAngle: Math.cos(spot.angle),
    radiance: [
      spot.color[0] * spot.intensity,
      spot.color[1] * spot.intensity,
      spot.color[2] * spot.intensity,
    ],
    hasSpotLight: true,
  };
}

function firstRectAreaLight(scene: Scene): {
  readonly position: readonly [number, number, number];
  readonly uAxis: readonly [number, number, number];
  readonly vAxis: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
  readonly hasRectAreaLight: boolean;
} {
  const rect = scene.emitters.find((e) => e.kind === 'rect-area');
  if (rect != null) {
    return {
      position: [rect.position[0], rect.position[1], rect.position[2]],
      uAxis: [rect.uAxis[0], rect.uAxis[1], rect.uAxis[2]],
      vAxis: [rect.vAxis[0], rect.vAxis[1], rect.vAxis[2]],
      radiance: [
        rect.color[0] * rect.intensity,
        rect.color[1] * rect.intensity,
        rect.color[2] * rect.intensity,
      ],
      hasRectAreaLight: true,
    };
  }
  const disc = scene.emitters.find((e) => e.kind === 'disc-area');
  if (disc != null) {
    if (!Number.isFinite(disc.radius) || disc.radius < 1e-8) {
      return {
        position: [0, 0, 0],
        uAxis: [0, 0, 0],
        vAxis: [0, 0, 0],
        radiance: [0, 0, 0],
        hasRectAreaLight: false,
      };
    }
    const d = discAreaPackedAsRect(disc);
    return {
      position: d.position,
      uAxis: [...d.uAxis],
      vAxis: [...d.vAxis],
      radiance: [...d.radiance],
      hasRectAreaLight: true,
    };
  }
  return {
    position: [0, 0, 0],
    uAxis: [0, 0, 0],
    vAxis: [0, 0, 0],
    radiance: [0, 0, 0],
    hasRectAreaLight: false,
  };
}

function firstMeshAreaLight(scene: Scene): {
  readonly triA: readonly [number, number, number];
  readonly triB: readonly [number, number, number];
  readonly triC: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
  readonly hasMeshAreaLight: boolean;
  readonly warnings: readonly string[];
} {
  const emitter = scene.emitters.find((e) => e.kind === 'mesh-area');
  if (emitter == null) {
    return {
      triA: [0, 0, 0],
      triB: [0, 0, 0],
      triC: [0, 0, 0],
      radiance: [0, 0, 0],
      hasMeshAreaLight: false,
      warnings: [],
    };
  }

  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') {
    return {
      triA: [0, 0, 0],
      triB: [0, 0, 0],
      triC: [0, 0, 0],
      radiance: [0, 0, 0],
      hasMeshAreaLight: false,
      warnings: [`Mesh-area emitter "${emitter.id}" references missing or non-mesh primitive "${emitter.meshId}".`],
    };
  }

  const positions = primitive.positions;
  const indices =
    primitive.indices ??
    (() => {
      const generated = new Uint32Array(positions.length / 3);
      for (let i = 0; i < generated.length; i += 1) generated[i] = i;
      return generated;
    })();
  if (indices.length < 3 || positions.length < 9) {
    return {
      triA: [0, 0, 0],
      triB: [0, 0, 0],
      triC: [0, 0, 0],
      radiance: [0, 0, 0],
      hasMeshAreaLight: false,
      warnings: [`Mesh-area emitter "${emitter.id}" references primitive "${emitter.meshId}" with no triangles.`],
    };
  }

  const i0 = indices[0] ?? 0;
  const i1 = indices[1] ?? 0;
  const i2 = indices[2] ?? 0;
  const fetchPos = (idx: number): [number, number, number] => [
    positions[idx * 3] ?? 0,
    positions[idx * 3 + 1] ?? 0,
    positions[idx * 3 + 2] ?? 0,
  ];
  let a = fetchPos(i0);
  let b = fetchPos(i1);
  let c = fetchPos(i2);
  const transform = primitive.kind === 'instanced-mesh' ? primitive.instances[0] : primitive.transform;
  if (transform != null) {
    a = transformPoint(transform, a);
    b = transformPoint(transform, b);
    c = transformPoint(transform, c);
  }
  return {
    triA: a,
    triB: b,
    triC: c,
    radiance: [
      emitter.color[0] * emitter.intensity,
      emitter.color[1] * emitter.intensity,
      emitter.color[2] * emitter.intensity,
    ],
    hasMeshAreaLight: true,
    warnings: [],
  };
}

function environmentParams(scene: Scene): {
  readonly tint: readonly [number, number, number];
  readonly sunDirection: readonly [number, number, number];
  readonly sunStrength: number;
  readonly hdriWidth: number;
  readonly hdriHeight: number;
  readonly hasHdri: boolean;
  readonly hdriTexels: Float32Array;
  readonly hdriCdf: Float32Array;
  readonly warnings: readonly string[];
} {
  if (scene.environment.kind === 'none') {
    return {
      tint: [1, 1, 1],
      sunDirection: [0, 1, 0],
      sunStrength: 0,
      hdriWidth: 0,
      hdriHeight: 0,
      hasHdri: false,
      hdriTexels: new Float32Array(0),
      hdriCdf: new Float32Array(0),
      warnings: [],
    };
  }
  if (scene.environment.kind === 'procedural-sky') {
    const d = scene.environment.sunDirection;
    const len = Math.hypot(d[0], d[1], d[2]);
    const sunDir: readonly [number, number, number] =
      len < 1e-8 ? [0, 1, 0] : [d[0] / len, d[1] / len, d[2] / len];
    const intensity = scene.environment.intensity ?? 1;
    const tintBoost = Math.max(0.2, 1 - scene.environment.mieCoefficient * 10);
    return {
      tint: [0.9 * tintBoost * intensity, 0.95 * intensity, 1.0 * intensity],
      sunDirection: sunDir,
      sunStrength: Math.max(0, intensity),
      hdriWidth: 0,
      hdriHeight: 0,
      hasHdri: false,
      hdriTexels: new Float32Array(0),
      hdriCdf: new Float32Array(0),
      warnings: [],
    };
  }
  type HdriLike = { width?: number; height?: number; data?: ArrayLike<number> };
  const hdri = scene.environment.hdri as HdriLike;
  const width = Number(hdri?.width ?? 0);
  const height = Number(hdri?.height ?? 0);
  const data = hdri?.data;
  if (
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 0 &&
    height > 0 &&
    data != null &&
    typeof data.length === 'number' &&
    data.length >= width * height * 3
  ) {
    const pixelCount = width * height;
    const texels = new Float32Array(pixelCount * 4);
    const cdf = new Float32Array(pixelCount + 1);
    let totalWeight = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const r = Number(data[i * 3] ?? 0);
      const g = Number(data[i * 3 + 1] ?? 0);
      const b = Number(data[i * 3 + 2] ?? 0);
      texels[i * 4] = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;
      const y = (i / width) | 0;
      const theta = ((y + 0.5) / height) * Math.PI;
      const weight = Math.max(0, (0.2126 * r + 0.7152 * g + 0.0722 * b) * Math.sin(theta));
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
    if (totalWeight > 1e-12) {
      const dOmegaBase = (2 * Math.PI / width) * (Math.PI / height);
      for (let i = 0; i < pixelCount; i += 1) {
        cdf[i + 1] = (cdf[i + 1] ?? 0) / totalWeight;
        const y = (i / width) | 0;
        const theta = ((y + 0.5) / height) * Math.PI;
        const sinTheta = Math.max(Math.sin(theta), 1e-5);
        const pmf = Math.max((cdf[i + 1] ?? 0) - (cdf[i] ?? 0), 0);
        texels[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
      }
      cdf[0] = 0;
      cdf[pixelCount] = 1;
      return {
        tint: [1, 1, 1],
        sunDirection: [0, 1, 0],
        sunStrength: scene.environment.intensity ?? 1,
        hdriWidth: width,
        hdriHeight: height,
        hasHdri: true,
        hdriTexels: texels,
        hdriCdf: cdf,
        warnings: [],
      };
    }
  }
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [
      'HDRI environment lacks CPU pixel data (width/height/data); falling back to procedural sky model.',
    ],
  };
}

function packEmitterArrays(scene: Scene): {
  readonly warnings: string[];
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
} {
  const warnings: string[] = [];
  const pointLightsData = new Float32Array(MAX_POINT_LIGHTS * POINT_LIGHT_FLOAT_STRIDE).fill(0);
  let pointLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'point') continue;
    if (pointLightCount >= MAX_POINT_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: point lights capped at ${MAX_POINT_LIGHTS}; emitter "${e.id}" and subsequent point emitters omitted.`,
      );
      break;
    }
    const o = pointLightCount * POINT_LIGHT_FLOAT_STRIDE;
    pointLightsData[o + 0] = e.position[0];
    pointLightsData[o + 1] = e.position[1];
    pointLightsData[o + 2] = e.position[2];
    pointLightsData[o + 4] = e.color[0] * e.intensity;
    pointLightsData[o + 5] = e.color[1] * e.intensity;
    pointLightsData[o + 6] = e.color[2] * e.intensity;
    pointLightCount += 1;
  }

  const spotLightsData = new Float32Array(MAX_SPOT_LIGHTS * SPOT_LIGHT_FLOAT_STRIDE).fill(0);
  let spotLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'spot') continue;
    if (spotLightCount >= MAX_SPOT_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: spot lights capped at ${MAX_SPOT_LIGHTS}; emitter "${e.id}" and subsequent spot emitters omitted.`,
      );
      break;
    }
    const d = e.direction;
    const len = Math.hypot(d[0], d[1], d[2]);
    const dir: readonly [number, number, number] =
      len < 1e-8 ? [0, -1, 0] : [d[0] / len, d[1] / len, d[2] / len];
    const o = spotLightCount * SPOT_LIGHT_FLOAT_STRIDE;
    spotLightsData[o + 0] = e.position[0];
    spotLightsData[o + 1] = e.position[1];
    spotLightsData[o + 2] = e.position[2];
    spotLightsData[o + 4] = dir[0];
    spotLightsData[o + 5] = dir[1];
    spotLightsData[o + 6] = dir[2];
    spotLightsData[o + 7] = Math.cos(e.angle);
    spotLightsData[o + 8] = e.color[0] * e.intensity;
    spotLightsData[o + 9] = e.color[1] * e.intensity;
    spotLightsData[o + 10] = e.color[2] * e.intensity;
    spotLightCount += 1;
  }

  const rectAreaLightsData = new Float32Array(MAX_RECT_AREA_LIGHTS * RECT_AREA_LIGHT_FLOAT_STRIDE).fill(0);
  let rectAreaLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'rect-area' && e.kind !== 'disc-area') continue;

    let position: readonly [number, number, number];
    let uAxis: readonly [number, number, number];
    let vAxis: readonly [number, number, number];
    let rgb: readonly [number, number, number];

    if (e.kind === 'rect-area') {
      position = [e.position[0], e.position[1], e.position[2]];
      uAxis = [e.uAxis[0], e.uAxis[1], e.uAxis[2]];
      vAxis = [e.vAxis[0], e.vAxis[1], e.vAxis[2]];
      rgb = [
        e.color[0] * e.intensity,
        e.color[1] * e.intensity,
        e.color[2] * e.intensity,
      ];
    } else {
      if (Number.isFinite(e.radius) && e.radius < 1e-8) {
        warnings.push(
          `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has near-zero radius; skipped.`,
        );
        continue;
      }
      const d = discAreaPackedAsRect(e);
      position = d.position;
      uAxis = d.uAxis;
      vAxis = d.vAxis;
      rgb = d.radiance;
    }

    if (rectAreaLightCount >= MAX_RECT_AREA_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: rect-/disc-area lights capped at ${MAX_RECT_AREA_LIGHTS}; emitter "${e.id}" and subsequent area emitters omitted.`,
      );
      break;
    }

    const o = rectAreaLightCount * RECT_AREA_LIGHT_FLOAT_STRIDE;
    rectAreaLightsData[o + 0] = position[0];
    rectAreaLightsData[o + 1] = position[1];
    rectAreaLightsData[o + 2] = position[2];
    rectAreaLightsData[o + 4] = uAxis[0];
    rectAreaLightsData[o + 5] = uAxis[1];
    rectAreaLightsData[o + 6] = uAxis[2];
    rectAreaLightsData[o + 8] = vAxis[0];
    rectAreaLightsData[o + 9] = vAxis[1];
    rectAreaLightsData[o + 10] = vAxis[2];
    rectAreaLightsData[o + 12] = rgb[0];
    rectAreaLightsData[o + 13] = rgb[1];
    rectAreaLightsData[o + 14] = rgb[2];
    rectAreaLightCount += 1;
  }

  const meshAreaLightsData = new Float32Array(MAX_MESH_AREA_LIGHTS * MESH_AREA_LIGHT_FLOAT_STRIDE).fill(0);
  let meshAreaLightCount = 0;
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    if (meshAreaLightCount >= MAX_MESH_AREA_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: mesh-area lights capped at ${MAX_MESH_AREA_LIGHTS}; emitter "${emitter.id}" and subsequent omitted.`,
      );
      break;
    }
    const packedOne = firstMeshAreaLight({ ...scene, emitters: [emitter] });
    if (!packedOne.hasMeshAreaLight) {
      warnings.push(...packedOne.warnings);
      continue;
    }
    const o = meshAreaLightCount * MESH_AREA_LIGHT_FLOAT_STRIDE;
    meshAreaLightsData[o + 0] = packedOne.triA[0];
    meshAreaLightsData[o + 1] = packedOne.triA[1];
    meshAreaLightsData[o + 2] = packedOne.triA[2];
    meshAreaLightsData[o + 4] = packedOne.triB[0];
    meshAreaLightsData[o + 5] = packedOne.triB[1];
    meshAreaLightsData[o + 6] = packedOne.triB[2];
    meshAreaLightsData[o + 8] = packedOne.triC[0];
    meshAreaLightsData[o + 9] = packedOne.triC[1];
    meshAreaLightsData[o + 10] = packedOne.triC[2];
    meshAreaLightsData[o + 12] = packedOne.radiance[0];
    meshAreaLightsData[o + 13] = packedOne.radiance[1];
    meshAreaLightsData[o + 14] = packedOne.radiance[2];
    meshAreaLightCount += 1;
  }

  return {
    warnings,
    pointLightCount,
    spotLightCount,
    rectAreaLightCount,
    meshAreaLightCount,
    pointLightsData,
    spotLightsData,
    rectAreaLightsData,
    meshAreaLightsData,
  };
}

export function buildPackedScene(scene: Scene): PackedSceneData {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const triMaterialIds: number[] = [];
  const materials: number[] = [];
  const analyticHeaders: number[] = [];
  const analyticParams: number[] = [];
  const analyticLocalToWorld: number[] = [];
  const analyticWorldToLocal: number[] = [];
  const warnings: string[] = [];
  for (const emitter of scene.emitters) {
    const k = emitter.kind;
    const supported =
      k === 'directional' ||
      k === 'point' ||
      k === 'spot' ||
      k === 'rect-area' ||
      k === 'disc-area' ||
      k === 'mesh-area';
    if (!supported) {
      warnings.push(
        `Emitter "${(emitter as SceneEmitter).id}" (${String(k)}) ignored; prototype supports directional, point, spot, rect-area, disc-area (packed as rect), and mesh-area emitters only.`,
      );
    }
  }

  let nextVertexOffset = 0;
  let nextMaterialId = 0;

  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic') {
      const shapeId = analyticShapeId(primitive.shape);
      if (shapeId === 0) {
        warnings.push(`Analytic primitive "${primitive.id}" has unsupported shape "${primitive.shape}".`);
        continue;
      }
      const matId = nextMaterialId++;
      materials.push(...materialToPackedVec4s(primitive.material));
      const transform = primitive.transform ?? IDENTITY_MAT4;
      const invTransform = invertMat4(transform) ?? IDENTITY_MAT4;
      const paramsOffset = Math.floor(analyticParams.length / 4);
      const p = primitive.params;
      analyticParams.push(
        p[0] ?? 0,
        p[1] ?? 0,
        p[2] ?? 0,
        p[3] ?? 0,
        p[4] ?? 0,
        p[5] ?? 0,
        p[6] ?? 0,
        p[7] ?? 0,
      );
      analyticHeaders.push(shapeId, matId, paramsOffset, 0);
      analyticLocalToWorld.push(...transform);
      analyticWorldToLocal.push(...invTransform);
      continue;
    }

    const matId = nextMaterialId++;
    materials.push(...materialToPackedVec4s(primitive.material));

    const basePositions = primitive.positions;
    const baseIndices =
      primitive.indices ??
      (() => {
        const generated = new Uint32Array(basePositions.length / 3);
        for (let i = 0; i < generated.length; i += 1) generated[i] = i;
        return generated;
      })();

    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];

    for (const transform of transforms) {
      const vertexCount = Math.floor(basePositions.length / 3);
      for (let i = 0; i < vertexCount; i += 1) {
        const p: [number, number, number] = [
          basePositions[i * 3] ?? 0,
          basePositions[i * 3 + 1] ?? 0,
          basePositions[i * 3 + 2] ?? 0,
        ];
        const n: [number, number, number] = [
          primitive.normals[i * 3] ?? 0,
          primitive.normals[i * 3 + 1] ?? 1,
          primitive.normals[i * 3 + 2] ?? 0,
        ];
        const tp = transform == null ? p : transformPoint(transform, p);
        const tn = transform == null ? n : transformDirection(transform, n);
        positions.push(tp[0], tp[1], tp[2], 0);
        normals.push(tn[0], tn[1], tn[2], 0);
      }

      const triCount = Math.floor(baseIndices.length / 3);
      for (let t = 0; t < triCount; t += 1) {
        const i0 = (baseIndices[t * 3] ?? 0) + nextVertexOffset;
        const i1 = (baseIndices[t * 3 + 1] ?? 0) + nextVertexOffset;
        const i2 = (baseIndices[t * 3 + 2] ?? 0) + nextVertexOffset;
        indices.push(i0, i1, i2, 0);
        triMaterialIds.push(matId);
      }

      nextVertexOffset += vertexCount;
    }
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedIndices = new Uint32Array(indices);
  const packedTriMaterialIds = new Uint32Array(triMaterialIds);
  const bvhBuild = buildCpuBvh(packedPositions, packedIndices, packedTriMaterialIds);
  const point = firstPointLight(scene);
  const spot = firstSpotLight(scene);
  const rectArea = firstRectAreaLight(scene);
  const meshArea = firstMeshAreaLight(scene);
  const emitArrays = packEmitterArrays(scene);
  const environment = environmentParams(scene);
  warnings.push(...environment.warnings);
  warnings.push(...meshArea.warnings);
  warnings.push(...emitArrays.warnings);

  return {
    positions: packedPositions,
    normals: packedNormals,
    indices: bvhBuild.reorderedIndices,
    triMaterialIds: bvhBuild.reorderedTriMaterialIds,
    materials: new Float32Array(materials),
    bvhNodes: bvhBuild.bvhNodes,
    analyticHeaders: new Float32Array(analyticHeaders),
    analyticParams: new Float32Array(analyticParams),
    analyticLocalToWorld: new Float32Array(analyticLocalToWorld),
    analyticWorldToLocal: new Float32Array(analyticWorldToLocal),
    triangleCount: bvhBuild.reorderedTriMaterialIds.length,
    analyticCount: Math.floor(analyticHeaders.length / 4),
    warnings,
    directionalLight: defaultDirectionalLight(scene),
    directionalIrradiance: defaultDirectionalIrradiance(scene),
    pointLightPosition: point.position,
    pointLightRadiance: point.radiance,
    hasPointLight: point.hasPointLight,
    spotLightPosition: spot.position,
    spotLightDirection: spot.direction,
    spotLightCosAngle: spot.cosAngle,
    spotLightRadiance: spot.radiance,
    hasSpotLight: spot.hasSpotLight,
    rectAreaPosition: rectArea.position,
    rectAreaUAxis: rectArea.uAxis,
    rectAreaVAxis: rectArea.vAxis,
    rectAreaRadiance: rectArea.radiance,
    hasRectAreaLight: rectArea.hasRectAreaLight,
    meshAreaTriA: meshArea.triA,
    meshAreaTriB: meshArea.triB,
    meshAreaTriC: meshArea.triC,
    meshAreaRadiance: meshArea.radiance,
    hasMeshAreaLight: meshArea.hasMeshAreaLight,
    pointLightCount: emitArrays.pointLightCount,
    spotLightCount: emitArrays.spotLightCount,
    rectAreaLightCount: emitArrays.rectAreaLightCount,
    meshAreaLightCount: emitArrays.meshAreaLightCount,
    pointLightsData: emitArrays.pointLightsData,
    spotLightsData: emitArrays.spotLightsData,
    rectAreaLightsData: emitArrays.rectAreaLightsData,
    meshAreaLightsData: emitArrays.meshAreaLightsData,
    environmentTint: environment.tint,
    environmentSunDirection: environment.sunDirection,
    environmentSunStrength: environment.sunStrength,
    environmentMapWidth: environment.hdriWidth,
    environmentMapHeight: environment.hdriHeight,
    hasEnvironmentMap: environment.hasHdri,
    environmentMapTexels: environment.hdriTexels,
    environmentMapCdf: environment.hdriCdf,
  };
}

export function uploadPackedScene(device: GPUDevice, packed: PackedSceneData): UploadedSceneBuffers {
  const positionsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.positions', packed.positions);
  const normalsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.normals', packed.normals);
  const indicesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.indices', packed.indices);
  const triMaterialIdsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.triMaterialIds', packed.triMaterialIds);
  const materialsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.materials', packed.materials);
  const bvhNodesBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.bvhNodes', packed.bvhNodes);
  const analyticHeadersBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticHeaders', packed.analyticHeaders);
  const analyticParamsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticParams', packed.analyticParams);
  const analyticLocalToWorldBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticLocalToWorld', packed.analyticLocalToWorld);
  const analyticWorldToLocalBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.analyticWorldToLocal', packed.analyticWorldToLocal);
  const environmentMapTexelsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.environmentMapTexels', packed.environmentMapTexels);
  const environmentMapCdfBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.environmentMapCdf', packed.environmentMapCdf);
  const pointLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.pointLights', packed.pointLightsData);
  const spotLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.spotLights', packed.spotLightsData);
  const rectAreaLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.rectAreaLights', packed.rectAreaLightsData);
  const meshAreaLightsBuffer = createStorageBuffer(device, 'vitrum.pt-webgpu.scene.meshAreaLights', packed.meshAreaLightsData);

  return {
    triangleCount: packed.triangleCount,
    bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
    materialCount: Math.floor(packed.materials.length / MATERIAL_FLOAT_STRIDE),
    analyticCount: packed.analyticCount,
    directionalLight: packed.directionalLight,
    directionalIrradiance: packed.directionalIrradiance,
    pointLightPosition: packed.pointLightPosition,
    pointLightRadiance: packed.pointLightRadiance,
    hasPointLight: packed.hasPointLight,
    spotLightPosition: packed.spotLightPosition,
    spotLightDirection: packed.spotLightDirection,
    spotLightCosAngle: packed.spotLightCosAngle,
    spotLightRadiance: packed.spotLightRadiance,
    hasSpotLight: packed.hasSpotLight,
    rectAreaPosition: packed.rectAreaPosition,
    rectAreaUAxis: packed.rectAreaUAxis,
    rectAreaVAxis: packed.rectAreaVAxis,
    rectAreaRadiance: packed.rectAreaRadiance,
    hasRectAreaLight: packed.hasRectAreaLight,
    meshAreaTriA: packed.meshAreaTriA,
    meshAreaTriB: packed.meshAreaTriB,
    meshAreaTriC: packed.meshAreaTriC,
    meshAreaRadiance: packed.meshAreaRadiance,
    hasMeshAreaLight: packed.hasMeshAreaLight,
    pointLightCount: packed.pointLightCount,
    spotLightCount: packed.spotLightCount,
    rectAreaLightCount: packed.rectAreaLightCount,
    meshAreaLightCount: packed.meshAreaLightCount,
    environmentTint: packed.environmentTint,
    environmentSunDirection: packed.environmentSunDirection,
    environmentSunStrength: packed.environmentSunStrength,
    environmentMapWidth: packed.environmentMapWidth,
    environmentMapHeight: packed.environmentMapHeight,
    hasEnvironmentMap: packed.hasEnvironmentMap,
    positionsBuffer,
    normalsBuffer,
    indicesBuffer,
    triMaterialIdsBuffer,
    materialsBuffer,
    bvhNodesBuffer,
    analyticHeadersBuffer,
    analyticParamsBuffer,
    analyticLocalToWorldBuffer,
    analyticWorldToLocalBuffer,
    environmentMapTexelsBuffer,
    environmentMapCdfBuffer,
    pointLightsBuffer,
    spotLightsBuffer,
    rectAreaLightsBuffer,
    meshAreaLightsBuffer,
    destroy: () => {
      positionsBuffer.destroy();
      normalsBuffer.destroy();
      indicesBuffer.destroy();
      triMaterialIdsBuffer.destroy();
      materialsBuffer.destroy();
      bvhNodesBuffer.destroy();
      analyticHeadersBuffer.destroy();
      analyticParamsBuffer.destroy();
      analyticLocalToWorldBuffer.destroy();
      analyticWorldToLocalBuffer.destroy();
      environmentMapTexelsBuffer.destroy();
      environmentMapCdfBuffer.destroy();
      pointLightsBuffer.destroy();
      spotLightsBuffer.destroy();
      rectAreaLightsBuffer.destroy();
      meshAreaLightsBuffer.destroy();
    },
  };
}
