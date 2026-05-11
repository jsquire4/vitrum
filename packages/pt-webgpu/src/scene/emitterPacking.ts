import type { DiscAreaEmitter, Scene } from '@vitrum/core';
import { transformPoint } from '../math/mat4.js';

export const MAX_POINT_LIGHTS = 16;
export const MAX_SPOT_LIGHTS = 8;
export const MAX_RECT_AREA_LIGHTS = 8;
export const MAX_MESH_AREA_LIGHTS = 8;

/** vec4 pairs per point light: position, radiance */
export const POINT_LIGHT_FLOAT_STRIDE = 8;
export const SPOT_LIGHT_FLOAT_STRIDE = 12;
export const RECT_AREA_LIGHT_FLOAT_STRIDE = 16;
export const MESH_AREA_LIGHT_FLOAT_STRIDE = 16;

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

export function defaultDirectionalLight(scene: Scene): readonly [number, number, number] {
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

export function defaultDirectionalIrradiance(scene: Scene): readonly [number, number, number] {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return [1, 1, 1];
  const scale = directional.intensity;
  return [
    directional.color[0] * scale,
    directional.color[1] * scale,
    directional.color[2] * scale,
  ];
}

export function firstPointLight(scene: Scene): {
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

export function firstSpotLight(scene: Scene): {
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

export function firstRectAreaLight(scene: Scene): {
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

export function firstMeshAreaLight(scene: Scene): {
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

export function packEmitterArrays(scene: Scene): {
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
