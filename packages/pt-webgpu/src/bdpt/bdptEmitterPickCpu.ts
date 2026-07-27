/**
 * CPU mirror of GPU `bdptEmitterCount` / `bdptEmitterPower` / `bdptPickEmitterFlat`
 * for test oracles and `fillBdptLightPathCpu`.
 */

import { luminance as luminance709 } from '@vitrum/shared-samplers';

import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  discArea,
  meshTriangleArea,
  rectQuadArea,
  walkPositionalEmitters,
  type PositionalEmitter,
} from './flatEmitterWalk.js';

const PI = Math.PI;
const DIRECTIONAL_LIGHT_FLOAT_STRIDE = 8;

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
function bdptLightLuminance(rgb: readonly [number, number, number]): number {
  return Math.max(luminance709(rgb[0], rgb[1], rgb[2]), 0);
}

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
function bdptHasEnvironmentEmitter(sb: UploadedSceneBuffers): boolean {
  if (sb.hasEnvironmentMap && sb.environmentMapWidth > 0 && sb.environmentMapHeight > 0) {
    return true;
  }
  return sb.environmentSunStrength > 0;
}

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
function bdptEnvironmentPower(sb: UploadedSceneBuffers): number {
  if (sb.hasEnvironmentMap && sb.environmentMapWidth > 0 && sb.environmentMapHeight > 0) {
    const count = sb.environmentMapWidth * sb.environmentMapHeight;
    if (sb.environmentMapCdf.length >= count + 1) {
      return Math.max(sb.environmentMapCdf[count]!, 0);
    }
  }
  if (sb.environmentSunStrength > 0) {
    return sb.environmentSunStrength * (4 * PI);
  }
  return 0;
}

function exactPositiveEmitterMeasure(value: number, label: string): number {
  if (!Number.isFinite(value) || !(value > 0)) {
    throw new RangeError(`${label} must be finite and positive; received ${String(value)}`);
  }
  return value;
}

function directionalRecord(
  sb: UploadedSceneBuffers,
  index: number,
): {
  readonly dir: [number, number, number];
  readonly irradiance: [number, number, number];
  readonly rawAngularDiameter: number;
} | null {
  const base = index * DIRECTIONAL_LIGHT_FLOAT_STRIDE;
  if (sb.directionalLightsData.length < base + DIRECTIONAL_LIGHT_FLOAT_STRIDE) {
    return null;
  }
  const dir = normalize3([
    sb.directionalLightsData[base] ?? 0,
    sb.directionalLightsData[base + 1] ?? 1,
    sb.directionalLightsData[base + 2] ?? 0,
  ]);
  return {
    dir,
    rawAngularDiameter: sb.directionalLightsData[base + 3] ?? 0,
    irradiance: [
      sb.directionalLightsData[base + 4] ?? 0,
      sb.directionalLightsData[base + 5] ?? 0,
      sb.directionalLightsData[base + 6] ?? 0,
    ],
  };
}

export function distantDirectEmitterCount(sb: UploadedSceneBuffers): number {
  return sb.directionalLightCount + (bdptHasEnvironmentEmitter(sb) ? 1 : 0);
}

export function distantDirectEmitterPower(
  sb: UploadedSceneBuffers,
  localIndex: number,
): number {
  const diskArea = exactPositiveEmitterMeasure(
    PI * sb.sceneRadius * sb.sceneRadius,
    'BDPT distant-emitter launch area',
  );
  if (localIndex < sb.directionalLightCount) {
    const record = directionalRecord(sb, localIndex);
    return diskArea * (record ? bdptLightLuminance(record.irradiance) : 0);
  }
  if (localIndex === sb.directionalLightCount && bdptHasEnvironmentEmitter(sb)) {
    return diskArea * bdptEnvironmentPower(sb);
  }
  return 0;
}

export function distantDirectEmitterGlobalIndex(
  sb: UploadedSceneBuffers,
  localIndex: number,
): number {
  if (localIndex < sb.directionalLightCount) return localIndex;
  return sb.directionalLightCount + sb.pointLightCount + sb.spotLightCount +
    sb.rectAreaLightCount + sb.meshAreaLightCount;
}

export function pickDistantDirectEmitter(
  sb: UploadedSceneBuffers,
  u: number,
): { readonly globalIndex: number; readonly invPdf: number } | null {
  const count = distantDirectEmitterCount(sb);
  if (count === 0) return null;
  let totalPower = 0;
  for (let i = 0; i < count; i += 1) totalPower += distantDirectEmitterPower(sb, i);
  if (!(totalPower > 0) || !Number.isFinite(totalPower)) return null;
  const target = Math.min(Math.max(u, 0), 1 - Number.EPSILON) * totalPower;
  let cumulative = 0;
  for (let i = 0; i < count; i += 1) {
    const power = distantDirectEmitterPower(sb, i);
    cumulative += power;
    if (power > 0 && target < cumulative) {
      return {
        globalIndex: distantDirectEmitterGlobalIndex(sb, i),
        invPdf: totalPower / power,
      };
    }
  }
  return null;
}

export function bdptEmitterCount(sb: UploadedSceneBuffers): number {
  return sb.directionalLightCount + sb.pointLightCount + sb.spotLightCount +
    sb.rectAreaLightCount + sb.meshAreaLightCount +
    (bdptHasEnvironmentEmitter(sb) ? 1 : 0);
}

/** Per-emitter luminous power for one positional light (matches the GPU term). */
function positionalEmitterPower(e: PositionalEmitter): number {
  switch (e.kind) {
    case 'point':
      return (4 * PI) * bdptLightLuminance(e.radiance);
    case 'spot': {
      const solidAngle = exactPositiveEmitterMeasure(
        2 * PI * (1 - 0.5 * (e.cosOuter + e.cosInner)),
        'BDPT spot-emitter solid angle',
      );
      return solidAngle * bdptLightLuminance(e.radiance);
    }
    case 'rect': {
      const isDisc = Math.abs(e.shapeTag - 1) < 0.5;
      const area = exactPositiveEmitterMeasure(
        isDisc ? discArea(e.uAxis) : rectQuadArea(e.uAxis, e.vAxis),
        `BDPT ${isDisc ? 'disc' : 'rect'} emitter area`,
      );
      return PI * area * bdptLightLuminance(e.radiance);
    }
    case 'mesh': {
      const area = exactPositiveEmitterMeasure(
        meshTriangleArea(e.triA, e.triB, e.triC),
        'BDPT mesh emitter area',
      );
      return PI * area * bdptLightLuminance(e.radiance);
    }
  }
}

export function bdptEmitterPower(sb: UploadedSceneBuffers, flatIdx: number): number {
  if (flatIdx < 0 || flatIdx >= bdptEmitterCount(sb)) return 0;
  if (flatIdx < sb.directionalLightCount) {
    return distantDirectEmitterPower(sb, flatIdx);
  }
  let cur = sb.directionalLightCount;
  for (const emitter of walkPositionalEmitters(sb)) {
    if (cur === flatIdx) return positionalEmitterPower(emitter);
    cur += 1;
  }
  if (cur === flatIdx && bdptHasEnvironmentEmitter(sb)) {
    return distantDirectEmitterPower(sb, sb.directionalLightCount);
  }
  return 0;
}

/** Sampled-mode direct-light PMF. BDPT root selection is deliberately uniform. */
export function distantDirectSelectionPdf(
  sb: UploadedSceneBuffers,
  globalIndex: number,
): number {
  const count = distantDirectEmitterCount(sb);
  if (count === 0) return 0;
  let localIndex = sb.directionalLightCount;
  if (globalIndex < sb.directionalLightCount) {
    localIndex = globalIndex;
  } else if (
    !bdptHasEnvironmentEmitter(sb) ||
    globalIndex !== distantDirectEmitterGlobalIndex(sb, sb.directionalLightCount)
  ) {
    return 0;
  }
  let totalPower = 0;
  for (let i = 0; i < count; i += 1) {
    totalPower += distantDirectEmitterPower(sb, i);
  }
  return totalPower > 0
    ? distantDirectEmitterPower(sb, localIndex) / totalPower
    : 0;
}

/** Exact rejection threshold used by `u32 % count` without modulo bias. */
export function bdptEmitterRejectionThreshold(emitterCount: number): number {
  if (!Number.isInteger(emitterCount) || emitterCount < 1 || emitterCount > 0xffffffff) {
    throw new RangeError('emitterCount must be an integer in 1..4294967295');
  }
  return 0x1_0000_0000 % emitterCount;
}

/**
 * Mirror one iteration of the WGSL full-u32 uniform pick. `null` means the
 * word lies in the rejection prefix and the shader must draw another word.
 */
export function bdptPickEmitterFlat(
  word: number,
  emitterCount: number,
): number | null {
  const threshold = bdptEmitterRejectionThreshold(emitterCount);
  const u32 = Math.trunc(word) >>> 0;
  return u32 < threshold ? null : u32 % emitterCount;
}

export type BdptBounce0Sample = {
  readonly emitPos: [number, number, number];
  readonly emitNormal: [number, number, number];
  readonly emitRad: [number, number, number];
  readonly pdfJoint: number;
  readonly pdfHemi: number;
  readonly selectionPdf?: number;
  readonly positionPdf?: number;
  readonly directionPdf?: number;
  readonly sourceDirectionWeight?: number;
  /** Exact sampled-mode p1 density: distant-light PMF × direction density. */
  readonly neePdf?: number;
  readonly directionIsDelta?: boolean;
  readonly nonConnectableEndpoint?: boolean;
  readonly lvMatId?: number;
  readonly endpointData?: [number, number, number];
};

function concentricDisc(u0: number, u1: number): readonly [number, number] {
  const a = 2 * clamp01(u0) - 1;
  const b = 2 * clamp01(u1) - 1;
  if (a === 0 && b === 0) return [0, 0];
  if (Math.abs(a) >= Math.abs(b)) {
    const phi = (PI / 4) * (b / a);
    return [a * Math.cos(phi), a * Math.sin(phi)];
  }
  const phi = PI / 2 - (PI / 4) * (a / b);
  return [b * Math.cos(phi), b * Math.sin(phi)];
}

function buildOnb(axis: readonly [number, number, number]): {
  tangent: [number, number, number]; bitangent: [number, number, number];
} {
  const up: [number, number, number] = Math.abs(axis[1]) < 0.999
    ? [0, 1, 0]
    : [1, 0, 0];
  const tangent = normalize3(cross3(up, axis));
  return { tangent, bitangent: normalize3(cross3(axis, tangent)) };
}

export function bdptDirectionalConePdf(angularDiameter: number): number {
  if (!(angularDiameter > 0)) return 1;
  const sinQuarter = Math.sin(angularDiameter * 0.25);
  const solidAngle = 4 * PI * sinQuarter * sinQuarter;
  return solidAngle > 0 ? 1 / solidAngle : 1;
}

export function bdptDirectionalSourceDirectionWeight(angularDiameter: number): number {
  return angularDiameter > 0 ? bdptDirectionalConePdf(angularDiameter) : 1;
}

function sampleCone(
  axis: readonly [number, number, number],
  angularDiameter: number,
  u0: number,
  u1: number,
): [number, number, number] {
  const normal = normalize3(axis);
  if (!(angularDiameter > 0)) return normal;
  const oneMinusCosHalf = 2 * Math.sin(angularDiameter * 0.25) ** 2;
  const cosTheta = 1 - clamp01(u0) * oneMinusCosHalf;
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = 2 * PI * clamp01(u1);
  const { tangent, bitangent } = buildOnb(normal);
  return normalize3([
    sinTheta * Math.cos(phi) * tangent[0] + sinTheta * Math.sin(phi) * bitangent[0] + cosTheta * normal[0],
    sinTheta * Math.cos(phi) * tangent[1] + sinTheta * Math.sin(phi) * bitangent[1] + cosTheta * normal[1],
    sinTheta * Math.cos(phi) * tangent[2] + sinTheta * Math.sin(phi) * bitangent[2] + cosTheta * normal[2],
  ]);
}

function sampleInfiniteDisk(
  sb: UploadedSceneBuffers,
  towardSourceIn: readonly [number, number, number],
  u0: number,
  u1: number,
): {
  position: [number, number, number];
  towardSource: [number, number, number];
  travelDirection: [number, number, number];
  positionPdf: number;
} {
  const towardSource = normalize3(towardSourceIn);
  const radius = Math.max(sb.sceneRadius, 1e-3);
  const center: [number, number, number] = [
    sb.sceneCenter[0] + towardSource[0] * radius,
    sb.sceneCenter[1] + towardSource[1] * radius,
    sb.sceneCenter[2] + towardSource[2] * radius,
  ];
  const [dx, dy] = concentricDisc(u0, u1);
  const { tangent, bitangent } = buildOnb(towardSource);
  return {
    position: [
      center[0] + radius * (dx * tangent[0] + dy * bitangent[0]),
      center[1] + radius * (dx * tangent[1] + dy * bitangent[1]),
      center[2] + radius * (dx * tangent[2] + dy * bitangent[2]),
    ],
    towardSource,
    travelDirection: [-towardSource[0], -towardSource[1], -towardSource[2]],
    positionPdf: 1 / (PI * radius * radius),
  };
}

/** Deterministic all-family bounce-0 endpoint oracle. */
export function sampleBdptBounce0Cpu(
  sb: UploadedSceneBuffers,
  flat: number,
  u0: number,
  u1 = 1 - u0,
): BdptBounce0Sample | null {
  const emitterCount = bdptEmitterCount(sb);
  if (emitterCount === 0 || flat < 0 || flat >= emitterCount) return null;
  const globalFlat = flat;
  const discretePdf = 1 / emitterCount;
  const finishEndpoint = (
    emitPos: [number, number, number],
    emitAxis: [number, number, number],
    endpointData: [number, number, number],
    emitRad: [number, number, number],
    pdfPosition: number,
    lvMatId: number,
  ): BdptBounce0Sample => {
    const pdf = pdfPosition;
    if (!(pdf > 0) || !Number.isFinite(pdf)) {
      throw new RangeError('bounce-0 position density must be finite and positive');
    }
    return {
      emitPos,
      emitNormal: emitAxis,
      endpointData,
      emitRad: [emitRad[0] / pdf, emitRad[1] / pdf, emitRad[2] / pdf],
      pdfJoint: pdf,
      pdfHemi: 0,
      selectionPdf: discretePdf,
      positionPdf: pdf / discretePdf,
      directionPdf: 0,
      sourceDirectionWeight: 1,
      lvMatId,
    };
  };

  if (globalFlat < sb.directionalLightCount) {
    const record = directionalRecord(sb, globalFlat);
    if (record == null) return null;
    const castShadowDisabled = record.rawAngularDiameter < 0;
    const angularDiameter = Math.max(
      castShadowDisabled ? -1 - record.rawAngularDiameter : record.rawAngularDiameter,
      0,
    );
    const towardSource = sampleCone(record.dir, angularDiameter, u0, u1);
    const launch = sampleInfiniteDisk(sb, towardSource, 1 - u0, u1);
    const directionPdf = bdptDirectionalConePdf(angularDiameter);
    const sourceDirectionWeight = bdptDirectionalSourceDirectionWeight(angularDiameter);
    const pdfJoint = discretePdf * launch.positionPdf;
    return {
      emitPos: launch.position,
      emitNormal: launch.towardSource,
      endpointData: launch.travelDirection,
      emitRad: record.irradiance.map(
        (channel) => channel * sourceDirectionWeight / pdfJoint,
      ) as [number, number, number],
      pdfJoint,
      pdfHemi: directionPdf,
      selectionPdf: discretePdf,
      positionPdf: launch.positionPdf,
      directionPdf,
      sourceDirectionWeight,
      neePdf: distantDirectSelectionPdf(sb, globalFlat) * directionPdf,
      directionIsDelta: angularDiameter <= 0,
      nonConnectableEndpoint: true,
      lvMatId: -8,
    };
  }

  let cur = sb.directionalLightCount;
  for (const e of walkPositionalEmitters(sb)) {
    if (cur === globalFlat) {
      switch (e.kind) {
        case 'point':
          // Direction sampling belongs to the first extension edge, not the endpoint.
          return finishEndpoint(
            e.position, [0, 1, 0], [0, 0, 0], e.radiance, discretePdf, -4,
          );
        case 'spot': {
          const spotDir = normalize3(e.axis);
          return finishEndpoint(
            e.position, spotDir, [e.cosInner, 0, 0],
            e.radiance, discretePdf, -5,
          );
        }
        case 'rect': {
          const ru = e.uAxis;
          const rv = e.vAxis;
          const isDisc = Math.abs((e.shapeTag ?? 0) - 1.0) < 0.5;
          let emitPos: [number, number, number];
          let area: number;
          if (isDisc) {
            const a = u0 * 2 - 1;
            const bb = u1 * 2 - 1;
            let cr: number;
            let cphi: number;
            if (Math.abs(a) >= Math.abs(bb)) {
              cr = a;
              cphi = (PI / 4) * (bb / Math.max(Math.abs(a), 1e-9));
            } else {
              cr = bb;
              cphi = PI / 2 - (PI / 4) * (a / Math.max(Math.abs(bb), 1e-9));
            }
            emitPos = [
              e.position[0] + ru[0] * (cr * Math.cos(cphi)) + rv[0] * (cr * Math.sin(cphi)),
              e.position[1] + ru[1] * (cr * Math.cos(cphi)) + rv[1] * (cr * Math.sin(cphi)),
              e.position[2] + ru[2] * (cr * Math.cos(cphi)) + rv[2] * (cr * Math.sin(cphi)),
            ];
            area = exactPositiveEmitterMeasure(
              discArea(ru), 'BDPT sampled disc emitter area',
            );
          } else {
            const u = u0 * 2 - 1;
            const v = u1 * 2 - 1;
            emitPos = [
              e.position[0] + ru[0] * u + rv[0] * v,
              e.position[1] + ru[1] * u + rv[1] * v,
              e.position[2] + ru[2] * u + rv[2] * v,
            ];
            area = exactPositiveEmitterMeasure(
              rectQuadArea(ru, rv), 'BDPT sampled rect emitter area',
            );
          }
          const emitNormal = normalize3(cross3(ru, rv));
          return finishEndpoint(
            emitPos, emitNormal, emitNormal, e.radiance, discretePdf / area, -2,
          );
        }
        case 'mesh': {
          const a = e.triA;
          const b = e.triB;
          const c = e.triC;
          const r1 = u0;
          const r2 = u1;
          const su = Math.sqrt(r1);
          const uu = 1 - su;
          const vv = r2 * su;
          const ww = 1 - uu - vv;
          const emitPos: [number, number, number] = [
            a[0] * uu + b[0] * vv + c[0] * ww,
            a[1] * uu + b[1] * vv + c[1] * ww,
            a[2] * uu + b[2] * vv + c[2] * ww,
          ];
          const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
          const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
          const n = cross3([e1[0]!, e1[1]!, e1[2]!], [e2[0]!, e2[1]!, e2[2]!]);
          const len = Math.hypot(n[0], n[1], n[2]);
          if (len < 1e-8) {
            return null;
          }
          const emitNormal: [number, number, number] = [n[0] / len, n[1] / len, n[2] / len];
          const area = exactPositiveEmitterMeasure(
            meshTriangleArea(a, b, c), 'BDPT sampled mesh emitter area',
          );
          return finishEndpoint(
            emitPos, emitNormal, emitNormal, e.radiance, discretePdf / area, -2,
          );
        }
      }
    }
    cur += 1;
  }
  if (bdptHasEnvironmentEmitter(sb) && cur === globalFlat) {
    let towardSource: [number, number, number];
    let radiance: [number, number, number];
    let directionPdf: number;
    const count = sb.environmentMapWidth * sb.environmentMapHeight;
    if (sb.hasEnvironmentMap && count > 0 && sb.environmentMapCdf.length >= count + 1) {
      const xi = clamp01(u0);
      let lo = 0;
      let hi = count;
      while (lo + 1 < hi) {
        const mid = (lo + hi) >>> 1;
        if ((sb.environmentMapCdf[mid] ?? 0) <= xi) lo = mid;
        else hi = mid;
      }
      const idx = Math.min(lo, count - 1);
      const x = idx % sb.environmentMapWidth;
      const y = Math.floor(idx / sb.environmentMapWidth);
      const phi = ((x + 0.5) / sb.environmentMapWidth - 0.5) * 2 * PI;
      const theta = ((y + 0.5) / sb.environmentMapHeight) * PI;
      const sinTheta = Math.sin(theta);
      towardSource = normalize3([
        Math.cos(phi) * sinTheta,
        Math.cos(theta),
        Math.sin(phi) * sinTheta,
      ]);
      const texel = idx * 4;
      const intensity = sb.environmentHdriIntensity ?? 1;
      radiance = [
        (sb.environmentMapTexels[texel] ?? 0) * intensity,
        (sb.environmentMapTexels[texel + 1] ?? 0) * intensity,
        (sb.environmentMapTexels[texel + 2] ?? 0) * intensity,
      ];
      directionPdf = sb.environmentMapTexels[texel + 3] ?? 0;
    } else {
      const z = 1 - 2 * clamp01(u0);
      const phi = 2 * PI * clamp01(u1);
      const radial = Math.sqrt(Math.max(0, 1 - z * z));
      towardSource = [radial * Math.cos(phi), z, radial * Math.sin(phi)];
      radiance = [
        sb.environmentTint[0] * sb.environmentSunStrength,
        sb.environmentTint[1] * sb.environmentSunStrength,
        sb.environmentTint[2] * sb.environmentSunStrength,
      ];
      directionPdf = 1 / (4 * PI);
    }
    if (!(directionPdf > 0) || !Number.isFinite(directionPdf)) return null;
    const launch = sampleInfiniteDisk(sb, towardSource, 1 - u0, u1);
    const pdfJoint = discretePdf * launch.positionPdf;
    return {
      emitPos: launch.position,
      emitNormal: launch.towardSource,
      endpointData: launch.travelDirection,
      emitRad: radiance.map((channel) => channel / pdfJoint) as [number, number, number],
      pdfJoint,
      pdfHemi: directionPdf,
      selectionPdf: discretePdf,
      positionPdf: launch.positionPdf,
      directionPdf,
      sourceDirectionWeight: 1,
      neePdf: distantDirectSelectionPdf(sb, globalFlat) * directionPdf,
      directionIsDelta: false,
      nonConnectableEndpoint: true,
      lvMatId: -9,
    };
  }
  return null;
}

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1 - Number.EPSILON);
}

function cross3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v: readonly [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
