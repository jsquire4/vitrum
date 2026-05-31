/**
 * CPU mirror of GPU `bdptEmitterCount` / `bdptEmitterPower` / `bdptPickEmitterFlat`
 * for test oracles and `fillBdptLightPathCpu`.
 */

import { luminance as luminance709 } from '@vitrum/shared-samplers';

import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  meshTriangleArea,
  rectQuadArea,
  walkPositionalEmitters,
  type PositionalEmitter,
} from './flatEmitterWalk.js';

const PI = Math.PI;

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
export function bdptLightLuminance(rgb: readonly [number, number, number]): number {
  return Math.max(luminance709(rgb[0], rgb[1], rgb[2]), 1e-20);
}

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
export function bdptHasEnvironmentEmitter(sb: UploadedSceneBuffers): boolean {
  if (sb.hasEnvironmentMap && sb.environmentMapWidth > 0 && sb.environmentMapHeight > 0) {
    return true;
  }
  const irr = sb.directionalIrradiance;
  return irr[0] + irr[1] + irr[2] > 1e-6 || sb.environmentSunStrength > 1e-6;
}

/** @internal Test-oracle CPU mirror of the GPU emitter-pick math; not public API. */
export function bdptEnvironmentPower(sb: UploadedSceneBuffers): number {
  if (sb.hasEnvironmentMap && sb.environmentMapWidth > 0 && sb.environmentMapHeight > 0) {
    const count = sb.environmentMapWidth * sb.environmentMapHeight;
    if (sb.environmentMapCdf.length >= count + 1) {
      return Math.max(sb.environmentMapCdf[count]!, 1e-20);
    }
  }
  const sunW = sb.environmentSunStrength > 1e-6
    ? sb.environmentSunStrength
    : sb.directionalIrradiance[0] + sb.directionalIrradiance[1] + sb.directionalIrradiance[2];
  if (sunW > 1e-6) {
    return Math.max(sunW, 1e-20) * (4 * PI);
  }
  return 1e-20;
}

export function bdptEmitterCount(sb: UploadedSceneBuffers): number {
  let n = 0;
  const irr = sb.directionalIrradiance;
  if (irr[0] + irr[1] + irr[2] > 1e-6) {
    n += 1;
  }
  n += sb.pointLightCount;
  n += sb.spotLightCount;
  n += sb.rectAreaLightCount;
  n += sb.meshAreaLightCount;
  if (bdptHasEnvironmentEmitter(sb)) {
    n += 1;
  }
  return n;
}

/** Per-emitter luminous power for one positional light (matches the GPU term). */
function positionalEmitterPower(e: PositionalEmitter): number {
  switch (e.kind) {
    case 'point':
    case 'spot':
      return bdptLightLuminance(e.radiance);
    case 'rect': {
      const area = Math.max(rectQuadArea(e.uAxis, e.vAxis), 1e-6);
      return area * bdptLightLuminance(e.radiance);
    }
    case 'mesh': {
      const area = Math.max(meshTriangleArea(e.triA, e.triB, e.triC), 1e-6);
      return area * bdptLightLuminance(e.radiance);
    }
  }
}

export function bdptEmitterPower(sb: UploadedSceneBuffers, flatIdx: number): number {
  let cur = 0;
  const irr = sb.directionalIrradiance;
  if (irr[0] + irr[1] + irr[2] > 1e-6) {
    if (cur === flatIdx) {
      return bdptLightLuminance([irr[0]!, irr[1]!, irr[2]!]);
    }
    cur += 1;
  }
  for (const e of walkPositionalEmitters(sb)) {
    if (cur === flatIdx) {
      return positionalEmitterPower(e);
    }
    cur += 1;
  }
  if (bdptHasEnvironmentEmitter(sb) && cur === flatIdx) {
    return bdptEnvironmentPower(sb);
  }
  return 1e-20;
}

/** Returns flat emitter index (0..count-1) for u in [0, totalPower). */
export function bdptPickEmitterFlat(
  sb: UploadedSceneBuffers,
  u: number,
  totalPower: number,
  emitterCount: number,
): number {
  if (emitterCount === 0) {
    return 0;
  }
  let cum = 0;
  for (let i = 0; i < emitterCount; i += 1) {
    cum += bdptEmitterPower(sb, i);
    if (u <= cum) {
      return i;
    }
  }
  return emitterCount - 1;
}

export type BdptBounce0Sample = {
  readonly emitPos: [number, number, number];
  readonly emitNormal: [number, number, number];
  readonly emitRad: [number, number, number];
  readonly pdfJoint: number;
  readonly pdfHemi: number;
};

/** Deterministic bounce-0 vertex (cosine hemisphere uses uHemi in [0,1)). */
export function sampleBdptBounce0Cpu(
  sb: UploadedSceneBuffers,
  flat: number,
  discretePdf: number,
  uHemi: number,
): BdptBounce0Sample | null {
  const finish = (
    emitPos: [number, number, number],
    emitNormal: [number, number, number],
    emitRad: [number, number, number],
    pdfLight: number,
  ): BdptBounce0Sample => {
    const n = emitNormal;
    const u1 = Math.max(uHemi, 1e-8);
    const u2 = 1 - u1 * 0.5;
    const r = Math.sqrt(u1);
    const phi = 2 * PI * u2;
    const x = r * Math.cos(phi);
    const y = r * Math.sin(phi);
    const z = Math.sqrt(Math.max(0, 1 - u1));
    const tx: [number, number, number] = Math.abs(n[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
    const t = normalize3(cross3(tx, n));
    const b = cross3(n, t);
    const wi = normalize3([
      t[0]! * x + b[0]! * y + n[0]! * z,
      t[1]! * x + b[1]! * y + n[1]! * z,
      t[2]! * x + b[2]! * y + n[2]! * z,
    ]);
    const cosEmit = Math.max(dot3(n, wi), 0);
    const pdfHemi = cosEmit / PI;
    const pdfJoint = Math.max(pdfLight * pdfHemi, 1e-8);
    const throughput: [number, number, number] = [
      (emitRad[0] * cosEmit) / pdfJoint,
      (emitRad[1] * cosEmit) / pdfJoint,
      (emitRad[2] * cosEmit) / pdfJoint,
    ];
    return { emitPos, emitNormal: n, emitRad: throughput, pdfJoint, pdfHemi };
  };

  let cur = 0;
  const irr = sb.directionalIrradiance;
  if (irr[0] + irr[1] + irr[2] > 1e-6) {
    if (cur === flat) {
      const len = Math.hypot(irr[0], irr[1], irr[2]) || 1;
      const lightDir: [number, number, number] = [irr[0] / len, irr[1] / len, irr[2] / len];
      return finish(
        [-lightDir[0] * 50, -lightDir[1] * 50, -lightDir[2] * 50],
        lightDir,
        [irr[0]!, irr[1]!, irr[2]!],
        discretePdf,
      );
    }
    cur += 1;
  }
  for (const e of walkPositionalEmitters(sb)) {
    if (cur === flat) {
      switch (e.kind) {
        case 'point':
          return finish(e.position, [0, 1, 0], e.radiance, discretePdf);
        case 'spot': {
          const spotDir = normalize3(e.axis);
          return finish(e.position, spotDir, e.radiance, discretePdf);
        }
        case 'rect': {
          const ru = e.uAxis;
          const rv = e.vAxis;
          const u = uHemi * 2 - 1;
          const v = (1 - uHemi) * 2 - 1;
          const emitPos: [number, number, number] = [
            e.position[0] + ru[0] * u + rv[0] * v,
            e.position[1] + ru[1] * u + rv[1] * v,
            e.position[2] + ru[2] * u + rv[2] * v,
          ];
          const emitNormal = normalize3(cross3(ru, rv));
          return finish(emitPos, emitNormal, e.radiance, discretePdf);
        }
        case 'mesh': {
          const a = e.triA;
          const b = e.triB;
          const c = e.triC;
          const r1 = uHemi;
          const r2 = 1 - uHemi * 0.5;
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
          return finish(emitPos, emitNormal, e.radiance, discretePdf);
        }
      }
    }
    cur += 1;
  }
  if (bdptHasEnvironmentEmitter(sb) && cur === flat) {
    if (sb.hasEnvironmentMap && sb.environmentMapWidth > 0 && sb.environmentMapHeight > 0) {
      const count = sb.environmentMapWidth * sb.environmentMapHeight;
      if (sb.environmentMapCdf.length >= count + 1 && sb.environmentMapTexels.length >= count * 4) {
        const xi = uHemi * sb.environmentMapCdf[count]!;
        let lo = 0;
        let hi = count;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if (sb.environmentMapCdf[mid]! <= xi) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        const idx = Math.min(lo, count - 1);
        const x = idx % sb.environmentMapWidth;
        const y = Math.floor(idx / sb.environmentMapWidth);
        const u = (x + 0.5) / sb.environmentMapWidth;
        const v = (y + 0.5) / sb.environmentMapHeight;
        const phi = (u - 0.5) * (2 * PI);
        const theta = v * PI;
        const sinTheta = Math.sin(theta);
        const dir = normalize3([
          Math.cos(phi) * sinTheta,
          Math.cos(theta),
          Math.sin(phi) * sinTheta,
        ]);
        const t = idx * 4;
        const texRgb: [number, number, number] = [
          sb.environmentMapTexels[t]!,
          sb.environmentMapTexels[t + 1]!,
          sb.environmentMapTexels[t + 2]!,
        ];
        const sunW = Math.max(sb.environmentSunStrength, 1);
        const value: [number, number, number] = [
          texRgb[0] * sunW,
          texRgb[1] * sunW,
          texRgb[2] * sunW,
        ];
        const pdf = Math.max(sb.environmentMapTexels[t + 3]!, 1e-8);
        const pdfLight = discretePdf * pdf;
        return finish([-dir[0] * 50, -dir[1] * 50, -dir[2] * 50], dir, value, pdfLight);
      }
    }
    if (sb.environmentSunStrength > 1e-6 || irr[0] + irr[1] + irr[2] > 1e-6) {
      const sd = sb.environmentSunDirection;
      const sunDir =
        Math.hypot(sd[0], sd[1], sd[2]) > 1e-6
          ? normalize3([sd[0], sd[1], sd[2]])
          : normalize3([
              sb.directionalLight[0],
              sb.directionalLight[1],
              sb.directionalLight[2],
            ]);
      const w = sb.environmentSunStrength > 1e-6 ? sb.environmentSunStrength : irr[0] + irr[1] + irr[2];
      return finish(
        [-sunDir[0] * 50, -sunDir[1] * 50, -sunDir[2] * 50],
        sunDir,
        [w, w, w],
        discretePdf,
      );
    }
    return null;
  }
  return null;
}

function dot3(a: readonly number[], b: readonly number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

function cross3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): [number, number, number] {
  return [
    a[1]! * b[2]! - a[2]! * b[1]!,
    a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!,
  ];
}

function normalize3(v: readonly [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
