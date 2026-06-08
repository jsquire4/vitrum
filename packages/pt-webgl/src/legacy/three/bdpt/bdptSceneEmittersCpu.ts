/**
 * CPU bounce-0 light-subpath vertex from @vitrum/core Scene emitters.
 * Mirrors pt-webgpu `bdptEmitterPickCpu` / `sampleBdptBounce0Cpu` for WebGL hosts.
 */

import type { Scene, SceneEmitter, Vec3 } from '@vitrum/core';
import { luminance as luminance709 } from '@vitrum/shared-samplers';

const PI = Math.PI;
const KIND_LIGHT = 0;

export interface BdptBounce0Vertex {
  readonly emitPos: Vec3;
  readonly emitNormal: Vec3;
  readonly throughput: Vec3;
  readonly pdfJoint: number;
  readonly pdfHemi: number;
}

function luminance(rgb: readonly [number, number, number]): number {
  return Math.max(luminance709(rgb[0], rgb[1], rgb[2]), 1e-20);
}

function emitterRadiance(e: SceneEmitter): Vec3 {
  const s = Math.max(e.intensity, 0);
  return [e.color[0] * s, e.color[1] * s, e.color[2] * s];
}

function emitterPower(e: SceneEmitter): number {
  return luminance(emitterRadiance(e));
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function hash01(seed: number, stream: number): number {
  let x = (seed ^ Math.imul(stream, 0x9e3779b9)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 2 ** 32;
}

/**
 * Right-handed orthonormal tangent/bitangent for a (normalized) surface normal,
 * picking a stable up-reference to avoid the degeneracy when `n ≈ ±Y`. Single
 * source for the two TBN constructions in this file (the cosine-hemisphere frame
 * in {@link finishBounce0} and the disc-area in-plane frame). The reference pick
 * (`abs(n.y) < 0.999 ? +Y : +X`) and cross order MUST stay byte-identical to the
 * GPU/pt-webgpu mirror — the emitter sample feeds RNG-correlated path building.
 */
function buildTangentFrame(n: Vec3): { t: Vec3; b: Vec3 } {
  const up: Vec3 = Math.abs(n[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const t = normalize3(cross3(up, n));
  const b = cross3(n, t);
  return { t, b };
}

function finishBounce0(
  emitPos: Vec3,
  emitNormal: Vec3,
  emitRad: Vec3,
  discretePdf: number,
  uHemi: number,
): BdptBounce0Vertex {
  const n = normalize3(emitNormal);
  const u1 = Math.max(uHemi, 1e-7);
  const u2 = (uHemi * 1.618033988749895) % 1;
  const r = Math.sqrt(u1);
  const phi = 2 * PI * u2;
  const x = r * Math.cos(phi);
  const y = r * Math.sin(phi);
  const z = Math.sqrt(Math.max(0, 1 - u1));
  const { t, b } = buildTangentFrame(n);
  const wi = normalize3([
    t[0] * x + b[0] * y + n[0] * z,
    t[1] * x + b[1] * y + n[1] * z,
    t[2] * x + b[2] * y + n[2] * z,
  ]);
  const cosEmit = Math.max(dot3(n, wi), 0);
  const pdfHemi = cosEmit / PI;
  const pdfJoint = Math.max(discretePdf * pdfHemi, 1e-8);
  const throughput: Vec3 = [
    (emitRad[0] * cosEmit) / pdfJoint,
    (emitRad[1] * cosEmit) / pdfJoint,
    (emitRad[2] * cosEmit) / pdfJoint,
  ];
  return { emitPos, emitNormal: n, throughput, pdfJoint, pdfHemi };
}

function pickEmitterFlat(scene: Scene, uPick: number): { emitter: SceneEmitter; discretePdf: number } | null {
  const emitters = scene.emitters;
  if (emitters.length === 0) return null;
  let total = 0;
  for (const e of emitters) total += emitterPower(e);
  if (total <= 1e-20) return null;
  let acc = 0;
  const target = uPick * total;
  for (const e of emitters) {
    const p = emitterPower(e);
    acc += p;
    if (target <= acc) {
      return { emitter: e, discretePdf: p / total };
    }
  }
  const last = emitters[emitters.length - 1]!;
  return { emitter: last, discretePdf: emitterPower(last) / total };
}

export function sampleBdptBounce0FromScene(
  scene: Scene,
  frameSeed: number,
): BdptBounce0Vertex | null {
  const uPick = ((frameSeed * 2654435761) >>> 0) / 2 ** 32;
  const uHemi = (((frameSeed + 1) * 1597334677) >>> 0) / 2 ** 32;
  const uArea0 = hash01(frameSeed, 2);
  const uArea1 = hash01(frameSeed, 3);
  const picked = pickEmitterFlat(scene, uPick);
  if (picked == null) return null;
  const { emitter, discretePdf } = picked;
  const rad = emitterRadiance(emitter);

  switch (emitter.kind) {
    case 'directional': {
      const d = normalize3(emitter.direction);
      return finishBounce0(
        [-d[0] * 50, -d[1] * 50, -d[2] * 50],
        d,
        rad,
        discretePdf,
        uHemi,
      );
    }
    case 'point':
      return finishBounce0(emitter.position, [0, 1, 0], rad, discretePdf, uHemi);
    case 'spot': {
      const axis = normalize3(emitter.direction);
      return finishBounce0(emitter.position, axis, rad, discretePdf, uHemi);
    }
    case 'rect-area': {
      const u = uArea0 * 2 - 1;
      const v = uArea1 * 2 - 1;
      const emitPos: Vec3 = [
        emitter.position[0] + emitter.uAxis[0] * u + emitter.vAxis[0] * v,
        emitter.position[1] + emitter.uAxis[1] * u + emitter.vAxis[1] * v,
        emitter.position[2] + emitter.uAxis[2] * u + emitter.vAxis[2] * v,
      ];
      const emitNormal = normalize3(cross3(emitter.uAxis, emitter.vAxis));
      return finishBounce0(emitPos, emitNormal, rad, discretePdf, uHemi);
    }
    case 'disc-area': {
      const r = Math.sqrt(uArea0) * emitter.radius;
      const angle = 2 * PI * uArea1;
      const n = normalize3(emitter.normal);
      const { t, b } = buildTangentFrame(n);
      const emitPos: Vec3 = [
        emitter.position[0] + (t[0] * Math.cos(angle) + b[0] * Math.sin(angle)) * r,
        emitter.position[1] + (t[1] * Math.cos(angle) + b[1] * Math.sin(angle)) * r,
        emitter.position[2] + (t[2] * Math.cos(angle) + b[2] * Math.sin(angle)) * r,
      ];
      return finishBounce0(emitPos, n, rad, discretePdf, uHemi);
    }
    case 'mesh-area':
      return null;
    default:
      return null;
  }
}

export const BDPT_KIND_INVALID = 3;
export const BDPT_KIND_LIGHT = KIND_LIGHT;
