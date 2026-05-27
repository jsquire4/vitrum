/**
 * CPU bounce-0 light-subpath vertex from @vitrum/core Scene emitters.
 * Mirrors pt-webgpu `bdptEmitterPickCpu` / `sampleBdptBounce0Cpu` for WebGL hosts.
 */

import type { Scene, SceneEmitter, Vec3 } from '@vitrum/core';

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
  return Math.max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1e-20);
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
  const tx: Vec3 = Math.abs(n[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const t = normalize3(cross3(tx, n));
  const b = cross3(n, t);
  const wi = normalize3([
    t[0] * x + b[0] * x + n[0] * z,
    t[1] * x + b[1] * x + n[1] * z,
    t[2] * x + b[2] * x + n[2] * z,
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
      const u = uHemi * 2 - 1;
      const v = (1 - uHemi) * 2 - 1;
      const emitPos: Vec3 = [
        emitter.position[0] + emitter.uAxis[0] * u + emitter.vAxis[0] * v,
        emitter.position[1] + emitter.uAxis[1] * u + emitter.vAxis[1] * v,
        emitter.position[2] + emitter.uAxis[2] * u + emitter.vAxis[2] * v,
      ];
      const emitNormal = normalize3(cross3(emitter.uAxis, emitter.vAxis));
      return finishBounce0(emitPos, emitNormal, rad, discretePdf, uHemi);
    }
    case 'disc-area': {
      const u = uHemi * 2 - 1;
      const v = (1 - uHemi) * 2 - 1;
      const r = Math.sqrt(u * u + v * v) * emitter.radius;
      const angle = Math.atan2(v, u);
      const n = normalize3(emitter.normal);
      const tx: Vec3 = Math.abs(n[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
      const t = normalize3(cross3(tx, n));
      const b = cross3(n, t);
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
