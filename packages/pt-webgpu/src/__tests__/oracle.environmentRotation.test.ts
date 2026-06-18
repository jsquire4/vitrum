/**
 * H55 proof oracle — pt-webgpu HDRI Y-rotation convention.
 *
 * Existing WGSL contract tests pin that rotateYNeg/rotateYPos are present in
 * the composed full/lite shaders. This file independently proves the behavior
 * those helpers must implement: world-space lookup is counter-rotated by
 * -rotationY, CDF-sampled map directions are rotated back by +rotationY, and
 * the two transforms are exact inverses that preserve direction length.
 */
import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from '../wgsl/pathTrace/connectCore.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

type Vec3 = readonly [number, number, number];

function rotateYNeg([x, y, z]: Vec3, rotY: number): Vec3 {
  const c = Math.cos(rotY);
  const s = Math.sin(rotY);
  return [c * x - s * z, y, s * x + c * z];
}

function rotateYPos([x, y, z]: Vec3, rotY: number): Vec3 {
  const c = Math.cos(rotY);
  const s = Math.sin(rotY);
  return [c * x + s * z, y, -s * x + c * z];
}

function length3([x, y, z]: Vec3): number {
  return Math.hypot(x, y, z);
}

function envU([x, , z]: Vec3): number {
  const raw = Math.atan2(z, x) / (2 * Math.PI) + 0.5;
  return raw - Math.floor(raw);
}

function expectVecClose(actual: Vec3, expected: Vec3, precision = 12): void {
  for (let i = 0; i < 3; i += 1) {
    expect(actual[i]!).toBeCloseTo(expected[i]!, precision);
  }
}

function extractFunctionBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}`);
  expect(start, `${name} exists`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', start);
  expect(open, `${name} body opens`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    const ch = src[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return src.slice(open + 1, i);
      }
    }
  }
  throw new Error(`${name} body did not close`);
}

describe('pt-webgpu HDRI rotation — independent Y-rotation oracle', () => {
  it('keeps zero rotation byte-semantics as identity', () => {
    const dirs: readonly Vec3[] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0.2, -0.4, 0.7],
    ];
    for (const dir of dirs) {
      expectVecClose(rotateYNeg(dir, 0), dir);
      expectVecClose(rotateYPos(dir, 0), dir);
    }
  });

  it('matches the documented +/-90 degree HDRI lookup convention', () => {
    const quarterTurn = Math.PI / 2;
    expectVecClose(rotateYNeg([1, 0, 0], quarterTurn), [0, 0, 1]);
    expectVecClose(rotateYPos([0, 0, 1], quarterTurn), [1, 0, 0]);

    // For an environment dome rotationY=+90deg, a world +X ray looks up the
    // unrotated map at +Z. Equirect U therefore moves from 0.5 (+X) to 0.75 (+Z).
    expect(envU([1, 0, 0])).toBeCloseTo(0.5, 12);
    expect(envU(rotateYNeg([1, 0, 0], quarterTurn))).toBeCloseTo(0.75, 12);
  });

  it('proves rotateYPos is rotateYNeg inverse and both preserve length', () => {
    const cases: ReadonlyArray<{ readonly dir: Vec3; readonly rotY: number }> = [
      { dir: [0.2, -0.4, 0.7], rotY: 0.37 },
      { dir: [-0.5, 0.25, 0.9], rotY: -1.2 },
      { dir: [0.01, 0.99, -0.3], rotY: Math.PI },
    ];

    for (const { dir, rotY } of cases) {
      const neg = rotateYNeg(dir, rotY);
      const roundTrip = rotateYPos(neg, rotY);
      expectVecClose(roundTrip, dir);
      expect(length3(neg)).toBeCloseTo(length3(dir), 12);
      expect(length3(rotateYPos(dir, rotY))).toBeCloseTo(length3(dir), 12);
    }
  });

  it('keeps production WGSL and both trace tiers wired to the oracle helpers', () => {
    const negBody = extractFunctionBody(PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL, 'rotateYNeg');
    const posBody = extractFunctionBody(PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL, 'rotateYPos');
    expect(negBody).toContain('c * dir.x - s * dir.z');
    expect(negBody).toContain('s * dir.x + c * dir.z');
    expect(posBody).toContain('c * dir.x + s * dir.z');
    expect(posBody).toContain('-s * dir.x + c * dir.z');

    for (const trace of [PT_WEBGPU_TRACE_WGSL, PT_WEBGPU_TRACE_LITE_WGSL]) {
      expect(trace).toContain('let lookupDir = rotateYNeg(dir, rotY);');
      expect(trace).toContain('rotateYPos(mapDir, rotY)');
    }
  });
});
