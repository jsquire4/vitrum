import { describe, expect, it } from 'vitest';

import {
  intersectOpticalTriangleWatertightF32,
  type OpticalV3,
} from '../../../shared-bvh/src/opticalWatertightTriangle.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';

function wgslFunction(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  expect(start, `missing WGSL function ${name}`).toBeGreaterThanOrEqual(0);
  const bodyStart = source.indexOf('{', start);
  expect(bodyStart, `missing body for WGSL function ${name}`).toBeGreaterThan(start);

  let depth = 0;
  for (let cursor = bodyStart; cursor < source.length; cursor += 1) {
    const char = source[cursor];
    if (char === '{') depth += 1;
    if (char === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, cursor + 1);
    }
  }
  throw new Error(`unterminated WGSL function ${name}`);
}

function wgslCallArguments(source: string, functionName: string): string[] {
  const calls: string[] = [];
  const needle = `${functionName}(`;
  let searchFrom = 0;
  while (searchFrom < source.length) {
    const start = source.indexOf(needle, searchFrom);
    if (start < 0) break;
    searchFrom = start + needle.length;

    // The definition is checked separately; only inspect production calls here.
    if (source.slice(0, start).trimEnd().endsWith('fn')) continue;

    const open = start + needle.length - 1;
    let depth = 0;
    for (let cursor = open; cursor < source.length; cursor += 1) {
      const char = source[cursor];
      if (char === '(') depth += 1;
      if (char === ')') {
        depth -= 1;
        if (depth === 0) {
          calls.push(source.slice(open + 1, cursor));
          searchFrom = cursor + 1;
          break;
        }
      }
    }
  }
  return calls;
}

function traceAnyCallArguments(source: string): string[] {
  return wgslCallArguments(source, 'traceAny');
}

function topLevelArgumentCount(argumentsSource: string): number {
  const normalized = argumentsSource.trim().replace(/,\s*$/, '');
  if (normalized === '') return 0;
  let nestedParentheses = 0;
  let commas = 0;
  for (const char of normalized) {
    if (char === '(') nestedParentheses += 1;
    if (char === ')') nestedParentheses -= 1;
    if (char === ',' && nestedParentheses === 0) commas += 1;
  }
  return commas + 1;
}

function nextUpF32(value: number): number {
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || rounded === Number.POSITIVE_INFINITY) {
    return rounded;
  }
  if (Object.is(rounded, -0)) return Number.MIN_VALUE;
  const bits = new Uint32Array(new Float32Array([rounded]).buffer);
  bits[0] = rounded >= 0 ? bits[0]! + 1 : bits[0]! - 1;
  return new Float32Array(bits.buffer)[0]!;
}

function planeHit(z: number) {
  const origin: OpticalV3 = [0, 0, 0];
  const direction: OpticalV3 = [0, 0, 1];
  const a: OpticalV3 = [-2, -2, z];
  const b: OpticalV3 = [2, -2, z];
  const c: OpticalV3 = [0, 2, z];
  return intersectOpticalTriangleWatertightF32(origin, direction, a, b, c);
}

describe('alpha-aware shadow traversal', () => {
  it('tests alpha with complete hit interpolation data in full-tier traceAny', () => {
    const traceAny = wgslFunction(PT_WEBGPU_TRACE_WGSL, 'traceAny');
    expect(traceAny).toContain('rng: ptr<function, PtRngState>');
    expect(traceAny).toMatch(
      /!alphaTestPassThrough\(\s*matId,\s*hit\.triIndex,\s*hit\.baryVW,\s*hit\.instanceIndex,\s*rng,\s*\)/,
    );
  });

  it('threads an RNG through every full, BDPT, and ReSTIR visibility call', () => {
    const fullVisibilityCalls = wgslCallArguments(
      PT_WEBGPU_TRACE_WGSL,
      'traceSurfaceVisibility',
    );
    expect(fullVisibilityCalls.length).toBeGreaterThan(0);
    for (const call of fullVisibilityCalls) {
      expect(topLevelArgumentCount(call), `traceSurfaceVisibility(${call})`).toBe(6);
      expect(call, 'full/BDPT visibility missing RNG').toMatch(/&?rng\s*,?\s*$/);
    }

    const sources = [
      ['ReSTIR producer', RESTIR_PT_PRODUCER_WGSL],
      ['ReSTIR temporal', RESTIR_PT_TEMPORAL_WGSL],
      ['ReSTIR spatial', RESTIR_PT_SPATIAL_WGSL],
    ] as const;

    for (const [label, source] of sources) {
      const calls = traceAnyCallArguments(source);
      expect(calls.length, `${label} has no traceAny coverage`).toBeGreaterThan(0);
      for (const call of calls) {
        expect(topLevelArgumentCount(call), `${label}: traceAny(${call})`).toBe(4);
        expect(call, `${label}: missing RNG`).toMatch(/&?rng\s*,?\s*$/);
      }
    }
  });

  it('keeps lite visibility on its three-argument opaque-only contract', () => {
    const traceAny = wgslFunction(PT_WEBGPU_TRACE_LITE_WGSL, 'traceAny');
    expect(traceAny).not.toContain('alphaTestPassThrough');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain(
      'fn alphaTestPassThrough(',
    );
    for (const call of traceAnyCallArguments(PT_WEBGPU_TRACE_LITE_WGSL)) {
      expect(topLevelArgumentCount(call)).toBe(3);
    }
  });

  it('retains an adjacent surface one f32 ULP after an alpha pass-through', () => {
    const firstZ = Math.fround(1);
    const secondZ = nextUpF32(firstZ);
    const first = planeHit(firstZ);
    const second = planeHit(secondZ);

    expect(first.hit).toBe(true);
    expect(second.hit).toBe(true);
    expect(first.t).toBe(firstZ);
    expect(second.t).toBe(secondZ);
    expect(second.t).toBeGreaterThan(first.t);

    // This is the exact exclusive-lower-bound walk used after the first
    // surface's alpha test passes through. An origin epsilon hop would skip
    // the one-ULP-adjacent second surface.
    const next = [first, second]
      .filter((hit) => hit.hit && hit.t > first.t)
      .sort((left, right) => left.t - right.t)[0];
    expect(next?.t).toBe(secondZ);

    const visibility = wgslFunction(PT_WEBGPU_TRACE_WGSL, 'traceSurfaceVisibility');
    expect(visibility).toContain('traceClosestRaw(ray, cursor, tMax)');
    expect(visibility).toContain('cursor = hit.dist;');
    expect(visibility).not.toMatch(/ray\.origin\s*=/);
  });
});
