import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  classifyAreaVectorF32,
  normalizeDirectionF32,
} from '../areaEmitterGeometry.js';
import {
  collectRectAreaEmitterTrisFromCore,
  packEmitterTrisForDDGI,
} from '../emitterHelpers.js';
import { EMITTER_SAMPLING_WGSL } from '../../shaders/emitterSampling.wgsl.js';

function rectScene(
  uAxis: [number, number, number],
  vAxis: [number, number, number],
): Scene {
  return {
    primitives: [],
    emitters: [{
      kind: 'rect-area',
      id: 'scaled-rect',
      position: [0, 0, 0],
      uAxis,
      vAxis,
      color: [1, 1, 1],
      intensity: 1,
    }],
    environment: { kind: 'none' },
  };
}

describe('walkaround-hybrid area-emitter Float32 scale contract', () => {
  it('retains tiny finite analytic triangles and publishes their recovered area', () => {
    const triangles = collectRectAreaEmitterTrisFromCore(
      rectScene([1e-18, 0, 0], [0, 1e-18, 0]),
    );
    expect(triangles).toHaveLength(2);
    expect(triangles[0]!.area / 2e-36).toBeCloseTo(1, 5);
    expect(triangles[1]!.area / 2e-36).toBeCloseTo(1, 5);

    const packed = packEmitterTrisForDDGI(triangles);
    expect(packed.count).toBe(2);
    expect(packed.data[15]! / 2e-36).toBeCloseTo(1, 5);
    expect(packed.data[35]! / 2e-36).toBeCloseTo(1, 5);
  });

  it('retains huge near-parallel finite rects that raw f32 cross arithmetic loses', () => {
    const base = Math.fround(1e20);
    const next = Math.fround(base * (1 + 1e-6));
    const measured = classifyAreaVectorF32([base, base, 0], [base, next, 0], 4);
    expect(measured.valid).toBe(true);
    if (!measured.valid) return;
    expect(measured.area).toBeGreaterThan(1e33);
    expect(measured.area).toBeLessThan(1e36);
    expect(measured.normal).toEqual([0, 0, 1]);
  });

  it('rejects exact or unrepresentable geometry and normalizes scale-free directions', () => {
    expect(classifyAreaVectorF32([1, 2, 3], [2, 4, 6], 4)).toEqual({
      valid: false,
      reason: 'degenerate',
    });
    expect(classifyAreaVectorF32([1e-20, 0, 0], [0, 1e-20, 0], 4)).toEqual({
      valid: false,
      reason: 'unrepresentable-area',
    });
    expect(normalizeDirectionF32([0, 1e-300, 0])).toEqual([0, 1, 0]);
    expect(() => collectRectAreaEmitterTrisFromCore(
      rectScene([1e20, 0, 0], [0, 1e20, 0]),
    )).toThrow(/unrepresentable-area/);
  });

  it('re-derives the packed Jacobian from published triangle geometry', () => {
    const packed = packEmitterTrisForDDGI([{
      vA: [0, 0, 0],
      vB: [1e-18, 0, 0],
      vC: [0, 1e-18, 0],
      normal: [0, 0, 1e300],
      area: 123,
      Le: [1, 1, 1],
    }]);
    expect(packed.data[12]).toBe(0);
    expect(packed.data[13]).toBe(0);
    expect(packed.data[14]).toBe(1);
    expect(packed.data[15]! / 5e-37).toBeCloseTo(1, 5);
  });

  it('publishes raw DDGI emitter radiance through the non-negative f32 envelope', () => {
    const base = {
      vA: [0, 0, 0] as [number, number, number],
      vB: [1, 0, 0] as [number, number, number],
      vC: [0, 1, 0] as [number, number, number],
      normal: [0, 0, 1] as [number, number, number],
      area: 0.5,
    };
    const packed = packEmitterTrisForDDGI([{
      ...base,
      Le: [0.1, 0.2, 0.3],
    }]);
    expect(Array.from(packed.data.slice(16, 19))).toEqual([
      Math.fround(0.1),
      Math.fround(0.2),
      Math.fround(0.3),
    ]);
    expect(() => packEmitterTrisForDDGI([{
      ...base,
      Le: [-1, 0, 0],
    }])).toThrow(/finite and non-negative/);
    expect(() => packEmitterTrisForDDGI([{
      ...base,
      Le: [Number.POSITIVE_INFINITY, 0, 0],
    }])).toThrow(/finite and non-negative/);
    expect(() => packEmitterTrisForDDGI([{
      ...base,
      Le: [Number.MIN_VALUE, 0, 0],
    }])).toThrow(/collapse completely to zero/);
  });

  it('fails the shader-side reciprocal Jacobian closed', () => {
    expect(EMITTER_SAMPLING_WGSL).toContain('inverseArea <= 3.402823e38');
    expect(EMITTER_SAMPLING_WGSL).toContain('result.pdfArea = 0.0');
  });
});
