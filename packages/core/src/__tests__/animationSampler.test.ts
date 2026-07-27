/**
 * animationSampler.test.ts — P3 CPU clip sampler (sampleAnimationClip).
 */
import { describe, it, expect } from 'vitest';
import {
  sampleAnimationClip,
  validateAnimationClip,
  type AnimationClip,
} from '../scene/animation.js';

function clip(channels: AnimationClip['channels']): AnimationClip {
  return { duration: 2, channels };
}

describe('sampleAnimationClip (P3)', () => {
  it('LINEAR translation lerps between keyframes', () => {
    const c = clip([{
      target: { node: 'n', path: 'translation' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 0, 0, 10, 20, 30]),
        interpolation: 'LINEAR',
      },
    }]);
    expect(Array.from(sampleAnimationClip(c, 0.5)[0]!.value)).toEqual([5, 10, 15]);
  });

  it('STEP holds the floor keyframe', () => {
    const c = clip([{
      target: { node: 'n', path: 'scale' },
      sampler: {
        times: new Float32Array([0, 1, 2]),
        values: new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        interpolation: 'STEP',
      },
    }]);
    expect(Array.from(sampleAnimationClip(c, 1.7)[0]!.value)).toEqual([2, 2, 2]);
  });

  it('clamps before first and after last keyframe', () => {
    const c = clip([{
      target: { node: 'n', path: 'translation' },
      sampler: {
        times: new Float32Array([1, 2]),
        values: new Float32Array([5, 5, 5, 9, 9, 9]),
        interpolation: 'LINEAR',
      },
    }]);
    expect(Array.from(sampleAnimationClip(c, 0)[0]!.value)).toEqual([5, 5, 5]);
    expect(Array.from(sampleAnimationClip(c, 99)[0]!.value)).toEqual([9, 9, 9]);
  });

  it('rotation uses quaternion slerp (45° about Z at the midpoint, unit-length)', () => {
    const q0 = [0, 0, 0, 1];
    const q1 = [0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4)]; // 90° about Z
    const c = clip([{
      target: { node: 'n', path: 'rotation' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([...q0, ...q1]),
        interpolation: 'LINEAR',
      },
    }]);
    const v = sampleAnimationClip(c, 0.5)[0]!.value;
    expect(Math.hypot(v[0]!, v[1]!, v[2]!, v[3]!)).toBeCloseTo(1, 5);
    expect(v[2]).toBeCloseTo(Math.sin(Math.PI / 8), 4); // 22.5°
    expect(v[3]).toBeCloseTo(Math.cos(Math.PI / 8), 4);
  });

  it('CUBICSPLINE returns the knot value at a keyframe', () => {
    const values = new Float32Array([
      0, 0, 0, 1, 2, 3, 0, 0, 0, // kf0: in, value, out
      0, 0, 0, 4, 5, 6, 0, 0, 0, // kf1
    ]);
    const c = clip([{
      target: { node: 'n', path: 'translation' },
      sampler: { times: new Float32Array([0, 1]), values, interpolation: 'CUBICSPLINE' },
    }]);
    expect(Array.from(sampleAnimationClip(c, 0)[0]!.value)).toEqual([1, 2, 3]);
    expect(Array.from(sampleAnimationClip(c, 1)[0]!.value)).toEqual([4, 5, 6]);
  });

  it('CUBICSPLINE rotation outputs are normalized after Hermite interpolation', () => {
    const values = new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, // kf0: in, value, out
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, // kf1
    ]);
    const c = clip([{
      target: { node: 'n', path: 'rotation' },
      sampler: { times: new Float32Array([0, 1]), values, interpolation: 'CUBICSPLINE' },
    }]);
    const v = sampleAnimationClip(c, 0.5)[0]!.value;
    expect(Math.hypot(v[0]!, v[1]!, v[2]!, v[3]!)).toBeCloseTo(1, 5);
    expect(v[2]).toBeCloseTo(Math.SQRT1_2, 5);
    expect(v[3]).toBeCloseTo(Math.SQRT1_2, 5);
  });

  it('STEP and clamped rotation knot values are normalized', () => {
    const c = clip([{
      target: { node: 'n', path: 'rotation' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 0, 0, 2, 0, 0, 0, 3]),
        interpolation: 'STEP',
      },
    }]);
    const step = sampleAnimationClip(c, 0.5)[0]!.value;
    const clamped = sampleAnimationClip(c, 99)[0]!.value;
    expect(Array.from(step)).toEqual([0, 0, 0, 1]);
    expect(Array.from(clamped)).toEqual([0, 0, 0, 1]);
  });

  it('rejects non-finite sample time and non-finite, unsorted, or negative key times', () => {
    const base = clip([{
      target: { node: 'n', path: 'translation' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 0, 0, 1, 1, 1]),
      },
    }]);
    expect(() => sampleAnimationClip(base, Number.NaN)).toThrow(/time must be finite/);
    expect(() => validateAnimationClip({
      ...base,
      channels: [{
        ...base.channels[0]!,
        sampler: { ...base.channels[0]!.sampler, times: new Float32Array([0, Number.NaN]) },
      }],
    })).toThrow(/times\[1\].*finite/);
    expect(() => validateAnimationClip({
      ...base,
      channels: [{
        ...base.channels[0]!,
        sampler: { ...base.channels[0]!.sampler, times: new Float32Array([1, 1]) },
      }],
    })).toThrow(/strictly greater/);
    expect(() => validateAnimationClip({
      ...base,
      channels: [{
        ...base.channels[0]!,
        sampler: { ...base.channels[0]!.sampler, times: new Float32Array([-1, 1]) },
      }],
    })).toThrow(/must be >= 0/);
  });

  it('rejects malformed output strides instead of floor-dividing and zero-filling', () => {
    expect(() => validateAnimationClip(clip([{
      target: { node: 'n', path: 'weights' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 1, 2]),
      },
    }]))).toThrow(/positive integral component count/);
    expect(() => validateAnimationClip(clip([{
      target: { node: 'n', path: 'rotation' },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 0, 0, 1, 0, 0, 1]),
      },
    }]))).toThrow(/expected 8/);
  });

  it('rejects zero rotation knots rather than silently replacing them with identity', () => {
    expect(() => validateAnimationClip(clip([{
      target: { node: 'n', path: 'rotation' },
      sampler: {
        times: new Float32Array([0]),
        values: new Float32Array([0, 0, 0, 0]),
      },
    }]))).toThrow(/zero-length rotation quaternion/);
  });

  it('requires pointer metadata only for pointer channels and rejects duplicate targets', () => {
    expect(() => validateAnimationClip(clip([{
      target: { node: 'p', path: 'pointer' },
      sampler: { times: new Float32Array([0]), values: new Float32Array([1]) },
    }]))).toThrow(/pointer must be a non-empty string/);
    const channel = {
      target: { node: 'n', path: 'scale' as const },
      sampler: { times: new Float32Array([0]), values: new Float32Array([1, 1, 1]) },
    };
    expect(() => validateAnimationClip(clip([channel, channel])))
      .toThrow(/duplicates an earlier channel target/);
  });
});
