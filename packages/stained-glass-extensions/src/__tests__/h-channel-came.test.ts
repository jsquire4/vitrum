import { describe, it, expect } from 'vitest';
import type { AnalyticPrimitive, AnalyticShape } from '@vitrum/core';
import {
  STAINED_GLASS_H_CHANNEL_CAME,
  STAINED_GLASS_H_CHANNEL_CAME_PARAMS,
  isHChannelCame,
} from '../h-channel-came.js';

describe('STAINED_GLASS_H_CHANNEL_CAME constant', () => {
  it('preserves the canonical discriminator string', () => {
    expect(STAINED_GLASS_H_CHANNEL_CAME).toBe('h-channel-came');
  });

  it('is assignable to the core AnalyticShape open-ended string union', () => {
    // The whole point of D2: the open-ended `(string & {})` variant in core
    // accepts host-defined tags. This is a compile-time assertion; it would
    // fail to compile if AnalyticShape were a closed union.
    const shape: AnalyticShape = STAINED_GLASS_H_CHANNEL_CAME;
    expect(shape).toBe('h-channel-came');
  });

  it('round-trips through an AnalyticPrimitive without widening to string', () => {
    const prim: AnalyticPrimitive = {
      kind: 'analytic',
      id: 'rail-0',
      shape: STAINED_GLASS_H_CHANNEL_CAME,
      params: new Float32Array([1.0, 0.012, 0.018, 0.004]),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    expect(prim.shape).toBe('h-channel-came');
  });
});

describe('isHChannelCame helper', () => {
  it('returns true for the canonical tag', () => {
    expect(isHChannelCame(STAINED_GLASS_H_CHANNEL_CAME)).toBe(true);
    expect(isHChannelCame('h-channel-came')).toBe(true);
  });

  it('returns false for the core analytic-shape tags', () => {
    expect(isHChannelCame('sphere')).toBe(false);
    expect(isHChannelCame('box')).toBe(false);
    expect(isHChannelCame('capsule')).toBe(false);
    expect(isHChannelCame('cylinder')).toBe(false);
  });

  it('returns false for unrelated strings', () => {
    expect(isHChannelCame('')).toBe(false);
    expect(isHChannelCame('h-channel')).toBe(false);
    expect(isHChannelCame('H-CHANNEL-CAME')).toBe(false);
    expect(isHChannelCame('came')).toBe(false);
  });
});

describe('STAINED_GLASS_H_CHANNEL_CAME_PARAMS layout', () => {
  it('exposes ordered field offsets', () => {
    expect(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.LENGTH).toBe(0);
    expect(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.RAIL_WIDTH).toBe(1);
    expect(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.BLOCK_HEIGHT).toBe(2);
    expect(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.WEB_THICKNESS).toBe(3);
    expect(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.PARAM_COUNT).toBe(4);
  });

  it('matches the parameter count of a real came primitive', () => {
    const params = new Float32Array([1.0, 0.012, 0.018, 0.004]);
    expect(params.length).toBe(STAINED_GLASS_H_CHANNEL_CAME_PARAMS.PARAM_COUNT);
  });
});
