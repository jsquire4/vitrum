import { describe, expect, it } from 'vitest';
import { giBilinearWeightsWgsl } from '../giBilinearGather.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../walkaroundUbo.wgsl.js';

interface WeightedCell {
  readonly index: number;
  readonly weight: number;
}

function centeredAxisWeights(
  pixel: number,
  fullExtent: number,
  stride: number,
): readonly WeightedCell[] {
  const gridExtent = Math.max(1, Math.floor(fullExtent / stride));
  const coordinate = Math.max(0, (pixel - Math.floor(stride / 2)) / stride);
  const base = Math.floor(coordinate);
  const fraction = coordinate - base;
  const weights = new Map<number, number>();
  const accumulate = (index: number, weight: number): void => {
    const clamped = Math.min(gridExtent - 1, index);
    weights.set(clamped, (weights.get(clamped) ?? 0) + weight);
  };
  accumulate(base, 1 - fraction);
  accumulate(base + 1, fraction);
  return [...weights]
    .filter(([, weight]) => weight > 0)
    .map(([index, weight]) => ({ index, weight }));
}

describe('ReSTIR-GI center-aligned bilinear gather', () => {
  it('pins the producer and consumer to the same sample-center convention', () => {
    expect(WALKAROUND_UBO_WGSL).toContain(
      'coord * stride + vec2u(stride / 2u)',
    );
    const gather = giBilinearWeightsWgsl();
    expect(gather).toContain('let giSampleCenter = f32(giStride / 2u);');
    expect(gather).toContain(
      '(vec2f(gid) - vec2f(giSampleCenter)) / f32(giStride)',
    );
    expect(gather).toContain('let halfPxF = max(');
  });

  it('is exact at stride-2 producer centers and clamps both image edges', () => {
    expect(centeredAxisWeights(0, 9, 2)).toEqual([
      { index: 0, weight: 1 },
    ]);
    expect(centeredAxisWeights(1, 9, 2)).toEqual([
      { index: 0, weight: 1 },
    ]);
    expect(centeredAxisWeights(2, 9, 2)).toEqual([
      { index: 0, weight: 0.5 },
      { index: 1, weight: 0.5 },
    ]);
    expect(centeredAxisWeights(3, 9, 2)).toEqual([
      { index: 1, weight: 1 },
    ]);
    expect(centeredAxisWeights(8, 9, 2)).toEqual([
      { index: 3, weight: 1 },
    ]);
  });

  it('is exact at scaled stride-4 centers, midpoint, and trailing edge', () => {
    expect(centeredAxisWeights(0, 13, 4)).toEqual([
      { index: 0, weight: 1 },
    ]);
    expect(centeredAxisWeights(2, 13, 4)).toEqual([
      { index: 0, weight: 1 },
    ]);
    expect(centeredAxisWeights(4, 13, 4)).toEqual([
      { index: 0, weight: 0.5 },
      { index: 1, weight: 0.5 },
    ]);
    expect(centeredAxisWeights(6, 13, 4)).toEqual([
      { index: 1, weight: 1 },
    ]);
    expect(centeredAxisWeights(10, 13, 4)).toEqual([
      { index: 2, weight: 1 },
    ]);
    expect(centeredAxisWeights(12, 13, 4)).toEqual([
      { index: 2, weight: 1 },
    ]);
  });
});
