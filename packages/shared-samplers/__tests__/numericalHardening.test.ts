import { describe, expect, it } from 'vitest';

import { xyzToLinearSRGB } from '../src/cieCmf.js';
import { readEnvironmentMapPixels } from '../src/environmentMapPixels.js';
import { rgbToSpectralCoefficients } from '../src/jakobHanika.js';
import {
  buildLightTree,
  FULL_SPHERE_CONE,
  packLightTreeForGPU,
  type LightTreeNode,
} from '../src/lightTree.js';
import { bakePreethamSkyEquirect } from '../src/preethamSky.js';
import { defineUbo } from '../src/uboCodegen.js';
import { tangentBasis } from '../src/vecMath.js';
import { wavelengthToRGB } from '../src/wavelengthSampling.js';
import { BSDF_PRIMITIVES_WGSL } from '../src/wgsl/bsdfPrimitives.wgsl.js';
import { PCG_HASH_TO_F32_WGSL, PCG_WGSL } from '../src/wgsl/pcg.wgsl.js';

describe('public numerical boundary hardening', () => {
  it('keeps a UBO destination byte-identical when a late field is rejected', () => {
    const ubo = defineUbo([
      { name: 'count', type: 'u32' },
      { name: 'direction', type: 'vec3f' },
    ] as const);
    const bytes = new Uint8Array(ubo.sizeBytes + 8);
    bytes.fill(0xa5);
    const before = bytes.slice();

    expect(() => ubo.pack(
      new DataView(bytes.buffer),
      4,
      { count: 7, direction: [1, Number.NaN, 3] },
    )).toThrow(RangeError);
    expect(bytes).toEqual(before);
  });

  it('commits validated UBO values once and deterministically zeroes padding', () => {
    const ubo = defineUbo([
      { name: 'count', type: 'u32' },
      { name: 'direction', type: 'vec3f' },
    ] as const);
    const bytes = new Uint8Array(ubo.sizeBytes + 8);
    bytes.fill(0xa5);
    ubo.pack(new DataView(bytes.buffer), 4, { count: 7, direction: [1, 2, 3] });

    expect([...bytes.slice(0, 4)]).toEqual([0xa5, 0xa5, 0xa5, 0xa5]);
    expect([...bytes.slice(4 + ubo.sizeBytes)]).toEqual([0xa5, 0xa5, 0xa5, 0xa5]);
    expect(bytes.slice(8, 20).every((value) => value === 0)).toBe(true);
  });

  it('rejects non-finite spectrum inputs instead of converting them to black', () => {
    expect(() => rgbToSpectralCoefficients(Number.NaN, 0.5, 0.5)).toThrow(RangeError);
    expect(() => rgbToSpectralCoefficients(0.5, Number.POSITIVE_INFINITY, 0.5)).toThrow(RangeError);
  });

  it('rejects finite inputs whose derived XYZ conversion overflows', () => {
    expect(() => xyzToLinearSRGB(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE))
      .toThrow(RangeError);
  });

  it('rejects sRGB environment values that overflow f32 after linearization', () => {
    expect(readEnvironmentMapPixels({
      width: 1,
      height: 1,
      data: [3e38, 0, 0],
      channels: 3,
      dataType: 'float32',
      colorSpace: 'srgb',
    })).toBeNull();
  });

  it('rejects sky parameters whose derived values overflow', () => {
    expect(() => bakePreethamSkyEquirect({
      width: 1,
      height: 1,
      mieCoefficient: Number.MAX_VALUE,
    })).toThrow(RangeError);
  });

  it('rejects non-finite wavelength reconstruction inputs even on the zero path', () => {
    expect(() => wavelengthToRGB(550, 1, Number.NEGATIVE_INFINITY)).toThrow(RangeError);
  });
});
describe('light-tree GPU encoding invariants', () => {
  const leaf = (overrides: Partial<LightTreeNode> = {}): LightTreeNode => ({
    emitterIndex: 0,
    totalPower: 1,
    aabbMin: [0, 0, 0],
    aabbMax: [1, 1, 1],
    leftChild: -1,
    rightChild: -1,
    cone: FULL_SPHERE_CONE,
    ...overrides,
  });

  it('rejects integer indices that cannot be represented exactly as f32', () => {
    expect(() => packLightTreeForGPU([leaf({ emitterIndex: 0x01000001 })]))
      .toThrow(RangeError);
  });

  it('rejects malformed node shape, bounds, and cone sentinels', () => {
    expect(() => packLightTreeForGPU([leaf({ leftChild: 0 })])).toThrow(RangeError);
    expect(() => packLightTreeForGPU([leaf({ aabbMin: [2, 0, 0] })])).toThrow(RangeError);
    expect(() => packLightTreeForGPU([
      leaf({ cone: { axis: [0, 0, 0], thetaO: 0, thetaE: Math.PI / 2 } }),
    ])).toThrow(RangeError);
  });

  it('accepts the rooted pre-order tree produced by the builder', () => {
    const { nodes } = buildLightTree({
      powers: [1, 2],
      centroids: [[0, 0, 0], [2, 0, 0]],
      aabbs: [
        { min: [-1, -1, -1], max: [1, 1, 1] },
        { min: [1, -1, -1], max: [3, 1, 1] },
      ],
    });
    expect(packLightTreeForGPU(nodes)).toHaveLength(nodes.length * 16);
  });

  it('normalizes non-zero cone axes independently of their authored scale', () => {
    const { nodes } = buildLightTree({
      powers: [1],
      centroids: [[0, 0, 0]],
      aabbs: [{ min: [0, 0, 0], max: [0, 0, 0] }],
      cones: [{ axis: [1e-30, 0, 0], thetaO: 0, thetaE: Math.PI / 2 }],
    });
    expect(nodes[0]!.cone.axis).toEqual([1, 0, 0]);

    const packed = packLightTreeForGPU([
      leaf({ cone: { axis: [0, -1e-30, 0], thetaO: 0, thetaE: Math.PI / 2 } }),
    ]);
    expect(Array.from(packed.slice(10, 13))).toEqual([0, -1, 0]);
  });
});

describe('CPU/WGSL sampler degeneracy contracts', () => {
  it('uses the same deterministic basis for a zero normal on CPU and WGSL', () => {
    expect(tangentBasis([0, 0, 0])).toEqual({ t: [0, 0, 1], b: [1, 0, 0] });
    expect(BSDF_PRIMITIVES_WGSL).toContain('return vec3f(0.0, 1.0, 0.0)');
    expect(BSDF_PRIMITIVES_WGSL).toContain('if (!(wiScale > 1e-12)');
    expect(BSDF_PRIMITIVES_WGSL).toContain('return 0.0;');
  });

  it('maps PCG output through the high 24 bits into the half-open unit interval', () => {
    expect(PCG_WGSL).toContain('pcgNext(state) >> 8u');
    expect(PCG_HASH_TO_F32_WGSL).toContain('>> 8u) / 16777216.0');
    expect(PCG_WGSL).not.toContain('/ f32(0xFFFFFFFFu)');
    expect(PCG_HASH_TO_F32_WGSL).not.toContain('/ 4294967295.0');
  });

  it('uses full-domain rejection for exact bounded integer selection', () => {
    expect(PCG_WGSL).toContain('fn rand_bounded_u32(');
    expect(PCG_WGSL).toContain('let threshold = (0u - bound) % bound;');
    expect(PCG_WGSL).toContain('if (value >= threshold) { return value % bound; }');
    expect(PCG_WGSL).toContain('fn represented_bernoulli_probability_f32(');
    expect(PCG_WGSL).toContain('floor(probability * 16777216.0 + 0.5)');
    const domain = 1n << 32n;
    for (const bound of [2n, 3n, 7n, 255n, 65537n, 0xffff_ffffn]) {
      const rejectedPrefix = domain % bound;
      const accepted = domain - rejectedPrefix;
      expect(rejectedPrefix).toBeLessThan(bound);
      expect(accepted % bound).toBe(0n);
    }
  });
});
