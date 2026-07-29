import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  classifyTriangleEmitterCore,
  emissiveMapTriangleSubdivisionLevel,
  forEachBarycentricSubTriangle,
  forEachEmissiveMapTexelSubTriangle,
  materialSpecEmissiveLe,
  materialSpecEmissiveLeAtUv,
  materialSpecSurfaceTextureId,
} from '../emitterClassify.js';

function material(partial: Partial<MaterialSpec>): MaterialSpec {
  return {
    baseColor: [0.2, 0.2, 0.2],
    roughness: 0.8,
    metallic: 0,
    ...partial,
  };
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

describe('materialSpecEmissiveLe', () => {
  it('retains the three-bit surfaceTextureId compatibility decoder', () => {
    expect(materialSpecSurfaceTextureId(material({
      extensions: { surfaceTextureId: 7 },
    }))).toBe(7);
    expect(materialSpecSurfaceTextureId(material({
      extensions: { surfaceTextureId: 8 },
    }))).toBe(0);
    expect(materialSpecSurfaceTextureId(material({
      extensions: { surfaceTextureId: 2.5 },
    }))).toBe(0);
    expect(materialSpecSurfaceTextureId(material({}))).toBe(0);
  });

  it('defaults missing emissiveIntensity to one', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [0.5, 0.25, 0.1],
    }))).toEqual([0.5, 0.25, 0.1]);
  });

  it('pre-multiplies authored emissiveIntensity into HDR radiance', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [0.5, 0.25, 0.1],
      emissiveIntensity: 4,
    }))).toEqual([2.0, 1.0, 0.4]);
  });

  it('rejects zero intensity as non-emissive', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveIntensity: 0,
    }))).toBeNull();
  });

  it('modulates Le by readable linear emissiveMap averages', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        0.25, 0.5, 1, 1,
        0.75, 0.25, 0.5, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    expect(materialSpecEmissiveLe(material({
      emissive: [2, 2, 2],
      emissiveIntensity: 3,
      emissiveMap: { handle },
    }))).toEqual([3, 2.25, 4.5]);
  });

  it('decodes readable sRGB emissiveMap handles before modulating Le', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 32, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'srgb' },
    };

    const le = materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveIntensity: 2,
      emissiveMap: { handle },
    }));

    expect(le?.[0]).toBeCloseTo(srgbToLinear(128 / 255) * 2, 6);
    expect(le?.[1]).toBeCloseTo(srgbToLinear(64 / 255) * 2, 6);
    expect(le?.[2]).toBeCloseTo(srgbToLinear(32 / 255) * 2, 6);
  });

  it('treats a readable black emissiveMap as non-emissive', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0, 0, 0, 1]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    expect(materialSpecEmissiveLe(material({
      emissive: [10, 10, 10],
      emissiveIntensity: 1,
      emissiveMap: { handle },
    }))).toBeNull();
  });

  it('samples readable emissiveMap texels at transformed and wrapped UVs', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 0, 1, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    const le = materialSpecEmissiveLeAtUv(material({
      emissive: [2, 2, 2],
      emissiveMap: { handle, wrapS: 'repeat', transform: { offset: [0.5, 0] } },
    }), [0.25, 0]);

    expect(le).toEqual([0, 0, 2]);
  });

  it('falls back to scalar emissive when emissiveMap targets an unsupported UV lane', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0, 0, 0, 1]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    const le = materialSpecEmissiveLeAtUv(material({
      emissive: [1, 1.5, 2],
      emissiveIntensity: 2,
      emissiveMap: { handle, texCoord: 2 },
    }), [0.5, 0.5]);

    expect(le).toEqual([2, 3, 4]);
  });

  it('samples an explicitly supplied arbitrary emissive UV lane', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    const le = materialSpecEmissiveLeAtUv(material({
      emissive: [2, 2, 2],
      emissiveMap: { handle, texCoord: 3 },
    }), [0, 0], undefined, [0.75, 0]);

    expect(le).toEqual([0, 2, 0]);
  });

  it('chooses a bounded subdivision level for CPU-readable emissive maps', () => {
    expect(emissiveMapTriangleSubdivisionLevel(material({ emissive: [1, 1, 1] }))).toBe(1);
    expect(emissiveMapTriangleSubdivisionLevel(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Float32Array([1, 1, 1, 1]),
          __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
        },
      },
    }))).toBe(1);
    expect(emissiveMapTriangleSubdivisionLevel(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 8,
          height: 2,
          data: new Float32Array(8 * 2 * 4).fill(1),
          __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
        },
      },
    }), 4)).toBe(4);
    expect(emissiveMapTriangleSubdivisionLevel(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        texCoord: 2,
        handle: {
          width: 8,
          height: 2,
          data: new Float32Array(8 * 2 * 4).fill(1),
          __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
        },
      },
    }), 4)).toBe(4);
  });

  it('enumerates barycentric micro-triangles with conserved area count', () => {
    const tris: string[] = [];
    forEachBarycentricSubTriangle(2, (a, b, c) => {
      tris.push(`${a.join(',')}|${b.join(',')}|${c.join(',')}`);
      for (const w of [a, b, c]) {
        expect(w[0] + w[1] + w[2]).toBeCloseTo(1, 12);
        expect(Math.min(w[0], w[1], w[2])).toBeGreaterThanOrEqual(0);
      }
    });

    expect(tris).toHaveLength(4);
    expect(tris[0]).toBe('1,0,0|0.5,0.5,0|0.5,0,0.5');
  });

  it('clips readable emissive maps into exact texel-cell sub-triangles', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };
    const patches: {
      areaRatio: number;
      radiance: readonly [number, number, number];
      texelX: number;
    }[] = [];
    const area2 = (
      a: readonly [number, number, number],
      b: readonly [number, number, number],
      c: readonly [number, number, number],
    ): number => Math.abs(
      (b[1] - a[1]) * (c[2] - a[2]) -
      (b[2] - a[2]) * (c[1] - a[1]),
    );

    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [2, 2, 2],
        emissiveMap: { handle },
      }),
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      (a, b, c, radiance, texelX) => {
        patches.push({ areaRatio: area2(a, b, c), radiance, texelX });
      },
    );

    expect(handled).toBe(true);
    expect(patches).toHaveLength(3);
    expect(patches.reduce((sum, p) => sum + p.areaRatio, 0)).toBeCloseTo(1, 12);
    expect(patches.filter((p) => p.texelX === 0).reduce((sum, p) => sum + p.areaRatio, 0))
      .toBeCloseTo(0.75, 12);
    expect(patches.filter((p) => p.texelX === 1).reduce((sum, p) => sum + p.areaRatio, 0))
      .toBeCloseTo(0.25, 12);
    expect(patches.some((p) => p.radiance[0] === 2 && p.radiance[1] === 0)).toBe(true);
    expect(patches.some((p) => p.radiance[0] === 0 && p.radiance[1] === 2)).toBe(true);
  });

  it('lets a backend inject exact decoded-texel storage and arithmetic', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0.25, 0.5, 1, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    let decoded: readonly [number, number, number] | undefined;
    let visited: readonly [number, number, number] | undefined;
    const mappedMaterial = material({
      emissive: [2, 2, 2],
      emissiveMap: { handle },
    });

    const handled = forEachEmissiveMapTexelSubTriangle(
      mappedMaterial,
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      (_a, _b, _c, radiance) => {
        visited = radiance;
      },
      4096,
      undefined,
      (resolvedMaterial, texelRgb, texelX, texelY) => {
        expect(resolvedMaterial).toBe(mappedMaterial);
        expect([texelX, texelY]).toEqual([0, 0]);
        decoded = texelRgb;
        return [texelRgb[0] + 1, texelRgb[1] + 2, texelRgb[2] + 3];
      },
    );

    expect(handled).toBe(true);
    expect(decoded).toEqual([0.25, 0.5, 1]);
    expect(visited).toEqual([1.25, 2.5, 4]);
  });

  it('does not split unsupported-UV emissive maps into exact texel sub-triangles', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };
    let visits = 0;

    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [2, 2, 2],
        emissiveMap: { handle, texCoord: 2 },
      }),
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      () => {
        visits += 1;
      },
    );

    expect(handled).toBe(false);
    expect(visits).toBe(0);
  });

  it('rejects enormous repeating UV spans before materializing cell intervals', () => {
    const handle = {
      width: 2,
      height: 2,
      data: new Float32Array(2 * 2 * 4).fill(1),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };
    let visits = 0;

    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [1, 1, 1],
        emissiveMap: { handle, wrapS: 'repeat', wrapT: 'repeat' },
      }),
      [0, 0],
      [1_000_000_000, 0],
      [0, 1_000_000_000],
      undefined,
      undefined,
      undefined,
      () => {
        visits += 1;
      },
      16,
    );

    expect(handled).toBe(false);
    expect(visits).toBe(0);
  });
});

describe('classifyTriangleEmitterCore', () => {
  it('classifies emissive core materials without an explicit intensity', () => {
    const emitter = classifyTriangleEmitterCore(
      material({ emissive: [0.25, 0.5, 1] }),
    );

    expect(emitter).toEqual({
      color: [0.25, 0.5, 1],
      intensity: 1,
    });
  });

  it('honors extensions.skipEmitter without suppressing camera-visible Le', () => {
    const skipped = material({
      emissive: [0.25, 0.5, 1],
      extensions: { skipEmitter: true },
    });
    expect(materialSpecEmissiveLe(skipped)).toEqual([0.25, 0.5, 1]);
    expect(classifyTriangleEmitterCore(skipped)).toBeNull();
  });

  it('treats skipped mapped emission as handled with no light-sampling patches', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([1, 1, 1, 1]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };
    let visits = 0;
    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [1, 1, 1],
        emissiveMap: { handle },
        extensions: { skipEmitter: true },
      }),
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      () => {
        visits += 1;
      },
    );
    expect(handled).toBe(true);
    expect(visits).toBe(0);
  });
});
