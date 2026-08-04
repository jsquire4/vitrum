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
    }))).toEqual([0.5, 0.25, Math.fround(0.1)]);
  });

  it('pre-multiplies authored emissiveIntensity into HDR radiance', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [0.5, 0.25, 0.1],
      emissiveIntensity: 4,
    }))).toEqual([2.0, 1.0, Math.fround(Math.fround(0.1) * 4)]);
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

  it('classifies unhinted Float32 emissive maps in the linear HDR domain', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Float32Array([0.25, 0.5, 4, 1]),
        },
      },
    }))).toEqual([0.25, 0.5, 4]);
  });

  it('normalizes public one/two-channel raw maps as RRR and RG0', () => {
    const oneChannel = materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Float32Array([0.25]),
          __vitrum_hint__: {
            channels: 1,
            dataType: 'float32',
            colorSpace: 'linear',
          },
        },
      },
    }));
    expect(oneChannel).toEqual([0.25, 0.25, 0.25]);

    const twoChannel = materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Float32Array([0.25, 4]),
          __vitrum_hint__: {
            channels: 2,
            dataType: 'float32',
            colorSpace: 'linear',
          },
        },
      },
    }));
    expect(twoChannel).toEqual([0.25, 4, 0]);
  });

  it('classifies a two-channel GPU mirror with native RG support, including green-only emission', () => {
    const source = material({
      emissive: [0, 2, 0],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          cpuMirror: {
            width: 1,
            height: 1,
            channels: 2,
            dataType: 'float32',
            colorSpace: 'linear',
            data: new Float32Array([0, 0.5]),
          },
        },
      },
    });
    expect(materialSpecEmissiveLe(source)).toEqual([0, 1, 0]);

    const visited: Array<readonly [number, number, number]> = [];
    expect(forEachEmissiveMapTexelSubTriangle(
      source,
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      (_a, _b, _c, radiance) => visited.push(radiance),
    )).toBe(true);
    expect(visited).toEqual([[0, 1, 0]]);
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

  it('falls back to scalar emission when the backend rejects the complete map payload', () => {
    const handle = {
      width: 1,
      height: 1,
      // The explicit RGB declaration makes the fourth word malformed rather
      // than an authored alpha lane. Atlas ingestion rejects this whole map.
      data: new Float32Array([0, 0, 0, 1]),
      __vitrum_hint__: {
        channels: 3,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };

    expect(materialSpecEmissiveLe(material({
      emissive: [2, 3, 4],
      emissiveMap: { handle },
    }))).toEqual([2, 3, 4]);
  });

  it('rejects a non-finite map atomically before classifying or visiting any texel', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 1, 1, 1,
        Number.NaN, 1, 1, 1,
      ]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const source = material({
      emissive: [2, 3, 4],
      emissiveMap: { handle, mipFilter: 'none' },
    });

    expect(materialSpecEmissiveLe(source)).toEqual([2, 3, 4]);

    let visits = 0;
    expect(forEachEmissiveMapTexelSubTriangle(
      source,
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      () => {
        visits += 1;
      },
    )).toBe(false);
    expect(visits).toBe(0);
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
        emissiveMap: { handle, mipFilter: 'none' },
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

  it('treats omitted mipFilter exactly like linear and reserves exact cells for explicit none', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' } as const,
    };
    const handled = (mipFilter?: 'linear' | 'none'): boolean =>
      forEachEmissiveMapTexelSubTriangle(
        material({
          emissive: [1, 1, 1],
          emissiveMap: {
            handle,
            ...(mipFilter !== undefined ? { mipFilter } : {}),
          },
        }),
        [0, 0], [1, 0], [0, 1],
        undefined, undefined, undefined,
        () => undefined,
      );

    expect(handled()).toBe(false);
    expect(handled('linear')).toBe(false);
    expect(handled('none')).toBe(true);
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
      emissiveMap: { handle, mipFilter: 'none' },
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

  it('keeps a late backend texel rejection callback-atomic', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
      ]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const resolverCalls: Array<readonly [number, number]> = [];
    let visits = 0;

    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [1, 1, 1],
        emissiveMap: { handle, mipFilter: 'none' },
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
      4096,
      undefined,
      (_resolvedMaterial, texelRgb, texelX, texelY) => {
        resolverCalls.push([texelX, texelY]);
        return texelX === 1 ? null : texelRgb;
      },
    );

    expect(handled).toBe(false);
    expect(resolverCalls).toEqual([[0, 0], [1, 0]]);
    expect(visits).toBe(0);
  });

  it('preserves successful mapped-emitter callback order and payloads after staging', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
      ]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const callbacks: Array<{
      readonly texelX: number;
      readonly texelY: number;
      readonly ordinal: number;
      readonly radiance: readonly [number, number, number];
    }> = [];

    const handled = forEachEmissiveMapTexelSubTriangle(
      material({
        emissive: [1, 1, 1],
        emissiveMap: { handle, mipFilter: 'none' },
      }),
      [0, 0],
      [1, 0],
      [0, 1],
      undefined,
      undefined,
      undefined,
      (_a, _b, _c, radiance, texelX, texelY, ordinal) => {
        callbacks.push({ texelX, texelY, ordinal, radiance });
      },
      4096,
      undefined,
      (_resolvedMaterial, _texelRgb, texelX, texelY) => [
        texelX + 1,
        texelY + 2,
        3,
      ],
    );

    expect(handled).toBe(true);
    expect(callbacks).toEqual([
      { texelX: 0, texelY: 0, ordinal: 0, radiance: [1, 2, 3] },
      { texelX: 0, texelY: 0, ordinal: 1, radiance: [1, 2, 3] },
      { texelX: 1, texelY: 0, ordinal: 2, radiance: [2, 2, 3] },
    ]);
  });

  it('rejects default mapped-texel f32 overflow and complete positive collapse', () => {
    const invoke = (emissive: number, texel: number): void => {
      forEachEmissiveMapTexelSubTriangle(
        material({
          emissive: [emissive, 0, 0],
          emissiveMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Float32Array([texel, 0, 0, 1]),
              __vitrum_hint__: {
                channels: 4,
                dataType: 'float32',
                colorSpace: 'linear',
              },
            },
            mipFilter: 'none',
          },
        }),
        [0, 0],
        [1, 0],
        [0, 1],
        undefined,
        undefined,
        undefined,
        () => undefined,
      );
    };

    expect(() => invoke(2, Math.fround(3.4028234663852886e38)))
      .toThrow(/remain finite in Float32/);
    expect(() => invoke(2 ** -80, 2 ** -80))
      .toThrow(/underflow completely to zero/);
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
        emissiveMap: { handle, texCoord: 2, mipFilter: 'none' },
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
        emissiveMap: { handle, wrapS: 'repeat', wrapT: 'repeat', mipFilter: 'none' },
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

  it('returns unit intensity when authored intensity is already folded into color', () => {
    const emitter = classifyTriangleEmitterCore(
      material({
        emissive: [0.5, 0.25, 0.1],
        emissiveIntensity: 4,
      }),
    );

    expect(emitter).toEqual({
      color: [2, 1, Math.fround(Math.fround(0.1) * 4)],
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
        emissiveMap: { handle, mipFilter: 'none' },
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
