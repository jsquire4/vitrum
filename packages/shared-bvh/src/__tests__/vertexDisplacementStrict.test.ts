import { describe, expect, it } from 'vitest';
import type { MaterialSpec, TextureRef } from '@vitrum/core';
import {
  maybeDisplaceMeshPositions,
  maybeMicrodisplaceMeshGeometry,
  resolveDisplacedGeometry,
} from '../vertexDisplacement.js';

function map(
  data: ArrayLike<number> = new Float32Array([1]),
  descriptor: Record<string, unknown> = {},
  sampler: Record<string, unknown> = {},
): TextureRef {
  return {
    handle: {
      width: 1,
      height: 1,
      data,
      ...descriptor,
    },
    mipFilter: 'none',
    ...sampler,
  };
}

function material(
  displacementMap: TextureRef,
  overrides: Partial<MaterialSpec> = {},
): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    displacementMap,
    displacementScale: 1,
    displacementBias: 0,
    ...overrides,
  };
}

function vertexInput(
  displacementMap: TextureRef,
  overrides: Partial<Parameters<typeof maybeDisplaceMeshPositions>[0]> = {},
): Parameters<typeof maybeDisplaceMeshPositions>[0] {
  return {
    primitiveId: 'strict-height',
    material: material(displacementMap),
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    ...overrides,
  };
}

function microInput(
  subdivisions: number,
  overrides: Partial<Parameters<typeof maybeMicrodisplaceMeshGeometry>[0]> = {},
): Parameters<typeof maybeMicrodisplaceMeshGeometry>[0] {
  const displacementMap = map();
  return {
    primitiveId: 'strict-micro-height',
    material: material(displacementMap, { displacementSubdivisions: subdivisions }),
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    ...overrides,
  };
}

describe('strict CPU displacement contract', () => {
  it.each([
    [{ width: 0 }, 'positive safe integers'],
    [{ width: 1.5 }, 'positive safe integers'],
    [{ height: Number.POSITIVE_INFINITY }, 'positive safe integers'],
    [{ channels: 5 }, 'channels must be an exact integer in [1, 4]'],
  ])('rejects malformed dimensions/channels %#', (descriptor, message) => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(new Float32Array([1]), descriptor))))
      .toThrow(message);
  });

  it('requires an exact pixel count with no short or trailing payload', () => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([0, 1, 2]),
      { width: 2, height: 1, channels: 1 },
    )))).toThrow('data length 3 must equal width*height*channels (2)');
  });

  it.each([
    [new Float64Array([1]), {}, 'pixel backing [object Float64Array] is unsupported'],
    [new Int8Array([1]), {}, 'pixel backing [object Int8Array] is unsupported'],
    [new Uint16Array([65535]), {}, 'require explicit dataType'],
    [new Uint8Array([255]), { dataType: 'float32' }, 'does not match [object Uint8Array]'],
  ])('rejects ambiguous or mismatched pixel backing %#', (data, descriptor, message) => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(data, descriptor))))
      .toThrow(message);
  });

  it('accepts explicitly typed normalized Uint16 pixels', () => {
    const result = maybeDisplaceMeshPositions(vertexInput(map(
      new Uint16Array([65535]),
      { dataType: 'uint16' },
    )));
    expect(result?.[2]).toBeCloseTo(1);
  });

  it.each([
    [new Float32Array([Number.NaN]), {}, 'decoded pixel[0]'],
    [new Float32Array([Number.POSITIVE_INFINITY]), {}, 'decoded pixel[0]'],
    [new Uint16Array([0x7c00]), { dataType: 'float16' }, 'decoded pixel[0]'],
  ])('rejects non-finite decoded heights %#', (data, descriptor, message) => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(data, descriptor))))
      .toThrow(message);
  });

  it('does not combine raw data with dimensions from an unrelated image payload', () => {
    const displacementMap = {
      handle: {
        data: new Float32Array([1]),
        image: { width: 1, height: 1, data: new Float32Array([0]) },
      },
      mipFilter: 'none',
    } as TextureRef;
    expect(() => maybeDisplaceMeshPositions(vertexInput(displacementMap)))
      .toThrow('width and height must be positive safe integers');
  });

  it('requires the exact authored UV stream and exact geometry stream lengths', () => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(), {
      uvs: null as unknown as Float32Array,
    })))
      .toThrow('requests TEXCOORD_0, but that exact UV channel is absent');
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(), {
      normals: new Float32Array([0, 0, 1]),
    }))).toThrow('normals.length must be exactly 9');
  });

  it('rejects zero normals instead of inventing a fallback direction', () => {
    const normals = new Float32Array([0, 0, 0, 0, 0, 1, 0, 0, 1]);
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(), { normals })))
      .toThrow('normals vertex 0 is zero-length');
  });

  it.each([
    [{ wrapS: 'border' }, 'wrapS has unsupported mode'],
    [{ magFilter: 'cubic' }, 'magFilter must be "nearest" or "linear"'],
    [{ magFilter: 'linear', minFilter: 'nearest' }, 'requires matching magFilter/minFilter'],
    [{ mipFilter: 'nearest' }, 'cannot be honored by the base-level CPU displacement sampler'],
    [{ mipFilter: 'linear' }, 'cannot be honored by the base-level CPU displacement sampler'],
  ])('rejects sampler semantics it cannot honor %#', (sampler, message) => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([1]),
      {},
      sampler,
    )))).toThrow(message);
  });

  it('treats omitted mipFilter as linear and requires explicit none for CPU displacement', () => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([1]),
      {},
      { mipFilter: undefined },
    )))).toThrow(/mipFilter "linear" cannot be honored/);
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([1]),
      {},
      { mipFilter: 'linear' },
    )))).toThrow(/mipFilter "linear" cannot be honored/);
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([1]),
      {},
      { mipFilter: 'none' },
    )))).not.toThrow();
  });

  it('honors nearest versus linear base-level sampling', () => {
    const data = new Float32Array([0, 1]);
    const descriptor = { width: 2, height: 1, channels: 1 };
    const uvs = new Float32Array([0.6, 0, 0.6, 0, 0.6, 0]);
    const nearest = maybeDisplaceMeshPositions(vertexInput(map(data, descriptor, {
      wrapS: 'clamp-to-edge',
      wrapT: 'clamp-to-edge',
      magFilter: 'nearest',
      minFilter: 'nearest',
    }), { uvs }));
    const linear = maybeDisplaceMeshPositions(vertexInput(map(data, descriptor, {
      wrapS: 'clamp-to-edge',
      wrapT: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    }), { uvs }));
    expect(nearest?.[2]).toBeCloseTo(1);
    // Normalized texture coordinates address texel centres at
    // (i + 0.5) / width, so u=0.6 lies 70% of the way from texel 0 to 1.
    expect(linear?.[2]).toBeCloseTo(0.7);
  });

  it('uses normalized-sampler texel centres for nearest filtering', () => {
    const data = new Float32Array([0, 0.25, 0.5, 1]);
    const descriptor = { width: 4, height: 1, channels: 1 };
    const uvs = new Float32Array([0.2, 0, 0.2, 0, 0.2, 0]);
    const displaced = maybeDisplaceMeshPositions(vertexInput(map(data, descriptor, {
      wrapS: 'clamp-to-edge',
      wrapT: 'clamp-to-edge',
      magFilter: 'nearest',
      minFilter: 'nearest',
    }), { uvs }));

    // floor(0.2 * 4) = texel 0. The old round(u * (width - 1)) path
    // incorrectly selected texel 1.
    expect(displaced?.[2]).toBeCloseTo(0);
  });

  it.each([
    ['repeat', 0, 0.5],
    ['repeat', 0.875, 0.75],
    ['clamp-to-edge', 0.25, 0],
    ['mirrored-repeat', 1.25, 1],
  ] as const)(
    'applies %s independently to linear-filter texel taps',
    (wrapS, u, expectedHeight) => {
      const data = new Float32Array([0, 1]);
      const descriptor = { width: 2, height: 1, channels: 1 };
      const uvs = new Float32Array([u, 0, u, 0, u, 0]);
      const displaced = maybeDisplaceMeshPositions(vertexInput(map(data, descriptor, {
        wrapS,
        wrapT: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
      }), { uvs }));
      expect(displaced?.[2]).toBeCloseTo(expectedHeight);
    },
  );

  it.each([
    ['repeat', 0, 0.5],
    ['repeat', -0.125, 0.75],
    ['clamp-to-edge', 0.25, 0],
    ['mirrored-repeat', 1.25, 1],
  ] as const)(
    'applies %s independently to vertical linear-filter texel taps',
    (wrapT, v, expectedHeight) => {
      const data = new Float32Array([0, 1]);
      const descriptor = { width: 1, height: 2, channels: 1 };
      const uvs = new Float32Array([0, v, 0, v, 0, v]);
      const displaced = maybeDisplaceMeshPositions(vertexInput(map(data, descriptor, {
        wrapS: 'clamp-to-edge',
        wrapT,
        magFilter: 'linear',
        minFilter: 'linear',
      }), { uvs }));
      expect(displaced?.[2]).toBeCloseTo(expectedHeight);
    },
  );

  it('decodes explicitly sRGB CPU pixels before filtering displacement heights', () => {
    const displacementMap = map(
      new Float32Array([0.5]),
      { __vitrum_hint__: { channels: 1, dataType: 'float32', colorSpace: 'srgb' } },
    );
    const displaced = maybeDisplaceMeshPositions(vertexInput(displacementMap));
    expect(displaced?.[2]).toBeCloseTo(0.21404114, 6);
  });

  it('rejects an unknown CPU pixel color space instead of guessing', () => {
    const displacementMap = map(
      new Float32Array([0.5]),
      { __vitrum_hint__: { channels: 1, dataType: 'float32', colorSpace: 'display-p3' } },
    );
    expect(() => maybeDisplaceMeshPositions(vertexInput(displacementMap)))
      .toThrow('colorSpace must be "linear" or "srgb"');
  });

  it('samples the red lane only from multi-channel height payloads', () => {
    const displacementMap = map(
      new Float32Array([0.25, 0.75, 1, 0]),
      { channels: 4, colorSpace: 'linear' },
    );
    const displaced = maybeDisplaceMeshPositions(vertexInput(displacementMap));
    expect(displaced?.[2]).toBeCloseTo(0.25);
  });

  it('rejects outputs that overflow float32', () => {
    const max = 3.4028234663852886e38;
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(), {
      material: material(map(), { displacementScale: max }),
      positions: new Float32Array([max, 0, 0, 0, 0, 0, 0, 1, 0]),
      normals: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 1]),
    }))).toThrow('output positions[0] must be finite and representable as float32');
  });

  it.each([-1, 1.5, 5, Number.NaN])(
    'rejects invalid displacementSubdivisions=%s',
    (subdivisions) => {
      expect(() => maybeMicrodisplaceMeshGeometry(microInput(subdivisions)))
        .toThrow('displacementSubdivisions must be an integer in [0, 4]');
    },
  );

  it('keeps subdivision zero as the vertex-only selection', () => {
    expect(maybeMicrodisplaceMeshGeometry(microInput(0))).toBeNull();
    expect(maybeDisplaceMeshPositions(vertexInput(map()))?.[2]).toBeCloseTo(1);
  });

  it('rejects malformed and out-of-range triangle indices', () => {
    expect(() => maybeMicrodisplaceMeshGeometry(microInput(1, {
      indices: new Uint32Array([0, 1]),
    }))).toThrow('indices.length must be divisible by 3');
    expect(() => maybeMicrodisplaceMeshGeometry(microInput(1, {
      indices: new Uint32Array([0, 1, 3]),
    }))).toThrow('is outside vertexCount 3');
  });

  it('welds regenerated normals across indexed source-triangle boundaries', () => {
    const invSqrt2 = Math.SQRT1_2;
    const result = maybeMicrodisplaceMeshGeometry(microInput(1, {
      material: material(map(new Float32Array([0])), {
        displacementSubdivisions: 1,
      }),
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]),
      normals: new Float32Array([
        0, invSqrt2, invSqrt2,
        0, invSqrt2, invSqrt2,
        0, 0, 1,
        0, 1, 0,
      ]),
      uvs: new Float32Array([
        0, 0,
        1, 0,
        0, 1,
        0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2, 1, 0, 3]),
    }));

    expect(result).not.toBeNull();
    const sharedMidpointNormals: number[][] = [];
    for (let vertex = 0; vertex < result!.positions.length / 3; vertex += 1) {
      const offset = vertex * 3;
      if (
        Math.abs(result!.positions[offset]! - 0.5) < 1e-6 &&
        Math.abs(result!.positions[offset + 1]!) < 1e-6 &&
        Math.abs(result!.positions[offset + 2]!) < 1e-6
      ) {
        sharedMidpointNormals.push(Array.from(result!.normals.subarray(offset, offset + 3)));
      }
    }
    expect(sharedMidpointNormals).toHaveLength(2);
    for (const normal of sharedMidpointNormals) {
      expect(normal[0]).toBeCloseTo(0, 6);
      expect(normal[1]).toBeCloseTo(invSqrt2, 6);
      expect(normal[2]).toBeCloseTo(invSqrt2, 6);
    }
  });

  it('preserves authored hard edges represented by duplicated source vertices', () => {
    const result = maybeMicrodisplaceMeshGeometry(microInput(1, {
      material: material(map(new Float32Array([0])), {
        displacementSubdivisions: 1,
      }),
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 0, 0,
        0, 0, 1,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
      ]),
      uvs: new Float32Array([
        0, 0,
        1, 0,
        0, 1,
        1, 1,
        0, 1,
        1, 0,
      ]),
      indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
    }));

    expect(result).not.toBeNull();
    const sharedMidpointNormals: number[][] = [];
    for (let vertex = 0; vertex < result!.positions.length / 3; vertex += 1) {
      const offset = vertex * 3;
      if (
        Math.abs(result!.positions[offset]! - 0.5) < 1e-6 &&
        Math.abs(result!.positions[offset + 1]!) < 1e-6 &&
        Math.abs(result!.positions[offset + 2]!) < 1e-6
      ) {
        sharedMidpointNormals.push(Array.from(result!.normals.subarray(offset, offset + 3)));
      }
    }
    expect(sharedMidpointNormals).toHaveLength(2);
    expect(sharedMidpointNormals).toEqual(
      expect.arrayContaining([
        [0, 0, 1],
        [0, 1, 0],
      ]),
    );
  });

  it('warns and falls back to authored-vertex displacement at the exact dicing cap', () => {
    const indices = new Uint32Array(1_025 * 3);
    for (let offset = 0; offset < indices.length; offset += 3) {
      indices[offset] = 0;
      indices[offset + 1] = 1;
      indices[offset + 2] = 2;
    }
    const warnings: string[] = [];
    const input = microInput(4, { indices, onWarning: (warning) => warnings.push(warning) });

    expect(maybeMicrodisplaceMeshGeometry(input)).toBeNull();
    expect(warnings).toEqual([
      'Primitive "strict-micro-height" displacementMap displacementSubdivisions=4 ' +
      'would generate 262400 triangles, above the shared-BVH safety cap 262144; ' +
      'falling back to vertex displacement.',
    ]);

    const resolved = resolveDisplacedGeometry({
      id: input.primitiveId,
      material: input.material,
      positions: input.positions,
      normals: input.normals,
      indices,
      ...(input.uvs != null ? { uvs: input.uvs } : {}),
    }, (warning) => warnings.push(warning));
    expect(resolved.microdisplaced).toBe(false);
    expect(Array.from(resolved.sourcePositions)).toEqual([
      0, 0, 1,
      1, 0, 1,
      0, 1, 1,
    ]);
    expect(warnings).toHaveLength(2);
  });
});
