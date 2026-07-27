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
    [{ magFilter: 'linear', minFilter: 'nearest' }, 'requires matching magFilter/minFilter'],
    [{ mipFilter: 'linear' }, 'cannot be honored by the base-level CPU displacement sampler'],
  ])('rejects sampler semantics it cannot honor %#', (sampler, message) => {
    expect(() => maybeDisplaceMeshPositions(vertexInput(map(
      new Float32Array([1]),
      {},
      sampler,
    )))).toThrow(message);
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
    expect(linear?.[2]).toBeCloseTo(0.6);
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
