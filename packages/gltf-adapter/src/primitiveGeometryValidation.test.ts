import { describe, expect, it } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import { GltfComponentType, type GltfAccessor, type GltfJson } from './gltfTypes.js';

interface PrimitiveFixtureOptions {
  positions: readonly number[];
  positionType?: GltfAccessor['type'];
  positionCount?: number;
  normals?: readonly number[];
  normalCount?: number;
  indices?: readonly number[];
}

function primitiveFixture(options: PrimitiveFixtureOptions): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const positionBytes = options.positions.length * 4;
  const normalBytes = (options.normals?.length ?? 0) * 4;
  const indexOffset = positionBytes + normalBytes;
  const indexBytes = (options.indices?.length ?? 0) * 2;
  const buffer = new ArrayBuffer(indexOffset + indexBytes);
  const view = new DataView(buffer);

  options.positions.forEach((value, i) => view.setFloat32(i * 4, value, true));
  options.normals?.forEach((value, i) => {
    view.setFloat32(positionBytes + i * 4, value, true);
  });
  options.indices?.forEach((value, i) => {
    view.setUint16(indexOffset + i * 2, value, true);
  });

  const accessors: GltfAccessor[] = [{
    bufferView: 0,
    componentType: GltfComponentType.FLOAT,
    count: options.positionCount ?? 3,
    type: options.positionType ?? 'VEC3',
  }];
  const bufferViews: NonNullable<GltfJson['bufferViews']> = [{
    buffer: 0,
    byteOffset: 0,
    byteLength: positionBytes,
  }];
  const attributes: Record<string, number> = { POSITION: 0 };

  if (options.normals) {
    attributes.NORMAL = accessors.length;
    accessors.push({
      bufferView: bufferViews.length,
      componentType: GltfComponentType.FLOAT,
      count: options.normalCount ?? 3,
      type: 'VEC3',
    });
    bufferViews.push({
      buffer: 0,
      byteOffset: positionBytes,
      byteLength: normalBytes,
    });
  }

  let indices: number | undefined;
  if (options.indices) {
    indices = accessors.length;
    accessors.push({
      bufferView: bufferViews.length,
      componentType: GltfComponentType.UNSIGNED_SHORT,
      count: options.indices.length,
      type: 'SCALAR',
    });
    bufferViews.push({
      buffer: 0,
      byteOffset: indexOffset,
      byteLength: indexBytes,
    });
  }

  return {
    buffers: new Map([[0, buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes, ...(indices === undefined ? {} : { indices }) }] }],
      accessors,
      bufferViews,
      buffers: [{ byteLength: buffer.byteLength }],
    },
  };
}

describe('glTF primitive geometry validation', () => {
  it('rejects a POSITION accessor that is not VEC3', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 1, 0, 0, 1],
      positionType: 'VEC2',
    });
    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.POSITION',
      })],
    });
  });

  it('rejects non-finite POSITION data before constructing a core primitive', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, Number.NaN, 1, 0],
    });
    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        message: expect.stringContaining('non-finite'),
      })],
    });
  });

  it('rejects indices outside the POSITION accessor count', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 3],
    });
    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].indices',
        message: expect.stringContaining('outside POSITION count'),
      })],
    });
  });

  it('rejects normalized primitive index accessors', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    });
    fixture.gltf.accessors![1]!.normalized = true;

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'unreadable-indices',
        path: 'meshes[0].primitives[0].indices',
        message: expect.stringContaining('must not be normalized'),
      })],
    });
  });

  it('rejects a primitive-restart sentinel in ordinary index data', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 0xffff],
    });

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'unreadable-indices',
        path: 'meshes[0].primitives[0].indices',
        message: expect.stringContaining('reserved primitive-restart value 65535'),
      })],
    });
  });

  it('rejects a primitive-restart sentinel introduced by a sparse index patch', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    });
    const sparseBytes = new Uint8Array([2, 0xff]).buffer;
    fixture.buffers.set(1, sparseBytes);
    fixture.gltf.buffers!.push({ byteLength: sparseBytes.byteLength });
    const sparseIndexView = fixture.gltf.bufferViews!.length;
    fixture.gltf.bufferViews!.push(
      { buffer: 1, byteOffset: 0, byteLength: 1 },
      { buffer: 1, byteOffset: 1, byteLength: 1 },
    );
    const accessor = fixture.gltf.accessors![1]!;
    delete accessor.bufferView;
    accessor.componentType = GltfComponentType.UNSIGNED_BYTE;
    accessor.sparse = {
      count: 1,
      indices: {
        bufferView: sparseIndexView,
        componentType: GltfComponentType.UNSIGNED_BYTE,
      },
      values: { bufferView: sparseIndexView + 1 },
    };

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'unreadable-indices',
        path: 'meshes[0].primitives[0].indices',
        message: expect.stringContaining('reserved primitive-restart value 255'),
      })],
    });
  });

  it('rejects a count-mismatched NORMAL accessor', async () => {
    const fixture = primitiveFixture({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      normals: [1, 0, 0, 1, 0, 0],
      normalCount: 2,
    });
    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.NORMAL',
      })],
    });
  });
});
