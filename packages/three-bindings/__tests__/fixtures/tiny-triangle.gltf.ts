// Hand-built minimum glTF 2.0 scene for unit tests. CC0 (public-domain math).
// One triangle in the XY plane, one PerspectiveCamera, one PBR material.
// Embedded via data URI so the test runs offline with no fixture file IO.

// Buffer layout: 3 × VEC3 positions (36 bytes) + 3 × SCALAR uint16 indices
// (6 bytes, padded to 8). Total 44 bytes.

function buildBuffer(): { base64: string; byteLength: number } {
  const positions = new Float32Array([0, 1, 0, -1, -1, 0, 1, -1, 0]);
  const indices = new Uint16Array([0, 1, 2, 0]); // pad to 8 bytes
  const bytes = new Uint8Array(positions.byteLength + indices.byteLength);
  bytes.set(new Uint8Array(positions.buffer), 0);
  bytes.set(new Uint8Array(indices.buffer), positions.byteLength);

  // Base64 encode (Node's Buffer is available in vitest's node environment).
  const b64 = Buffer.from(bytes).toString('base64');
  return { base64: b64, byteLength: bytes.byteLength };
}

export function tinyTriangleGltfJson(): string {
  const buf = buildBuffer();
  const gltf = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes: [
      { name: 'Triangle', mesh: 0 },
      { name: 'Camera', camera: 0, translation: [0, 0, 5] },
    ],
    meshes: [
      {
        name: 'TriMesh',
        primitives: [
          {
            attributes: { POSITION: 0 },
            indices: 1,
            material: 0,
          },
        ],
      },
    ],
    cameras: [
      {
        type: 'perspective',
        perspective: { yfov: Math.PI / 4, aspectRatio: 1, znear: 0.1, zfar: 100 },
      },
    ],
    materials: [
      {
        name: 'Red',
        pbrMetallicRoughness: {
          baseColorFactor: [1, 0, 0, 1],
          metallicFactor: 0,
          roughnessFactor: 0.5,
        },
      },
    ],
    buffers: [
      {
        byteLength: buf.byteLength,
        uri: `data:application/octet-stream;base64,${buf.base64}`,
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 },
    ],
    accessors: [
      {
        bufferView: 0,
        byteOffset: 0,
        componentType: 5126, // FLOAT
        count: 3,
        type: 'VEC3',
        min: [-1, -1, 0],
        max: [1, 1, 0],
      },
      {
        bufferView: 1,
        byteOffset: 0,
        componentType: 5123, // UNSIGNED_SHORT
        count: 3,
        type: 'SCALAR',
      },
    ],
  };
  return JSON.stringify(gltf);
}

/** Variant without an embedded camera — used to test the no-camera branch. */
export function tinyTriangleGltfJsonNoCamera(): string {
  const json = JSON.parse(tinyTriangleGltfJson());
  delete json.cameras;
  json.nodes = json.nodes.filter((n: { camera?: number }) => n.camera === undefined);
  json.scenes[0].nodes = [0];
  return JSON.stringify(json);
}
