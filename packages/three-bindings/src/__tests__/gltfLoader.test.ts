import { describe, it, expect } from 'vitest';

// three.js's loader callbacks reference globalThis.ProgressEvent (a DOM
// type). Node test environment doesn't define it; polyfill with a minimal
// stand-in BEFORE importing the loader so the lazy reference resolves.
if (typeof (globalThis as { ProgressEvent?: unknown }).ProgressEvent === 'undefined') {
  (globalThis as { ProgressEvent?: unknown }).ProgressEvent = class {
    loaded = 0;
    total = 0;
    lengthComputable = false;
    type: string;
    constructor(type: string) {
      this.type = type;
    }
  };
}

const { loadGltfScene } = await import('../gltfLoader.js');
const { tinyTriangleGltfJson, tinyTriangleGltfJsonNoCamera } =
  await import('../../__tests__/fixtures/tiny-triangle.gltf.js');

// GLTFLoader.parse accepts a JSON string for text-form glTF, OR an
// ArrayBuffer for GLB. We feed JSON-as-Blob (and ArrayBuffer) to verify
// both source-form code paths in loadGltfScene.

function jsonAsBlob(json: string): Blob {
  return new Blob([json], { type: 'model/gltf+json' });
}

function jsonAsArrayBuffer(json: string): ArrayBuffer {
  const enc = new TextEncoder();
  return enc.encode(json).buffer;
}

describe('loadGltfScene', () => {
  it('parses a glTF blob and returns a vitrum Scene', async () => {
    const result = await loadGltfScene(jsonAsBlob(tinyTriangleGltfJson()));
    expect(result.scene.primitives.length).toBe(1);
    const prim = result.scene.primitives[0]!;
    expect(prim.kind).toBe('mesh');
    expect(prim.material.baseColor).toEqual([1, 0, 0]);
    expect(result.scene.emitters.length).toBe(0);
  });

  it('accepts an ArrayBuffer source', async () => {
    const result = await loadGltfScene(jsonAsArrayBuffer(tinyTriangleGltfJson()));
    expect(result.scene.primitives.length).toBe(1);
  });

  it('surfaces the first embedded camera with view+proj matrices', async () => {
    const result = await loadGltfScene(jsonAsBlob(tinyTriangleGltfJson()));
    expect(result.camera).toBeDefined();
    const cam = result.camera!;
    // Camera at (0, 0, 5) → cameraPosition reads world translation.
    expect(cam.cameraPosition[0]).toBeCloseTo(0, 5);
    expect(cam.cameraPosition[1]).toBeCloseTo(0, 5);
    expect(cam.cameraPosition[2]).toBeCloseTo(5, 5);
    // View matrix is the inverse of the camera world transform; for a pure
    // translation (0,0,5) the view-translation column should be (0,0,-5).
    expect(cam.viewMatrix[12]).toBeCloseTo(0, 5);
    expect(cam.viewMatrix[13]).toBeCloseTo(0, 5);
    expect(cam.viewMatrix[14]).toBeCloseTo(-5, 5);
    // Projection matrix is non-trivially populated (yfov = π/4, aspect = 1).
    expect(cam.projMatrix.length).toBe(16);
    expect(cam.projMatrix[0]).not.toBe(0);
    expect(cam.projMatrix[5]).not.toBe(0);
    expect(cam.projMatrix[10]).not.toBe(0);
  });

  it('omits camera field when the glTF has no cameras', async () => {
    const result = await loadGltfScene(jsonAsBlob(tinyTriangleGltfJsonNoCamera()));
    expect(result.camera).toBeUndefined();
    expect(result.scene.primitives.length).toBe(1);
  });

  it('rejects non-Blob non-ArrayBuffer non-string sources', async () => {
    await expect(loadGltfScene(42 as unknown as string)).rejects.toThrow(TypeError);
  });
});
