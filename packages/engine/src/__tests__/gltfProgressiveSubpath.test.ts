import { describe, expect, it, vi } from 'vitest';
import type { GltfJson } from '@vitrum/gltf-adapter';

const createProgressiveEngineMock = vi.hoisted(() => vi.fn());

vi.mock('../createProgressiveEngine.js', () => ({
  createProgressiveEngine: createProgressiveEngineMock,
}));

import {
  loadGltfWithProgressiveEngine,
  type LoadGltfWithProgressiveEngineOptions,
} from '../gltf.js';

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function makeInlineTriangleGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    },
    buffers: new Map([[0, positions]]),
  };
}

describe('@vitrum/engine/gltf progressive helper', () => {
  it('loads a glTF scene and wires its controller into createProgressiveEngine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const canvas = {} as HTMLCanvasElement;
    const handle = {
      coordinator: {},
      realtime: {},
      converged: {},
      dispose: vi.fn(),
    };
    createProgressiveEngineMock.mockResolvedValueOnce(handle);

    const options: LoadGltfWithProgressiveEngineOptions = {
      buffers,
      engineOptions: {
        canvas,
        seedFromRealtime: false,
      },
    };
    const result = await loadGltfWithProgressiveEngine(gltf, options);

    expect(result.backend).toBe('pt-webgpu');
    expect(result.engine).toBe(handle);
    expect(result.attached).toBe(true);
    expect(createProgressiveEngineMock).toHaveBeenCalledTimes(1);
    expect(createProgressiveEngineMock.mock.calls[0]![0]).toEqual(expect.objectContaining({
      canvas,
      seedFromRealtime: false,
      scene: result.asset.scene,
      controller: result.controller,
    }));
  });
});
