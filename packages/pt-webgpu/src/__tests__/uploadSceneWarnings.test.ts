import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { uploadPackedScene, buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function oneMeshSceneWithBadBaseColorTexture(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: { image: { width: 0, height: 0 } } },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('uploadPackedScene warning propagation', () => {
  it('routes material texture upload warnings through UploadedSceneBuffers.warnings', () => {
    installGpuConstStubs();
    const device = {
      queue: {
        writeBuffer: vi.fn(),
        writeTexture: vi.fn(),
        copyExternalImageToTexture: vi.fn(),
      },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        label: desc.label,
        size: desc.size,
        destroy: vi.fn(),
      })),
      ...textureStubMethods(),
      limits: { maxTextureDimension2D: 8192 },
    } as unknown as GPUDevice;
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const uploaded = uploadPackedScene(device, buildPackedScene(oneMeshSceneWithBadBaseColorTexture()));
      expect(uploaded.warnings).toContain(
        'material texture array (sRGB): [materialTextureArray] source 0 has no usable image; layer left black.',
      );
      expect(consoleWarn).not.toHaveBeenCalled();
    } finally {
      consoleWarn.mockRestore();
    }
  });
});
