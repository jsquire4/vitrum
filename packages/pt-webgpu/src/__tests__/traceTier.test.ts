import { describe, expect, it } from 'vitest';
import { resolvePtWebgpuTraceTier, selectPtWebgpuTraceTier } from '../traceTier.js';
import {
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from '../webgpuLimits.js';

function deviceWithLimits(
  maxStorageBuffersPerShaderStage: number,
  maxStorageTexturesPerShaderStage = 8,
): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage,
      maxStorageTexturesPerShaderStage,
    },
  } as GPUDevice;
}

describe('selectPtWebgpuTraceTier', () => {
  it('selects full when buffers and textures meet the full layout', () => {
    expect(
      selectPtWebgpuTraceTier(
        deviceWithLimits(
          PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
          PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
        ),
      ),
    ).toBe('full');
  });

  it('selects lite on SwiftShader-class caps', () => {
    expect(
      selectPtWebgpuTraceTier(
        deviceWithLimits(PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 4),
      ),
    ).toBe('lite');
  });

  it('throws when below lite tier', () => {
    expect(() => selectPtWebgpuTraceTier(deviceWithLimits(4, 4))).toThrow(
      /below the lite tier/i,
    );
  });

  it('resolvePtWebgpuTraceTier honors force lite on a full-capable stub', () => {
    expect(
      resolvePtWebgpuTraceTier(
        deviceWithLimits(PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP, 8),
        'lite',
      ),
    ).toBe('lite');
  });

  it('resolvePtWebgpuTraceTier throws when force full on a lite-only stub', () => {
    expect(() =>
      resolvePtWebgpuTraceTier(deviceWithLimits(10, 4), 'full'),
    ).toThrow(/traceTier=full requested/i);
  });
});
