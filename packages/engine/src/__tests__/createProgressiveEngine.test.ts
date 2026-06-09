import { describe, expect, it } from 'vitest';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from '@vitrum/pt-webgpu';
import { computeProgressiveLimitUnion } from '../createProgressiveEngine.js';

// The createProgressiveEngine factory itself is GPU-only (it mints a real device
// and stands up two real engines); it is validated end-to-end via the wsl-gpu
// harness `progressive-handoff-e2e.ts`. Here we pin only the pure, GPU-free
// limit-union math — the contract the device request and the adapter preflight
// both depend on.
describe('computeProgressiveLimitUnion', () => {
  it('is the per-key MAX of the hybrid-full and pt-webgpu-full floors', () => {
    const union = computeProgressiveLimitUnion();
    expect(union['maxStorageBuffersPerShaderStage']).toBe(
      Math.max(
        HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!,
        PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      ),
    );
    expect(union['maxStorageTexturesPerShaderStage']).toBe(
      Math.max(
        HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!,
        PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
      ),
    );
  });

  it('dominates BOTH backend floors on every key (a union device satisfies each)', () => {
    const union = computeProgressiveLimitUnion();
    const ptFull: Record<string, number> = {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    };
    for (const floor of [HYBRID_WEBGPU_REQUIRED_LIMITS, ptFull]) {
      for (const [key, val] of Object.entries(floor)) {
        expect(union[key]).toBeGreaterThanOrEqual(val);
      }
    }
  });

  it('combines the pt-webgpu buffer floor with the walkaround texture floor', () => {
    const union = computeProgressiveLimitUnion();
    expect(union['maxStorageBuffersPerShaderStage']).toBe(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
    expect(union['maxStorageTexturesPerShaderStage']).toBe(8);
  });

  it('raises the shared-device buffer floor when converged ReSTIR-PT reuse is enabled', () => {
    const union = computeProgressiveLimitUnion({ restirPtReuse: true });
    expect(union['maxStorageBuffersPerShaderStage']).toBe(
      PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
    expect(union['maxStorageTexturesPerShaderStage']).toBe(8);
  });
});
