import { describe, expect, it } from 'vitest';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE,
  PT_WEBGPU_LITE_HDRI_STORAGE_BUFFERS_NEEDED,
  PT_WEBGPU_LITE_AREA_LIGHT_STORAGE_BUFFERS_NEEDED,
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  mergeAdapterRequiredLimits,
  ptWebgpuRequiredLimitsForAdapter,
} from '../webgpuLimits.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

describe('webgpuLimits', () => {
  it('PT full limits request the stage-level buffer and texture floors', () => {
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageTexturesPerShaderStage).toBe(
      PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it('mergeAdapterRequiredLimits clamps to adapter caps', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 10,
        maxStorageTexturesPerShaderStage: 4,
      },
    } as GPUAdapter;
    const merged = mergeAdapterRequiredLimits(adapter, {
      maxStorageBuffersPerShaderStage: 23,
      maxStorageTexturesPerShaderStage: 8,
    });
    expect(merged.maxStorageBuffersPerShaderStage).toBe(10);
    expect(merged.maxStorageTexturesPerShaderStage).toBe(4);
  });

  it('ptWebgpuRequiredLimitsForAdapter picks lite caps on SwiftShader-class adapters', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 10,
        maxStorageTexturesPerShaderStage: 4,
      },
    } as GPUAdapter;
    expect(ptWebgpuRequiredLimitsForAdapter(adapter)).toEqual({
      maxStorageBuffersPerShaderStage: PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 4,
    });
  });

  it('ptWebgpuRequiredLimitsForAdapter requests the ReSTIR-PT reuse buffer floor when opted in', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 64,
        maxStorageTexturesPerShaderStage: 8,
      },
    } as GPUAdapter;
    expect(ptWebgpuRequiredLimitsForAdapter(adapter, { restirPtReuse: true })).toEqual({
      maxStorageBuffersPerShaderStage: PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    });
  });

  it('ptWebgpuRequiredLimitsForAdapter falls back to lite when ReSTIR-PT reuse cannot fit', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1,
        maxStorageTexturesPerShaderStage: 8,
      },
    } as GPUAdapter;
    expect(ptWebgpuRequiredLimitsForAdapter(adapter, { restirPtReuse: true })).toEqual({
      maxStorageBuffersPerShaderStage: PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 4,
    });
  });
});

// ── B12 — lite-tier binding-budget proof (fidelity-cliff arithmetic) ─────────
// PINS the storage-buffer arithmetic that proves WHY the lite tier cannot restore
// HDRI importance sampling as storage buffers within its 8-buffer budget, and
// that area-light MIS fits with exactly zero headroom. If a future change frees or
// consumes a lite storage-buffer slot, or moves HDRI to textures, update both the
// constants and this proof in lockstep (see B12 in road-to-100.md).
describe('B12 lite-tier binding-budget proof', () => {
  it('the lite WGSL declares exactly the proven number of group-0 storage buffers', () => {
    // Count `@group(0) @binding(N) var<storage, ...>` declarations in the composed
    // lite trace string — the authoritative source of what the lite layout binds.
    const storageDecls = (
      PT_WEBGPU_TRACE_LITE_WGSL.match(/@group\(0\) @binding\(\d+\) var<storage/g) ?? []
    ).length;
    expect(storageDecls).toBe(PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE); // 7
  });

  it('HDRI importance does NOT fit the lite storage-buffer budget (needs texture packing)', () => {
    expect(
      PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE + PT_WEBGPU_LITE_HDRI_STORAGE_BUFFERS_NEEDED,
    ).toBeGreaterThan(PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE); // 7 + 2 = 9 > 8
  });

  it('area-light MIS fits the lite budget but with zero headroom', () => {
    expect(
      PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE + PT_WEBGPU_LITE_AREA_LIGHT_STORAGE_BUFFERS_NEEDED,
    ).toBe(PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE); // 7 + 1 = 8 == cap
  });
});
