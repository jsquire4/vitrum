import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { GpuResources } from '../gpuResources.js';
import { buildPackedScene, uploadPackedScene } from '../scene/uploadSceneBuffers.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { createSizeValidatingGpuDeviceStub } from './gpuStub.js';

function oneMeshScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.7, 0.7, 0.7], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function addPipelineStubs(device: GPUDevice): void {
  Object.assign(device as unknown as Record<string, unknown>, {
    createShaderModule: vi.fn(() => ({})),
    createPipelineLayout: vi.fn((desc: { label?: string; bindGroupLayouts: unknown[] }) => ({
      label: desc.label,
      bindGroupLayouts: desc.bindGroupLayouts,
    })),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
  });
}

function boundBuffer(entry: GPUBindGroupEntry): unknown {
  const resource = entry.resource as GPUBufferBinding | GPUBuffer;
  if (typeof resource === 'object' && resource != null && 'buffer' in resource) {
    return resource.buffer;
  }
  return resource;
}

describe('pt-webgpu CWBVH traversal wiring', () => {
  it('keeps the default full and lite shaders on binary BVH traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('cwbvhNodeBounds');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('traceTlasClosestCwbvh');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('cwbvhNodeBounds');
  });

  it('composes an explicit full-tier CWBVH closest-hit variant', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).toContain('fn cwbvhIntersectFirstHitFromRoot(');
    expect(wgsl).toContain('cwbvhIntersectFirstHitRangeFromRoot(');
    expect(wgsl).toContain('tMin,');
    expect(wgsl).toContain('tMaxBound,');
    expect(wgsl).toContain('@group(3) @binding(12) var<storage, read> cwbvhNodeBounds');
    expect(wgsl).toContain('@group(3) @binding(16) var<storage, read> cwbvhTlasBlasRoots');
    expect(wgsl).toContain('fn traceMeshCwbvhClosest(');
    expect(wgsl).toContain('fn traceTlasClosestCwbvh(');
    expect(wgsl).toContain('_ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);');
  });

  it('retains binary any-hit traversal for shadow predicate parity', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).toContain('fn traceTlasAny(ray: Ray, tMin: f32, tMax: f32) -> bool');
    expect(wgsl).toContain('traceMeshBvh(localRay, tMin, localTMax, false, &localHit, blasRoot, false)');
    expect(wgsl).toContain('if (triShadowCastDisabled(t))');
  });

  it('binds uploaded CWBVH buffers into the opt-in full-tier renderer group 3', () => {
    const stub = createSizeValidatingGpuDeviceStub({ maxBufferSize: 64 * 1024 * 1024 });
    addPipelineStubs(stub.device);

    const sceneBuffers = uploadPackedScene(stub.device, buildPackedScene(oneMeshScene()));
    const gpu = new GpuResources(
      stub.device,
      'full',
      false,
      false,
      undefined,
      'pcg',
      'cwbvh-closest-experimental',
    );
    gpu.ensurePipeline();
    gpu.ensureAccumResources(1, 1);
    gpu.ensureBdptEyeStack(1, 1, 1, false);
    gpu.ensureSppmBuffers(false);
    const bdptLightPathBuffer = stub.device.createBuffer({
      label: 'vitrum.pt-webgpu.bdpt.lightPath.placeholder.test',
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    gpu.buildBindGroups(sceneBuffers, () => bdptLightPathBuffer);

    const group3 = stub.bindGroups.find((group) => group.label === 'vitrum.pt-webgpu.pathTrace.bindgroup3.full');
    expect(group3).toBeDefined();
    const byBinding = new Map(group3!.entries.map((entry) => [entry.binding, entry]));
    expect(boundBuffer(byBinding.get(12)!)).toBe(sceneBuffers.cwbvhNodeBoundsBuffer);
    expect(boundBuffer(byBinding.get(13)!)).toBe(sceneBuffers.cwbvhChildBoundsPackedBuffer);
    expect(boundBuffer(byBinding.get(14)!)).toBe(sceneBuffers.cwbvhChildMetaBuffer);
    expect(boundBuffer(byBinding.get(15)!)).toBe(sceneBuffers.cwbvhChildCountBuffer);
    expect(boundBuffer(byBinding.get(16)!)).toBe(sceneBuffers.cwbvhTlasBlasRootsBuffer);
  });
});
