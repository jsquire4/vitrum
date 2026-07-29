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
    expect(wgsl).toContain(
      'traceMeshCwbvhClosest(localRay, localTMin, localTMax, &localHit, blasRoot, binaryBlasRoot, true)',
    );
    expect(wgsl).not.toContain('localHit.dist < (*hit).dist');
    expect(wgsl).toContain('fn traceTlasClosestCwbvh(');
    expect(wgsl).toContain('_ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);');
  });

  it('uses transpose(worldToLocal) for instanced normals under rotation and shear', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).toContain('dot(w2l0.xyz, nLocal)');
    expect(wgsl).toContain('dot(w2l1.xyz, nLocal)');
    expect(wgsl).toContain('dot(w2l2.xyz, nLocal)');
    expect(wgsl).not.toContain('let row0 = vec3f(w2l0.x, w2l1.x, w2l2.x)');
    expect(wgsl).not.toContain('dot(row0, nLocal)');
  });

  it('routes visibility through sided CWBVH closest candidates', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).toContain('fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit');
    expect(wgsl).toContain('_ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);');
    expect(wgsl).toContain('let hit = traceClosestRaw(ray, cursor, tMax);');
    expect(wgsl).toContain('materialAcceptsSidedHit(matId, hit.frontFace) &&');
    expect(wgsl).toContain('!materialShadowCastDisabled(matId)');
    expect(wgsl).not.toContain('fn traceTlasAnyCwbvh(');
    expect(wgsl).not.toContain('fn traceMeshCwbvhAny(');
  });

  it('validates paired roots before closest-hit use and restarts canonical traversal with original bounds', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).toContain('cwbvhTlasBlasRoots: array<vec4u>');
    expect(wgsl).toContain('rootPair.x != CWBVH_ROOT_PAIR_MAGIC');
    expect(wgsl).toContain('rootPair.y != tlasBlasRoots[instIdx]');
    expect(wgsl).toContain('rootPair.y >= min(params.bvhNodeCount, arrayLength(&bvhNodes))');
    expect(wgsl).toContain('rootPair.z >= arrayLength(&cwbvhChildCount)');
    expect(wgsl).toContain('rootPair.y * CWBVH_BINARY_ROOT_FACTOR');
    expect(wgsl).toContain('rootPair.z * CWBVH_WIDE_ROOT_FACTOR');
    expect(wgsl).toContain('cHit.status != CWBVH_STATUS_COMPLETE');
    expect(wgsl).toContain(
      'return traceMeshBvh(ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails);',
    );
    expect(wgsl).toContain('let fallbackHit = traceTlasClosest(ray, tMin, tMax, hit);');
    expect(wgsl).toContain('tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;');
    expect(wgsl).not.toContain('prefer occlusion over a light leak');
    expect(wgsl.match(/let rootPair = cwbvhTlasBlasRoots\[instIdx\];/g)).toHaveLength(1);

    const closestPairLoad = wgsl.indexOf('let rootPair = cwbvhTlasBlasRoots[instIdx];');
    const closestPairCheck = wgsl.indexOf('rootPair.x != CWBVH_ROOT_PAIR_MAGIC', closestPairLoad);
    const closestWideUse = wgsl.indexOf('let blasRoot = rootPair.z;', closestPairLoad);
    expect(closestPairLoad).toBeGreaterThan(-1);
    expect(closestPairCheck).toBeGreaterThan(closestPairLoad);
    expect(closestWideUse).toBeGreaterThan(closestPairCheck);
  });

  it('initializes every binary traversal output before an early return', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    const meshStart = wgsl.indexOf('fn traceMeshBvh(');
    const meshEnd = wgsl.indexOf('\nfn traceAnalyticShapes(', meshStart);
    const meshBody = wgsl.slice(meshStart, meshEnd);
    const initialization = meshBody.indexOf('(*hit).didHit = false;');
    const invalidRootReturn = meshBody.indexOf('if (params.bvhNodeCount == 0u');
    expect(initialization).toBeGreaterThan(-1);
    expect(initialization).toBeLessThan(invalidRootReturn);
    expect(meshBody).toContain('(*hit).dist = tMaxBound;');
    expect(meshBody).toContain('(*hit).triIndex = 0u;');
    expect(meshBody).toContain('(*hit).normal = vec3f(0.0, 1.0, 0.0);');
    expect(meshBody).toContain('(*hit).baryVW = vec2f(0.0);');
    expect(meshBody).toContain('(*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;');
    expect(meshBody).toContain('var stack: array<u32, 60>;');
    expect(meshBody).toContain('if (stackPtr + 2u <= 60u)');
    expect(meshBody).toContain('return select(true, (*hit).didHit, closest);');
  });

  it('does not compose a second CWBVH any-hit walker', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(wgsl).not.toContain('fn cwbvhIntersectAny(');
    expect(wgsl).not.toContain('fn traceMeshBinaryAny(');
    expect(wgsl).not.toContain('fn traceMeshCwbvhAny(');
    expect(wgsl).not.toContain('fn traceTlasAnyCwbvh(');

    const anyStart = wgsl.indexOf('fn traceAny(');
    const anyEnd = wgsl.indexOf('\nfn hitMaterialId(', anyStart);
    const anyBody = wgsl.slice(anyStart, anyEnd);
    expect(anyStart).toBeGreaterThan(-1);
    expect(anyEnd).toBeGreaterThan(anyStart);
    expect(anyBody).toContain('let hit = traceClosestRaw(ray, cursor, tMax);');
    expect(anyBody).toContain('materialAcceptsSidedHit(matId, hit.frontFace)');
    expect(anyBody).toContain('!materialShadowCastDisabled(matId)');
    expect(anyBody).toContain('!alphaTestPassThrough(');
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
      'cwbvh-closest',
    );
    gpu.ensurePipeline();
    gpu.ensureAccumResources(1, 1);
    gpu.ensureSppmBuffers(false);
    gpu.buildBindGroups(sceneBuffers);

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
