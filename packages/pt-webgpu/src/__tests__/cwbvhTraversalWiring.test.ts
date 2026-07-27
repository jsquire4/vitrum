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
    expect(wgsl).not.toContain('if (traceTlasAnyCwbvh(ray, tMin, tMax))');
  });

  it('validates paired roots before use and restarts canonical traversal with original bounds', () => {
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
    expect(wgsl).toContain(
      'return traceMeshBinaryAny(ray, tMin, tMaxBound, binaryRootNode);',
    );
    expect(wgsl).toContain('let fallbackHit = traceTlasClosest(ray, tMin, tMax, hit);');
    expect(wgsl).toContain('let fallbackHit = traceTlasAny(ray, tMin, tMax);');
    expect(wgsl).toContain('tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;');
    expect(wgsl).not.toContain('prefer occlusion over a light leak');

    const closestPairLoad = wgsl.indexOf('let rootPair = cwbvhTlasBlasRoots[instIdx];');
    const closestPairCheck = wgsl.indexOf('rootPair.x != CWBVH_ROOT_PAIR_MAGIC', closestPairLoad);
    const closestWideUse = wgsl.indexOf('let blasRoot = rootPair.z;', closestPairLoad);
    expect(closestPairLoad).toBeGreaterThan(-1);
    expect(closestPairCheck).toBeGreaterThan(closestPairLoad);
    expect(closestWideUse).toBeGreaterThan(closestPairCheck);

    const anyPairLoad = wgsl.indexOf('let rootPair = cwbvhTlasBlasRoots[instIdx];', closestPairLoad + 1);
    const anyPairCheck = wgsl.indexOf('rootPair.x != CWBVH_ROOT_PAIR_MAGIC', anyPairLoad);
    const anyWideUse = wgsl.indexOf('let blasRoot = rootPair.z;', anyPairLoad);
    expect(anyPairCheck).toBeGreaterThan(anyPairLoad);
    expect(anyWideUse).toBeGreaterThan(anyPairCheck);
  });

  it('initializes every binary traversal output and falls back exactly once on CWBVH stack overflow', () => {
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

    const anyStart = wgsl.indexOf('fn traceMeshCwbvhAny(');
    const anyEnd = wgsl.indexOf('\nfn traceTlasClosestCwbvh(', anyStart);
    const anyBody = wgsl.slice(anyStart, anyEnd);
    const overflowStart = anyBody.indexOf('if (stackPtr >= CWBVH_INTERSECT_STACK_DEPTH)');
    const overflowEnd = anyBody.indexOf('stack[stackPtr] = childInfo.indexOrOffset;', overflowStart);
    const overflowBranch = anyBody.slice(overflowStart, overflowEnd);
    expect(overflowStart).toBeGreaterThan(-1);
    expect(overflowEnd).toBeGreaterThan(overflowStart);
    expect(overflowBranch.match(/return traceMeshBinaryAny\(/g)).toHaveLength(1);
    expect(overflowBranch).not.toContain('return false');
  });

  it('restarts binary any-hit for every corrupt live-child shape', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    const anyStart = wgsl.indexOf('fn traceMeshCwbvhAny(');
    const anyEnd = wgsl.indexOf('\nfn traceTlasClosestCwbvh(', anyStart);
    const anyBody = wgsl.slice(anyStart, anyEnd);
    expect(anyBody).toContain('if (!cwbvhBoundsAreValid(parentMin, parentMax))');
    expect(anyBody).toContain('if (!cwbvhBoundsAreValid(bounds.boundsMin, bounds.boundsMax))');
    expect(anyBody).toContain('childInfo.triCount == 0u ||');
    expect(anyBody).not.toMatch(
      /if \(childInfo\.kind == CWBVH_CHILD_EMPTY\) \{\s*continue;/,
    );
    for (const marker of [
      'if (!cwbvhBoundsAreValid(parentMin, parentMax))',
      'if (childInfo.kind == CWBVH_CHILD_EMPTY)',
      'if (!cwbvhBoundsAreValid(bounds.boundsMin, bounds.boundsMax))',
      'childInfo.triCount == 0u ||',
    ]) {
      const start = anyBody.indexOf(marker);
      expect(start).toBeGreaterThan(-1);
      expect(anyBody.slice(start, start + 420)).toContain(
        'return traceMeshBinaryAny(ray, tMin, tMaxBound, binaryRootNode);',
      );
    }
  });

  it('uses overflow-safe CWBVH capacity and leaf-range guards before any-hit loads', () => {
    const wgsl = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    const anyStart = wgsl.indexOf('fn traceMeshCwbvhAny(');
    const anyEnd = wgsl.indexOf('\nfn traceTlasClosestCwbvh(', anyStart);
    const anyBody = wgsl.slice(anyStart, anyEnd);
    expect(anyBody).toContain(
      'nodeCount > arrayLength(&cwbvhChildBoundsPacked) / (CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32)',
    );
    expect(anyBody).toContain(
      'nodeCount > arrayLength(&cwbvhChildMeta) / CWBVH_CHILDREN',
    );
    expect(anyBody).not.toContain('arrayLength(&cwbvhChildBoundsPacked) < nodeCount *');
    expect(anyBody).toContain('childInfo.indexOrOffset > triangleCount');
    expect(anyBody).toContain('childInfo.triCount > triangleCount - childInfo.indexOrOffset');
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
