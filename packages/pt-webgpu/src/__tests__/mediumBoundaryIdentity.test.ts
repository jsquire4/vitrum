import { describe, expect, it } from 'vitest';
import { asMat4, type Mat4, type Scene } from '@vitrum/core';
import {
  BDPT_MEDIUM_BOUNDARY_KIND_ANALYTIC_CPU,
  BDPT_MEDIUM_BOUNDARY_KIND_INVALID_CPU,
  BDPT_MEDIUM_BOUNDARY_KIND_TLAS_CPU,
  transitionBdptMediumStackCpu,
  type BdptMediumBoundaryLayerCpu,
} from '../bdpt/bdptMediumTransportCpu.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import {
  composePtWebgpuPathTraceIntersectionWgsl,
  PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL,
} from '../wgsl/pathTrace/intersection.wgsl.js';
import {
  composePathTraceKernelWgsl,
  PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
} from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

function transform(scale: number): Mat4 {
  return asMat4(new Float32Array([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1]));
}

function instancedVolume(instances: readonly Mat4[]): Scene {
  return {
    primitives: [
      {
        kind: 'instanced-mesh',
        id: 'nested-volume-shells',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0,
          metallic: 0,
          transmission: 1,
          thickness: 1,
          scatteringCoefficient: 0.2,
        },
        instances,
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu exact medium-boundary identity', () => {
  it('keeps repeated-material nested shells distinct and rejects out-of-order exits', () => {
    const outer = {
      matId: 7,
      boundaryKind: BDPT_MEDIUM_BOUNDARY_KIND_TLAS_CPU,
      boundaryIndex: 4,
    };
    const inner = {
      matId: 7,
      boundaryKind: BDPT_MEDIUM_BOUNDARY_KIND_TLAS_CPU,
      boundaryIndex: 9,
    };
    let stack: readonly BdptMediumBoundaryLayerCpu[] = [];

    const enteredOuter = transitionBdptMediumStackCpu(stack, {
      ...outer,
      entering: true,
    });
    expect(enteredOuter.ok).toBe(true);
    if (!enteredOuter.ok) throw new Error('test setup: outer entry rejected');
    stack = enteredOuter.stack;

    const enteredInner = transitionBdptMediumStackCpu(stack, {
      ...inner,
      entering: true,
    });
    expect(enteredInner.ok).toBe(true);
    if (!enteredInner.ok) throw new Error('test setup: inner entry rejected');
    stack = enteredInner.stack;

    const beforeOutOfOrderExit = stack;
    const rejectedOuterExit = transitionBdptMediumStackCpu(stack, {
      ...outer,
      entering: false,
    });
    expect(rejectedOuterExit).toEqual({
      ok: false,
      reason: 'boundary-mismatch',
      stack: beforeOutOfOrderExit,
    });
    expect(rejectedOuterExit.stack).toBe(beforeOutOfOrderExit);

    const exitedInner = transitionBdptMediumStackCpu(stack, {
      ...inner,
      entering: false,
    });
    expect(exitedInner.ok).toBe(true);
    if (!exitedInner.ok) throw new Error('test setup: inner exit rejected');
    stack = exitedInner.stack;
    const exitedOuter = transitionBdptMediumStackCpu(stack, {
      ...outer,
      entering: false,
    });
    expect(exitedOuter).toEqual({ ok: true, stack: [] });
  });

  it('uses unique analytic ids and refuses identity-free merged boundaries', () => {
    const analyticA = {
      matId: 3,
      boundaryKind: BDPT_MEDIUM_BOUNDARY_KIND_ANALYTIC_CPU,
      boundaryIndex: 0,
    };
    const analyticB = { ...analyticA, boundaryIndex: 1 };
    const enteredA = transitionBdptMediumStackCpu([], {
      ...analyticA,
      entering: true,
    });
    expect(enteredA.ok).toBe(true);
    if (!enteredA.ok) throw new Error('test setup: analytic A entry rejected');
    const enteredB = transitionBdptMediumStackCpu(enteredA.stack, {
      ...analyticB,
      entering: true,
    });
    expect(enteredB.ok).toBe(true);
    if (!enteredB.ok) throw new Error('test setup: analytic B entry rejected');
    expect(
      transitionBdptMediumStackCpu(enteredB.stack, {
        ...analyticA,
        entering: false,
      }),
    ).toMatchObject({ ok: false, reason: 'boundary-mismatch' });
    expect(
      transitionBdptMediumStackCpu(enteredB.stack, {
        matId: 3,
        boundaryKind: BDPT_MEDIUM_BOUNDARY_KIND_INVALID_CPU,
        boundaryIndex: BDPT_MEDIUM_BOUNDARY_KIND_INVALID_CPU,
        entering: true,
      }),
    ).toEqual({
      ok: false,
      reason: 'invalid-boundary',
      stack: enteredB.stack,
    });
  });

  it('carries the identity and strict top-of-stack rule through every transport path', () => {
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'return vec2u(MEDIUM_BOUNDARY_KIND_TLAS, instanceIndex);',
    );
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'return vec2u(MEDIUM_BOUNDARY_KIND_ANALYTIC, analyticIndex);',
    );
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'return vec2u(MEDIUM_BOUNDARY_KIND_INVALID);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('boundaryKind: u32,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('boundaryIndex: u32,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'mediumBoundaryMatches(layer.boundaryKind, layer.boundaryIndex, boundary)',
    );

    const nonVolumetricKernel = composePathTraceKernelWgsl({
      volumetricSss: false,
    });
    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      nonVolumetricKernel,
      PT_WEBGPU_MEDIUM_NEE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      SPPM_PHOTON_PASS_WGSL,
    ]) {
      expect(source).toContain('mediumBoundaryIdentity(');
      expect(source).toContain('mediumBoundaryIsValid(');
    }
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'bdptMediumLayerMatchesBoundary(stack[depth - 1u], matId, boundary)',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptMediumLayerMatchesBoundary(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonMediumBoundaryKinds[photonMediumDepth - 1u]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonMediumBoundaryIndices[photonMediumDepth - 1u]');
    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      nonVolumetricKernel,
      PT_WEBGPU_MEDIUM_NEE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      SPPM_PHOTON_PASS_WGSL,
    ]) {
      expect(source).not.toContain('removeIndex');
      expect(source).not.toContain('sideMatchIndex');
    }
  });

  it('gives every renderable full-tier mesh instance a stable TLAS token', () => {
    const packed = buildPackedScene(instancedVolume([transform(1), transform(0.5)]), {
      geometryMode: 'tlas',
    });
    expect(packed.tlasNodes.length).toBeGreaterThan(0);
    expect(packed.tlasInstanceIndices).toHaveLength(2);
    expect(new Set(packed.tlasInstanceIndices).size).toBe(2);
    expect(packed.tlasBlasRoots).toHaveLength(2);
  });

  it('never revives skipped full-tier meshes through the identity-free root-0 fallback', () => {
    const packed = buildPackedScene(instancedVolume([transform(0)]), {
      geometryMode: 'tlas',
    });
    expect(packed.triangleCount).toBe(1);
    expect(packed.tlasNodes).toHaveLength(0);
    expect(packed.warnings.some((warning) => warning.includes('non-invertible'))).toBe(true);

    const binary = composePtWebgpuPathTraceIntersectionWgsl();
    const cwbvh = composePtWebgpuPathTraceIntersectionWgsl({ cwbvhClosest: true });
    for (const source of [binary, cwbvh]) {
      expect(source).toContain('tlasResetSceneHit(hit, tMax);\n    return false;');
      expect(source).not.toContain('return traceMeshBvh(ray, tMin, tMax, true, hit, 0u, true);');
      expect(source).not.toContain(
        'return traceMeshCwbvhClosest(ray, tMin, tMax, hit, 0u, 0u, true);',
      );
    }
  });
});
