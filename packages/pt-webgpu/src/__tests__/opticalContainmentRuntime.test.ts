import { describe, expect, it } from 'vitest';

import {
  groupOpticalBoundaryEventCandidates,
  intersectOpticalTriangleWatertightF32,
  type OpticalBoundaryEventCandidate,
  type OpticalV3,
} from '../../../shared-bvh/src/opticalWatertightTriangle.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from '../wgsl/pathTrace/intersectionCore.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

interface PlaneBoundary {
  readonly z: number;
  readonly boundary: number;
  readonly outward: -1 | 1;
}

function boundaryEvents(
  origin: OpticalV3,
  planes: readonly PlaneBoundary[],
): OpticalBoundaryEventCandidate[] {
  const direction: OpticalV3 = [0, 0, 1];
  const candidates: OpticalBoundaryEventCandidate[] = [];
  for (const plane of planes) {
    const a: OpticalV3 = [-8, -7, plane.z];
    const b: OpticalV3 = [9, -7, plane.z];
    const c: OpticalV3 = [0.5, 10, plane.z];
    const hit = plane.outward > 0
      ? intersectOpticalTriangleWatertightF32(origin, direction, a, b, c)
      : intersectOpticalTriangleWatertightF32(origin, direction, a, c, b);
    if (!hit.hit || hit.side === 0) continue;
    candidates.push({
      t: hit.t,
      encodedBoundaryId: plane.boundary,
      side: hit.side,
    });
  }
  return candidates.sort((left, right) => left.t - right.t);
}

function reconstructContainment(
  events: readonly OpticalBoundaryEventCandidate[],
  stackLimit = 8,
): { readonly valid: boolean; readonly stack: readonly number[] } {
  const entered: number[] = [];
  const contained: number[] = [];
  let cursor = 0;
  while (true) {
    const grouped = groupOpticalBoundaryEventCandidates(
      events.filter((event) => event.t > cursor),
    );
    if (grouped.kind === 'none') {
      return entered.length === 0
        ? { valid: true, stack: contained.reverse() }
        : { valid: false, stack: [] };
    }
    if (grouped.kind === 'invalid-input' || grouped.kind === 'invalid-tie') {
      return { valid: false, stack: [] };
    }
    cursor = grouped.t;
    if (grouped.kind === 'tangent') continue;
    if (grouped.side > 0) {
      if (entered.length >= stackLimit) return { valid: false, stack: [] };
      entered.push(grouped.encodedBoundaryId);
      continue;
    }
    if (entered.length > 0) {
      if (entered.at(-1) !== grouped.encodedBoundaryId) {
        return { valid: false, stack: [] };
      }
      entered.pop();
      continue;
    }
    if (contained.length >= stackLimit) return { valid: false, stack: [] };
    contained.push(grouped.encodedBoundaryId);
  }
}

describe('pt-webgpu optical start-inside reconstruction', () => {
  it('reconstructs nested launch media in outer-to-inner LIFO order', () => {
    const events = boundaryEvents([0, 0, 0], [
      { z: 1, boundary: 3, outward: 1 },
      { z: 2, boundary: 2, outward: 1 },
      { z: 3, boundary: 1, outward: 1 },
    ]);
    expect(events.map((event) => event.side)).toEqual([-1, -1, -1]);
    expect(reconstructContainment(events)).toEqual({
      valid: true,
      stack: [1, 2, 3],
    });
  });

  it('permits more than eight total crossings while bounding only live nesting', () => {
    const planes: PlaneBoundary[] = [];
    for (let boundary = 1; boundary <= 12; boundary += 1) {
      planes.push(
        { z: boundary * 2, boundary, outward: -1 },
        { z: boundary * 2 + 1, boundary, outward: 1 },
      );
    }
    expect(reconstructContainment(boundaryEvents([0, 0, 0], planes))).toEqual({
      valid: true,
      stack: [],
    });
  });

  it('fails closed when the live authored nesting exceeds eight', () => {
    const events = boundaryEvents(
      [0, 0, 0],
      Array.from({ length: 9 }, (_, index) => ({
        z: index + 1,
        boundary: 9 - index,
        outward: 1 as const,
      })),
    );
    expect(reconstructContainment(events)).toEqual({ valid: false, stack: [] });
  });

  it('wires the exact boundary payload into every production launch family', () => {
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'fn opticalContainmentAlongRay(',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'There is deliberately no crossing-count ceiling',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'containedTriIndices[containedDepth] = event.triIndex;',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'containedInstanceIndices[containedDepth] = event.instanceIndex;',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'containedBaryVWs[containedDepth] = event.baryVW;',
    );

    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      SPPM_PHOTON_PASS_WGSL,
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL,
    ]) {
      expect(source).toContain('opticalContainmentAlongRay(');
    }
    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      SPPM_PHOTON_PASS_WGSL,
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL,
    ]) {
      expect(source).toContain('materialAtOpticalBoundary(');
    }
  });
});
