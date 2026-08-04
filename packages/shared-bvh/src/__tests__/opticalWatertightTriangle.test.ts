import { describe, expect, it } from 'vitest';
import {
  createOpticalSourceFeature,
  groupOpticalBoundaryEventCandidates,
  intersectOpticalTriangleWatertightF32,
  opticalSourceFeatureSuppressesCandidate,
  type OpticalSourceFeatureCandidate,
  type OpticalV3,
} from '../opticalWatertightTriangle.js';

const ORIGIN: OpticalV3 = [0, 0, 0];
const FORWARD: OpticalV3 = [0, 0, 1];

function nextF32(value: number): number {
  const floats = new Float32Array([value]);
  const bits = new Uint32Array(floats.buffer);
  bits[0] = bits[0]! + 1;
  return floats[0]!;
}

function candidate(
  triangleIndex: number,
  vertices: readonly [OpticalV3, OpticalV3, OpticalV3],
  overrides: Partial<OpticalSourceFeatureCandidate> = {},
): OpticalSourceFeatureCandidate {
  return {
    encodedBoundaryId: 7,
    representedPrimitiveInstanceId: 3,
    triangleIndex,
    vertices,
    ...overrides,
  };
}

describe('optical watertight triangle oracle', () => {
  it('gives a planar face diagonal exact tied t and one uniform crossing', () => {
    const first = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD,
      [-1, -1, 1], [1, -1, 1], [1, 1, 1],
    );
    const second = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD,
      [-1, -1, 1], [1, 1, 1], [-1, 1, 1],
    );
    expect(first.hit).toBe(true);
    expect(second.hit).toBe(true);
    expect(Object.is(first.t, second.t)).toBe(true);
    expect(first.zeroEdgeMask).not.toBe(0);
    expect(second.zeroEdgeMask).not.toBe(0);
    expect(groupOpticalBoundaryEventCandidates([
      { t: first.t, encodedBoundaryId: 1, side: first.side as -1 | 1 },
      { t: second.t, encodedBoundaryId: 1, side: second.side as -1 | 1 },
    ])).toEqual({ kind: 'crossing', t: 1, encodedBoundaryId: 1, side: -1 });
  });

  it('gives convex edge and vertex incident faces exact tied t', () => {
    const edgeX = intersectOpticalTriangleWatertightF32(
      [2, 2, 0], [-1, -1, 0],
      [1, -1, -1], [1, 1, -1], [1, 1, 1],
    );
    const edgeY = intersectOpticalTriangleWatertightF32(
      [2, 2, 0], [-1, -1, 0],
      [-1, 1, -1], [1, 1, 1], [1, 1, -1],
    );
    expect([edgeX.hit, edgeY.hit]).toEqual([true, true]);
    expect(Object.is(edgeX.t, edgeY.t)).toBe(true);
    expect([edgeX.side, edgeY.side]).toEqual([1, 1]);
    expect(groupOpticalBoundaryEventCandidates([
      { t: edgeX.t, encodedBoundaryId: 2, side: edgeX.side as -1 | 1 },
      { t: edgeY.t, encodedBoundaryId: 2, side: edgeY.side as -1 | 1 },
    ])).toMatchObject({ kind: 'crossing', t: 1, encodedBoundaryId: 2, side: 1 });

    const vertexFaces = [
      [[1, -1, -1], [1, 1, -1], [1, 1, 1]],
      [[-1, 1, -1], [1, 1, 1], [1, 1, -1]],
      [[-1, -1, 1], [1, -1, 1], [1, 1, 1]],
    ] as const;
    const vertexHits = vertexFaces.map(([a, b, c]) =>
      intersectOpticalTriangleWatertightF32([2, 2, 2], [-1, -1, -1], a, b, c));
    expect(vertexHits.every((hit) => hit.hit && hit.zeroEdgeMask >= 3)).toBe(true);
    expect(new Set(vertexHits.map((hit) => hit.t)).size).toBe(1);
    expect(new Set(vertexHits.map((hit) => hit.side))).toEqual(new Set([1]));
  });

  it('classifies silhouette edge and vertex ties as tangencies', () => {
    const edgeFront = intersectOpticalTriangleWatertightF32(
      [2, 0, 0], [-1, 1, 0],
      [1, -1, -1], [1, 1, -1], [1, 1, 1],
    );
    const edgeBack = intersectOpticalTriangleWatertightF32(
      [2, 0, 0], [-1, 1, 0],
      [-1, 1, -1], [1, 1, 1], [1, 1, -1],
    );
    expect([edgeFront.side, edgeBack.side]).toEqual([1, -1]);
    expect(Object.is(edgeFront.t, edgeBack.t)).toBe(true);
    expect(groupOpticalBoundaryEventCandidates([
      { t: edgeFront.t, encodedBoundaryId: 3, side: 1 },
      { t: edgeBack.t, encodedBoundaryId: 3, side: -1 },
    ])).toEqual({ kind: 'tangent', t: 1, encodedBoundaryId: 3 });

    expect(groupOpticalBoundaryEventCandidates([
      { t: 1, encodedBoundaryId: 3, side: 1 },
      { t: 1, encodedBoundaryId: 3, side: 1 },
      { t: 1, encodedBoundaryId: 3, side: -1 },
      { t: 1, encodedBoundaryId: 3, side: -1 },
    ])).toEqual({ kind: 'tangent', t: 1, encodedBoundaryId: 3 });
  });

  it('preserves a same-side concave manifold edge as one crossing', () => {
    const event = groupOpticalBoundaryEventCandidates([
      { t: 4, encodedBoundaryId: 11, side: -1 },
      { t: 4, encodedBoundaryId: 11, side: -1 },
    ]);
    expect(event).toEqual({ kind: 'crossing', t: 4, encodedBoundaryId: 11, side: -1 });
  });

  it('keeps adjacent f32 nested boundaries as two distinct events', () => {
    const outerT = 1;
    const innerT = nextF32(outerT);
    const outer = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD, [-1, -1, outerT], [1, -1, outerT], [0, 1, outerT],
    );
    const inner = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD, [-1, -1, innerT], [1, -1, innerT], [0, 1, innerT], outer.t,
    );
    expect(outer.hit).toBe(true);
    expect(inner.hit).toBe(true);
    expect(outer.t).toBe(outerT);
    expect(inner.t).toBe(innerT);
    expect(groupOpticalBoundaryEventCandidates([
      { t: outer.t, encodedBoundaryId: 4, side: outer.side as -1 | 1 },
      { t: inner.t, encodedBoundaryId: 5, side: inner.side as -1 | 1 },
    ])).toMatchObject({ kind: 'crossing', t: outerT, encodedBoundaryId: 4 });
  });

  it('carries one canonical shared-edge point across unrelated third vertices and a one-ULP layer', () => {
    // At this scale the subtraction/projection/interpolation sequence performs
    // substantial f32 cancellation. The incident faces deliberately have
    // third vertices at radically different distances, so reconstructing the
    // edge point from either face's barycentrics is not a shared calculation.
    const base = 67_108_864;
    const edgeA: OpticalV3 = [base - 64, -base - 32, 1];
    const edgeB: OpticalV3 = [base + 128, -base + 64, 1];
    const origin: OpticalV3 = [base, -base, 0];
    const nearA = intersectOpticalTriangleWatertightF32(
      origin, FORWARD,
      edgeA, edgeB, [base - 8_192, -base + 16_777_216, 1],
    );
    const nearB = intersectOpticalTriangleWatertightF32(
      origin, FORWARD,
      edgeB, edgeA, [base + 33_554_432, -base - 4_096, 1],
    );
    expect([nearA.hit, nearB.hit]).toEqual([true, true]);
    expect(nearA.zeroEdgeMask).not.toBe(0);
    expect(nearB.zeroEdgeMask).not.toBe(0);
    expect(nearA.point).toEqual(nearB.point);
    expect(Object.is(nearA.t, nearB.t)).toBe(true);

    const nextZ = nextF32(1);
    const liftedA: OpticalV3 = [edgeA[0], edgeA[1], nextZ];
    const liftedB: OpticalV3 = [edgeB[0], edgeB[1], nextZ];
    const lifted = intersectOpticalTriangleWatertightF32(
      origin, FORWARD,
      liftedA, liftedB, [base - 8_192, -base + 16_777_216, nextZ],
      nearA.t,
    );
    expect(lifted.hit).toBe(true);
    expect(lifted.t).toBe(nextZ);
    expect(lifted.point[2]).toBe(nextZ);
    expect(lifted.point[2]).not.toBe(nearA.point[2]);
  });

  it('uses exact edge zeros and an exclusive minT', () => {
    const nearEdge = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD,
      [0, 0, 1], [1, 0, 1], [0, 1, 1],
      0,
    );
    expect(nearEdge.hit).toBe(true);
    expect(nearEdge.zeroEdgeMask).toBe(6);

    const interiorNearEdge = intersectOpticalTriangleWatertightF32(
      [0.5, 0.499_999_94, 0], FORWARD,
      [0, 0, 1], [1, 0, 1], [0, 1, 1],
    );
    expect(interiorNearEdge.hit).toBe(true);
    expect(interiorNearEdge.zeroEdgeMask).toBe(0);

    const exact = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD, [-1, -1, 1], [1, -1, 1], [0, 1, 1], 1,
    );
    const next = nextF32(1);
    const immediatelyAfter = intersectOpticalTriangleWatertightF32(
      ORIGIN, FORWARD, [-1, -1, next], [1, -1, next], [0, 1, next], 1,
    );
    expect(exact.hit).toBe(false);
    expect(immediatelyAfter.hit).toBe(true);
    expect(immediatelyAfter.t).toBe(next);
  });

  it('keeps side classification finite when unscaled edge cross products overflow', () => {
    const hit = intersectOpticalTriangleWatertightF32(
      ORIGIN,
      [1, 0, 0],
      [1, 0, 0],
      [1e30, 1e10, 0],
      [1, 0, 1e10],
    );
    expect(hit.hit).toBe(true);
    expect(hit.t).toBe(1);
    expect(hit.side).toBe(-1);
  });
});

describe('exact optical boundary event grouping', () => {
  it('fails closed for distinct IDs and unbalanced mixed sides at equal t', () => {
    expect(groupOpticalBoundaryEventCandidates([
      { t: 2, encodedBoundaryId: 1, side: 1 },
      { t: 2, encodedBoundaryId: 2, side: 1 },
    ])).toEqual({ kind: 'invalid-tie', t: 2 });
    expect(groupOpticalBoundaryEventCandidates([
      { t: 2, encodedBoundaryId: 1, side: 1 },
      { t: 2, encodedBoundaryId: 1, side: 1 },
      { t: 2, encodedBoundaryId: 1, side: -1 },
    ])).toEqual({ kind: 'invalid-tie', t: 2 });
  });

  it('normalizes output t to f32 and rejects invalid candidate parameters', () => {
    const onePlusF64Only = 1 + Number.EPSILON;
    expect(groupOpticalBoundaryEventCandidates([
      { t: onePlusF64Only, encodedBoundaryId: 1, side: 1 },
      { t: 1, encodedBoundaryId: 1, side: 1 },
    ])).toEqual({ kind: 'crossing', t: 1, encodedBoundaryId: 1, side: 1 });
    expect(groupOpticalBoundaryEventCandidates([
      { t: Number.NaN, encodedBoundaryId: 1, side: 1 },
    ])).toEqual({ kind: 'invalid-input' });
    expect(groupOpticalBoundaryEventCandidates([
      { t: 0, encodedBoundaryId: 1, side: 1 },
    ])).toEqual({ kind: 'invalid-input' });
  });
});

describe('optical source-feature suppression', () => {
  const a: OpticalV3 = [0, 0, 1];
  const b: OpticalV3 = [1, 0, 1];
  const c: OpticalV3 = [0, 1, 1];

  it('keeps a face token scoped to packed triangle and instance', () => {
    const source = candidate(5, [a, b, c]);
    const feature = createOpticalSourceFeature(source, 0);
    expect(opticalSourceFeatureSuppressesCandidate(feature, source)).toBe(true);
    expect(opticalSourceFeatureSuppressesCandidate(feature, candidate(6, [a, b, c]))).toBe(false);
    expect(opticalSourceFeatureSuppressesCandidate(
      feature,
      candidate(5, [a, b, c], { representedPrimitiveInstanceId: 4 }),
    )).toBe(false);
  });

  it('scopes boundary-zero thin sheets by represented primitive-instance identity', () => {
    const source = candidate(5, [a, b, c], {
      encodedBoundaryId: 0,
      representedPrimitiveInstanceId: 1,
    });
    const feature = createOpticalSourceFeature(source, 4);
    const unrelatedSheet = candidate(6, [a, b, [0, -1, 1]], {
      encodedBoundaryId: 0,
      representedPrimitiveInstanceId: 2,
    });
    expect(opticalSourceFeatureSuppressesCandidate(feature, unrelatedSheet)).toBe(false);
  });

  it('suppresses a duplicated-index UV seam across the whole exact edge feature', () => {
    const source = candidate(5, [a, b, c]);
    const feature = createOpticalSourceFeature(source, 4);
    const seamNeighbor = candidate(6, [[1, 0, 1], [0, 0, 1], [0, -1, 1]]);
    expect(feature.kind).toBe('edge');
    expect(opticalSourceFeatureSuppressesCandidate(feature, seamNeighbor)).toBe(true);
  });

  it('does not suppress an arbitrarily close distinct opposite face', () => {
    const source = candidate(5, [a, b, c]);
    const feature = createOpticalSourceFeature(source, 0);
    const z = nextF32(1);
    const opposite = candidate(6, [[0, 0, z], [0, 1, z], [1, 0, z]]);
    expect(opticalSourceFeatureSuppressesCandidate(feature, opposite)).toBe(false);
  });

  it('treats a near-edge interior hit as a face, not an edge', () => {
    const source = candidate(5, [a, b, c]);
    const feature = createOpticalSourceFeature(source, 0);
    expect(feature.kind).toBe('face');
    expect(opticalSourceFeatureSuppressesCandidate(
      feature,
      candidate(6, [a, b, [0, -1, 1]]),
    )).toBe(false);
  });

  it('rejects impossible exact zero-edge masks', () => {
    expect(() => createOpticalSourceFeature(candidate(5, [a, b, c]), 7)).toThrow(RangeError);
    expect(() => createOpticalSourceFeature(candidate(5, [a, b, c], {
      representedPrimitiveInstanceId: 0,
    }), 0)).toThrow(RangeError);
  });
});
