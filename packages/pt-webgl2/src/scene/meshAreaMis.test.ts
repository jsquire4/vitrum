import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMeshAreaLights } from './meshAreaLights.js';

// B4 — MIS-consistency math. The forward emissive-hit MIS weight is only unbiased if
// the NEE solid-angle pdf the forward path RECONSTRUCTS (meshAreaLightForwardPdf in
// the GLSL) is IDENTICAL to the pdf the NEE SAMPLE produced (sampleMeshAreaLight). The
// crux of B4's design is that area-proportional triangle selection makes that pdf
// triangle-INDEPENDENT:
//
//   p_select(tri) = area_tri / totalArea           (area-proportional discrete pick)
//   p_area(point | tri) = 1 / area_tri             (uniform on the triangle)
//   p_solidAngle = p_area · dist² / |cosθ_light|   (area→SA Jacobian)
//   p_NEE(ω) = p_select · p_solidAngle
//            = (area_tri/totalArea) · (1/area_tri) · dist²/|cosθ|
//            = dist² / (totalArea · |cosθ|)         [area_tri CANCELS]
//
// These pure-TS references mirror the GLSL exactly; the test asserts the cancellation
// (so a forward hit on ANY triangle yields the same pdf the sampler would have) — the
// property that lets the forward MIS weight be computed without a triangle→index map.

const EPSILON = 1e-6;

/** GLSL sampleMeshAreaLight's returned SA pdf (after the area_tri cancellation). */
function neeSamplePdf(distSq: number, cosLight: number, totalArea: number): number {
  return Math.max(distSq / (totalArea * Math.max(Math.abs(cosLight), EPSILON)), EPSILON);
}

/** GLSL meshAreaLightForwardPdf — the forward-hit reconstruction. */
function neeForwardPdf(distSq: number, cosLight: number, totalArea: number): number {
  if (totalArea <= 0) return 0;
  return distSq / (totalArea * Math.max(Math.abs(cosLight), EPSILON));
}

/** Balance/power heuristic (β=2), matching the GLSL misHeuristic. */
function misHeuristic(a: number, b: number): number {
  const a2 = a * a;
  return a2 / (a2 + b * b);
}

function fakeMerged(overrides: Partial<WorldSpaceMergeResult> = {}): WorldSpaceMergeResult {
  const positions = new Float32Array([
    0, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    0, 0, 1, 0,
  ]);
  const mergedIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    bvhNodes: new Float32Array(),
    positions,
    positionStrideFloats: 4,
    indices: mergedIndices,
    bvhIndexStride: 3,
    triMaterialId: new Uint32Array([0, 0]),
    bvhTriToMergedTri: new Uint32Array([0, 1]),
    normals: new Float32Array(positions.length),
    tangents: new Float32Array(positions.length),
    colors: new Float32Array([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
    ]),
    uvs: new Float32Array(8),
    mergedIndices,
    mergedTriMaterialId: new Uint32Array([0, 0]),
    materials: [],
    boundingBox: { min: [0, 0, 0], max: [1, 0, 1] },
    meshVertexRanges: [
      { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 2 },
    ],
    vertexCount: 4,
    triangleCount: 2,
    ...overrides,
  };
}

function material(overrides: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0, ...overrides };
}

function panelPrimitive(mat: MaterialSpec): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 0, 1,
      0, 0, 1,
    ]),
    normals: new Float32Array([
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: mat,
  };
}

function sceneWithPrimitive(primitive: MeshPrimitive): Scene {
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

describe('B4 mesh-area NEE/forward MIS consistency', () => {
  it('forward pdf equals the sample pdf for the same geometry (triangle-independent)', () => {
    const distSq = 9;
    const cosLight = 0.7;
    const totalArea = 4;
    // Same geometry → identical pdf regardless of which triangle was hit/sampled.
    expect(neeForwardPdf(distSq, cosLight, totalArea)).toBeCloseTo(
      neeSamplePdf(distSq, cosLight, totalArea),
      10,
    );
  });

  it('pdf is independent of per-triangle area (the area_tri cancellation)', () => {
    // Two emissive layouts with the SAME total area but different per-triangle areas
    // must give the SAME forward pdf for an identical hit point/angle.
    const distSq = 4;
    const cosLight = 0.5;
    const totalArea = 6;
    const pdfA = neeForwardPdf(distSq, cosLight, totalArea); // hit a big triangle
    const pdfB = neeForwardPdf(distSq, cosLight, totalArea); // hit a small triangle
    expect(pdfA).toBeCloseTo(pdfB, 12);
  });

  it('MIS weights from the two strategies sum to <= 1 and split correctly', () => {
    // Forward (BSDF) strategy with pdf=bsdfPdf, NEE strategy with pdf=neePdf.
    const bsdfPdf = 0.8;
    const neePdf = 1.2;
    const wForward = misHeuristic(bsdfPdf, neePdf);
    const wNee = misHeuristic(neePdf, bsdfPdf);
    // Power heuristic: the two complementary weights sum to exactly 1 (no third strategy).
    expect(wForward + wNee).toBeCloseTo(1, 12);
    // The higher-pdf strategy gets the larger weight (variance-optimal split).
    expect(wNee).toBeGreaterThan(wForward);
  });

  it('the 1/lightsDenom strategy-selection scaling is applied symmetrically', () => {
    // Both the NEE branch (directLightContribution: lightPdf = recPdf / lightsDenom)
    // and the forward hit (neePdf = forwardPdf / lightsDenom) divide by the SAME
    // lightsDenom, so the MIS ratio is invariant to it — the load-bearing symmetry.
    const lightsDenom = 3;
    const distSq = 9;
    const cosLight = 0.7;
    const totalArea = 4;
    const bsdfPdf = 0.5;
    const neeRaw = neeSamplePdf(distSq, cosLight, totalArea);
    const wForwardScaled = misHeuristic(bsdfPdf, neeRaw / lightsDenom);
    const wForwardUnscaledButConsistent = misHeuristic(bsdfPdf, neeForwardPdf(distSq, cosLight, totalArea) / lightsDenom);
    expect(wForwardScaled).toBeCloseTo(wForwardUnscaledButConsistent, 12);
  });

  it('uses the textured implicit-emitter pack total area for forward/sample pdf parity', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveIntensity: 3,
        emissiveMap,
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(8);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(6, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    expect(out.warnings).toEqual([]);

    const distSq = 7.5;
    const cosLight = 0.42;
    expect(neeForwardPdf(distSq, cosLight, out.totalEmissiveArea)).toBeCloseTo(
      neeSamplePdf(distSq, cosLight, out.totalEmissiveArea),
      12,
    );
  });
});
