import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMeshAreaLights } from './meshAreaLights.js';

// B4 — MIS-consistency math. The forward emissive-hit MIS weight is only unbiased if
// the NEE solid-angle pdf the forward path RECONSTRUCTS (meshAreaLightForwardPdf in
// the GLSL) is IDENTICAL to the pdf the NEE SAMPLE produced (sampleMeshAreaLight). The
// crux of B4's current design is emitted-power-proportional triangle selection:
//
//   power_tri = luminance(radiance_tri) · area_tri
//   p_select(tri) = power_tri / totalPower         (emitted-power discrete pick)
//   p_area(point | tri) = 1 / area_tri             (uniform on the triangle)
//   p_solidAngle = p_area · dist² / |cosθ_light|   (area→SA Jacobian)
//   p_NEE(ω) = p_select · p_solidAngle
//            = (power_tri/totalPower) · (1/area_tri) · dist²/|cosθ|
//            = luminance(radiance_tri)/totalPower · dist²/|cosθ|
//
// These pure-TS references mirror the GLSL exactly; the test asserts the cancellation
// (so a forward hit can recover the same area density from surf.emission) — the
// property that lets the forward MIS weight be computed without a triangle→index map.

const EPSILON = 1e-6;

function luminance(rgb: readonly [number, number, number]): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/** GLSL sampleMeshAreaLight's returned SA pdf after power/area density reduction. */
function neeSamplePdf(
  distSq: number,
  cosLight: number,
  triPower: number,
  triArea: number,
  totalPower: number,
): number {
  const areaDensity = Math.max(triPower, 0) / (Math.max(triArea, EPSILON) * totalPower);
  return Math.max(areaDensity * distSq / Math.max(Math.abs(cosLight), EPSILON), EPSILON);
}

/** GLSL meshAreaLightForwardPdf — the forward-hit reconstruction. */
function neeForwardPdf(
  distSq: number,
  cosLight: number,
  totalPower: number,
  emission: readonly [number, number, number],
): number {
  if (totalPower <= 0) return 0;
  const areaDensity = Math.max(luminance(emission), 0) / totalPower;
  return areaDensity * distSq / Math.max(Math.abs(cosLight), EPSILON);
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
    warnings: [],
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
  it('forward pdf equals the sample pdf for the same emitted-power density', () => {
    const distSq = 9;
    const cosLight = 0.7;
    const emission: [number, number, number] = [2, 4, 8];
    const triArea = 0.25;
    const triPower = luminance(emission) * triArea;
    const totalPower = 9;
    expect(neeForwardPdf(distSq, cosLight, totalPower, emission)).toBeCloseTo(
      neeSamplePdf(distSq, cosLight, triPower, triArea, totalPower),
      10,
    );
  });

  it('pdf is independent of per-triangle area for equal radiance (the area_tri cancellation)', () => {
    // Two emitters with the SAME radiance but different triangle areas produce the
    // same local area density under power-proportional selection.
    const distSq = 4;
    const cosLight = 0.5;
    const emission: [number, number, number] = [3, 3, 3];
    const totalPower = 12;
    const pdfA = neeSamplePdf(distSq, cosLight, luminance(emission) * 2, 2, totalPower);
    const pdfB = neeSamplePdf(distSq, cosLight, luminance(emission) * 0.25, 0.25, totalPower);
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
    const emission: [number, number, number] = [2, 2, 2];
    const triArea = 0.5;
    const totalPower = 4;
    const bsdfPdf = 0.5;
    const neeRaw = neeSamplePdf(distSq, cosLight, luminance(emission) * triArea, triArea, totalPower);
    const wForwardScaled = misHeuristic(bsdfPdf, neeRaw / lightsDenom);
    const wForwardUnscaledButConsistent = misHeuristic(
      bsdfPdf,
      neeForwardPdf(distSq, cosLight, totalPower, emission) / lightsDenom,
    );
    expect(wForwardScaled).toBeCloseTo(wForwardUnscaledButConsistent, 12);
  });

  it('uses the textured implicit-emitter pack total power for forward/sample pdf parity', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
      magFilter: 'nearest' as const,
      minFilter: 'nearest' as const,
      mipFilter: 'none' as const,
    };
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveIntensity: 3,
        emissiveMap,
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.totalEmissivePower).toBeGreaterThan(0);
    expect(out.data![4]).toBeCloseTo(6, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    const distSq = 7.5;
    const cosLight = 0.42;
    const firstTriArea = out.data![15]!;
    const firstTriPower = out.data![16]!;
    expect(neeForwardPdf(distSq, cosLight, out.totalEmissivePower, [6, 0, 0])).toBeCloseTo(
      neeSamplePdf(distSq, cosLight, firstTriPower, firstTriArea, out.totalEmissivePower),
      5,
    );
  });
});
