import { describe, expect, it } from 'vitest';
import { asMat4 } from '@vitrum/core';
import {
  bdptExplicitConnectionStrategyIsValid,
  maskBDPTExplicitConnectionStrategyPDFs,
} from '@vitrum/shared-samplers';
import {
  evaluateBdptCameraSplatContribution,
  projectBdptCameraSplat,
} from '../bdpt/bdptCameraSplatCpu.js';
import {
  assembleMergedConnectionPath,
  assembleMergedCameraSplatPath,
  bdptCameraSplatMisFull,
  buildStrategyPdfs,
  powerHeuristicWeight,
  type LightStackVertex,
  type Vec3,
} from '../bdpt/bdptConnectionMisFull.js';
import { invertMat4 } from '../math/mat4.js';

// 90° vertical FOV, aspect=2, near=1, far=10, camera at the origin looking -Z.
// The one-unit image-plane rectangle is x=[-2,2], y=[-1,1], so A=8.
const VIEW_PROJECTION = [
  0.5, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, -11 / 9, -1,
  0, 0, -20 / 9, 0,
] as const;
const INVERSE_VIEW_PROJECTION = [
  2, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, -9 / 20,
  0, 0, -1, 11 / 20,
] as const;

// Same near/far/extents as above, but with the one-unit image-plane window
// shifted from x=[-2,2] to x=[-1,3]. Its raster-center ray points diagonally,
// while the perspective camera's optical forward remains world -Z.
const OFF_AXIS_VIEW_PROJECTION = [
  0.5, 0, 0, 0,
  0, 1, 0, 0,
  0.5, 0, -11 / 9, -1,
  0, 0, -20 / 9, 0,
] as const;
const OFF_AXIS_INVERSE_VIEW_PROJECTION = invertMat4(
  asMat4(new Float32Array(OFF_AXIS_VIEW_PROJECTION)),
)!;

function lightFixture(): {
  lightChain: LightStackVertex[];
  camera: { position: Vec3; normal: Vec3 };
  lightBrdfPdf: (wo: Vec3, wi: Vec3) => number;
} {
  return {
    lightChain: [
      {
        position: [-1, 2, -3],
        normal: [0, -1, 0],
        pdfFwd: 0.31,
        pdfRev: 0.17,
        isSpecular: false,
      },
      {
        position: [0, 0, -2],
        normal: [0, 0, 1],
        pdfFwd: 0.43,
        pdfRev: 0.29,
        isSpecular: false,
      },
    ],
    camera: {
      position: [0, 0, 0],
      normal: [0, 0, -1],
    },
    lightBrdfPdf: (wo, wi) =>
      0.1 +
      0.2 * Math.abs(wo[2]) +
      0.3 * Math.abs(wi[1]) +
      0.05 * Math.abs(wo[0] * wi[2]),
  };
}

describe('BDPT t=1 camera-splat CPU algebra', () => {
  it('projects the center ray and reproduces PerspectiveCamera Sample_Wi / pdf', () => {
    const projection = projectBdptCameraSplat({
      viewProjection: VIEW_PROJECTION,
      inverseViewProjection: INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [0, 0, -2],
      width: 4,
      height: 2,
    });
    expect(projection).not.toBeNull();
    expect(projection!.pixelX).toBe(2);
    expect(projection!.pixelY).toBe(1);
    expect(projection!.pixelIndex).toBe(6);
    expect(projection!.cameraForward).toEqual([0, 0, -1]);
    expect(projection!.imagePlaneArea).toBeCloseTo(8, 13);
    expect(projection!.cosTheta).toBeCloseTo(1, 13);
    expect(projection!.distanceSquared).toBeCloseTo(4, 13);
    expect(projection!.cameraDirectionalPdf).toBeCloseTo(1 / 8, 13);
    expect(projection!.sampleWiOverPdf).toBeCloseTo(1 / 32, 13);
  });

  it('uses the cos^-3 perspective density off axis and rejects sensor misses', () => {
    const projection = projectBdptCameraSplat({
      viewProjection: VIEW_PROJECTION,
      inverseViewProjection: INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [1, 0, -2],
      width: 8,
      height: 4,
    });
    const cosTheta = 2 / Math.sqrt(5);
    expect(projection).not.toBeNull();
    expect(projection!.cosTheta).toBeCloseTo(cosTheta, 13);
    expect(projection!.cameraDirectionalPdf).toBeCloseTo(
      1 / (8 * cosTheta ** 3),
      13,
    );
    expect(projectBdptCameraSplat({
      viewProjection: VIEW_PROJECTION,
      inverseViewProjection: INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [0, 0, 2],
      width: 8,
      height: 4,
    })).toBeNull();
    expect(projectBdptCameraSplat({
      viewProjection: VIEW_PROJECTION,
      inverseViewProjection: INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [5, 0, -2],
      width: 8,
      height: 4,
    })).toBeNull();
  });

  it('derives optical forward from the image plane for asymmetric projections', () => {
    const opticalAxis = projectBdptCameraSplat({
      viewProjection: OFF_AXIS_VIEW_PROJECTION,
      inverseViewProjection: OFF_AXIS_INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [0, 0, -2],
      width: 8,
      height: 4,
    });
    expect(opticalAxis).not.toBeNull();
    expect(opticalAxis!.pixelX).toBe(2);
    expect(opticalAxis!.pixelY).toBe(2);
    expect(opticalAxis!.cameraForward[0]).toBeCloseTo(0, 7);
    expect(opticalAxis!.cameraForward[1]).toBeCloseTo(0, 7);
    expect(opticalAxis!.cameraForward[2]).toBeCloseTo(-1, 7);
    expect(opticalAxis!.imagePlaneArea).toBeCloseTo(8, 6);
    expect(opticalAxis!.cosTheta).toBeCloseTo(1, 7);
    expect(opticalAxis!.cameraDirectionalPdf).toBeCloseTo(1 / 8, 6);

    const rasterCenter = projectBdptCameraSplat({
      viewProjection: OFF_AXIS_VIEW_PROJECTION,
      inverseViewProjection: OFF_AXIS_INVERSE_VIEW_PROJECTION,
      cameraPosition: [0, 0, 0],
      vertexPosition: [2, 0, -2],
      width: 8,
      height: 4,
    });
    const centerCos = 1 / Math.sqrt(2);
    expect(rasterCenter).not.toBeNull();
    expect(rasterCenter!.pixelX).toBe(4);
    expect(rasterCenter!.cosTheta).toBeCloseTo(centerCos, 7);
    expect(rasterCenter!.cameraDirectionalPdf).toBeCloseTo(
      1 / (8 * centerCos ** 3),
      6,
    );
  });

  it('multiplies throughput, light scatter, receiver cosine, camera measure, and MIS once', () => {
    const contribution = evaluateBdptCameraSplatContribution({
      lightThroughput: [2, 3, 5],
      lightScatter: [0.25, 0.5, 0.75],
      surfaceCosine: 0.8,
      sampleWiOverPdf: 0.125,
      misWeight: 0.4,
    });
    const expected = [
      2 * 0.25 * 0.8 * 0.125 * 0.4,
      3 * 0.5 * 0.8 * 0.125 * 0.4,
      5 * 0.75 * 0.8 * 0.125 * 0.4,
    ];
    for (let channel = 0; channel < 3; channel += 1) {
      expect(contribution[channel]).toBeCloseTo(expected[channel]!, 14);
    }
  });

  it('assembles s=n-1 with the two camera-connection reverse-density overrides', () => {
    const fixture = lightFixture();
    const cameraDirectionalPdf = 0.125;
    const { vertices, selectedS } = assembleMergedCameraSplatPath({
      ...fixture,
      cameraDirectionalPdf,
    });
    expect(vertices).toHaveLength(3);
    expect(selectedS).toBe(2);
    expect(vertices[1]!.pdfRev).toBe(cameraDirectionalPdf);
    const lcToCamera: Vec3 = [0, 0, 1];
    const toPrev = [-1, 2, -1] as const;
    const invLen = 1 / Math.hypot(...toPrev);
    const lcToPrev: Vec3 = [
      toPrev[0] * invLen,
      toPrev[1] * invLen,
      toPrev[2] * invLen,
    ];
    expect(vertices[0]!.pdfRev).toBeCloseTo(
      fixture.lightBrdfPdf(lcToCamera, lcToPrev),
      13,
    );
  });

  it('admits t=1 in the bounded explicit family and keeps s=0 separately owned', () => {
    const limits = { maxLightVertices: 8, maxEyeVertices: 8 };
    expect(bdptExplicitConnectionStrategyIsValid(3, 0, limits)).toBe(false);
    expect(bdptExplicitConnectionStrategyIsValid(3, 1, limits)).toBe(true);
    expect(bdptExplicitConnectionStrategyIsValid(3, 2, limits)).toBe(true);
    expect(bdptExplicitConnectionStrategyIsValid(3, 3, limits)).toBe(false);
    expect(
      bdptExplicitConnectionStrategyIsValid(
        4,
        3,
        { maxLightVertices: 2, maxEyeVertices: 8 },
      ),
    ).toBe(false);
  });

  it('includes the t=1 strategy in the denominator and is pRef-scale invariant', () => {
    const fixture = lightFixture();
    const assembled = assembleMergedCameraSplatPath({
      ...fixture,
      cameraDirectionalPdf: 0.125,
    });
    const mask = (pRef: number) => maskBDPTExplicitConnectionStrategyPDFs(
      buildStrategyPdfs(assembled.vertices, assembled.selectedS, pRef),
      { maxLightVertices: 8, maxEyeVertices: 8 },
    );
    const pdfs = mask(0.013);
    expect(pdfs[0]).toBe(0);
    expect(pdfs[1]).toBeGreaterThan(0);
    expect(pdfs[2]).toBeGreaterThan(0);
    const weightSum =
      powerHeuristicWeight(pdfs, 1, 2) +
      powerHeuristicWeight(pdfs, 2, 2);
    expect(weightSum).toBeCloseTo(1, 13);

    const wA = bdptCameraSplatMisFull({
      ...fixture,
      cameraDirectionalPdf: 0.125,
      pRef: 0.013,
    });
    const wB = bdptCameraSplatMisFull({
      ...fixture,
      cameraDirectionalPdf: 0.125,
      pRef: 7.9,
    });
    expect(wA).toBeGreaterThan(0);
    expect(wA).toBeLessThan(1);
    expect(wA).toBeCloseTo(wB, 13);
  });

  it('uses the same camera directional density from both sides of the MIS partition', () => {
    const camera = {
      position: [0, 0, 0] as Vec3,
      normal: [0, 0, -1] as Vec3,
    };
    const l0Position: Vec3 = [0, 1, -3];
    const l1Position: Vec3 = [0, 0, -2];
    const endpointToSurface: Vec3 = [
      0,
      -1 / Math.sqrt(2),
      1 / Math.sqrt(2),
    ];
    const cameraDirectionalPdf = 0.2;
    const lightVertexReversePdf = 0.23;
    const endpointDirectionalPdf = 1 / Math.PI;
    const t1 = assembleMergedCameraSplatPath({
      lightChain: [
        {
          position: l0Position,
          normal: endpointToSurface,
          pdfFwd: 0.4,
          pdfRev: 0.7,
          isSpecular: false,
        },
        {
          position: l1Position,
          normal: [0, 0, 1],
          pdfFwd: endpointDirectionalPdf,
          pdfRev: 0.6,
          isSpecular: false,
        },
      ],
      camera,
      cameraDirectionalPdf,
      lightBrdfPdf: () => lightVertexReversePdf,
    });
    const s1 = assembleMergedConnectionPath({
      lightChain: [{
        position: l0Position,
        normal: endpointToSurface,
        pdfFwd: 0.4,
        pdfRev: 0.7,
        isSpecular: false,
      }],
      eyeChain: [{
        position: l1Position,
        normal: [0, 0, 1],
        pdfFwd: 0.9,
        // This is the production eye-stack value for E0: Pdf_We of the
        // primary ray, not the historical unit placeholder.
        pdfRev: cameraDirectionalPdf,
        isSpecular: false,
      }],
      camera,
      eyeBrdfPdf: () => lightVertexReversePdf,
    });
    expect(s1.vertices).toEqual(t1.vertices);

    const t1Pdfs = buildStrategyPdfs(t1.vertices, t1.selectedS, 1);
    const s1Pdfs = buildStrategyPdfs(
      s1.vertices,
      s1.selectedS,
      t1Pdfs[s1.selectedS]!,
    );
    expect(Array.from(s1Pdfs)).toEqual(Array.from(t1Pdfs));
    const masked = maskBDPTExplicitConnectionStrategyPDFs(t1Pdfs, {
      maxLightVertices: 8,
      maxEyeVertices: 8,
    });
    expect(
      powerHeuristicWeight(masked, 1, 2) +
      powerHeuristicWeight(masked, 2, 2),
    ).toBeCloseTo(1, 13);
  });
});
