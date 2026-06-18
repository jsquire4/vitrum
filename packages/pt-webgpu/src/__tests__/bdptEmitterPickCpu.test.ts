import { describe, expect, it } from 'vitest';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  bdptEmitterCount,
  bdptEmitterPower,
  bdptPickEmitterFlat,
  sampleBdptBounce0Cpu,
} from '../bdpt/bdptEmitterPickCpu.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';

const REC709_R = 0.2126;
const REC709_G = 0.7152;
const REC709_B = 0.0722;
const LUMINANCE_FLOOR = 1e-20;

function independentLuminance(rgb: readonly [number, number, number]): number {
  return rgb[0] * REC709_R + rgb[1] * REC709_G + rgb[2] * REC709_B;
}

function independentLightLuminance(rgb: readonly [number, number, number]): number {
  return Math.max(independentLuminance(rgb), LUMINANCE_FLOOR);
}

function expectRelative(actual: number, expected: number, relative = 1e-6): void {
  const tolerance = Math.max(1e-30, Math.abs(expected) * relative);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

function stubScene(partial: Partial<UploadedSceneBuffers>): UploadedSceneBuffers {
  const base = {
    directionalLight: [0, -1, 0],
    directionalIrradiance: [0, 0, 0],
    directionalLightCount: 0,
    directionalLightsData: new Float32Array(0),
    sceneCenter: [0, 0, 0],
    sceneRadius: 1,
    pointLightCount: 0,
    spotLightCount: 0,
    rectAreaLightCount: 0,
    meshAreaLightCount: 0,
    pointLightsData: new Float32Array(0),
    spotLightsData: new Float32Array(0),
    rectAreaLightsData: new Float32Array(0),
    meshAreaLightsData: new Float32Array(0),
    environmentTint: [1, 1, 1],
    environmentSunDirection: [0, 1, 0],
    environmentSunStrength: 0,
    environmentMapWidth: 0,
    environmentMapHeight: 0,
    hasEnvironmentMap: false,
    environmentMapTexels: new Float32Array(0),
    environmentMapCdf: new Float32Array(0),
    triangleCount: 0,
    analyticCount: 0,
    warnings: [],
    tlasNodeCount: 0,
    primitiveTlasBindings: [],
    analyticHeaders: new Float32Array(0),
    analyticParams: new Float32Array(0),
    analyticLocalToWorld: new Float32Array(0),
    analyticWorldToLocal: new Float32Array(0),
    bvhNodesBuffer: null as unknown as GPUBuffer,
    bvhIndexBuffer: null as unknown as GPUBuffer,
    bvhPositionsBuffer: null as unknown as GPUBuffer,
    bvhBeerColorsBuffer: null as unknown as GPUBuffer,
    emittersBuffer: null as unknown as GPUBuffer,
    emitterCdfBuffer: null as unknown as GPUBuffer,
    pointLightsBuffer: null as unknown as GPUBuffer,
    spotLightsBuffer: null as unknown as GPUBuffer,
    rectAreaLightsBuffer: null as unknown as GPUBuffer,
    meshAreaLightsBuffer: null as unknown as GPUBuffer,
    environmentMapTexelsBuffer: null as unknown as GPUBuffer,
    environmentMapCdfBuffer: null as unknown as GPUBuffer,
    tlasNodesBuffer: null as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: null as unknown as GPUBuffer,
    tlasBlasRootsBuffer: null as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: null as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: null as unknown as GPUBuffer,
    bvhNodes: new Uint32Array(0),
    bvhIndex: new Uint32Array(0),
    bvhPositions: new Float32Array(0),
    bvhBeerColors: new Float32Array(0),
    emitters: new Float32Array(0),
    emitterCdf: new Float32Array(0),
    tlasNodes: new Uint32Array(0),
    tlasInstanceIndices: new Uint32Array(0),
    tlasBlasRoots: new Uint32Array(0),
    tlasInstanceWorldToLocal: new Float32Array(0),
    tlasInstanceLocalToWorld: new Float32Array(0),
  } as unknown as UploadedSceneBuffers;
  return { ...base, ...partial };
}

describe('bdptEmitterPickCpu', () => {
  it('uses Rec.709 luminance with a positive floor for directional, point, and spot power', () => {
    const sameAsRedViaGreen: [number, number, number] = [0, REC709_R / REC709_G, 0];
    const red: [number, number, number] = [1, 0, 0];
    const negative: [number, number, number] = [-10, -1, -10];

    const spot = new Float32Array(16);
    spot.set([0, 0, 0, 0, 0, -1, 0, 0.5, sameAsRedViaGreen[0], sameAsRedViaGreen[1], sameAsRedViaGreen[2], 0.25], 0);
    const sb = stubScene({
      directionalLightCount: 1,
      directionalLightsData: new Float32Array([0, -1, 0, 0, negative[0], negative[1], negative[2], 0]),
      pointLightCount: 1,
      pointLightsData: new Float32Array([0, 0, 0, 0, red[0], red[1], red[2], 0, 0, 0, 0, 0]),
      spotLightCount: 1,
      spotLightsData: spot,
    });

    expect(bdptEmitterCount(sb)).toBe(3);
    expect(bdptEmitterPower(sb, 0)).toBe(LUMINANCE_FLOOR);
    expectRelative(bdptEmitterPower(sb, 1), independentLightLuminance(red));
    expectRelative(bdptEmitterPower(sb, 2), independentLightLuminance(sameAsRedViaGreen));
    expectRelative(bdptEmitterPower(sb, 2), bdptEmitterPower(sb, 1));
  });

  it('multiplies finite-emitter power by independent geometric area and luminance', () => {
    const rectRgb: [number, number, number] = [1, 0, 0];
    const discRgb: [number, number, number] = [-1, -1, -1];
    const meshRgb: [number, number, number] = [0, 0, 1];
    const rect = new Float32Array(32);
    // Rect: area = 4 * |u x v| = 4 * 6 = 24.
    rect.set([0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, rectRgb[0], rectRgb[1], rectRgb[2], 0], 0);
    // Disc: shapeTag=1, radius=2 from |u|, area = 4π; negative luminance floors.
    rect.set([0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, discRgb[0], discRgb[1], discRgb[2], 1], 16);
    const mesh = new Float32Array(16);
    // Mesh triangle: area = 0.5 * |(4,0,0) x (0,3,0)| = 6.
    mesh.set([0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, meshRgb[0], meshRgb[1], meshRgb[2], 0], 0);

    const sb = stubScene({
      rectAreaLightCount: 2,
      rectAreaLightsData: rect,
      meshAreaLightCount: 1,
      meshAreaLightsData: mesh,
    });

    expect(bdptEmitterCount(sb)).toBe(3);
    expectRelative(bdptEmitterPower(sb, 0), 24 * independentLightLuminance(rectRgb));
    expectRelative(bdptEmitterPower(sb, 1), 4 * Math.PI * independentLightLuminance(discRgb));
    expectRelative(bdptEmitterPower(sb, 2), 6 * independentLightLuminance(meshRgb));
  });

  it('keeps the GPU BDPT emitter-power helper linked to canonical luminance with the floor', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fn bdptLightLuminance(c: vec3f) -> f32');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('return max(luminance(c), 1e-20);');
  });

  it('counts directional + point + spot + rect + mesh + env', () => {
    const sb = stubScene({
      directionalLightCount: 1,
      directionalLightsData: new Float32Array([0, -1, 0, 0, 1, 1, 1, 1]),
      pointLightCount: 1,
      spotLightCount: 1,
      rectAreaLightCount: 1,
      meshAreaLightCount: 1,
      pointLightsData: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0]),
      spotLightsData: new Float32Array(12).fill(0),
      rectAreaLightsData: new Float32Array(16).fill(0),
      meshAreaLightsData: new Float32Array(16).fill(0),
      environmentSunStrength: 2,
    });
    expect(bdptEmitterCount(sb)).toBe(6);
  });

  it('power-weighted pick prefers brighter mesh over dim point', () => {
    const mesh = new Float32Array(16);
    mesh[0] = 0;
    mesh[1] = 0;
    mesh[2] = 0;
    mesh[4] = 2;
    mesh[5] = 0;
    mesh[6] = 0;
    mesh[8] = 0;
    mesh[9] = 2;
    mesh[10] = 0;
    mesh[12] = 100;
    mesh[13] = 100;
    mesh[14] = 100;

    const sb = stubScene({
      pointLightCount: 1,
      meshAreaLightCount: 1,
      pointLightsData: new Float32Array([0, 1, 0, 0, 0.01, 0.01, 0.01, 0]),
      meshAreaLightsData: mesh,
    });

    const pointPower = bdptEmitterPower(sb, 0);
    const meshPower = bdptEmitterPower(sb, 1);
    expect(meshPower).toBeGreaterThan(pointPower * 10);

    const flat = bdptPickEmitterFlat(sb, pointPower + meshPower * 0.5, meshPower + pointPower, 2);
    expect(flat).toBe(1);

    const sample = sampleBdptBounce0Cpu(sb, flat, 1, 0.37);
    expect(sample).not.toBeNull();
    expect(sample!.pdfJoint).toBeGreaterThan(0);
  });

  // Regression for the bounce-0 tangent-frame bug: the cosine hemisphere
  // direction must be `wi = t*x + b*y + n*z`. The original code scaled the
  // bitangent `b` by `x` instead of `y` (the `y = r*sin(phi)` local was
  // dead), collapsing the two tangent-plane components to be equal. The
  // mesh triangle a=[0,0,0], b=[2,0,0], c=[0,2,0] has normal +Z, so with
  // uHemi=0.37 the sample gives distinct x=0.241576, y=-0.558248 and the
  // derived hemisphere pdf differs between the buggy and corrected formulas.
  it('builds the bounce-0 direction with the y-scaled bitangent (mesh emitter)', () => {
    const mesh = new Float32Array(16);
    mesh[4] = 2; // b = a + (2,0,0)
    mesh[9] = 2; // c = a + (0,2,0)
    mesh[12] = 100;
    mesh[13] = 100;
    mesh[14] = 100;
    const sb = stubScene({ meshAreaLightCount: 1, meshAreaLightsData: mesh });

    const sample = sampleBdptBounce0Cpu(sb, 0, 1, 0.37);
    expect(sample).not.toBeNull();
    // Triangle normal is +Z.
    expect(sample!.emitNormal[0]).toBeCloseTo(0, 6);
    expect(sample!.emitNormal[1]).toBeCloseTo(0, 6);
    expect(sample!.emitNormal[2]).toBeCloseTo(1, 6);
    // pdfHemi = cosEmit / PI with the CORRECTED (b*y) direction.
    // The buggy (b*x) variant yields 0.2923763342 instead.
    expect(sample!.pdfHemi).toBeCloseTo(0.25265063960867545, 9);
    // Finite area emitters store pdfPick*pdfArea for the emitter vertex; this
    // mesh triangle has area 2 and discretePdf=1.
    expect(sample!.pdfJoint).toBeCloseTo(0.5, 9);
    expect(sample!.pdfHemi).not.toBeCloseTo(0.2923763342268401, 9);
  });

  // Characterization golden for the single-sourced flat-emitter stride-walk
  // (`walkPositionalEmitters`). A scene mixing directional + point + spot + rect
  // + mesh + env exercises every kind in the canonical NEE walk order
  //   directional · point[8] · spot[12] · rect[16] · mesh[16] · env.
  // The flat index feeds the RNG-correlated power-weighted pick, so BOTH the walk
  // ORDER and the per-emitter values MUST stay byte-identical. These goldens were
  // captured from the pre-dedup unrolled walks (Theme J refactor, 2026-05-30).
  it('golden: mixed-scene flat enumeration is byte-stable across the walk dedup', () => {
    const point = new Float32Array([1, 2, 3, 0, 4, 5, 6, 0]);
    const spot = new Float32Array(12);
    spot.set([7, 8, 9, 0, 0, -1, 0, 0.5, 10, 11, 12], 0);
    const rect = new Float32Array(16);
    rect.set([0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 13, 14, 15], 0);
    const mesh = new Float32Array(16);
    mesh.set([0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 16, 17, 18], 0);
    const sb = stubScene({
      directionalLightCount: 1,
      directionalLightsData: new Float32Array([0, -1, 0, 0, 0.9, 0.8, 0.7, 0.8]),
      sceneCenter: [10, 20, 30],
      sceneRadius: 5,
      pointLightCount: 1,
      spotLightCount: 1,
      rectAreaLightCount: 1,
      meshAreaLightCount: 1,
      pointLightsData: point,
      spotLightsData: spot,
      rectAreaLightsData: rect,
      meshAreaLightsData: mesh,
      environmentSunStrength: 2,
    });

    // 6 selectable lights: directional, point, spot, rect, mesh, env.
    expect(bdptEmitterCount(sb)).toBe(6);

    const goldenPowers = [
      0.8140400025963783,
      4.8595999999999995,
      10.8596,
      332.63039999999995,
      33.7192,
      25.132741228718345,
    ];
    for (let i = 0; i < goldenPowers.length; i += 1) {
      expect(bdptEmitterPower(sb, i)).toBeCloseTo(goldenPowers[i]!, 9);
    }

    // Sampled bounce-0 vertices, one per flat index (discretePdf=0.5, uHemi=0.37).
    const goldenSamples = [
      { emitPos: [10, 40, 30], emitNormal: [0, -1, 0], pdfHemi: 0.2526506396086754 },
      // A9 — point emitter is ISOTROPIC (uniform sphere): emitNormal = sampled sphere
      // dir, pdfHemi = 1/(4π) = 0.0795774715… (was cosine-up about [0,1,0]).
      { emitPos: [1, 2, 3], emitNormal: [-0.5316159505093274, 0.566113487883891, -0.6299999999999999], pdfHemi: 0.07957747154594767 },
      { emitPos: [7, 8, 9], emitNormal: [0, -1, 0], pdfHemi: 0.25265063960867545 },
      { emitPos: [-0.52, 5, 0.78], emitNormal: [0, -1, 0], pdfHemi: 0.25265063960867545 },
      { emitPos: [0.9914902924386096, 0.22506221362103418, 0], emitNormal: [0, 0, 1], pdfHemi: 0.25265063960867545 },
      { emitPos: [10, 0, 30], emitNormal: [0, 1, 0], pdfHemi: 0.25265063960867545 },
    ] as const;
    for (let i = 0; i < goldenSamples.length; i += 1) {
      const s = sampleBdptBounce0Cpu(sb, i, 0.5, 0.37);
      expect(s).not.toBeNull();
      const g = goldenSamples[i]!;
      expect(s!.emitPos[0]).toBeCloseTo(g.emitPos[0], 9);
      expect(s!.emitPos[1]).toBeCloseTo(g.emitPos[1], 9);
      expect(s!.emitPos[2]).toBeCloseTo(g.emitPos[2], 9);
      expect(s!.emitNormal[0]).toBeCloseTo(g.emitNormal[0], 9);
      expect(s!.emitNormal[1]).toBeCloseTo(g.emitNormal[1], 9);
      expect(s!.emitNormal[2]).toBeCloseTo(g.emitNormal[2], 9);
      expect(s!.pdfHemi).toBeCloseTo(g.pdfHemi, 9);
    }
  });
});
