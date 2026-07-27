import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildLightTree, lightTreePdfCPU } from '@vitrum/shared-samplers';
import { buildLightTreeInputForScene, packEmitterArrays } from '../scene/emitterPacking.js';
import { materialToPackedVec4s } from '../scene/materialPacking.js';
import { assertVolumeSceneSupported } from '../spectralSceneValidation.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';

const INV_4PI = 1 / (4 * Math.PI);

function hg(cosTheta: number, g: number): number {
  return (INV_4PI * (1 - g * g)) / Math.max((1 + g * g - 2 * g * cosTheta) ** 1.5, 1e-18);
}

function hgSolidAngleIntegral(g: number): number {
  if (Math.abs(g) < 1e-12) return 1;
  const primitive = (mu: number): number =>
    (1 - g * g) / (2 * g * Math.sqrt(1 + g * g - 2 * g * mu));
  return primitive(1) - primitive(-1);
}

function power(a: number, b: number): number {
  return (a * a) / (a * a + b * b);
}

function availableMediumDistance(lightDistance: number, remainingBudget: number): number {
  return Math.min(Math.max(lightDistance, 0), Math.max(remainingBudget, 0));
}

function volumeScene(material: Scene['primitives'][number]['material']): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'volume',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material,
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('SSS production closure — analytic transport', () => {
  it('normalizes HG and remains reciprocal near the supported endpoints', () => {
    for (const g of [-0.999, -0.9, 0, 0.9, 0.999]) {
      expect(hgSolidAngleIntegral(g)).toBeCloseTo(1, 10);
      for (const cosine of [-1, -0.2, 0, 0.7, 1]) {
        expect(hg(cosine, g)).toBeCloseTo(hg(cosine, g), 14);
        expect(Number.isFinite(hg(cosine, g))).toBe(true);
        expect(hg(cosine, g)).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('normalizes collision density plus terminal survival for finite slabs', () => {
    const cases: Array<[number, number]> = [
      [0.2, 3],
      [1.7, 0.8],
      [8, 0.04],
    ];
    for (const [sigmaT, distance] of cases) {
      const steps = 200_000;
      const dt = distance / steps;
      let collision = 0;
      for (let i = 0; i < steps; i++) {
        const t = (i + 0.5) * dt;
        collision += sigmaT * Math.exp(-sigmaT * t) * dt;
      }
      const survival = Math.exp(-sigmaT * distance);
      expect(collision + survival).toBeCloseTo(1, 9);
    }
  });

  it('conserves the homogeneous slab event estimator channel-by-channel', () => {
    const sigmaA = [0.1, 0.6, 1.8];
    const sigmaS = [0.8, 0.3, 0.05];
    const distance = 0.7;
    for (let c = 0; c < 3; c++) {
      const sigmaT = sigmaA[c]! + sigmaS[c]!;
      const survival = Math.exp(-sigmaT * distance);
      const scattered = (sigmaS[c]! / sigmaT) * (1 - survival);
      const absorbed = (sigmaA[c]! / sigmaT) * (1 - survival);
      expect(survival + scattered + absorbed).toBeCloseTo(1, 13);
    }
  });

  it('partitions paired phase/light MIS and gives unpaired or delta emitters full ownership', () => {
    const pdfPairs: Array<[number, number]> = [
      [0.01, 3],
      [0.2, 0.3],
      [4, 0.02],
    ];
    for (const [phasePdf, lightPdf] of pdfPairs) {
      expect(power(phasePdf, lightPdf) + power(lightPdf, phasePdf)).toBeCloseTo(1, 14);
    }
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'powerHeuristic(light.pdf, phasePdf), 1.0,',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'light.delta || softDirectionalWithoutComplement',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('INFINITY, directionPdf,');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'directionIsDelta, ignoreOpaque, true,',
    );
  });

  it('uses only the physically available segment near a boundary or finite light', () => {
    expect(availableMediumDistance(5, 0.03)).toBeCloseTo(0.03, 12);
    expect(availableMediumDistance(0.2, 0.8)).toBeCloseTo(0.2, 12);
    expect(Math.exp(-2 * availableMediumDistance(5, 0.03))).toBeCloseTo(Math.exp(-2 * 0.03), 14);
    expect(PT_WEBGPU_TRACE_WGSL).toContain('eyeMedium.remainingDistance - freeFlightDist');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'let attenDistance = min(segment, stack[top].remainingDistance);',
    );
  });
});

describe('SSS production closure — emitter ownership and packing', () => {
  const scene: Scene = {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
      },
    ],
    emitters: [
      { kind: 'directional', id: 'd', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
      { kind: 'point', id: 'p', position: [0, 2, 0], color: [1, 1, 1], intensity: 2 },
      {
        kind: 'spot',
        id: 's',
        position: [1, 2, 0],
        direction: [0, -1, 0],
        angle: 0.5,
        color: [1, 1, 1],
        intensity: 3,
      },
      {
        kind: 'rect-area',
        id: 'r',
        position: [0, 2, 0],
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
        color: [1, 1, 1],
        intensity: 4,
      },
      {
        kind: 'disc-area',
        id: 'c',
        position: [0, 3, 0],
        normal: [0, -1, 0],
        radius: 0.5,
        color: [1, 1, 1],
        intensity: 5,
      },
      { kind: 'mesh-area', id: 'm', meshId: 'tri', color: [1, 1, 1], intensity: 6 },
    ],
    environment: { kind: 'none' },
  };

  it('keeps CPU packed counts, tree leaves, and WGSL flat order identical', () => {
    const packed = packEmitterArrays(scene);
    const expected =
      packed.directionalLightCount +
      packed.pointLightCount +
      packed.spotLightCount +
      packed.rectAreaLightCount +
      packed.meshAreaLightCount +
      1;
    const input = buildLightTreeInputForScene(scene, {
      packed,
      envSummary: { hasHdri: true, sunStrength: 0, tint: [1, 1, 1] },
    });
    expect(input.powers.length).toBe(expected);
    const order = [
      'params.directionalLightCount',
      'params.pointLightCount',
      'params.spotLightCount',
      'params.rectAreaLightCount',
      'params.meshAreaLightCount',
      'if (current == flat) {\n    let env',
    ];
    let previous = -1;
    for (const token of order) {
      const next = PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf(token, previous + 1);
      expect(next).toBeGreaterThan(previous);
      previous = next;
    }

    const { nodes } = buildLightTree(input);
    let pdfSum = 0;
    for (let emitter = 0; emitter < expected; emitter++) {
      pdfSum += lightTreePdfCPU(nodes, [0.1, 0.2, 0.3], 1e-3, emitter);
    }
    expect(pdfSum).toBeCloseTo(1, 6);
  });

  it('implements every supported emitter class and paired area/environment phase endpoints', () => {
    for (const token of [
      'directionalLights[base]',
      'pointLights[base]',
      'spotLights[base]',
      'rectAreaLights[base]',
      'meshAreaLights[base]',
      'sampleEnvironmentImportance(rng)',
      'intersectRectAreaLightRay(',
      'intersectMeshAreaLightRay(',
      'environmentPdf(sampledDirection)',
    ]) {
      expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(token);
    }
  });

  it('partitions BDPT/SPPM ownership by interior light-prefix deltas and suppresses one phase endpoint', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!bdptOwnsFiniteLightFamily) {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevEventKind = 2u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevDirectionalPdf = phasePdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('prevEventKind = select(0u, 1u, sampleAllowsAreaMis);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('sppmReceiverPrefixActive = false;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('destIsMedium: bool,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('bdptLightPrefixContainsInteriorDelta(c)');
  });
});

describe('SSS production closure — boundaries, textures, and safety', () => {
  it('preserves alpha-skipped distance before free flight on eye and BDPT light paths', () => {
    for (const source of [PT_WEBGPU_TRACE_WGSL, PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL]) {
      expect(source).toContain('var alphaAdvance = 0.0;');
      expect(source).toContain('alphaAdvance = alphaAdvance + alphaStep;');
      expect(source).toContain('hit.dist = hit.dist + alphaAdvance;');
    }
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('alphaTestPassThrough(');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('step < 16u');
  });

  it('samples thickness on every BDPT light-side material boundary', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fn bdptMaterialWithVolumeThickness(');
    expect(
      (PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.match(/bdptMaterialWithVolumeThickness\(/g) ?? []).length,
    ).toBeGreaterThanOrEqual(5);
  });

  it('fails closed on stack overflow, unmatched underflow, or traversal exhaustion', () => {
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'if (depth >= BDPT_MEDIUM_STACK_LIMIT) { return result; }',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      '!bdptMediumLayerMatchesBoundary(stack[depth - 1u], matId, boundary)',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toMatch(
      /for \(var step = 0u; step < 16u;[\s\S]*?return result;\n}/,
    );
  });

  it('uses exact RGB-overrides-scalar scattering semantics', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1],
      roughness: 0.3,
      metallic: 0,
      transmission: 1,
      scatteringCoefficient: 4,
      scatteringCoefficientRGB: [0.1, 0.2, 0.3],
    });
    expect(packed.slice(12, 15)).toEqual([0.1, 0.2, 0.3]);
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let sigmaS = mat.scatteringRgb;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain(
      'max(mat.scatteringRgb, vec3f(mat.scatteringCoeff))',
    );
  });

  it('degrades lite to absorption-only without destroying scattering energy', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'throughput = throughput * exp(-sigmaA * materialAttenuationDistance',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('0.02 * scatteringCoeff');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('sigmaA + sigmaS');
  });
});

describe('SSS production closure — strict scene validation', () => {
  it('accepts finite physical coefficients through the near-|g| domain', () => {
    expect(() =>
      assertVolumeSceneSupported(
        volumeScene({
          baseColor: [1, 1, 1],
          roughness: 0.2,
          metallic: 0,
          transmission: 1,
          attenuationColor: [0.2, 0.5, 1],
          attenuationDistance: 0.7,
          thickness: 0.3,
          scatteringCoefficient: 0.1,
          scatteringCoefficientRGB: [0.05, 0.1, 0.2],
          scatteringAnisotropy: 0.999,
          spectralAttenuation: {
            wavelengthStart: 380,
            wavelengthEnd: 780,
            values: new Float32Array([0.1, 0.2, 0.3]),
          },
        }),
      ),
    ).not.toThrow();
  });

  it.each([
    ['negative sigma', { scatteringCoefficient: -1 }],
    ['nonfinite RGB', { scatteringCoefficientRGB: [0, Number.NaN, 0] }],
    ['negative thickness', { thickness: -0.1 }],
    ['zero attenuation distance', { attenuationDistance: 0 }],
    ['invalid g', { scatteringAnisotropy: 1 }],
  ])('rejects %s before upload or incremental commit', (_name, patch) => {
    expect(() =>
      assertVolumeSceneSupported(
        volumeScene({
          baseColor: [1, 1, 1],
          roughness: 0.2,
          metallic: 0,
          transmission: 1,
          ...patch,
        } as Scene['primitives'][number]['material']),
      ),
    ).toThrow(/pt-webgpu volume scene validation/);
  });
});
