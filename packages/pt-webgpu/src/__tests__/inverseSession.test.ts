// inverseSession.test.ts — WS5 InverseSession contract + Phase-0 loop tests.
//
// The GPU render+readback is faked here (it needs a real device; V24 is the
// hardware A/B). What IS exercised on the CPU: the contract shape, parameter
// path resolution + validation, the finite-difference + Adam optimizer loop
// (loss provably decreases on a fittable fake forward model), the
// frozen-RNG-replay determinism the FD gradient relies on, idempotent dispose,
// and the pure loss/Adam helpers.

import { describe, it, expect } from 'vitest';
import { asMat4 } from '@vitrum/core';
import type { Scene, InverseSessionOptions, MaterialSpec, SceneEmitter } from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type InverseEngineHooks,
  type AdjointGradientRequest,
} from '../inverse/inverseSession.js';
import { l2Loss, l1Loss, lossValue, Adam, parseParamPath, paramLength } from '../inverse/optimizer.js';
import { MESH_AREA_LIGHT_TRI_CAP } from '../scene/emitterPacking.js';

// ── a fittable fake forward model ─────────────────────────────────────────────
//
// The "render" is a deterministic function of one material's baseColor: every
// pixel is exactly the material baseColor. So fitting baseColor to a target
// constant-colour image is a convex L2 problem the optimizer must drive to ~0.
// The fake also keys its output on the LIVE scene the hooks expose, so when the
// FD probe perturbs baseColor the rendered image changes accordingly — exactly
// what the real GPU path does, but synchronous + exact (no MC noise).

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.2, 0.2, 0.2], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [
      { kind: 'point', id: 'lamp', color: [1, 1, 1], intensity: 2, position: [0, 1, 0] },
    ],
    environment: { kind: 'none' },
  };
}

interface FakeEngine {
  hooks: InverseEngineHooks;
  scene: Scene;
  renderCount: number;
  lastSeedSequences: number[][];
}

/** Build hooks over a mutable scene; the render maps baseColor → flat image.
 *  `seedLog` records (per render) the per-sample seed sequence the session
 *  would use — we model the determinism contract: same scene + same seed
 *  sequence ⇒ identical image. */
function makeFakeEngine(W = 2, H = 2): FakeEngine {
  const fake: FakeEngine = {
    scene: makeScene(),
    renderCount: 0,
    lastSeedSequences: [],
    hooks: {} as InverseEngineHooks,
  };
  fake.hooks = {
    getScene: () => fake.scene,
    renderAndReadback: async (width, height, _samples) => {
      fake.renderCount += 1;
      const mat = fake.scene.primitives[0]!.material;
      const transmission = mat.transmission ?? 0;
      const thickness = mat.thickness ?? 0;
      const opacity = mat.alphaMode === 'mask' || mat.alphaMode === 'blend'
        ? mat.opacity ?? 1
        : 0;
      const alphaCutoff = mat.alphaMode === 'mask'
        ? mat.alphaCutoff ?? 0.5
        : 0;
      const attenuationDistance = mat.attenuationDistance ?? 0;
      const attenuationColor = mat.attenuationColor ?? [0, 0, 0];
      const iridescenceThicknessRange = mat.iridescenceThicknessRange ?? [0, 0];
      const anisotropyRotation = mat.anisotropyRotation ?? 0;
      const normalOrBumpScale = (mat.normalScale ?? 0) + (mat.bumpScale ?? 0) + (mat.clearcoatNormalScale ?? 0);
      const materialMapIntensity = (mat.aoMapIntensity ?? 0) + (mat.lightMapIntensity ?? 0) + (mat.envMapIntensity ?? 0);
      const dispersionAbbeNumber = mat.dispersionAbbeNumber ?? 0;
      const scatteringCoefficient = mat.scatteringCoefficient ?? 0;
      const scatteringAnisotropy = mat.scatteringAnisotropy ?? 0;
      const scatteringCoefficientRGB = mat.scatteringCoefficientRGB ?? [0, 0, 0];
      const rgb = new Float32Array(width * height * 3);
      for (let p = 0; p < width * height; p++) {
        rgb[p * 3 + 0] = mat.baseColor[0] + transmission + opacity + normalOrBumpScale + attenuationColor[0] + iridescenceThicknessRange[0] / 1000 + dispersionAbbeNumber / 100 + scatteringCoefficient + scatteringCoefficientRGB[0];
        rgb[p * 3 + 1] = mat.baseColor[1] + thickness + attenuationDistance + alphaCutoff + materialMapIntensity + attenuationColor[1] + iridescenceThicknessRange[1] / 1000 + scatteringAnisotropy + scatteringCoefficientRGB[1];
        rgb[p * 3 + 2] = mat.baseColor[2] + attenuationColor[2] + anisotropyRotation + scatteringCoefficientRGB[2];
      }
      return { rgb, channels: 3 as const };
    },
    patchMaterial: (id: string, patch: Partial<MaterialSpec>) => {
      fake.scene = {
        ...fake.scene,
        primitives: fake.scene.primitives.map((pr) =>
          pr.id === id ? { ...pr, material: { ...pr.material, ...patch } } : pr,
        ),
      };
    },
    patchEmitter: (id: string, patch: Partial<SceneEmitter>) => {
      fake.scene = {
        ...fake.scene,
        emitters: fake.scene.emitters.map((e) =>
          e.id === id ? ({ ...e, ...patch } as SceneEmitter) : e,
        ),
      };
    },
  };
  void W; void H;
  return fake;
}

function targetImage(W: number, H: number, color: [number, number, number]) {
  const data = new Float32Array(W * H * 3);
  for (let p = 0; p < W * H; p++) {
    data[p * 3 + 0] = color[0];
    data[p * 3 + 1] = color[1];
    data[p * 3 + 2] = color[2];
  }
  return { data, width: W, height: H, channels: 3 as const };
}

describe('InverseSession — contract shape', () => {
  it('reports parameterCount and the resolved method', () => {
    const fake = makeFakeEngine();
    const opts: InverseSessionOptions = {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
    };
    const session = new PtWebgpuInverseSession(fake.hooks, opts);
    expect(session.parameterCount).toBe(1);
    // No method requested + no adjoint hook ⇒ default finite-difference.
    expect(session.method).toBe('finite-difference');
    expect(session.currentValues()).toHaveLength(1);
    expect(session.currentValues()[0]).toHaveLength(3);
    session.dispose();
  });

  it('seeds currentValues from the scene when no `initial` override is given', () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
    });
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.2, 6), expect.closeTo(0.2, 6), expect.closeTo(0.2, 6),
    ]);
    session.dispose();
  });

  it('honours an `initial` override', () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar', initial: [0.9] }],
    });
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.9, 6)]);
    session.dispose();
  });
});

describe('InverseSession — path resolution + validation throws', () => {
  it('throws on an unknown primitive id', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.nope.baseColor', kind: 'rgb' }],
    })).toThrow(/no primitive with id "nope"/);
  });

  it('throws on a non-optimizable field', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.displacementScale', kind: 'scalar' }],
    })).toThrow(/not optimizable/);
  });

  it('throws when the declared kind disagrees with the resolved field', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'scalar' }],
    })).toThrow(/declared kind 'scalar'/);
  });

  it('throws when a vec2 material field is declared as scalar', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.iridescenceThicknessRange', kind: 'scalar' }],
    })).toThrow(/declared kind 'scalar'.*'vec2'/);
  });

  it('throws on an empty parameter list', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [],
    })).toThrow(/at least one parameter/);
  });

  it("throws on the reserved 'texture' kind (Phase 2, not yet differentiable)", () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.baseColorMap', kind: 'texture' }],
    })).toThrow(/texture/);
  });

  it("throws on a reserved perceptual loss ('lpips')", () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      loss: 'lpips',
    })).toThrow(/perceptual loss/);
  });
});

describe('InverseSession — Phase-0 finite-difference loop converges', () => {
  it('optimizes scalar transport controls through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                transmission: 0.3,
                thickness: 0.6,
                attenuationDistance: 0.9,
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.transmission', kind: 'scalar', max: 0.2 },
        { path: 'materials.panel.thickness', kind: 'scalar', max: 0.5 },
        { path: 'materials.panel.attenuationDistance', kind: 'scalar', max: 0.4 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.3, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.6, 6)]);
    expect(session.currentValues()[2]).toEqual([expect.closeTo(0.9, 6)]);

    const result = await session.step();
    expect(result.gradient[0]![0]).toBeGreaterThan(0);
    expect(result.gradient[1]![0]).toBeGreaterThan(0);
    expect(result.gradient[2]![0]).toBeGreaterThan(0);
    expect(session.currentValues()[0]![0]).toBeLessThanOrEqual(0.2);
    expect(session.currentValues()[1]![0]).toBeLessThanOrEqual(0.5);
    expect(session.currentValues()[2]![0]).toBeLessThanOrEqual(0.400001);
    expect(fake.scene.primitives[0]!.material.transmission).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    expect(fake.scene.primitives[0]!.material.thickness).toBeCloseTo(session.currentValues()[1]![0]!, 6);
    expect(fake.scene.primitives[0]!.material.attenuationDistance).toBeCloseTo(session.currentValues()[2]![0]!, 6);
    session.dispose();
  });

  it('does not clamp anisotropyRotation to a half-turn by default', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, anisotropy: 0.75, anisotropyRotation: 4.0 } }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 5.2]),
      parameters: [{ path: 'materials.panel.anisotropyRotation', kind: 'scalar' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.25, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]![0]).toBeGreaterThan(Math.PI);
    const result = await session.step();
    expect(result.gradient[0]![0]).toBeLessThan(0);
    expect(session.currentValues()[0]![0]).toBeGreaterThan(4.0);
    expect(session.currentValues()[0]![0]).toBeGreaterThan(Math.PI);
    expect(fake.scene.primitives[0]!.material.anisotropyRotation).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    session.dispose();
  });

  it('does not clamp anisotropyRotation to a non-negative angle by default', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, anisotropy: 0.75, anisotropyRotation: -0.25 } }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, -1.1]),
      parameters: [{ path: 'materials.panel.anisotropyRotation', kind: 'scalar' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.25, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]![0]).toBeLessThan(0);
    const result = await session.step();
    expect(result.gradient[0]![0]).toBeGreaterThan(0);
    expect(session.currentValues()[0]![0]).toBeLessThan(-0.25);
    expect(fake.scene.primitives[0]!.material.anisotropyRotation).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    session.dispose();
  });

  it('optimizes attenuationColor through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                attenuationColor: [0.8, 0.7, 0.6],
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.attenuationColor', kind: 'rgb', max: 0.5 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.8, 6),
      expect.closeTo(0.7, 6),
      expect.closeTo(0.6, 6),
    ]);

    const result = await session.step();
    for (const grad of result.gradient[0]!) {
      expect(grad).toBeGreaterThan(0);
    }
    for (const value of session.currentValues()[0]!) {
      expect(value).toBeLessThanOrEqual(0.500001);
    }
    expect(fake.scene.primitives[0]!.material.attenuationColor).toEqual([
      expect.closeTo(session.currentValues()[0]![0]!, 6),
      expect.closeTo(session.currentValues()[0]![1]!, 6),
      expect.closeTo(session.currentValues()[0]![2]!, 6),
    ]);
    session.dispose();
  });

  it('optimizes vec2 iridescence thickness ranges through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescenceThicknessRange: [500, 700],
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.iridescenceThicknessRange', kind: 'vec2', max: 300 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(500, 6),
      expect.closeTo(700, 6),
    ]);

    const result = await session.step();
    expect(result.gradient[0]![0]).toBeGreaterThan(0);
    expect(result.gradient[0]![1]).toBeGreaterThan(0);
    expect(session.currentValues()[0]![0]).toBeLessThanOrEqual(300.000001);
    expect(session.currentValues()[0]![1]).toBeLessThanOrEqual(300.000001);
    expect(fake.scene.primitives[0]!.material.iridescenceThicknessRange).toEqual([
      expect.closeTo(session.currentValues()[0]![0]!, 6),
      expect.closeTo(session.currentValues()[0]![1]!, 6),
    ]);
    session.dispose();
  });

  it('optimizes scalar alpha coverage controls through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                alphaMode: 'mask',
                opacity: 0.7,
                alphaCutoff: 0.8,
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.opacity', kind: 'scalar', max: 0.5 },
        { path: 'materials.panel.alphaCutoff', kind: 'scalar', max: 0.4 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.7, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.8, 6)]);

    const result = await session.step();
    expect(result.gradient[0]![0]).toBeGreaterThan(0);
    expect(result.gradient[1]![0]).toBeGreaterThan(0);
    expect(session.currentValues()[0]![0]).toBeLessThanOrEqual(0.5);
    expect(session.currentValues()[1]![0]).toBeLessThanOrEqual(0.400001);
    expect(fake.scene.primitives[0]!.material.opacity).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    expect(fake.scene.primitives[0]!.material.alphaCutoff).toBeCloseTo(session.currentValues()[1]![0]!, 6);
    session.dispose();
  });

  it('optimizes renderer-consumed scalar map controls through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalScale: 0.8,
                bumpScale: 0.7,
                clearcoatNormalScale: 0.6,
                aoMapIntensity: 0.9,
                lightMapIntensity: 0.5,
                envMapIntensity: 0.4,
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.normalScale', kind: 'scalar', max: 0.4 },
        { path: 'materials.panel.bumpScale', kind: 'scalar', max: 0.4 },
        { path: 'materials.panel.clearcoatNormalScale', kind: 'scalar', max: 0.4 },
        { path: 'materials.panel.aoMapIntensity', kind: 'scalar', max: 0.4 },
        { path: 'materials.panel.lightMapIntensity', kind: 'scalar', max: 0.4 },
        { path: 'materials.panel.envMapIntensity', kind: 'scalar', max: 0.4 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.8, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.7, 6)]);
    expect(session.currentValues()[2]).toEqual([expect.closeTo(0.6, 6)]);
    expect(session.currentValues()[3]).toEqual([expect.closeTo(0.9, 6)]);
    expect(session.currentValues()[4]).toEqual([expect.closeTo(0.5, 6)]);
    expect(session.currentValues()[5]).toEqual([expect.closeTo(0.4, 6)]);

    const result = await session.step();
    for (const grad of result.gradient) {
      expect(grad[0]).toBeGreaterThan(0);
    }
    for (const value of session.currentValues()) {
      expect(value[0]).toBeLessThanOrEqual(0.400001);
    }
    const mat = fake.scene.primitives[0]!.material;
    expect(mat.normalScale).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    expect(mat.bumpScale).toBeCloseTo(session.currentValues()[1]![0]!, 6);
    expect(mat.clearcoatNormalScale).toBeCloseTo(session.currentValues()[2]![0]!, 6);
    expect(mat.aoMapIntensity).toBeCloseTo(session.currentValues()[3]![0]!, 6);
    expect(mat.lightMapIntensity).toBeCloseTo(session.currentValues()[4]![0]!, 6);
    expect(mat.envMapIntensity).toBeCloseTo(session.currentValues()[5]![0]!, 6);
    session.dispose();
  });

  it('optimizes renderer-consumed dispersion and scattering controls through finite differences', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                dispersionAbbeNumber: 45,
                scatteringCoefficient: 0.8,
                scatteringAnisotropy: 0.5,
                scatteringCoefficientRGB: [0.6, 0.7, 0.8],
              },
            }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.2, 0.2, 0.2]),
      parameters: [
        { path: 'materials.panel.dispersionAbbeNumber', kind: 'scalar', max: 20 },
        { path: 'materials.panel.scatteringCoefficient', kind: 'scalar', max: 0.3 },
        { path: 'materials.panel.scatteringAnisotropy', kind: 'scalar', max: 0.25 },
        { path: 'materials.panel.scatteringCoefficientRGB', kind: 'rgb', max: 0.25 },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(45, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.8, 6)]);
    expect(session.currentValues()[2]).toEqual([expect.closeTo(0.5, 6)]);
    expect(session.currentValues()[3]).toEqual([
      expect.closeTo(0.6, 6),
      expect.closeTo(0.7, 6),
      expect.closeTo(0.8, 6),
    ]);

    const result = await session.step();
    for (const grad of result.gradient) {
      for (const component of grad) expect(component).toBeGreaterThan(0);
    }
    expect(session.currentValues()[0]![0]).toBeLessThanOrEqual(20.000001);
    expect(session.currentValues()[1]![0]).toBeLessThanOrEqual(0.300001);
    expect(session.currentValues()[2]![0]).toBeLessThanOrEqual(0.250001);
    for (const value of session.currentValues()[3]!) {
      expect(value).toBeLessThanOrEqual(0.250001);
    }
    const mat = fake.scene.primitives[0]!.material;
    expect(mat.dispersionAbbeNumber).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    expect(mat.scatteringCoefficient).toBeCloseTo(session.currentValues()[1]![0]!, 6);
    expect(mat.scatteringAnisotropy).toBeCloseTo(session.currentValues()[2]![0]!, 6);
    expect(mat.scatteringCoefficientRGB).toEqual([
      expect.closeTo(session.currentValues()[3]![0]!, 6),
      expect.closeTo(session.currentValues()[3]![1]!, 6),
      expect.closeTo(session.currentValues()[3]![2]!, 6),
    ]);
    session.dispose();
  });

  it('loss strictly decreases and baseColor approaches the target over steps', async () => {
    const fake = makeFakeEngine();
    const target: [number, number, number] = [0.8, 0.15, 0.4];
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(4, 4, target),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    const losses: number[] = [];
    for (let i = 0; i < 40; i++) {
      const r = await session.step();
      losses.push(r.loss);
      expect(r.step).toBe(i);
    }
    // Monotone-ish decrease: final loss is far below the first.
    expect(losses[losses.length - 1]!).toBeLessThan(losses[0]! * 0.05);
    // The fitted baseColor lands near the target.
    const fit = session.currentValues()[0]!;
    for (let c = 0; c < 3; c++) expect(fit[c]!).toBeCloseTo(target[c]!, 1);
    session.dispose();
  });

  it('gradient sign is correct (loss-reducing direction)', async () => {
    const fake = makeFakeEngine();
    // Target brighter than the 0.2 start ⇒ to reduce L2, baseColor must INCREASE,
    // so dLoss/dbaseColor must be NEGATIVE (the optimizer moves opposite to grad).
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.9, 0.9, 0.9]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.1, fdEpsilon: 1e-3 },
    });
    const r = await session.step();
    for (let c = 0; c < 3; c++) expect(r.gradient[0]![c]!).toBeLessThan(0);
    session.dispose();
  });

  it('resolves + patches an emitter rgb (color) field', async () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.5, 0.5, 0.5]),
      parameters: [{ path: 'emitters.lamp.color', kind: 'rgb' }],
      samplesPerStep: 1,
    });
    // starting value read from the scene emitter color [1,1,1]
    expect(session.currentValues()[0]).toEqual([1, 1, 1]);
    const r = await session.step();
    expect(r.values[0]).toHaveLength(3);
    // the emitter PATCH path ran (the scene emitter still carries an rgb color
    // tuple of length 3; the fake render ignores color so the gradient is 0 and
    // the value is unchanged — that is correct, exercises patchEmitter + packing).
    const lamp = fake.scene.emitters[0]!;
    expect(lamp.color).toHaveLength(3);
    expect(r.gradient[0]).toEqual([0, 0, 0]);
    session.dispose();
  });

  it('runs the l1 loss path end-to-end', async () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.7, 0.3, 0.5]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      loss: 'l1',
      samplesPerStep: 1,
      optimizer: { learningRate: 0.1 },
    });
    const first = await session.step();
    const second = await session.step();
    expect(second.loss).toBeLessThanOrEqual(first.loss);
    session.dispose();
  });

  it('optimizes an emitter scalar (intensity) toward a brighter target', async () => {
    const fake = makeFakeEngine();
    // The fake render ignores emitter intensity, so this exercises the emitter
    // PATCH path + scalar packing without asserting convergence on intensity.
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.5, 0.5, 0.5]),
      parameters: [{ path: 'emitters.lamp.intensity', kind: 'scalar', max: 100 }],
      samplesPerStep: 1,
    });
    const r = await session.step();
    expect(r.values[0]).toHaveLength(1);
    session.dispose();
  });
});

describe('InverseSession — frozen-RNG replay determinism (the FD precondition)', () => {
  it('same scene + same render call ⇒ identical image (deterministic replay)', async () => {
    const fake = makeFakeEngine();
    const a = await fake.hooks.renderAndReadback(3, 3, 4);
    const b = await fake.hooks.renderAndReadback(3, 3, 4);
    expect(Array.from(a.rgb)).toEqual(Array.from(b.rgb));
  });

  it('different parameter ⇒ different contribution (the FD signal is real)', async () => {
    const fake = makeFakeEngine();
    const before = await fake.hooks.renderAndReadback(3, 3, 4);
    fake.hooks.patchMaterial('panel', { baseColor: [0.9, 0.1, 0.1] });
    const after = await fake.hooks.renderAndReadback(3, 3, 4);
    expect(Array.from(after.rgb)).not.toEqual(Array.from(before.rgb));
    // and only the changed channel moved by the expected amount
    expect(after.rgb[0]! - before.rgb[0]!).toBeCloseTo(0.7, 6);
  });
});

describe('InverseSession — dispose is idempotent + blocks further steps', () => {
  it('dispose twice is a no-op; step after dispose throws', async () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.5, 0.5, 0.5]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      samplesPerStep: 1,
    });
    session.dispose();
    expect(() => session.dispose()).not.toThrow();
    await expect(session.step()).rejects.toThrow(/disposed/);
  });
});

describe('InverseSession — Phase-1 path-replay adjoint wire', () => {
  const eligibleOpts = (): InverseSessionOptions => ({
    target: targetImage(2, 2, [0.8, 0.1, 0.1]),
    parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
    method: 'path-replay',
    samplesPerStep: 4,
  });

  it('resolves to path-replay when the engine provides the hook + every param is eligible', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, eligibleOpts());
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps path-replay for a single-bounce RGB render context', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayRenderContext: () => ({ bounces: 1, spectral: false }),
      computeAdjointGradient: async () => new Float32Array(3),
    };
    const session = new PtWebgpuInverseSession(hooks, eligibleOpts());
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('degrades path-replay when the forward baseline used multiple bounces', () => {
    const fake = makeFakeEngine();
    const diagnostics: unknown[] = [];
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayRenderContext: () => ({ bounces: 2, spectral: false }),
      computeAdjointGradient: async () => new Float32Array(3),
    };
    const session = new PtWebgpuInverseSession(hooks, {
      ...eligibleOpts(),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-render-regime',
      details: expect.objectContaining({ bounces: 2, supportedBounces: 1 }),
    }));
    expect(diagnostics).toEqual(session.diagnostics);
    session.dispose();
  });

  it('degrades path-replay when the forward baseline used spectral transport', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayRenderContext: () => ({ bounces: 1, spectral: true }),
      computeAdjointGradient: async () => new Float32Array(3),
    };
    const session = new PtWebgpuInverseSession(hooks, eligibleOpts());
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-render-regime',
      details: expect.objectContaining({ spectral: true }),
    }));
    session.dispose();
  });

  for (const [label, context, detail] of [
    ['BDPT', { bounces: 1, spectral: false, bdpt: true }, { bdpt: true, unsupportedFeature: 'bdpt' }],
    [
      'ReSTIR-PT reuse',
      { bounces: 1, spectral: false, restirPtReuse: true },
      { restirPtReuse: true, unsupportedFeature: 'restir-pt-reuse' },
    ],
    [
      'MNEE caustics',
      { bounces: 1, spectral: false, causticStrategy: 'manifold-nee' as const },
      { causticStrategy: 'manifold-nee', unsupportedFeature: 'caustic-strategy' },
    ],
    [
      'SPPM caustics',
      { bounces: 1, spectral: false, causticStrategy: 'photon-map' as const },
      { causticStrategy: 'photon-map', unsupportedFeature: 'caustic-strategy' },
    ],
  ] as const) {
    it(`degrades path-replay when the forward baseline used ${label}`, () => {
      const fake = makeFakeEngine();
      const hooks: InverseEngineHooks = {
        ...fake.hooks,
        getPathReplayRenderContext: () => context,
        computeAdjointGradient: async () => new Float32Array(3),
      };
      const session = new PtWebgpuInverseSession(hooks, eligibleOpts());
      expect(session.method).toBe('finite-difference');
      expect(session.diagnostics).toContainEqual(expect.objectContaining({
        code: 'path-replay-unsupported-render-regime',
        details: expect.objectContaining(detail),
      }));
      session.dispose();
    });
  }

  it('keeps path-replay for map-free unlit baseColor without requiring scene lights', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([0.1, 0.2, 0.3, 1]) } },
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, shadingModel: 'unlit' as const, alphaMode: 'opaque' as const, opacity: 0.25 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps unlit mapped baseColor on path-replay but unlit non-baseColor on finite-difference', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                shadingModel: 'unlit' as const,
                baseColorMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const mappedBase = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(mappedBase.method).toBe('path-replay');
    mappedBase.dispose();

    const roughness = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(roughness.method).toBe('finite-difference');
    roughness.dispose();
  });

  it('degrades to finite-difference when the engine provides NO adjoint hook', () => {
    const fake = makeFakeEngine();
    const diagnostics: unknown[] = [];
    const session = new PtWebgpuInverseSession(fake.hooks, {
      ...eligibleOpts(),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-hook-missing',
    }));
    expect(diagnostics).toEqual(session.diagnostics);
    session.dispose();
  });

  it('uses adjoint for eligible slots and FD only for unsupported holdouts', async () => {
    const fake = makeFakeEngine();
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.25, -0.5, 0.75, 999]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb' },
        { path: 'materials.panel.ior', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.ior',
    }));

    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'materials', id: 'panel', field: 'baseColor', offset: 0, length: 3 },
    ]);
    // Baseline render + one FD probe for the ior holdout. The baseColor slot
    // came from the adjoint hook instead of three more render probes.
    expect(fake.renderCount).toBe(2);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.25, 6), expect.closeTo(-0.5, 6), expect.closeTo(0.75, 6)],
      [expect.closeTo(0, 6)],
    ]);
    session.dispose();
  });

  it('keeps path-replay for deterministic point emitter color/intensity and passes hook offsets', async () => {
    const fake = makeFakeEngine();
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.1, 0.2, 0.3, 0.4]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'emitters.lamp.color', kind: 'rgb' },
        { path: 'emitters.lamp.intensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'lamp', field: 'color', offset: 0, length: 3 },
      { domain: 'emitters', id: 'lamp', field: 'intensity', offset: 3, length: 1 },
    ]);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)],
      [expect.closeTo(0.4, 6)],
    ]);
    session.dispose();
  });

  it('keeps path-replay for uncapped explicit mesh-area emitter color/intensity and passes hook offsets', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [{
        kind: 'mesh-area',
        id: 'mesh-light',
        color: [0.25, 0.5, 1],
        intensity: 4,
        meshId: 'panel',
      }],
    };
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.1, 0.2, 0.3, 0.4]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'emitters.mesh-light.color', kind: 'rgb' },
        { path: 'emitters.mesh-light.intensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'mesh-light', field: 'color', offset: 0, length: 3 },
      { domain: 'emitters', id: 'mesh-light', field: 'intensity', offset: 3, length: 1 },
    ]);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)],
      [expect.closeTo(0.4, 6)],
    ]);
    session.dispose();
  });

  it('keeps mapped mesh-area emitter color/intensity on path-replay through source-factor derivatives', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 0.25, 1]) } },
              },
            }
          : pr,
      ),
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-mesh-light',
        color: [0.25, 0.5, 1],
        intensity: 4,
        meshId: 'panel',
      }],
    };
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.1, 0.2, 0.3, 0.4]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'emitters.mapped-mesh-light.color', kind: 'rgb' },
        { path: 'emitters.mapped-mesh-light.intensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'mapped-mesh-light', field: 'color', offset: 0, length: 3 },
      { domain: 'emitters', id: 'mapped-mesh-light', field: 'intensity', offset: 3, length: 1 },
    ]);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)],
      [expect.closeTo(0.4, 6)],
    ]);
    session.dispose();
  });

  it('keeps zero-intensity mapped mesh-area emitter intensity on path-replay through adjoint replay factors', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 0.25, 1]) } },
              },
            }
          : pr,
      ),
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-dark-mesh-light',
        color: [0.25, 0.5, 1],
        intensity: 0,
        meshId: 'panel',
      }],
    };
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.4]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'emitters.mapped-dark-mesh-light.intensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'mapped-dark-mesh-light', field: 'intensity', offset: 0, length: 1 },
    ]);
    expect(result.gradient).toEqual([[expect.closeTo(0.4, 6)]]);
    session.dispose();
  });

  it('keeps mapped mesh-area emitter color on path-replay when an authored color channel is zero', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 0.25, 1]) } },
              },
            }
          : pr,
      ),
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-mesh-light',
        color: [0, 0.5, 1],
        intensity: 4,
        meshId: 'panel',
      }],
    };
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.1, 0.2, 0.3]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'emitters.mapped-mesh-light.color', kind: 'rgb' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'mapped-mesh-light', field: 'color', offset: 0, length: 3 },
    ]);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)],
    ]);
    session.dispose();
  });

  it('keeps capped mesh-area emitter targets on path-replay through adjoint owner tags', async () => {
    const fake = makeFakeEngine();
    const triangleCount = MESH_AREA_LIGHT_TRI_CAP + 1;
    const positions = new Float32Array(triangleCount * 9);
    const normals = new Float32Array(triangleCount * 9);
    for (let tri = 0; tri < triangleCount; tri += 1) {
      const x = tri * 2;
      const o = tri * 9;
      positions.set([x, 0, 0, x + 1, 0, 0, x, 1, 0], o);
      normals.set([0, 0, 1, 0, 0, 1, 0, 0, 1], o);
    }
    fake.scene = {
      ...fake.scene,
      primitives: [{
        kind: 'mesh',
        id: 'panel',
        positions,
        normals,
        material: { baseColor: [0.2, 0.2, 0.2], roughness: 0.5, metallic: 0 },
      }],
      emitters: [{
        kind: 'mesh-area',
        id: 'mesh-light',
        color: [1, 1, 1],
        intensity: 1,
        meshId: 'panel',
      }],
    };
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.6]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'emitters.mesh-light.intensity', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();
    expect(captured).not.toBeNull();
    expect(captured!.params).toEqual([
      { domain: 'emitters', id: 'mesh-light', field: 'intensity', offset: 0, length: 1 },
    ]);
    expect(result.gradient).toEqual([[expect.closeTo(0.6, 6)]]);
    session.dispose();
  });

  it.each([
    ['soft directional', [{
      kind: 'directional' as const,
      id: 'soft-sun',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      direction: [0, -1, 0] as [number, number, number],
      angularDiameter: 0.01,
    }], 'emitters.soft-sun.intensity', 'scalar' as const],
    ['zero-intensity directional intensity', [{
      kind: 'directional' as const,
      id: 'dark-sun',
      color: [1, 0.8, 0.6] as [number, number, number],
      intensity: 0,
      direction: [0, -1, 0] as [number, number, number],
    }], 'emitters.dark-sun.intensity', 'scalar' as const],
    ['black directional color', [{
      kind: 'directional' as const,
      id: 'black-sun',
      color: [0, 0, 0] as [number, number, number],
      intensity: 2,
      direction: [0, -1, 0] as [number, number, number],
    }], 'emitters.black-sun.color', 'rgb' as const],
  ])('keeps %s emitter targets on path-replay now that directional replay is mirrored', (_label, emitters, path, kind) => {
    const fake = makeFakeEngine();
    fake.scene = { ...fake.scene, emitters };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path, kind }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({ path }));
    session.dispose();
  });

  it('keeps emitter path-replay when receiver materials use replayed top-level normal maps', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'emitters.lamp.intensity', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      path: 'emitters.lamp.intensity',
    }));
    session.dispose();
  });

  it('keeps emitter path-replay when receiver materials use replayed bump maps', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                bumpMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 0.5, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'emitters.lamp.intensity', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      path: 'emitters.lamp.intensity',
    }));
    session.dispose();
  });

  it('keeps path-replay for camera-direct emissive params with an emissiveMap', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissive: [0.25, 0.5, 0.75],
                emissiveIntensity: 2,
                emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 0.25, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.emissive', kind: 'rgb' },
        { path: 'materials.panel.emissiveIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([0.25, 0.5, 0.75]);
    expect(session.currentValues()[1]).toEqual([2]);
    session.dispose();
  });

  it('keeps unlit emissive params on path-replay because emission is a primary-hit term', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'none' },
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                shadingModel: 'unlit' as const,
                emissive: [0.25, 0.5, 0.75],
                emissiveIntensity: 2,
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.emissive', kind: 'rgb' },
        { path: 'materials.panel.emissiveIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps camera-direct emissive params on path-replay with normal-only maps', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissive: [0.25, 0.5, 0.75],
                emissiveIntensity: 2,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
                bumpMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 0.5, 1]) } },
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.emissive', kind: 'rgb' },
        { path: 'materials.panel.emissiveIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps alpha-coverage emissive params on finite-difference because visibility is not replayed', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissive: [1, 1, 1],
                alphaMode: 'mask',
                alphaMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-visibility',
      path: 'materials.panel.emissive',
      details: expect.objectContaining({
        field: 'alphaMode',
        finiteDifferenceReason: 'visibility',
      }),
    }));
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses baseColorMap in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                clearcoat: 0.4,
                baseColorMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses an AO map in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                baseColor: [0.8, 0.7, 0.6],
                aoMapIntensity: 0.75,
                aoMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses roughness/metallic maps in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                roughness: 0.7,
                metallic: 0.5,
                roughnessMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 0.6, 1, 1]) } },
                metallicMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 0.4, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.roughness', kind: 'scalar' },
        { path: 'materials.panel.metallic', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses specular maps in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                specularColor: [0.8, 0.7, 0.6],
                specularIntensity: 0.9,
                specularColorMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.75, 1, 1]) } },
                specularIntensityMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 0.4]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.specularColor', kind: 'rgb' },
        { path: 'materials.panel.specularIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses clearcoat/sheen maps in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                clearcoat: 0.6,
                clearcoatRoughness: 0.35,
                sheen: 0.5,
                sheenColor: [0.8, 0.7, 0.6],
                sheenRoughness: 0.4,
                clearcoatMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 1, 1]) } },
                clearcoatRoughnessMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 0.45, 1, 1]) } },
                sheenColorMap: { handle: { width: 1, height: 1, data: new Float32Array([0.8, 0.7, 0.6, 1]) } },
                sheenRoughnessMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 0.55]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(6) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.clearcoat', kind: 'scalar' },
        { path: 'materials.panel.clearcoatRoughness', kind: 'scalar' },
        { path: 'materials.panel.sheenColor', kind: 'rgb' },
        { path: 'materials.panel.sheenRoughness', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.6, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.35, 6)]);
    expect(session.currentValues()[2]).toEqual([
      expect.closeTo(0.8, 6),
      expect.closeTo(0.7, 6),
      expect.closeTo(0.6, 6),
    ]);
    expect(session.currentValues()[3]).toEqual([expect.closeTo(0.4, 6)]);
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses iridescence maps in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescence: 0.7,
                iridescenceIor: 1.45,
                iridescenceThicknessRange: [120, 420] as [number, number],
                iridescenceMap: { handle: { width: 1, height: 1, data: new Float32Array([0.65, 1, 1, 1]) } },
                iridescenceThicknessMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 0.5, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.iridescence', kind: 'scalar' },
        { path: 'materials.panel.iridescenceIor', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.7, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(1.45, 6)]);
    session.dispose();
  });

  it('keeps path-replay when a lit BRDF material uses an anisotropy map in the scoped adjoint domain', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                anisotropy: 0.55,
                anisotropyRotation: 0.3,
                anisotropyMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 0.5, 0.8, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.anisotropy', kind: 'scalar' },
        { path: 'materials.panel.anisotropyRotation', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.55, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.3, 6)]);
    session.dispose();
  });

  it('keeps path-replay when additive emissive/light maps are present on a BRDF target', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissive: [0.2, 0.15, 0.1],
                emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 0.25, 1]) } },
                lightMapIntensity: 2,
                lightMap: { handle: { width: 1, height: 1, data: new Float32Array([0.1, 0.2, 0.3, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps transport params on finite-difference until path replay mirrors transport', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                ior: 1.37,
                transmission: 0.25,
                thickness: 0.75,
                attenuationColor: [0.8, 0.7, 0.6],
                attenuationDistance: 1.25,
                dispersionAbbeNumber: 45,
                scatteringCoefficient: 0.8,
                scatteringAnisotropy: 0.5,
                scatteringCoefficientRGB: [0.6, 0.7, 0.8],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.transmission', kind: 'scalar' },
        { path: 'materials.panel.thickness', kind: 'scalar' },
        { path: 'materials.panel.ior', kind: 'scalar' },
        { path: 'materials.panel.attenuationDistance', kind: 'scalar' },
        { path: 'materials.panel.attenuationColor', kind: 'rgb' },
        { path: 'materials.panel.dispersionAbbeNumber', kind: 'scalar' },
        { path: 'materials.panel.scatteringCoefficient', kind: 'scalar' },
        { path: 'materials.panel.scatteringAnisotropy', kind: 'scalar' },
        { path: 'materials.panel.scatteringCoefficientRGB', kind: 'rgb' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.25, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.75, 6)]);
    expect(session.currentValues()[2]).toEqual([expect.closeTo(1.37, 6)]);
    expect(session.currentValues()[3]).toEqual([expect.closeTo(1.25, 6)]);
    expect(session.currentValues()[4]).toEqual([
      expect.closeTo(0.8, 6),
      expect.closeTo(0.7, 6),
      expect.closeTo(0.6, 6),
    ]);
    expect(session.currentValues()[5]).toEqual([expect.closeTo(45, 6)]);
    expect(session.currentValues()[6]).toEqual([expect.closeTo(0.8, 6)]);
    expect(session.currentValues()[7]).toEqual([expect.closeTo(0.5, 6)]);
    expect(session.currentValues()[8]).toEqual([
      expect.closeTo(0.6, 6),
      expect.closeTo(0.7, 6),
      expect.closeTo(0.8, 6),
    ]);
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.transmission',
      details: expect.objectContaining({ field: 'transmission', finiteDifferenceReason: 'transport' }),
    }));
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.thickness',
      details: expect.objectContaining({ field: 'thickness', finiteDifferenceReason: 'transport' }),
    }));
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.ior',
      details: expect.objectContaining({ field: 'ior', finiteDifferenceReason: 'transport' }),
    }));
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.attenuationDistance',
      details: expect.objectContaining({ field: 'attenuationDistance', finiteDifferenceReason: 'transport' }),
    }));
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-transport',
      path: 'materials.panel.attenuationColor',
      details: expect.objectContaining({ field: 'attenuationColor', finiteDifferenceReason: 'transport' }),
    }));
    for (const field of [
      'dispersionAbbeNumber',
      'scatteringCoefficient',
      'scatteringAnisotropy',
      'scatteringCoefficientRGB',
    ]) {
      expect(session.diagnostics).toContainEqual(expect.objectContaining({
        code: 'path-replay-unsupported-transport',
        path: `materials.panel.${field}`,
        details: expect.objectContaining({ field, finiteDifferenceReason: 'transport' }),
      }));
    }
    session.dispose();
  });

  it('resolves map-free vec2 iridescenceThicknessRange to path-replay', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescence: 1,
                iridescenceIor: 1.5,
                iridescenceThicknessRange: [120, 420],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.iridescenceThicknessRange', kind: 'vec2' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(120, 6),
      expect.closeTo(420, 6),
    ]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps mapped iridescence thickness ranges on path-replay through texture-space range gradients', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescence: 1,
                iridescenceIor: 1.5,
                iridescenceThicknessRange: [120, 420],
                iridescenceThicknessMap: {
                  handle: { width: 1, height: 1, data: new Float32Array([0, 0.5, 0, 1]) },
                },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.iridescenceThicknessRange', kind: 'vec2' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(120, 6),
      expect.closeTo(420, 6),
    ]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps alpha coverage params on finite-difference until path replay mirrors visibility', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                alphaMode: 'mask',
                opacity: 0.75,
                alphaCutoff: 0.4,
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.opacity', kind: 'scalar' },
        { path: 'materials.panel.alphaCutoff', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.75, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.4, 6)]);
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-visibility',
      path: 'materials.panel.opacity',
      details: expect.objectContaining({ field: 'opacity', finiteDifferenceReason: 'visibility' }),
    }));
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-visibility',
      path: 'materials.panel.alphaCutoff',
      details: expect.objectContaining({ field: 'alphaCutoff', finiteDifferenceReason: 'visibility' }),
    }));
    session.dispose();
  });

  it('keeps opaque-material baseColor on path-replay when opacity is inert', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                alphaMode: 'opaque',
                opacity: 0.25,
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps AO map intensity on path-replay through the local base-color chain factor', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                baseColor: [0.8, 0.7, 0.6],
                aoMapIntensity: 0.5,
                aoMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.aoMapIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.5, 6)]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps unlit AO map intensity on path-replay despite irrelevant BRDF lobe flags', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'none' },
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                shadingModel: 'unlit' as const,
                baseColor: [0.8, 0.7, 0.6],
                aoMapIntensity: 0.5,
                aoMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 1, 1, 1]) } },
                anisotropy: 0.25,
                iridescence: 0.4,
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.aoMapIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps light map intensity on path-replay as primary-hit baked radiance', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                lightMapIntensity: 0.4,
                lightMap: { handle: { width: 1, height: 1, data: new Float32Array([0.2, 0.3, 0.4, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.lightMapIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.4, 6)]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps light map intensity on path-replay with normal-only maps', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                lightMapIntensity: 0.4,
                lightMap: { handle: { width: 1, height: 1, data: new Float32Array([0.2, 0.3, 0.4, 1]) } },
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
                bumpMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 0.5, 1]) } },
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.lightMapIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps top-level normalScale on path-replay for replayed normal-map direct lighting', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalScale: 0.8,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.7, 0.45, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.normalScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()).toEqual([[expect.closeTo(0.8, 6)]]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps bumpScale on path-replay for replayed bump-map direct lighting', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                bumpScale: 0.7,
                bumpMap: { handle: { width: 2, height: 1, data: new Float32Array([0.1, 0, 0, 1, 0.9, 0, 0, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.bumpScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()).toEqual([[expect.closeTo(0.7, 6)]]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps normalScale on path-replay when chained through a bump-map tangent frame', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalScale: 0.8,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.7, 0.45, 1, 1]) } },
                bumpMap: { handle: { width: 2, height: 1, data: new Float32Array([0.1, 0, 0, 1, 0.9, 0, 0, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.normalScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps clearcoatNormalScale on path-replay for replayed clearcoat-normal direct lighting', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                clearcoat: 0.65,
                clearcoatNormalScale: 0.6,
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.7, 0.45, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.clearcoatNormalScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()).toEqual([
      [expect.closeTo(0.6, 6)],
    ]);
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps normalScale on path-replay when chained through a clearcoat-normal tangent frame', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalScale: 0.8,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.7, 0.45, 1, 1]) } },
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.45, 0.7, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.normalScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps bumpScale on path-replay when chained through a clearcoat-normal tangent frame', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                bumpScale: 0.7,
                bumpMap: { handle: { width: 2, height: 1, data: new Float32Array([0.1, 0, 0, 1, 0.9, 0, 0, 1]) } },
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.45, 0.7, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.bumpScale', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps envMapIntensity on scoped path-replay for direct HDRI environment NEE', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, envMapIntensity: 0.3 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.envMapIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.currentValues()).toEqual([[expect.closeTo(0.3, 6)]]);
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-environment',
    }));
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-lighting',
    }));
    session.dispose();
  });

  it('keeps path-replay for emissive params when a baked light map is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                emissive: [0.25, 0.5, 1],
                emissiveIntensity: 2,
                lightMap: { handle: { width: 1, height: 1, data: new Float32Array([0.1, 0.2, 0.3, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.emissive', kind: 'rgb' },
        { path: 'materials.panel.emissiveIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('keeps base BRDF controls on path-replay when anisotropy is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                anisotropy: 0.55,
                anisotropyRotation: 0.3,
                anisotropyMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 0.5, 0.8, 1]) } },
                specularColor: [0.7, 0.8, 0.9],
                specularIntensity: 0.65,
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(9) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb' },
        { path: 'materials.panel.roughness', kind: 'scalar' },
        { path: 'materials.panel.metallic', kind: 'scalar' },
        { path: 'materials.panel.specularColor', kind: 'rgb' },
        { path: 'materials.panel.specularIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-material',
    }));
    session.dispose();
  });

  it('keeps base BRDF controls on path-replay when a top-level normalMap is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                normalScale: 0.8,
                normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.7, 0.45, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps base BRDF controls on path-replay when a clearcoat-normal map is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                clearcoat: 0.6,
                clearcoatRoughness: 0.25,
                clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.45, 0.7, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps path-replay when an alpha map is dormant under opaque alphaMode', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                alphaMode: 'opaque',
                alphaMap: { handle: { width: 1, height: 1, data: new Float32Array([0.25, 0.25, 0.25, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps path-replay when transmission and thickness maps are dormant at zero transmission', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                transmission: 0,
                transmissionMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0, 0, 1]) } },
                thickness: 0.75,
                thicknessMap: { handle: { width: 1, height: 1, data: new Float32Array([0, 0.5, 0, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it('keeps path-replay when spectral and scattering metadata are dormant on an opaque material', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                transmission: 0,
                dispersionAbbeNumber: 42,
                spectralAttenuation: {
                  wavelengthStart: 380,
                  wavelengthEnd: 700,
                  values: new Float32Array([0.1, 0.2, 0.3]),
                },
                scatteringCoefficient: 0.3,
                scatteringCoefficientRGB: [0.1, 0.2, 0.3],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).toEqual([]);
    session.dispose();
  });

  it.each([
    ['alpha mode', { alphaMode: 'mask' as const }, 'path-replay-unsupported-visibility', 'visibility'],
    ['blend opacity', { alphaMode: 'blend' as const, opacity: 0.75 }, 'path-replay-unsupported-visibility', 'visibility'],
    ['transmission', { transmission: 0.25 }, 'path-replay-unsupported-transport', 'transport'],
    ['alpha map', { alphaMode: 'mask' as const, alphaMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } } }, 'path-replay-unsupported-visibility', 'visibility'],
    ['transmission map', { transmission: 0.25, transmissionMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } } }, 'path-replay-unsupported-transport', 'transport'],
    ['thickness map', { transmission: 0.25, thicknessMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } } }, 'path-replay-unsupported-transport', 'transport'],
    ['displacement map', { displacementMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } } }, 'path-replay-unsupported-material', 'geometry'],
    ['front layer', { frontLayer: { transmission: [0.9, 0.8, 0.7] as [number, number, number] } }, 'path-replay-unsupported-transport', 'transport'],
    ['thin-film stack', { thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 180 }] } }, 'path-replay-unsupported-transport', 'transport'],
    ['spectral attenuation', {
      transmission: 0.25,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
    }, 'path-replay-unsupported-transport', 'transport'],
    ['volume scattering', { transmission: 0.25, scatteringCoefficientRGB: [0.1, 0.2, 0.3] as [number, number, number] }, 'path-replay-unsupported-transport', 'transport'],
  ])('degrades to finite-difference for path-replay material with %s', (_label, patch, expectedCode, expectedReason) => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, ...patch } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: expectedCode,
      path: 'materials.panel.baseColor',
      ...(expectedReason != null
        ? { details: expect.objectContaining({ finiteDifferenceReason: expectedReason }) }
        : {}),
    }));
    session.dispose();
  });

  it('keeps material path-replay when a contributing HDRI environment is the direct-light source', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-light-selection',
    }));
    session.dispose();
  });

  it('degrades material path-replay when multiple direct-light candidates require selection replay', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-light-selection',
      path: 'materials.panel.baseColor',
      details: expect.objectContaining({
        candidateCount: 2,
        directLighting: 'sampled-selection',
        candidates: expect.arrayContaining([
          'emitter:lamp:point',
          'environment:hdri',
        ]),
      }),
    }));
    session.dispose();
  });

  it('keeps material path-replay for multiple direct-light candidates when the forward context sums expectations', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } },
    };
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayRenderContext: () => ({ directLighting: 'summed-expectation' }),
      computeAdjointGradient: async () => new Float32Array(3),
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-light-selection',
    }));
    session.dispose();
  });

  it.each([
    ['zero-intensity HDRI', { kind: 'hdri' as const, hdri: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) }, intensity: 0 }],
    ['zero-intensity procedural sky', {
      kind: 'procedural-sky' as const,
      sunDirection: [0, 1, 0] as [number, number, number],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 0,
    }],
  ])('keeps path-replay for %s because the environment contributes no radiance', (_label, environment) => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      environment,
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-lighting',
    }));
    session.dispose();
  });

  it('keeps path-replay for rect-area direct-light scenes', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [{
        kind: 'rect-area',
        id: 'panel-light',
        color: [1, 1, 1],
        intensity: 1,
        position: [0, 1, 0],
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
      }],
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it.each([
    ['directional', [{
      kind: 'directional' as const,
      id: 'sun',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      direction: [0, -1, 0] as [number, number, number],
    }]],
    ['soft directional', [{
      kind: 'directional' as const,
      id: 'soft-sun',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      direction: [0, -1, 0] as [number, number, number],
      angularDiameter: 0.01,
    }]],
    ['spot', [{
      kind: 'spot' as const,
      id: 'spot',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      position: [0, 1, 0] as [number, number, number],
      direction: [0, -1, 0] as [number, number, number],
      angle: 0.5,
    }]],
    ['disc-area', [{
      kind: 'disc-area' as const,
      id: 'disc-light',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      position: [0, 1, 0] as [number, number, number],
      normal: [0, -1, 0] as [number, number, number],
      radius: 0.5,
    }]],
    ['mesh-area', [{
      kind: 'mesh-area' as const,
      id: 'mesh-light',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      meshId: 'panel',
    }]],
  ])('keeps path-replay for %s direct-light scenes now covered by the adjoint pass', (_label, emitters) => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters,
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('resolves KHR_materials_specular controls to path-replay when the adjoint hook is present', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.specularColor', kind: 'rgb' },
        { path: 'materials.panel.specularIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([1, 1, 1]);
    expect(session.currentValues()[1]).toEqual([1]);
    session.dispose();
  });

  it('resolves metallic to path-replay in the base direct-light adjoint domain', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.metallic', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([0]);
    session.dispose();
  });

  it('resolves map-free clearcoat controls to path-replay when the adjoint hook is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, clearcoat: 0.35, clearcoatRoughness: 0.42 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.clearcoat', kind: 'scalar' },
        { path: 'materials.panel.clearcoatRoughness', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.35, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.42, 6)]);
    session.dispose();
  });

  it('resolves map-free sheen controls to path-replay when the adjoint hook is present', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                sheen: 0.44,
                sheenRoughness: 0.57,
                sheenColor: [0.8, 0.3, 0.15],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(5) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.sheen', kind: 'scalar' },
        { path: 'materials.panel.sheenColor', kind: 'rgb' },
        { path: 'materials.panel.sheenRoughness', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.44, 6)]);
    expect(session.currentValues()[1]).toEqual([
      expect.closeTo(0.8, 6),
      expect.closeTo(0.3, 6),
      expect.closeTo(0.15, 6),
    ]);
    expect(session.currentValues()[2]).toEqual([expect.closeTo(0.57, 6)]);
    session.dispose();
  });

  it('resolves map-free scalar iridescence to path-replay when it is the optimized field', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescence: 0.41,
                iridescenceIor: 1.35,
                iridescenceThicknessRange: [120, 360] as [number, number],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.iridescence', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.41, 6)]);
    session.dispose();
  });

  it('resolves map-free scalar iridescenceIor to path-replay when it is the optimized field', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                iridescence: 0.41,
                iridescenceIor: 1.35,
                iridescenceThicknessRange: [120, 360] as [number, number],
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.iridescenceIor', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(1.35, 6)]);
    session.dispose();
  });

  it('keeps coupled BRDF params on finite-difference while iridescence is being optimized', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, iridescence: 0.0 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb' },
        { path: 'materials.panel.iridescence', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference');
    session.dispose();
  });

  it('keeps coupled BRDF params on finite-difference while iridescenceIor is being optimized', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, iridescence: 0.25, iridescenceIor: 1.35 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(4) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb' },
        { path: 'materials.panel.iridescenceIor', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference');
    session.dispose();
  });

  it('resolves map-free scalar anisotropy controls to path-replay when optimized directly', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, anisotropy: 0.42, anisotropyRotation: 0.31 } }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.anisotropy', kind: 'scalar' },
        { path: 'materials.panel.anisotropyRotation', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    expect(session.currentValues()[0]).toEqual([expect.closeTo(0.42, 6)]);
    expect(session.currentValues()[1]).toEqual([expect.closeTo(0.31, 6)]);
    session.dispose();
  });

  it('keeps single-field anisotropy-map materials on path-replay for anisotropy replay targets', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? {
              ...pr,
              material: {
                ...pr.material,
                anisotropy: 0.42,
                anisotropyMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
              },
            }
          : pr,
      ),
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.anisotropy', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('path-replay');
    session.dispose();
  });

  it('degrades to finite-difference for primitive targets the adjoint pass cannot replay as triangles', () => {
    const fake = makeFakeEngine();
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    const cases: Scene['primitives'] = [
      {
        kind: 'analytic',
        id: 'panel',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: baseMesh.material,
      },
    ];
    for (const primitive of cases) {
      fake.scene = { ...fake.scene, primitives: [primitive] };
      const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };
      const session = new PtWebgpuInverseSession(hooks, {
        target: targetImage(2, 2, [0.8, 0.1, 0.1]),
        parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
        method: 'path-replay',
      });
      expect(session.method).toBe('finite-difference');
      expect(session.diagnostics).toContainEqual(expect.objectContaining({
        code: 'path-replay-unsupported-primitive',
        path: 'materials.panel.baseColor',
      }));
      session.dispose();
    }
  });

  it('keeps path-replay for instanced mesh targets because the adjoint pass bakes every instance', () => {
    const fake = makeFakeEngine();
    const identity = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]));
    const translated = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, 0, 0, 1,
    ]));
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = {
      ...fake.scene,
      primitives: [{
        kind: 'instanced-mesh',
        id: 'panel',
        positions: baseMesh.positions,
        normals: baseMesh.normals,
        material: baseMesh.material,
        instances: [identity, translated],
      }],
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-primitive',
      path: 'materials.panel.baseColor',
    }));
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-scene-geometry',
      path: 'materials.panel.baseColor',
    }));
    session.dispose();
  });

  it('degrades to finite-difference for zero-instance instanced mesh targets', () => {
    const fake = makeFakeEngine();
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = {
      ...fake.scene,
      primitives: [{
        kind: 'instanced-mesh',
        id: 'panel',
        positions: baseMesh.positions,
        normals: baseMesh.normals,
        material: baseMesh.material,
        instances: [],
      }],
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-primitive',
      path: 'materials.panel.baseColor',
      details: expect.objectContaining({ instanceCount: 0 }),
    }));
    session.dispose();
  });

  it('keeps path-replay for transformed mesh targets because the adjoint pass bakes a world-space replay stream', () => {
    const fake = makeFakeEngine();
    const translated = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, 0, 0, 1,
    ]));
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = { ...fake.scene, primitives: [{ ...baseMesh, transform: translated }] };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-primitive',
      path: 'materials.panel.baseColor',
    }));
    session.dispose();
  });

  it('degrades to finite-difference when non-flat scene geometry would affect path replay visibility', () => {
    const fake = makeFakeEngine();
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = {
      ...fake.scene,
      primitives: [
        baseMesh,
        {
          kind: 'analytic',
          id: 'sphere-occluder',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: baseMesh.material,
        },
      ],
    };
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(3) };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-scene-geometry',
      path: 'materials.panel.baseColor',
    }));
    session.dispose();
  });

  it('keeps path-replay for supported analytic scene geometry via tessellated replay', () => {
    const fake = makeFakeEngine();
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = {
      ...fake.scene,
      primitives: [
        baseMesh,
        {
          kind: 'analytic',
          id: 'sphere-occluder',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: baseMesh.material,
        },
      ],
    };
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayGeometryCapabilities: () => ({ supportedAnalyticShapes: new Set(['sphere']) }),
      computeAdjointGradient: async () => new Float32Array(3),
    };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-scene-geometry',
      path: 'materials.panel.baseColor',
    }));
    session.dispose();
  });

  it('keeps path-replay for supported analytic material targets via tessellated replay', () => {
    const fake = makeFakeEngine();
    const baseMesh = fake.scene.primitives[0]!;
    if (baseMesh.kind !== 'mesh') throw new Error('test fixture expected a mesh primitive');
    fake.scene = {
      ...fake.scene,
      primitives: [{
        kind: 'analytic',
        id: 'panel',
        shape: 'box',
        params: new Float32Array([0, 0, 0, 1, 1, 1]),
        material: baseMesh.material,
      }],
    };
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getPathReplayGeometryCapabilities: () => ({ supportedAnalyticShapes: new Set(['box']) }),
      computeAdjointGradient: async () => new Float32Array(3),
    };

    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      method: 'path-replay',
    });

    expect(session.method).toBe('path-replay');
    expect(session.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'path-replay-unsupported-primitive',
      path: 'materials.panel.baseColor',
    }));
    session.dispose();
  });

  it('passes specular adjoint params to the hook with the expected flat offsets', async () => {
    const fake = makeFakeEngine();
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.1, 0.2, 0.3, 0.4]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [
        { path: 'materials.panel.specularColor', kind: 'rgb' },
        { path: 'materials.panel.specularIntensity', kind: 'scalar' },
      ],
      method: 'path-replay',
    });
    const result = await session.step();
    const req = captured as AdjointGradientRequest | null;
    expect(req).not.toBeNull();
    expect(req!.gradientLength).toBe(4);
    expect(req!.params).toEqual([
      { domain: 'materials', id: 'panel', field: 'specularColor', offset: 0, length: 3 },
      { domain: 'materials', id: 'panel', field: 'specularIntensity', offset: 3, length: 1 },
    ]);
    expect(result.gradient).toEqual([
      [expect.closeTo(0.1, 6), expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)],
      [expect.closeTo(0.4, 6)],
    ]);
    session.dispose();
  });

  it('step() calls the hook with dLoss_dRendered + uses its gradient (one render, no FD probes)', async () => {
    const fake = makeFakeEngine();
    let captured: AdjointGradientRequest | null = null;
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      computeAdjointGradient: async (req) => {
        captured = req;
        return new Float32Array([0.5, -0.25, 0.125]);
      },
    };
    const session = new PtWebgpuInverseSession(hooks, eligibleOpts());
    const before = fake.renderCount;
    const result = await session.step();
    // Exactly ONE render (the baseline) — the N-render FD probe loop did NOT run.
    expect(fake.renderCount - before).toBe(1);
    expect(captured).not.toBeNull();
    expect(captured!.dLoss_dRendered.length).toBe(2 * 2 * 3); // per-pixel loss gradient
    expect(captured!.samples).toBe(4); // consumed by the engine adjoint replay loop
    expect(captured!.gradientLength).toBe(3);
    expect(captured!.params[0]).toMatchObject({ domain: 'materials', id: 'panel', field: 'baseColor', offset: 0, length: 3 });
    expect(result.gradient[0]).toEqual([0.5, -0.25, 0.125]); // used the hook's gradient verbatim (pre-Adam)
    session.dispose();
  });

  it('throws if the adjoint hook returns a wrong-length gradient', async () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(2) };
    const session = new PtWebgpuInverseSession(hooks, eligibleOpts()); // expects length 3
    await expect(session.step()).rejects.toThrow(/gradient length/);
    session.dispose();
  });
});

describe('inverse optimizer helpers — pure unit math', () => {
  it('l2Loss: identical images ⇒ zero loss and zero gradient', () => {
    const img = new Float32Array([0.5, 0.2, 0.1, 0.5, 0.2, 0.1]);
    const t = { data: img.slice(), width: 2, height: 1, channels: 3 as const };
    const { loss, dLoss_dRendered } = l2Loss(img, 3, t);
    expect(loss).toBeCloseTo(0, 12);
    for (const g of dLoss_dRendered) expect(g).toBeCloseTo(0, 12);
  });

  it('l2Loss: gradient = 2·(rendered−target)/N', () => {
    const rendered = new Float32Array([1, 0, 0, 0, 0, 0]); // 2px RGB
    const t = { data: new Float32Array(6), width: 2, height: 1, channels: 3 as const };
    const { loss, dLoss_dRendered } = l2Loss(rendered, 3, t);
    const N = 6;
    expect(loss).toBeCloseTo(1 / N, 12);
    // dLoss is stored f32, so compare at f32 precision.
    expect(dLoss_dRendered[0]!).toBeCloseTo((2 * 1) / N, 6);
  });

  it('l2Loss: 4-channel rendered image (alpha ignored) reads first 3 channels', () => {
    // 1px RGBA rendered, RGB target. The alpha (4th) channel must be skipped.
    const rendered = new Float32Array([0.5, 0.25, 0.1, 999]); // alpha = 999 must NOT leak
    const t = { data: new Float32Array([0.5, 0.25, 0.1]), width: 1, height: 1, channels: 3 as const };
    expect(l2Loss(rendered, 4, t).loss).toBeCloseTo(0, 12);
  });

  it('lossValue: scalar-only path equals the full loss (l2 + l1, no gradient alloc)', () => {
    const rendered = new Float32Array([0.7, 0.2, 0.1, 0.3, 0.5, 0.9]);
    const t = { data: new Float32Array([0.5, 0.2, 0.1, 0.3, 0.4, 1.0]), width: 2, height: 1, channels: 3 as const };
    expect(lossValue(rendered, 3, t, 'l2')).toBeCloseTo(l2Loss(rendered, 3, t).loss, 12);
    expect(lossValue(rendered, 3, t, 'l1')).toBeCloseTo(l1Loss(rendered, 3, t).loss, 12);
  });

  it('loss is finite-safe: non-finite rendered pixels (firefly ±Inf / NaN) map to 0, never poison the loss', () => {
    // A path tracer can produce ±Inf firefly pixels (representable in rgba16float)
    // or NaN from a degenerate sample. Without the guard a SINGLE bad pixel makes
    // the mean image loss Inf/NaN, which NaNs the finite-difference gradient and
    // the Adam step — silently stalling the optimizer (surfaced by the V24 GPU run).
    const t = { data: new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), width: 2, height: 1, channels: 3 as const };
    const withInf = new Float32Array([Infinity, 0.5, 0.5, 0.5, 0.5, 0.5]);
    const withNaN = new Float32Array([NaN, 0.5, 0.5, 0.5, 0.5, 0.5]);
    const withNegInf = new Float32Array([-Infinity, 0.5, 0.5, 0.5, 0.5, 0.5]);
    // Each bad pixel is treated as 0 → only that channel contributes (0−0.5)²; all
    // others match exactly. So the loss is finite and equals (0.25)/6 for l2.
    for (const bad of [withInf, withNaN, withNegInf]) {
      const l2 = l2Loss(bad, 3, t);
      expect(Number.isFinite(l2.loss)).toBe(true);
      expect(l2.loss).toBeCloseTo(0.25 / 6, 12);
      expect(l2.dLoss_dRendered.every((x) => Number.isFinite(x))).toBe(true);
      expect(Number.isFinite(lossValue(bad, 3, t, 'l2'))).toBe(true);
      expect(Number.isFinite(lossValue(bad, 3, t, 'l1'))).toBe(true);
      const l1 = l1Loss(bad, 3, t);
      expect(Number.isFinite(l1.loss)).toBe(true);
      expect(l1.dLoss_dRendered.every((x) => Number.isFinite(x))).toBe(true);
    }
  });

  it('l1Loss: mean absolute error + sign gradient', () => {
    const rendered = new Float32Array([1, 0, 0, 0, 0, 0]);
    const t = { data: new Float32Array(6), width: 2, height: 1, channels: 3 as const };
    const { loss, dLoss_dRendered } = l1Loss(rendered, 3, t);
    expect(loss).toBeCloseTo(1 / 6, 12);
    expect(dLoss_dRendered[0]!).toBeCloseTo(1 / 6, 6);
  });

  it('Adam: descends a 1-D quadratic toward the minimum', () => {
    // Minimize f(x) = (x − 3)², grad = 2(x−3). Start at 0.
    const p = new Float32Array([0]);
    const adam = new Adam(1, { learningRate: 0.3, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 });
    for (let i = 0; i < 200; i++) {
      const g = new Float32Array([2 * (p[0]! - 3)]);
      adam.step(p, g);
    }
    expect(p[0]!).toBeCloseTo(3, 1);
  });

  it('parseParamPath splits domain / id / field (id may contain dots)', () => {
    expect(parseParamPath('materials.panel-1.roughness')).toEqual({
      domain: 'materials', id: 'panel-1', field: 'roughness',
    });
    expect(parseParamPath('emitters.sun.intensity')).toEqual({
      domain: 'emitters', id: 'sun', field: 'intensity',
    });
    expect(parseParamPath('materials.a.b.c.baseColor')).toEqual({
      domain: 'materials', id: 'a.b.c', field: 'baseColor',
    });
    expect(() => parseParamPath('bogus.x.y')).toThrow(/unknown domain/);
    expect(() => parseParamPath('materials.x')).toThrow(/must be/);
  });

  it('paramLength: scalar=1, rgb=3, texture throws', () => {
    expect(paramLength({ path: 'p', kind: 'scalar' })).toBe(1);
    expect(paramLength({ path: 'p', kind: 'rgb' })).toBe(3);
    expect(() => paramLength({ path: 'p', kind: 'texture' })).toThrow(/texture/);
  });
});
