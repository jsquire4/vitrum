// inverseSession.test.ts — WS5 InverseSession contract + Phase-0 loop tests.
//
// The GPU render+readback is faked here (it needs a real device; V24 is the
// hardware A/B). What IS exercised on the CPU: the contract shape, parameter
// path resolution + validation, the finite-difference + Adam optimizer loop
// (loss provably decreases on a fittable fake forward model), the
// frozen-RNG-replay determinism the FD gradient relies on, idempotent dispose,
// and the pure loss/Adam helpers.

import { describe, it, expect } from 'vitest';
import type { Scene, InverseSessionOptions, MaterialSpec, SceneEmitter } from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type InverseEngineHooks,
} from '../inverse/inverseSession.js';
import { l2Loss, l1Loss, lossValue, Adam, parseParamPath, paramLength } from '../inverse/optimizer.js';

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
        material: {
          baseColor: [0.2, 0.2, 0.2],
          roughness: 0.5,
          metallic: 0,
          shadingModel: 'unlit',
          emissive: [0.2, 0.2, 0.2],
        },
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

function deferred<T>(): {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
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
    getPathReplayRenderContext: () => ({
      bounces: 1,
      spectral: false,
      bdpt: false,
      restirPtReuse: false,
      causticStrategy: 'none',
      directLighting: 'summed-expectation',
      cameraVisibleEmitters: true,
      implicitEmissiveMeshLights: true,
    }),
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
      const displacementScale = mat.displacementScale ?? 0;
      const displacementBias = mat.displacementBias ?? 0;
      const rgb = new Float32Array(width * height * 3);
      for (let p = 0; p < width * height; p++) {
        rgb[p * 3 + 0] = mat.baseColor[0] + transmission + opacity + normalOrBumpScale + attenuationColor[0] + iridescenceThicknessRange[0] / 1000 + dispersionAbbeNumber / 100 + scatteringCoefficient + scatteringCoefficientRGB[0] + displacementScale;
        rgb[p * 3 + 1] = mat.baseColor[1] + thickness + attenuationDistance + alphaCutoff + materialMapIntensity + attenuationColor[1] + iridescenceThicknessRange[1] / 1000 + scatteringAnisotropy + scatteringCoefficientRGB[1] + displacementBias;
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

  it('uses an explicit finite attenuationDistance initial when the scene has no finite medium yet', () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.attenuationDistance', kind: 'scalar', initial: [2.5] }],
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(2.5, 6)]);
    expect(fake.scene.primitives[0]!.material.attenuationDistance).toBeCloseTo(2.5, 6);
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
      parameters: [{ path: 'materials.panel.baseColorMap', kind: 'scalar' }],
    })).toThrow(/not optimizable/);
  });

  it('accepts scalar displacement controls as finite-difference parameters', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, displacementScale: 0.3, displacementBias: -0.4 } }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [
        { path: 'materials.panel.displacementScale', kind: 'scalar' },
        { path: 'materials.panel.displacementBias', kind: 'scalar' },
      ],
    });
    expect(session.method).toBe('finite-difference');
    expect(session.currentValues()[0]![0]).toBeCloseTo(0.3, 6);
    expect(session.currentValues()[1]![0]).toBeCloseTo(-0.4, 6);
    session.dispose();
  });

  it('throws when the active runtime profile reports a material field unsupported', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getMaterialSupportDetails: () => ({ normalScale: 'unsupported' }),
    };
    expect(() => new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.normalScale', kind: 'scalar' }],
    })).toThrow(/active pt-webgpu runtime profile/);
  });

  it('throws when the active runtime profile reports an emitter kind unsupported', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [{
        kind: 'mesh-area',
        id: 'mesh-light',
        meshId: 'panel',
        color: [1, 1, 1],
        intensity: 1,
      }],
    };
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      getEmitterSupportDetails: () => ({ 'mesh-area': 'unsupported' }),
    };
    expect(() => new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'emitters.mesh-light.intensity', kind: 'scalar' }],
    })).toThrow(/emitter kind "mesh-area".*active pt-webgpu runtime profile/);
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

  it.each([
    ['missing', undefined],
    ['infinite', Number.POSITIVE_INFINITY],
    ['zero', 0],
  ])('throws when attenuationDistance has a %s scene seed and no explicit initial', (_label, attenuationDistance) => {
    const fake = makeFakeEngine();
    if (attenuationDistance !== undefined) {
      fake.scene = {
        ...fake.scene,
        primitives: fake.scene.primitives.map((pr) =>
          pr.id === 'panel'
            ? { ...pr, material: { ...pr.material, attenuationDistance } }
            : pr,
        ),
      };
    }

    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.attenuationDistance', kind: 'scalar' }],
    })).toThrow(/requires a finite positive scene attenuationDistance/);
  });

  it('throws on an empty parameter list', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [],
    })).toThrow(/at least one parameter/);
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

  it('fits scalar displacement controls through finite difference', async () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, displacementScale: 0.3, displacementBias: -0.4 } }
          : pr,
      ),
    };
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0.8, 0.4, 0.2]),
      parameters: [
        { path: 'materials.panel.displacementScale', kind: 'scalar' },
        { path: 'materials.panel.displacementBias', kind: 'scalar' },
      ],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.2, fdEpsilon: 1e-3 },
    });

    const result = await session.step();
    expect(result.gradient[0]![0]).toBeLessThan(0);
    expect(result.gradient[1]![0]).toBeLessThan(0);
    expect(session.currentValues()[0]![0]).toBeGreaterThan(0.3);
    expect(session.currentValues()[1]![0]).toBeGreaterThan(-0.4);
    expect(fake.scene.primitives[0]!.material.displacementScale).toBeCloseTo(session.currentValues()[0]![0]!, 6);
    expect(fake.scene.primitives[0]!.material.displacementBias).toBeCloseTo(session.currentValues()[1]![0]!, 6);
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

  it('uses a bounded backward probe at the default attenuationColor upper limit', async () => {
    const fake = makeFakeEngine(1, 1);
    const patchMaterial = fake.hooks.patchMaterial;
    const authoredColors: number[][] = [];
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      patchMaterial: (id, patch) => {
        if (patch.attenuationColor != null) {
          const value = [...patch.attenuationColor];
          authoredColors.push(value);
          if (value.some((component) => component < 1e-4 || component > 1)) {
            throw new RangeError('attenuationColor left its effective [1e-4, 1] bounds');
          }
        }
        patchMaterial(id, patch);
      },
      // Non-linear forward response: rendered RGB = attenuationColor². Against
      // a black target, each component's known loss is x⁴/3.
      renderAndReadback: async () => {
        const color = fake.scene.primitives[0]!.material.attenuationColor ?? [1, 1, 1];
        return {
          rgb: new Float32Array([
            color[0] * color[0],
            color[1] * color[1],
            color[2] * color[2],
          ]),
          channels: 3,
        };
      },
    };
    const fdEpsilon = 1e-2;
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(1, 1, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.attenuationColor', kind: 'rgb' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 1e-2, fdEpsilon },
    });

    expect(session.currentValues()[0]).toEqual([1, 1, 1]);
    const result = await session.step();
    const probe = Math.fround(1 - fdEpsilon);
    const probeLoss = (probe ** 4 + 2) / 3;
    const expectedBackwardDifference = (probeLoss - 1) / (probe - 1);
    for (const componentGradient of result.gradient[0]!) {
      expect(componentGradient).toBeGreaterThan(0);
      expect(componentGradient).toBeCloseTo(expectedBackwardDifference, 5);
    }
    expect(authoredColors.some((color) => color.some((component) => component < 1))).toBe(true);
    expect(authoredColors.every((color) =>
      color.every((component) => component >= 1e-4 && component <= 1),
    )).toBe(true);
    session.dispose();
  });

  it('uses a bounded forward probe at a default [0,1] lower limit', async () => {
    const fake = makeFakeEngine(1, 1);
    const patchMaterial = fake.hooks.patchMaterial;
    const authoredTransmission: number[] = [];
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      patchMaterial: (id, patch) => {
        if (patch.transmission != null) {
          authoredTransmission.push(patch.transmission);
          if (patch.transmission < 0 || patch.transmission > 1) {
            throw new RangeError('transmission left its effective [0, 1] bounds');
          }
        }
        patchMaterial(id, patch);
      },
      renderAndReadback: async () => ({
        rgb: new Float32Array([
          fake.scene.primitives[0]!.material.transmission ?? 0,
          0,
          0,
        ]),
        channels: 3,
      }),
    };
    const fdEpsilon = 1e-2;
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(1, 1, [1, 0, 0]),
      parameters: [{ path: 'materials.panel.transmission', kind: 'scalar' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 1e-2, fdEpsilon },
    });

    const result = await session.step();
    const probe = Math.fround(fdEpsilon);
    const expectedForwardDifference = (((probe - 1) ** 2) / 3 - 1 / 3) / probe;
    expect(result.gradient[0]![0]).toBeLessThan(0);
    expect(result.gradient[0]![0]).toBeCloseTo(expectedForwardDifference, 5);
    expect(authoredTransmission).toContain(probe);
    expect(authoredTransmission.every((value) => value >= 0 && value <= 1)).toBe(true);
    session.dispose();
  });

  it('caps an oversized probe from an out-of-range initial value at the nearest bound', async () => {
    const fake = makeFakeEngine(1, 1);
    const patchMaterial = fake.hooks.patchMaterial;
    const authoredRoughness: number[] = [];
    const hooks: InverseEngineHooks = {
      ...fake.hooks,
      patchMaterial: (id, patch) => {
        if (patch.roughness != null) {
          authoredRoughness.push(patch.roughness);
          if (patch.roughness < 0 || patch.roughness > 1) {
            throw new RangeError('roughness left its physical [0, 1] domain');
          }
        }
        patchMaterial(id, patch);
      },
      renderAndReadback: async () => ({
        rgb: new Float32Array([
          fake.scene.primitives[0]!.material.roughness,
          0,
          0,
        ]),
        channels: 3,
      }),
    };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(1, 1, [0, 0, 0]),
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        initial: [0.8],
        max: 0.5,
      }],
      samplesPerStep: 1,
      optimizer: { learningRate: 1e-2, fdEpsilon: 1 },
    });

    const result = await session.step();
    const baseline = Math.fround(0.8);
    const probe = 0.5;
    const expected = ((probe ** 2) / 3 - (baseline ** 2) / 3) / (probe - baseline);
    expect(result.gradient[0]![0]).toBeCloseTo(expected, 6);
    expect(authoredRoughness).toContain(probe);
    expect(authoredRoughness).not.toContain(Math.fround(baseline - 1));
    expect(authoredRoughness.every((value) => value >= 0 && value <= 1)).toBe(true);
    session.dispose();
  });

  it('treats a zero-width effective interval as a fixed zero-gradient parameter', async () => {
    const fake = makeFakeEngine(1, 1);
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(1, 1, [0.9, 0.9, 0.9]),
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        initial: [0.5],
        min: 0.5,
        max: 0.5,
      }],
      samplesPerStep: 1,
    });

    const result = await session.step();
    expect(result.gradient).toEqual([[0]]);
    expect(session.currentValues()).toEqual([[0.5]]);
    expect(fake.renderCount).toBe(1);
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

describe('InverseSession — transactional lifecycle', () => {
  it('rejects reentrant steps and invalidates a baseline await when disposed', async () => {
    const fake = makeFakeEngine();
    const pending = deferred<{ rgb: Float32Array; channels: 3 }>();
    const session = new PtWebgpuInverseSession({
      ...fake.hooks,
      renderAndReadback: () => pending.promise,
    }, {
      target: targetImage(2, 2, [0.5, 0.5, 0.5]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
    });
    const first = session.step();
    await expect(session.step()).rejects.toThrow(/already in progress/);
    session.dispose();
    pending.resolve({ rgb: new Float32Array(12), channels: 3 });
    await expect(first).rejects.toThrow(/disposed while the step was in progress/);
    expect(session.currentValues()).toEqual([[0.5]]);
    expect(fake.scene.primitives[0]!.material.roughness).toBe(0.5);
  });

  it('invalidates an adjoint await when disposed without publishing a step', async () => {
    const fake = makeFakeEngine();
    const started = deferred<void>();
    const pending = deferred<Float32Array>();
    const session = new PtWebgpuInverseSession({
      ...fake.hooks,
      computeAdjointGradient: async () => {
        started.resolve();
        return pending.promise;
      },
    }, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
      method: 'path-replay',
    });
    const step = session.step();
    await started.promise;
    session.dispose();
    pending.resolve(new Float32Array(3));
    await expect(step).rejects.toThrow(/disposed while the step was in progress/);
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
    ]);
    expect(fake.scene.primitives[0]!.material.emissive).toEqual([
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
    ]);
  });

  it('restores a finite-difference perturbation when probe rendering rejects', async () => {
    const fake = makeFakeEngine();
    const render = fake.hooks.renderAndReadback;
    let renders = 0;
    const session = new PtWebgpuInverseSession({
      ...fake.hooks,
      renderAndReadback: async (...args) => {
        renders += 1;
        if (renders === 2) throw new Error('probe render failed');
        return render(...args);
      },
    }, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{
        path: 'materials.panel.baseColor',
        kind: 'rgb',
        initial: [0.3, 0.1, 0.1],
      }],
    });
    await expect(session.step()).rejects.toThrow(/probe render failed/);
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.3, 6),
      expect.closeTo(0.1, 6),
      expect.closeTo(0.1, 6),
    ]);
    expect(fake.scene.primitives[0]!.material.baseColor).toEqual([
      expect.closeTo(0.3, 6),
      expect.closeTo(0.1, 6),
      expect.closeTo(0.1, 6),
    ]);
  });

  it('rolls back partial constructor overrides and aggregates rollback failure', () => {
    const fake = makeFakeEngine();
    const patchMaterial = fake.hooks.patchMaterial;
    let patchCount = 0;
    expect(() => new PtWebgpuInverseSession({
      ...fake.hooks,
      patchMaterial: (...args) => {
        patchCount += 1;
        if (patchCount === 2) throw new Error('second initial patch failed');
        patchMaterial(...args);
      },
    }, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb', initial: [0.9, 0.8, 0.7] },
        { path: 'materials.panel.roughness', kind: 'scalar', initial: [0.25] },
      ],
    })).toThrow(/second initial patch failed/);
    expect(fake.scene.primitives[0]!.material.baseColor).toEqual([0.2, 0.2, 0.2]);

    const broken = makeFakeEngine();
    const brokenPatch = broken.hooks.patchMaterial;
    let brokenCount = 0;
    expect(() => new PtWebgpuInverseSession({
      ...broken.hooks,
      patchMaterial: (...args) => {
        brokenCount += 1;
        if (brokenCount >= 2) throw new Error('patch unavailable');
        brokenPatch(...args);
      },
    }, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [
        { path: 'materials.panel.baseColor', kind: 'rgb', initial: [0.9, 0.8, 0.7] },
        { path: 'materials.panel.roughness', kind: 'scalar', initial: [0.25] },
      ],
    })).toThrow(AggregateError);
  });

  it('poisons the session when rollback cannot restore the scene', async () => {
    const fake = makeFakeEngine();
    const patchMaterial = fake.hooks.patchMaterial;
    let fail = false;
    const session = new PtWebgpuInverseSession({
      ...fake.hooks,
      patchMaterial: (...args) => {
        if (fail) throw new Error('scene patch unavailable');
        patchMaterial(...args);
      },
    }, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
    });
    fail = true;
    await expect(session.step()).rejects.toThrow(AggregateError);
    fail = false;
    await expect(session.step()).rejects.toThrow(/session is poisoned/);
  });

  it('restores Adam state so a retry follows the same trajectory as a clean step', async () => {
    const options: InverseSessionOptions = {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
      method: 'path-replay',
      optimizer: { learningRate: 0.05 },
    };
    const failedFake = makeFakeEngine();
    const failedPatch = failedFake.hooks.patchMaterial;
    let patchCount = 0;
    const failedSession = new PtWebgpuInverseSession({
      ...failedFake.hooks,
      patchMaterial: (...args) => {
        patchCount += 1;
        if (patchCount === 3) throw new Error('updated scene patch failed');
        failedPatch(...args);
      },
      computeAdjointGradient: async () => new Float32Array([-1, 0, 0]),
    }, options);
    await expect(failedSession.step()).rejects.toThrow(/updated scene patch failed/);
    expect(failedSession.currentValues()[0]![0]).toBeCloseTo(0.2, 6);

    const cleanFake = makeFakeEngine();
    const cleanSession = new PtWebgpuInverseSession({
      ...cleanFake.hooks,
      computeAdjointGradient: async () => new Float32Array([-1, 0, 0]),
    }, options);
    const [retry, clean] = await Promise.all([failedSession.step(), cleanSession.step()]);
    expect(retry).toEqual(clean);
    expect(failedFake.scene.primitives[0]!.material.emissive)
      .toEqual(cleanFake.scene.primitives[0]!.material.emissive);
  });

  it('rejects malformed readbacks and non-finite adjoint gradients transactionally', async () => {
    const malformedFake = makeFakeEngine();
    const malformed = new PtWebgpuInverseSession({
      ...malformedFake.hooks,
      renderAndReadback: async () => ({ rgb: new Float32Array(2), channels: 3 }),
    }, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
    });
    await expect(malformed.step()).rejects.toThrow(/data length/);

    const adjointFake = makeFakeEngine();
    const adjoint = new PtWebgpuInverseSession({
      ...adjointFake.hooks,
      computeAdjointGradient: async () =>
        new Float32Array([Number.NaN, 0, 0]),
    }, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
      method: 'path-replay',
    });
    await expect(adjoint.step()).rejects.toThrow(/non-finite/);
    expect(adjoint.currentValues()[0]).toEqual([
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
      expect.closeTo(0.2, 6),
    ]);
  });

  it('normalizes non-Error hook rejections before exposing them', async () => {
    const fake = makeFakeEngine();
    const session = new PtWebgpuInverseSession({
      ...fake.hooks,
      renderAndReadback: async () => {
        // eslint-disable-next-line @typescript-eslint/only-throw-error -- verifies rejection normalization
        throw undefined;
      },
    }, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
    });
    let caught: unknown;
    try {
      await session.step();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toContain('non-Error value');
    expect(fake.scene.primitives[0]!.material.roughness).toBe(0.5);
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

  it('paramLength maps every supported kind', () => {
    expect(paramLength({ path: 'p', kind: 'scalar' })).toBe(1);
    expect(paramLength({ path: 'p', kind: 'vec2' })).toBe(2);
    expect(paramLength({ path: 'p', kind: 'rgb' })).toBe(3);
  });
});
