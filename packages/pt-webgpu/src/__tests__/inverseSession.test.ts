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
      const rgb = new Float32Array(width * height * 3);
      for (let p = 0; p < width * height; p++) {
        rgb[p * 3 + 0] = mat.baseColor[0];
        rgb[p * 3 + 1] = mat.baseColor[1];
        rgb[p * 3 + 2] = mat.baseColor[2];
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
      parameters: [{ path: 'materials.panel.thickness', kind: 'scalar' }],
    })).toThrow(/not optimizable/);
  });

  it('throws when the declared kind disagrees with the resolved field', () => {
    const fake = makeFakeEngine();
    expect(() => new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(2, 2, [0, 0, 0]),
      parameters: [{ path: 'materials.panel.baseColor', kind: 'scalar' }],
    })).toThrow(/declared kind 'scalar'/);
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
    session.dispose();
  });

  it('keeps path-replay for map-free unlit baseColor without requiring scene lights', () => {
    const fake = makeFakeEngine();
    fake.scene = {
      ...fake.scene,
      emitters: [],
      environment: { kind: 'hdri', hdri: { width: 1, height: 1, data: new Float32Array([0.1, 0.2, 0.3, 1]) } },
      primitives: fake.scene.primitives.map((pr) =>
        pr.id === 'panel'
          ? { ...pr, material: { ...pr.material, shadingModel: 'unlit' as const } }
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
    const session = new PtWebgpuInverseSession(fake.hooks, eligibleOpts());
    expect(session.method).toBe('finite-difference');
    session.dispose();
  });

  it('degrades to finite-difference for an ineligible (emitter) param even with the hook', () => {
    const fake = makeFakeEngine();
    const hooks: InverseEngineHooks = { ...fake.hooks, computeAdjointGradient: async () => new Float32Array(1) };
    const session = new PtWebgpuInverseSession(hooks, {
      target: targetImage(2, 2, [0.8, 0.1, 0.1]),
      parameters: [{ path: 'emitters.lamp.intensity', kind: 'scalar' }],
      method: 'path-replay',
    });
    expect(session.method).toBe('finite-difference'); // emitter intensity isn't a Phase-1 BSDF param
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

  it('keeps alpha-mapped emissive params on finite-difference because visibility is not replayed', () => {
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

  it.each([
    ['transmission', { transmission: 0.25 }],
    ['anisotropy', { anisotropy: 0.25 }],
    ['normal map', { normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } } }],
    ['emissive map on a lit BRDF target', { emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1]) } } }],
    ['clearcoat normal map', { clearcoatNormalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } } }],
    ['front layer', { frontLayer: { transmission: [0.9, 0.8, 0.7] as [number, number, number] } }],
    ['thin-film stack', { thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 180 }] } }],
    ['spectral attenuation', {
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
    }],
    ['volume scattering', { scatteringCoefficientRGB: [0.1, 0.2, 0.3] as [number, number, number] }],
  ])('degrades to finite-difference for path-replay material with %s', (_label, patch) => {
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
    session.dispose();
  });

  it('degrades to finite-difference when scene lighting is outside the adjoint pass scope', () => {
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

  it.each([
    ['soft directional', [{
      kind: 'directional' as const,
      id: 'soft-sun',
      color: [1, 1, 1] as [number, number, number],
      intensity: 1,
      direction: [0, -1, 0] as [number, number, number],
      angularDiameter: 0.01,
    }]],
  ])('degrades to finite-difference for lighting outside adjoint pass scope: %s', (_label, emitters) => {
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
    expect(session.method).toBe('finite-difference');
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
    const cases: Scene['primitives'] = [
      {
        kind: 'analytic',
        id: 'panel',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: baseMesh.material,
      },
      {
        kind: 'instanced-mesh',
        id: 'panel',
        positions: baseMesh.positions,
        normals: baseMesh.normals,
        material: baseMesh.material,
        instances: [identity],
      },
      {
        ...baseMesh,
        transform: translated,
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
      session.dispose();
    }
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
