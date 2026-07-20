import { describe, it, expect } from 'vitest';
import {
  Adam,
  DEFAULT_ADAM,
  lossValue,
  l2Loss,
  l1Loss,
  paramLength,
  parseParamPath,
  clampParams,
  defaultClampRange,
  materialPatch,
  emitterPatch,
  readSceneValue,
  MATERIAL_PARAM_DESCRIPTORS,
  EMITTER_PARAM_DESCRIPTORS,
  MATERIAL_RGB_FIELDS,
  MATERIAL_VEC2_FIELDS,
  MATERIAL_SCALAR_FIELDS,
  EMITTER_RGB_FIELDS,
  EMITTER_SCALAR_FIELDS,
  validateParam,
  type ParamLayoutEntry,
} from '../inverse-scaffolding.js';
import type { InverseParam, InverseTargetImage, Scene } from '../index.js';

// ── Adam parity: identical gradients ⇒ identical steps ─────────────────────────
// The whole point of the shared scaffolding is that both backends run the SAME
// optimizer; this pins that two Adam instances fed identical gradients produce
// byte-identical parameter trajectories.

describe('shared inverse scaffolding — Adam optimizer parity', () => {
  it('two instances given identical gradients step identically', () => {
    const a = new Adam(3, DEFAULT_ADAM);
    const b = new Adam(3, DEFAULT_ADAM);
    const pa = new Float32Array([0.5, 0.5, 0.5]);
    const pb = new Float32Array([0.5, 0.5, 0.5]);
    const grads = [
      new Float32Array([0.1, -0.2, 0.3]),
      new Float32Array([0.05, 0.05, -0.1]),
      new Float32Array([-0.4, 0.2, 0.0]),
    ];
    for (const g of grads) {
      a.step(pa, g);
      b.step(pb, g);
    }
    expect(Array.from(pa)).toEqual(Array.from(pb));
    // and it actually moved (not a no-op)
    expect(pa[0]).not.toBe(0.5);
  });

  it('DEFAULT_ADAM matches the documented Kingma & Ba defaults', () => {
    expect(DEFAULT_ADAM).toEqual({
      learningRate: 1e-2,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    });
  });
});

// ── loss ───────────────────────────────────────────────────────────────────────

describe('shared inverse scaffolding — image loss', () => {
  const target: InverseTargetImage = {
    data: new Float32Array([0, 0, 0, 1, 1, 1]),
    width: 2,
    height: 1,
    channels: 3,
  };

  it('l2 scalar loss maps non-finite samples to 0 and averages over RGB', () => {
    const rendered = new Float32Array([0, 0, 0, 1, 1, 1]);
    expect(lossValue(rendered, 3, target, 'l2')).toBe(0);
    const off = new Float32Array([0, 0, 0, 0, 0, 0]);
    expect(lossValue(off, 3, target, 'l2')).toBeCloseTo(3 / 6, 10);
  });

  it('l2Loss / l1Loss return per-RGB gradient arrays for the adjoint pass', () => {
    const rendered = new Float32Array([0.5, 0, 0, 1, 1, 1]);
    const { loss, dLoss_dRendered } = l2Loss(rendered, 3, target);
    expect(dLoss_dRendered.length).toBe(6);
    expect(loss).toBeGreaterThan(0);
    const l1 = l1Loss(rendered, 3, target);
    expect(l1.dLoss_dRendered.length).toBe(6);
  });
});

// ── param path + length ──────────────────────────────────────────────────────

describe('shared inverse scaffolding — parseParamPath / paramLength', () => {
  it('parses materials.<id>.<field> with dotted ids', () => {
    expect(parseParamPath('materials.panel.a.roughness')).toEqual({
      domain: 'materials',
      id: 'panel.a',
      field: 'roughness',
    });
    expect(parseParamPath('emitters.sun.intensity')).toEqual({
      domain: 'emitters',
      id: 'sun',
      field: 'intensity',
    });
  });

  it('rejects malformed paths and unknown domains', () => {
    expect(() => parseParamPath('materials.x')).toThrow(/must be/);
    expect(() => parseParamPath('lights.x.intensity')).toThrow(/unknown domain/);
  });

  it('paramLength maps kinds and attributes the texture error to the backend', () => {
    expect(paramLength({ path: 'm.x.roughness', kind: 'scalar' }, 'pt-webgl2')).toBe(1);
    expect(paramLength({ path: 'm.x.iridescenceThicknessRange', kind: 'vec2' }, 'pt-webgl2')).toBe(2);
    expect(paramLength({ path: 'm.x.baseColor', kind: 'rgb' }, 'pt-webgpu')).toBe(3);
    expect(() =>
      paramLength({ path: 'm.x.baseColorMap', kind: 'texture' }, 'pt-webgl2'),
    ).toThrow(/pt-webgl2/);
    expect(() =>
      paramLength({ path: 'm.x.baseColorMap', kind: 'texture' }, 'pt-webgpu'),
    ).toThrow(/pt-webgpu/);
  });
});

// ── descriptor table is the single source of truth ────────────────────────────

describe('shared inverse scaffolding — descriptor table drives all four ops', () => {
  it('kind field-sets are derived from the descriptor table', () => {
    for (const [field, d] of Object.entries(MATERIAL_PARAM_DESCRIPTORS)) {
      if (d.kind === 'rgb') expect(MATERIAL_RGB_FIELDS.has(field)).toBe(true);
      if (d.kind === 'vec2') expect(MATERIAL_VEC2_FIELDS.has(field)).toBe(true);
      if (d.kind === 'scalar') expect(MATERIAL_SCALAR_FIELDS.has(field)).toBe(true);
    }
    expect(EMITTER_RGB_FIELDS.has('color')).toBe(true);
    expect(EMITTER_SCALAR_FIELDS.has('intensity')).toBe(true);
  });

  it('defaultClampRange resolves off the descriptor table', () => {
    expect(defaultClampRange('roughness')).toEqual([0, 1]);
    expect(defaultClampRange('ior')).toEqual([1, 2.5]);
    expect(defaultClampRange('attenuationColor')).toEqual([1e-4, 1]);
    expect(defaultClampRange('scatteringAnisotropy')).toEqual([-0.95, 0.95]);
    expect(defaultClampRange('emissive')).toEqual([0, Infinity]);
    expect(defaultClampRange('intensity')).toEqual([0, Infinity]);
    // unknown field falls back
    expect(defaultClampRange('nonexistent')).toEqual([0, Infinity]);
  });

  it('materialPatch / emitterPatch build the incremental update record', () => {
    expect(materialPatch('roughness', [0.3])).toEqual({ roughness: 0.3 });
    expect(materialPatch('baseColor', [0.1, 0.2, 0.3])).toEqual({ baseColor: [0.1, 0.2, 0.3] });
    expect(materialPatch('iridescenceThicknessRange', [50, 200])).toEqual({
      iridescenceThicknessRange: [50, 200],
    });
    expect(emitterPatch('intensity', [2])).toEqual({ intensity: 2 });
    expect(emitterPatch('color', [1, 0, 0])).toEqual({ color: [1, 0, 0] });
    expect(() => materialPatch('bogus', [0])).toThrow(/unsupported material field/);
    expect(() => emitterPatch('bogus', [0])).toThrow(/unsupported emitter field/);
  });

  it('readSceneValue round-trips through the descriptor read fns', () => {
    const scene: Scene = {
      primitives: [
        {
          id: 'p',
          geometry: { kind: 'analytic', shape: { kind: 'sphere', radius: 1 } },
          material: { baseColor: [0.2, 0.4, 0.6], roughness: 0.7, metallic: 0.1 },
        } as unknown as Scene['primitives'][number],
      ],
      emitters: [
        { id: 'e', kind: 'point', color: [1, 1, 1], intensity: 3 } as unknown as Scene['emitters'][number],
      ],
    } as unknown as Scene;
    expect(readSceneValue(scene, { domain: 'materials', id: 'p', field: 'baseColor' }, 3)).toEqual([
      0.2, 0.4, 0.6,
    ]);
    expect(readSceneValue(scene, { domain: 'materials', id: 'p', field: 'roughness' }, 1)).toEqual([
      0.7,
    ]);
    expect(readSceneValue(scene, { domain: 'emitters', id: 'e', field: 'intensity' }, 1)).toEqual([
      3,
    ]);
  });
});

// ── clampParams ────────────────────────────────────────────────────────────────

describe('shared inverse scaffolding — clampParams', () => {
  it('clamps per-slot to the field-aware default range, param min/max overrides', () => {
    const flat = new Float32Array([2, -0.5, 5]);
    const params: InverseParam[] = [
      { path: 'materials.p.roughness', kind: 'scalar' }, // default [0,1]
      { path: 'materials.p.baseColor', kind: 'scalar', min: -1 }, // override min
    ];
    const layout: ParamLayoutEntry[] = [
      { offset: 0, length: 1, defaultMin: 0, defaultMax: 1 },
      { offset: 1, length: 2, defaultMin: 0, defaultMax: 1 },
    ];
    clampParams(flat, params, layout);
    expect(flat[0]).toBe(1); // 2 → clamped to 1
    expect(flat[1]).toBe(-0.5); // min override -1, stays
    expect(flat[2]).toBe(1); // 5 → clamped to 1
  });
});

// ── validateParam per-backend availability gate ───────────────────────────────

describe('shared inverse scaffolding — validateParam per-backend gate', () => {
  const scene: Scene = {
    primitives: [
      {
        id: 'p',
        geometry: { kind: 'analytic', shape: { kind: 'sphere', radius: 1 } },
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      } as unknown as Scene['primitives'][number],
    ],
    emitters: [
      { id: 'e', kind: 'point', color: [1, 1, 1], intensity: 1 } as unknown as Scene['emitters'][number],
    ],
  } as unknown as Scene;

  it('accepts an optimizable field with no support-details (pt-webgl2 style)', () => {
    expect(() =>
      validateParam(scene, { path: 'materials.p.roughness', kind: 'scalar' }, {
        domain: 'materials',
        id: 'p',
        field: 'roughness',
      }, { backend: 'pt-webgl2' }),
    ).not.toThrow();
  });

  it('rejects an unsupported field ONLY when support-details flag it (pt-webgpu style)', () => {
    const target = { domain: 'materials', id: 'p', field: 'roughness' } as const;
    const param: InverseParam = { path: 'materials.p.roughness', kind: 'scalar' };
    // no gate → accepted
    expect(() => validateParam(scene, param, target, { backend: 'pt-webgpu' })).not.toThrow();
    // gate says unsupported → rejected with backend attribution
    expect(() =>
      validateParam(scene, param, target, {
        backend: 'pt-webgpu',
        materialSupportDetails: { roughness: 'unsupported' },
      }),
    ).toThrow(/pt-webgpu runtime profile/);
  });

  it('throwOnTextureKind controls the texture-kind raise site', () => {
    const target = { domain: 'materials', id: 'p', field: 'baseColorMap' } as const;
    const param: InverseParam = { path: 'materials.p.baseColorMap', kind: 'texture' };
    expect(() =>
      validateParam(scene, param, target, { backend: 'pt-webgpu', throwOnTextureKind: true }),
    ).toThrow(/texture/);
    // pt-webgl2 defers to paramLength (also throws /texture/)
    expect(() =>
      validateParam(scene, param, target, { backend: 'pt-webgl2' }),
    ).toThrow(/texture/);
  });
});
