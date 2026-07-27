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
  MATERIAL_RGB_FIELDS,
  MATERIAL_VEC2_FIELDS,
  MATERIAL_SCALAR_FIELDS,
  EMITTER_RGB_FIELDS,
  EMITTER_SCALAR_FIELDS,
  assertFiniteArray,
  invokeInverseHook,
  normalizeInverseError,
  validateInverseReadback,
  validateInverseSessionOptions,
  validateInitialSceneValue,
  validateParam,
  type ParamLayoutEntry,
} from '../inverse-scaffolding.js';
import type {
  InverseParam,
  InverseSessionOptions,
  InverseTargetImage,
  Scene,
} from '../index.js';

describe('shared inverse scaffolding — hook error boundary', () => {
  it('preserves Error objects and normalizes every non-Error throw', () => {
    const existing = new Error('existing');
    expect(normalizeInverseError(existing, 'operation')).toBe(existing);

    for (const thrown of ['string failure', undefined, null, { code: 7 }]) {
      let caught: unknown;
      try {
        invokeInverseHook('test hook', () => {
          // eslint-disable-next-line @typescript-eslint/only-throw-error -- verifies the hook boundary
          throw thrown;
        });
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(Error);
      expect((caught as Error).message).toContain('test hook');
      expect((caught as Error).message).toContain('non-Error value');
    }
  });
});

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

  it('snapshots/restores moments and rejects a non-finite step atomically', () => {
    const adam = new Adam(1, DEFAULT_ADAM);
    const params = new Float32Array([0.5]);
    const initial = adam.snapshot();
    adam.step(params, new Float32Array([0.25]));
    expect(params[0]).not.toBe(0.5);
    adam.restore(initial);
    params[0] = 0.5;
    const before = adam.snapshot();
    expect(() => adam.step(params, new Float32Array([Number.NaN]))).toThrow(/non-finite/);
    expect(params[0]).toBe(0.5);
    expect(adam.snapshot()).toEqual(before);
  });
});

describe('shared inverse scaffolding — strict session validation', () => {
  const valid = (): InverseSessionOptions => ({
    target: {
      data: new Float32Array([0, 0, 0]),
      width: 1,
      height: 1,
      channels: 3,
    },
    parameters: [{ path: 'materials.panel.roughness', kind: 'scalar' }],
  });

  it.each([
    ['zero width', { width: 0 }, /positive safe integers/],
    ['fractional width', { width: 1.5 }, /positive safe integers/],
    ['NaN height', { height: Number.NaN }, /positive safe integers/],
    ['infinite height', { height: Number.POSITIVE_INFINITY }, /positive safe integers/],
  ])('rejects %s', (_name, targetPatch, message) => {
    const options = valid();
    expect(() => validateInverseSessionOptions({
      ...options,
      target: { ...options.target, ...targetPatch },
    }, 'test')).toThrow(message);
  });

  it('rejects invalid channel counts, target lengths, and non-finite targets', () => {
    const options = valid();
    expect(() => validateInverseSessionOptions({
      ...options,
      target: { ...options.target, channels: 2 as 3 },
    }, 'test')).toThrow(/channels/);
    expect(() => validateInverseSessionOptions({
      ...options,
      target: { ...options.target, data: new Float32Array(2) },
    }, 'test')).toThrow(/data length/);
    expect(() => validateInverseSessionOptions({
      ...options,
      target: { ...options.target, data: new Float32Array([0, Number.NaN, 0]) },
    }, 'test')).toThrow(/non-finite/);
  });

  it.each([0, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 4097])(
    'rejects samplesPerStep=%s',
    (samplesPerStep) => {
      expect(() => validateInverseSessionOptions({
        ...valid(),
        samplesPerStep,
      }, 'test')).toThrow(/samplesPerStep/);
    },
  );

  it.each([
    [{ learningRate: 0 }, /learningRate/],
    [{ beta1: -0.1 }, /beta1/],
    [{ beta1: 1 }, /beta1/],
    [{ beta2: Number.NaN }, /beta2/],
    [{ epsilon: 0 }, /epsilon/],
    [{ fdEpsilon: Number.POSITIVE_INFINITY }, /fdEpsilon/],
  ])('rejects invalid optimizer config %#', (optimizer, message) => {
    expect(() => validateInverseSessionOptions({
      ...valid(),
      optimizer,
    }, 'test')).toThrow(message);
  });

  it('rejects duplicate paths, invalid bounds, and invalid initial values', () => {
    expect(() => validateInverseSessionOptions({
      ...valid(),
      parameters: [
        { path: 'materials.panel.roughness', kind: 'scalar' },
        { path: 'materials.panel.roughness', kind: 'scalar' },
      ],
    }, 'test')).toThrow(/duplicate or overlapping/);
    expect(() => validateInverseSessionOptions({
      ...valid(),
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        min: 2,
        max: 1,
      }],
    }, 'test')).toThrow(/min must not exceed max/);
    expect(() => validateInverseSessionOptions({
      ...valid(),
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        initial: [Number.NaN],
      }],
    }, 'test')).toThrow(/initial value must be finite/);
  });

  it('defensively copies target pixels, parameters, and initial arrays', () => {
    const targetData = new Float32Array([0.1, 0.2, 0.3]);
    const initial = [0.4];
    const options: InverseSessionOptions = {
      target: { data: targetData, width: 1, height: 1 },
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        initial,
      }],
    };
    const validated = validateInverseSessionOptions(options, 'test');
    targetData[0] = 9;
    initial[0] = 9;
    expect(validated.target.data[0]).toBeCloseTo(0.1);
    expect(validated.parameters[0]!.initial).toEqual([0.4]);
  });

  it('rejects unknown and accessor-backed session fields without invoking accessors', () => {
    expect(() => validateInverseSessionOptions({
      ...valid(),
      typo: true,
    } as unknown as InverseSessionOptions, 'test')).toThrow(/unknown key.*typo/i);

    let getterCalls = 0;
    const accessorOptions: Record<string, unknown> = { ...valid() };
    Object.defineProperty(accessorOptions, 'loss', {
      enumerable: true,
      configurable: true,
      get: () => {
        getterCalls += 1;
        return 'l2';
      },
    });
    expect(() => validateInverseSessionOptions(
      accessorOptions as unknown as InverseSessionOptions,
      'test',
    )).toThrow(/own data property/i);
    expect(getterCalls).toBe(0);
  });

  it('rejects sparse arrays and unknown nested configuration fields', () => {
    const sparseParameters = new Array(1) as InverseSessionOptions['parameters'];
    expect(() => validateInverseSessionOptions({
      ...valid(),
      parameters: sparseParameters,
    }, 'test')).toThrow(/dense.*index 0/i);

    expect(() => validateInverseSessionOptions({
      ...valid(),
      optimizer: { learningRate: 0.1, typo: 1 },
    } as unknown as InverseSessionOptions, 'test')).toThrow(/unknown key.*typo/i);

    expect(() => validateInverseSessionOptions({
      ...valid(),
      parameters: [{
        path: 'materials.panel.roughness',
        kind: 'scalar',
        typo: true,
      }],
    } as unknown as InverseSessionOptions, 'test')).toThrow(/unknown key.*typo/i);
  });

  it('validates backend readback shape and rejects non-finite radiance', () => {
    expect(() => validateInverseReadback(new Float32Array(4), 4, 1, 1, 'readback'))
      .not.toThrow();
    expect(() => validateInverseReadback(new Float32Array(3), 2, 1, 1, 'readback'))
      .toThrow(/channels/);
    expect(() => validateInverseReadback(new Float32Array(3), 4, 1, 1, 'readback'))
      .toThrow(/data length/);
    expect(() => validateInverseReadback(
      new Float32Array([0, Number.POSITIVE_INFINITY, 0]),
      3,
      1,
      1,
      'readback',
    )).toThrow(/non-finite/);
    expect(() => assertFiniteArray(new Float32Array([0, Number.POSITIVE_INFINITY]), 'gradient'))
      .toThrow(/non-finite/);
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

  it('paramLength maps every supported kind', () => {
    expect(paramLength({ path: 'm.x.roughness', kind: 'scalar' }, 'pt-webgl2')).toBe(1);
    expect(paramLength({ path: 'm.x.iridescenceThicknessRange', kind: 'vec2' }, 'pt-webgl2')).toBe(2);
    expect(paramLength({ path: 'm.x.baseColor', kind: 'rgb' }, 'pt-webgpu')).toBe(3);
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
    expect(defaultClampRange('dispersionAbbeNumber')).toEqual([1e-6, Infinity]);
    expect(defaultClampRange('sheenColor')).toEqual([0, 1]);
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
    expect(() => materialPatch('baseColor', [0.1, 0.2])).toThrow(/exactly 3 components/);
    expect(() => materialPatch('iridescenceThicknessRange', [100])).toThrow(/exactly 2 components/);
    expect(() => emitterPatch('intensity', [])).toThrow(/exactly 1 component/);
    expect(() => emitterPatch('color', [1, Number.NaN, 0])).toThrow(/must be finite/);
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

  it('requires a positive seed before fitting disabled dispersion', () => {
    const slot = {
      param: { path: 'materials.p.dispersionAbbeNumber', kind: 'scalar' },
      target: { domain: 'materials', id: 'p', field: 'dispersionAbbeNumber' },
      offset: 0,
      length: 1,
    } as const;
    expect(() => validateInitialSceneValue(slot, [0], false, 'test-backend'))
      .toThrow(/explicit positive parameter\.initial/);
    expect(() => validateInitialSceneValue(slot, [0], true, 'test-backend'))
      .toThrow(/finite positive initial/);
    expect(() => validateInitialSceneValue(slot, [20], true, 'test-backend'))
      .not.toThrow();
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

});
