import { describe, expect, it } from 'vitest';
import {
  asMat4,
  type InverseSessionOptions,
  type MaterialSpec,
  type Scene,
  type SceneEmitter,
} from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type AdjointGradientRequest,
  type InverseEngineHooks,
  type InversePathReplayRenderContext,
} from '../inverse/inverseSession.js';
import {
  emissiveReplayTargetIssue,
} from '../inverse/emissivePathReplayDomain.js';

function targetImage(): InverseSessionOptions['target'] {
  return {
    data: new Float32Array(12).fill(0.5),
    width: 2,
    height: 2,
    channels: 3,
  };
}

function triangle(
  id: string,
  material: Partial<MaterialSpec> = {},
): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      -1, -1, -1,
      1, -1, -1,
      0, 1, -1,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2]),
    material: {
      baseColor: [0, 0, 0],
      roughness: 1,
      metallic: 0,
      shadingModel: 'unlit',
      emissive: [0.2, 0.3, 0.4],
      ...material,
    },
  };
}

function exactContext(
  patch: Partial<InversePathReplayRenderContext> = {},
): InversePathReplayRenderContext {
  return {
    bounces: 1,
    spectral: false,
    bdpt: false,
    restirPtReuse: false,
    causticStrategy: 'none',
    cameraVisibleEmitters: true,
    ...patch,
  };
}

interface Fake {
  hooks: InverseEngineHooks;
  scene: Scene;
  patches: number;
  renders: number;
  adjoints: number;
  lastAdjoint: AdjointGradientRequest | null;
}

function fakeEngine(
  scene: Scene = {
    primitives: [triangle('target')],
    emitters: [],
    environment: { kind: 'none' },
  },
  context: InversePathReplayRenderContext | null = exactContext(),
): Fake {
  const fake: Fake = {
    hooks: {} as InverseEngineHooks,
    scene,
    patches: 0,
    renders: 0,
    adjoints: 0,
    lastAdjoint: null,
  };
  fake.hooks = {
    getScene: () => fake.scene,
    ...(context == null
      ? {}
      : { getPathReplayRenderContext: () => context }),
    renderAndReadback: async (width, height) => {
      fake.renders += 1;
      return {
        rgb: new Float32Array(width * height * 3),
        channels: 3,
      };
    },
    patchMaterial: (id, patch) => {
      fake.patches += 1;
      fake.scene = {
        ...fake.scene,
        primitives: fake.scene.primitives.map((primitive) =>
          primitive.id === id
            ? {
                ...primitive,
                material: { ...primitive.material, ...patch },
              }
            : primitive,
        ),
      };
    },
    patchEmitter: (id, patch) => {
      fake.patches += 1;
      fake.scene = {
        ...fake.scene,
        emitters: fake.scene.emitters.map((emitter) =>
          emitter.id === id
            ? ({ ...emitter, ...patch } as SceneEmitter)
            : emitter,
        ),
      };
    },
    computeAdjointGradient: async (request) => {
      fake.adjoints += 1;
      fake.lastAdjoint = request;
      return new Float32Array(request.gradientLength).fill(0.25);
    },
  };
  return fake;
}

function pathReplayOptions(
  patch: Partial<InverseSessionOptions> = {},
): InverseSessionOptions {
  return {
    target: targetImage(),
    parameters: [{
      path: 'materials.target.emissive',
      kind: 'rgb',
    }],
    method: 'path-replay',
    samplesPerStep: 4,
    ...patch,
  };
}

describe('InverseSession certified emissive path replay', () => {
  it('reports emissive maps as an explicit certified-domain boundary', () => {
    const primitive = triangle('mapped-target', {
      emissiveMap: { handle: {} },
    });
    const scene: Scene = {
      primitives: [primitive],
      emitters: [],
      environment: { kind: 'none' },
    };

    expect(emissiveReplayTargetIssue(scene, primitive)).toMatchObject({
      code: 'path-replay-unsupported-material',
      message:
        'primitive "mapped-target" uses an emissive map outside the certified spatially constant emission domain',
      details: {
        primitiveId: 'mapped-target',
        feature: 'emissive-map',
      },
    });
  });

  it('dispatches one all-analytic emissive gradient after one baseline render', async () => {
    const fake = fakeEngine();
    const session = new PtWebgpuInverseSession(
      fake.hooks,
      pathReplayOptions(),
    );

    expect(session.method).toBe('path-replay');
    expect(session.parameterMethods).toEqual(['path-replay']);
    expect(session.diagnostics).toEqual([]);
    const result = await session.step();

    expect(fake.renders).toBe(1);
    expect(fake.adjoints).toBe(1);
    expect(fake.lastAdjoint?.samples).toBe(4);
    expect(fake.lastAdjoint?.params).toEqual([{
      domain: 'materials',
      id: 'target',
      field: 'emissive',
      offset: 0,
      length: 3,
    }]);
    expect(result.gradient).toEqual([[0.25, 0.25, 0.25]]);
  });

  it.each([
    {
      name: 'more than one bounce',
      fake: () => fakeEngine(undefined, exactContext({ bounces: 2 })),
      options: () => pathReplayOptions({
        parameters: [{
          path: 'materials.target.emissive',
          kind: 'rgb',
          initial: [0.8, 0.7, 0.6],
        }],
      }),
      code: 'path-replay-unsupported-render-regime',
    },
    {
      name: 'camera-visible primitive emission disabled',
      fake: () => fakeEngine(
        undefined,
        exactContext({ cameraVisibleEmitters: false }),
      ),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-render-regime',
    },
    {
      name: 'missing forward regime',
      fake: () => fakeEngine(undefined, null),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-render-regime',
    },
    {
      name: 'analytic primitive',
      fake: () => fakeEngine({
        primitives: [{
          kind: 'analytic',
          id: 'target',
          shape: 'sphere',
          params: new Float32Array([0, 0, -1, 1]),
          material: triangle('unused').material,
        }],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-primitive',
    },
    {
      name: 'singular transform omitted by forward TLAS',
      fake: () => {
        const primitive = triangle('target');
        if (primitive.kind !== 'mesh') throw new Error('fixture');
        return fakeEngine({
          primitives: [{
            ...primitive,
            transform: asMat4(new Float32Array([
              0, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
            ])),
          }],
          emitters: [],
          environment: { kind: 'none' },
        });
      },
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-scene-geometry',
    },
    {
      name: 'lit receiver with implicit-emitter coupling',
      fake: () => fakeEngine({
        primitives: [
          triangle('target'),
          triangle('receiver', { shadingModel: 'pbr' }),
        ],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-receiver',
    },
    {
      name: 'cross-primitive equal-distance tie cannot be excluded',
      fake: () => fakeEngine({
        primitives: [
          triangle('target'),
          triangle('overlapping-occluder', { emissive: [0, 0, 0] }),
        ],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-scene-geometry',
    },
    {
      name: 'emissive map',
      fake: () => fakeEngine({
        primitives: [triangle('target', { emissiveMap: { handle: {} } })],
        emitters: [],
        environment: { kind: 'none' },
        }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-material',
    },
    {
      name: 'clearcoat',
      fake: () => fakeEngine({
        primitives: [triangle('target', { clearcoat: 0.5 })],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-material',
    },
    {
      name: 'transmission',
      fake: () => fakeEngine({
        primitives: [triangle('target', { transmission: 0.5 })],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-transport',
    },
    {
      name: 'layered transport',
      fake: () => fakeEngine({
        primitives: [triangle('target', {
          frontLayer: { transmission: [0.9, 0.9, 0.9], roughness: 0.1 },
        })],
        emitters: [],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions(),
      code: 'path-replay-unsupported-transport',
    },
    {
      name: 'non-emissive material field',
      fake: () => fakeEngine(),
      options: () => pathReplayOptions({
        parameters: [{
          path: 'materials.target.baseColor',
          kind: 'rgb',
          initial: [0.8, 0.7, 0.6],
        }],
      }),
      code: 'path-replay-unsupported-field',
    },
    {
      name: 'emitter parameter',
      fake: () => fakeEngine({
        primitives: [triangle('target')],
        emitters: [{
          kind: 'point',
          id: 'lamp',
          color: [1, 1, 1],
          intensity: 1,
          position: [0, 0, 1],
        }],
        environment: { kind: 'none' },
      }),
      options: () => pathReplayOptions({
        parameters: [{
          path: 'emitters.lamp.intensity',
          kind: 'scalar',
          initial: [3],
        }],
      }),
      code: 'path-replay-unsupported-param-domain',
    },
  ])('rejects $name before mutation, render, or adjoint dispatch', ({
    fake: makeFake,
    options,
    code,
  }) => {
    const fake = makeFake();
    const diagnostics: string[] = [];
    const opts = {
      ...options(),
      onDiagnostic: (diagnostic: { readonly code: string }) => {
        diagnostics.push(diagnostic.code);
      },
    };

    expect(() => new PtWebgpuInverseSession(fake.hooks, opts)).toThrow(
      /outside the certified pt-webgpu domain/,
    );
    expect(diagnostics).toContain(code);
    expect(fake.patches).toBe(0);
    expect(fake.renders).toBe(0);
    expect(fake.adjoints).toBe(0);
  });

  it('keeps malformed adjoint readback transactional', async () => {
    const fake = fakeEngine();
    fake.hooks.computeAdjointGradient = async () => new Float32Array(2);
    const session = new PtWebgpuInverseSession(
      fake.hooks,
      pathReplayOptions(),
    );
    const before = session.currentValues();

    await expect(session.step()).rejects.toThrow(/gradient length/);
    expect(session.currentValues()).toEqual(before);
    expect(fake.scene.primitives[0]!.material.emissive).toEqual(
      before[0],
    );
  });

  it('retains explicit finite difference for parameters outside replay', async () => {
    const fake = fakeEngine({
      primitives: [triangle('target', { shadingModel: 'pbr' })],
      emitters: [],
      environment: { kind: 'none' },
    }, null);
    delete fake.hooks.computeAdjointGradient;
    const session = new PtWebgpuInverseSession(fake.hooks, {
      target: targetImage(),
      parameters: [{
        path: 'materials.target.baseColor',
        kind: 'rgb',
      }],
      method: 'finite-difference',
      samplesPerStep: 1,
    });

    expect(session.method).toBe('finite-difference');
    await session.step();
    expect(fake.renders).toBe(4);
    expect(fake.adjoints).toBe(0);
  });
});
