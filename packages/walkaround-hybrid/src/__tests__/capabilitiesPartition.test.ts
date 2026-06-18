/**
 * Capability-set reconciliation + partitionSceneBySupport wiring for
 * walkaround-hybrid.
 *
 * Pins two invariants the 2026-05-28 reconciliation locked in:
 *   1. The declared `supported*Kinds` match what the engine genuinely
 *      RENDERS, not just what it ingests:
 *        - primitives: mesh / skinned-mesh / instanced-mesh / analytic.
 *          Instanced-mesh is genuine via the TLAS per-instance traversal;
 *          analytic is accepted through a generated MeshPrimitive fallback
 *          before the core-native BVH+DDGI path.
 *        - emitters: directional / rect-area / disc-area / point / spot /
 *          mesh-area (rect/disc → ReSTIR-DI tris + DDGI fixtures; mesh-area →
 *          mesh emissive; point/spot → shade/OIT analytic direct-light terms
 *          plus DDGI/RC fixture lights; directional → DDGI `sun` light via
 *          coreEmittersToDDGILights, carrying the emitter's real
 *          direction/intensity/colour, single-counted by the host's
 *          sun-intensity multiplier=1). A scene directional is KEPT
 *          (not warn-skipped) and reaches the DDGI sun path.
 *   2. `setScene` filters the scene through `partitionSceneBySupport(scene,
 *      this.capabilities)` BEFORE BVH ingestion, so an
 *      unsupported nodes are warn-skipped (NOT thrown / NOT silently flowed
 *      through), while supported analytics stay in authored `_lastScene` and
 *      appear as generated meshes in `_renderScene`.
 *
 * The engine is constructed directly (not via the factory, which bootstraps
 * with an empty scene) against a minimal duck-typed GPUDevice stub. Only the
 * SYNCHRONOUS `setScene` filter + the declared capability sets are asserted;
 * the fire-and-forget async init chain is aborted via dispose() right after.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../neural/unetArchitecture.js';
import type { LayerWeights, ModelWeights } from '../neural/weights.js';

/** Minimal GPUDevice stub — HybridEngine's constructor only stores the device
 *  (DDGI is CPU-side) and the factory's duck-type check needs
 *  `createCommandEncoder`. The async init chain is aborted via dispose() before
 *  it touches any of these. */
function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    createTexture: vi.fn(),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 3,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function makeZeroNeuralWeights(): ModelWeights {
  const layers: LayerWeights[] = [];
  for (const layer of WALKAROUND_DENOISER_UNET_SPEC.layers) {
    if (layer.kind !== 'conv2d' && layer.kind !== 'transposedConv2d') continue;
    const kH = layer.params.kH ?? 3;
    const kW = layer.params.kW ?? 3;
    layers.push({
      name: layer.name,
      weights: new Float32Array(layer.params.outC * layer.params.inC * kH * kW),
      biases: new Float32Array(layer.params.outC),
    });
  }
  return { layers };
}

function meshPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function instancedMeshPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    instances: [
      asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    ],
  };
}

function analyticPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'analytic',
    id,
    shape: 'sphere',
    params: new Float32Array([0, 0, 0, 1]),
    material: { baseColor: [1, 0, 0], roughness: 0.3, metallic: 0 },
  };
}

function rectEmitter(id: string): SceneEmitter {
  return {
    kind: 'rect-area',
    id,
    position: [0, 5, 0],
    uAxis: [1, 0, 0],
    vAxis: [0, 0, 1],
    color: [1, 1, 1],
    intensity: 4,
  };
}

function directionalEmitter(id: string): SceneEmitter {
  return {
    kind: 'directional',
    id,
    direction: [0, -1, 0],
    color: [1, 1, 1],
    intensity: 3,
  };
}

function sceneWith(primitives: ScenePrimitive[], emitters: SceneEmitter[] = []): Scene {
  return { primitives, emitters, environment: { kind: 'none' } };
}

describe('walkaround-hybrid capability/partition reconciliation', () => {
  let warnSpy: MockInstance | undefined;
  // The fire-and-forget async init chain fails in this non-WebGPU test env
  // (`GPUBufferUsage` is undefined) and logs via console.error before the
  // chain's catch sets state to 'error'. That's expected and orthogonal to
  // the synchronous partition behaviour under test — silence it so the
  // expected noise doesn't pollute output. (The partition *warnings* go
  // through console.warn, which the relevant tests spy on explicitly.)
  let errorSpy: MockInstance | undefined;
  beforeEach(() => {
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy?.mockRestore();
    errorSpy?.mockRestore();
  });

  it('declares mesh / skinned-mesh / instanced-mesh and analytic fallback support', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      const kinds = engine.capabilities.supportedPrimitiveKinds!;
      expect(kinds.has('mesh')).toBe(true);
      expect(kinds.has('skinned-mesh')).toBe(true);
      expect(kinds.has('instanced-mesh')).toBe(true);
      expect(kinds.has('analytic')).toBe(true);
      expect(engine.capabilities.supportedAnalyticShapes).toEqual(
        new Set(['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came']),
      );
    } finally {
      engine.dispose();
    }
  });

  it('declares directional / rect-area / disc-area / point / spot / mesh-area emitters', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      const kinds = engine.capabilities.supportedEmitterKinds;
      for (const k of ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'] as const) {
        expect(kinds.has(k)).toBe(true);
      }
      // directional now drives the DDGI sun via coreEmittersToDDGILights
      // (single-counted), so it is a genuinely-rendered emitter kind.
      expect(kinds.has('directional')).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning when NRC is explicitly enabled', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: unknown[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      nrcEnabled: true,
      onWarning: (warning) => warnings.push(warning),
    });
    try {
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.nrc-experimental-biased',
          backend: 'walkaround-hybrid',
          phase: 'construction',
          method: 'createWalkaroundEngine_Hybrid',
          details: expect.objectContaining({
            nrcEnabled: true,
            defaultEnabled: false,
            estimator: 'biased',
          }),
        }),
      ]);
      expect(warnSpy.mock.calls.flat().join('\n')).toContain('nrcEnabled:true');
    } finally {
      engine.dispose();
    }
  });

  it('routes DDGI constructor warnings through the engine warning surface', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: unknown[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      ddgiMaxMaterials: 0,
      onWarning: (warning) => warnings.push(warning),
    });
    try {
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.ddgi-invalid-max-materials',
          backend: 'walkaround-hybrid',
          phase: 'construction',
          method: 'ProbeUpdatePass.constructor',
          details: expect.objectContaining({ requested: 0, clampedTo: 1 }),
        }),
      ]);
      expect(warnSpy.mock.calls.flat().join('\n')).toContain('maxMaterials=0');
    } finally {
      engine.dispose();
    }
  });

  it('keeps learned/research paths out of experimentalFeatures until explicitly enabled', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      expect(engine.capabilities.experimentalFeatures).toEqual(new Set(['svgf-real-conservative-objid']));
    } finally {
      engine.dispose();
    }
  });

  it('declares opt-in learned/research paths as experimental features', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: unknown[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      restirPtReuse: true,
      ppgEnabled: true,
      nrcEnabled: true,
      denoiser: 'neural',
      neuralWeights: makeZeroNeuralWeights(),
      onWarning: (warning) => warnings.push(warning),
    });
    try {
      expect(engine.capabilities.experimentalFeatures).toEqual(new Set([
        'svgf-real-conservative-objid',
        'walkaround-hybrid-gris-unbiased-reuse',
        'walkaround-hybrid-ppg-guided-gi',
        'walkaround-hybrid-nrc-biased-cache',
        'walkaround-hybrid-neural-denoiser-host-weights',
      ]));
      expect(warnings).toEqual(expect.arrayContaining([
        expect.objectContaining({
          code: 'walkaround-hybrid.nrc-experimental-biased',
          details: expect.objectContaining({ estimator: 'biased', defaultEnabled: false }),
        }),
        expect.objectContaining({
          code: 'walkaround-hybrid.neural-host-weights-required',
          details: expect.objectContaining({
            denoiser: 'neural',
            weightsRequired: true,
            packageProvidesProductionWeights: false,
            defaultEnabled: false,
          }),
        }),
      ]));
    } finally {
      engine.dispose();
    }
  });

  it('keeps an instanced-mesh + rect-area scene intact (instanced-mesh is genuine via TLAS)', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(sceneWith([instancedMeshPrimitive('inst-a')], [rectEmitter('rect-a')]));

      // No skip-warning for the supported nodes.
      expect(warnSpy.mock.calls.flat().some((m) => String(m).includes('not supported'))).toBe(false);

      // The stored authored scene retains the supported primitive + emitter,
      // and the render-ingestion predicate treats an instanced-only core scene
  // as a real triangle source for TLAS init.
      const internals = engine as unknown as {
        _lastScene: Scene;
        _renderScene: Scene;
        _coreSceneSuppliesMeshes(): boolean;
      };
      const stored = internals._lastScene;
      expect(stored.primitives.map((p) => String(p.id))).toContain('inst-a');
      expect(stored.emitters.map((e) => String(e.id))).toContain('rect-a');
      expect(internals._renderScene.primitives.map((p) => String(p.id))).toEqual(['inst-a']);
      expect(internals._coreSceneSuppliesMeshes()).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('keeps a directional emitter (it drives the DDGI sun via coreEmittersToDDGILights)', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(
        sceneWith([meshPrimitive('mesh-a')], [rectEmitter('rect-a'), directionalEmitter('sun-a')]),
      );

      // No skip-warning for the directional emitter — it is now a supported
      // emitter kind (mapped to a DDGI `sun` light), so partitionSceneBySupport
      // retains it rather than warn-skipping.
      const warned = warnSpy.mock.calls.flat().map(String);
      expect(warned.some((m) => m.includes('sun-a') && m.includes('not supported'))).toBe(false);

      // The stored scene keeps BOTH the rect-area emitter and the directional.
      const stored = (engine as unknown as { _lastScene: Scene })._lastScene;
      expect(stored.emitters.map((e) => String(e.id)).sort()).toEqual(['rect-a', 'sun-a']);
      expect(stored.emitters.some((e) => e.kind === 'directional')).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('keeps authored analytics and exposes generated mesh fallback to render ingestion', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      expect(() =>
        engine.setScene(sceneWith([meshPrimitive('mesh-a'), analyticPrimitive('sphere-a')])),
      ).not.toThrow();

      const warned = warnSpy.mock.calls.flat().map(String);
      expect(warned.some((m) => m.includes('sphere-a') && m.includes('not supported'))).toBe(false);

      const internals = engine as unknown as { _lastScene: Scene; _renderScene: Scene };
      expect(internals._lastScene.primitives.map((p) => p.kind)).toEqual(['mesh', 'analytic']);
      expect(internals._lastScene.primitives.map((p) => String(p.id))).toEqual(['mesh-a', 'sphere-a']);

      const render = internals._renderScene;
      expect(render.primitives.map((p) => String(p.id))).toEqual(['mesh-a', 'sphere-a']);
      expect(render.primitives.some((p) => p.kind === 'analytic')).toBe(false);
      const generated = render.primitives.find((p) => p.id === 'sphere-a');
      expect(generated?.kind).toBe('mesh');
      if (generated?.kind !== 'mesh') throw new Error('expected analytic fallback to be a mesh');
      expect(generated.positions.length).toBeGreaterThan(0);
      expect(generated.normals.length).toBe(generated.positions.length);
      expect(generated.material.baseColor).toEqual([1, 0, 0]);
    } finally {
      engine.dispose();
    }
  });
});
