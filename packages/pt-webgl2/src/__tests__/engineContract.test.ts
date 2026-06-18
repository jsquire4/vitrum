import { describe, expect, it, vi } from 'vitest';
import type { AnalyticPrimitive, EngineWarning, FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import type { PTEngineWebGL2Options } from '../index.js';
import { createMockGl } from './mockGl.js';

// The mock GL drives the full pipeline without a GPU — it verifies the engine
// ORCHESTRATION (accumulation loop, convergence, FrameOutput shape, lifecycle),
// NOT pixel correctness (that is the real-GPU capture-host A/B, plan 06).

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };
function triScene(): Scene {
  const prim: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

function tri(id: string, x: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([x, -1, 0, x + 1, -1, 0, x + 1, 1, 0, x, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
}

function sceneWithEmitter(): Scene {
  return {
    primitives: [tri('tri', 0)],
    emitters: [
      {
        kind: 'point',
        id: 'point-a',
        position: [1, 2, 3],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

function sceneWithSoftDirectionalEmitter(): Scene {
  return {
    primitives: [tri('tri', 0)],
    emitters: [
      {
        kind: 'directional',
        id: 'sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 2,
        angularDiameter: 0.01,
      },
    ],
    environment: { kind: 'none' },
  };
}

function sceneWithMeshAreaEmitter(): Scene {
  return {
    primitives: [tri('panel', 0)],
    emitters: [
      {
        kind: 'mesh-area',
        id: 'panel-light',
        meshId: 'panel',
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

function hdriScene(): Scene {
  return {
    primitives: [tri('tri', 0)],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([0.25, 0.5, 1]),
      },
      intensity: 1,
    },
  };
}

function opts(): PTEngineWebGL2Options {
  return { device: createMockGl() };
}

function frame(spp: number): FrameInput {
  // an identity-ish view + a finite invertible projection (packFrameParams inverts both)
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 64, height: 64, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

describe('PTEngineWebGL2 — contract conformance + accumulation orchestration', () => {
  it('factory returns an engine in state "ready"; rejects a non-WebGL2 device', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.state).toBe('ready');
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(/WebGL2RenderingContext/);
  });

  it('advertises the contract capabilities (offscreen-texture, accumulates, caustic field)', async () => {
    const c = (await createPTEngine_WebGL2(opts())).capabilities;
    expect(c.presentationMode).toBe('offscreen-texture');
    expect(c.accumulates).toBe(true);
    expect(c.causticStrategy).toBe('none');
    expect(c.supportedPrimitiveKinds?.has('mesh')).toBe(true);
    expect(c.supportedPrimitiveKinds?.has('analytic')).toBe(true);
    expect(c.supportedAnalyticShapes?.has('sphere')).toBe(true);
    expect(c.supportedAnalyticShapes?.has('box')).toBe(true);
    expect(c.supportDetails?.primitives.analytic).toBe('fallback-generated-mesh');
    expect(c.supportDetails?.analyticShapes.sphere).toBe('fallback-generated-mesh');
    expect(c.supportsIncrementalScene).toBe(true);
    expect(c.incrementalPatchSupport).toEqual({
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    });
    expect(c.supportsAddRemovePrimitive).toBe(true);
    expect(c.supportDetails?.mutations.material).toBe('native');
    expect(c.supportDetails?.mutations.emitter).toBe('native');
    expect(c.supportDetails?.mutations.environment).toBe('native');
    expect(c.supportDetails?.mutations.positions).toBe('fallback-rebuild');
    expect(c.supportDetails?.mutations.resize).toBe('native');
    expect(c.supportDetails?.denoisers).toEqual({
      none: 'native',
      atrous: 'unsupported',
      'atrous-variance': 'unsupported',
      'svgf-real': 'unsupported',
      bmfr: 'unsupported',
      'oidn-final': 'native',
      neural: 'unsupported',
    });
  });

  it('surfaces caustic approximations without advertising native MNEE support', async () => {
    for (const causticStrategy of ['manifold-nee', 'photon-map'] as const) {
      const c = (await createPTEngine_WebGL2({ ...opts(), causticStrategy })).capabilities;
      expect(c.causticStrategy).toBe(causticStrategy);
      expect(c.experimentalFeatures?.has('pt-webgl2-manifold-nee')).not.toBe(true);
      expect(c.experimentalFeatures?.has('pt-webgl2-photon-map')).not.toBe(true);
    }
  });

  it('exposes scene mutation methods', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.updatePrimitive).toBe('function');
    expect(typeof e.updateEmitter).toBe('function');
    expect(typeof e.updateEnvironment).toBe('function');
    expect(typeof e.addPrimitive).toBe('function');
    expect(typeof e.removePrimitive).toBe('function');
  });

  it('setScene ingests via shared-bvh; getScene returns the filtered scene', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['tri']);
    expect(e._debugGeoPack?.triangleCount).toBe(2);
  });

  it('setScene tessellates analytic primitives to generated mesh fallbacks', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const sphere: AnalyticPrimitive = {
        kind: 'analytic',
        id: 'sphere-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: GREY,
      };

      e.setScene({ primitives: [sphere], emitters: [], environment: { kind: 'none' } });

      const converted = e.getScene?.()?.primitives[0];
      expect(converted?.id).toBe('sphere-a');
      expect(converted?.kind).toBe('mesh');
      if (converted?.kind !== 'mesh') {
        throw new Error('expected analytic primitive to be converted before pt-webgl2 ingestion');
      }
      expect(converted.indices?.length).toBeGreaterThan(0);
      expect(e._debugGeoPack?.triangleCount).toBeGreaterThan(0);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.scene-upload-warning' &&
        String(w.details?.warning).includes('tessellated to a generated MeshPrimitive fallback'),
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('emits structured warnings for unreadable material textures and HDRIs', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            baseColorMap: { handle: { id: 'unreadable-map' } },
          },
        }],
        environment: { kind: 'hdri', hdri: { mock: true }, intensity: 1 },
      };
      e.setScene(scene);

      expect(structured.some((w) =>
        w.code === 'pt-webgl2.texture-unreadable' &&
        w.phase === 'setScene' &&
        w.method === 'setScene' &&
        w.details?.colorSpace === 'srgb',
      )).toBe(true);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.hdri-unreadable' &&
        w.phase === 'setScene' &&
        w.method === 'setScene' &&
        w.details?.width === 0 &&
        w.details?.height === 0,
      )).toBe(true);
      expect(warn.mock.calls.flat().map(String).some((m) => m.includes('texture handle is not readable'))).toBe(true);
      expect(warn.mock.calls.flat().map(String).some((m) => m.includes('HDRI environment is present'))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('emits structured HDRI warnings on the updateEnvironment fast path', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      structured.length = 0;
      warn.mockClear();

      e.updateEnvironment?.({ kind: 'hdri', hdri: { mock: true }, intensity: 1 });

      expect(structured.some((w) =>
        w.code === 'pt-webgl2.hdri-unreadable' &&
        w.phase === 'mutation' &&
        w.method === 'updateEnvironment',
      )).toBe(true);
      expect(warn.mock.calls.flat().map(String).some((m) => m.includes('HDRI environment is present'))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('warns when displacement material fields are supplied', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            displacementMap: { handle: { id: 'height' } },
            displacementScale: 0.2,
            displacementBias: -0.1,
          },
        }],
      };
      e.setScene(scene);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('displacementMap') &&
        m.includes('displacementScale') &&
        m.includes('displacementBias'),
      )).toBe(true);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-displacement-material' &&
        Array.isArray(w.details?.fields) &&
        w.details.fields.includes('displacementMap') &&
        w.details.fields.includes('displacementScale') &&
        w.details.fields.includes('displacementBias') &&
        Array.isArray(w.details?.primitiveIds) &&
        w.details.primitiveIds.includes('tri') &&
        Array.isArray(w.details?.primitiveFields) &&
        w.details.primitiveFields.some((entry) =>
          entry.primitiveId === 'tri' &&
          entry.fields.includes('displacementMap') &&
          entry.fields.includes('displacementScale') &&
          entry.fields.includes('displacementBias')
        ),
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('warns when scalar displacement fields are supplied through the material mutation fast path', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.42,
          displacementScale: 0.2,
          displacementBias: -0.1,
        },
      } as never);

      const displacementWarnings = structured.filter((w) =>
        w.code === 'pt-webgl2.unsupported-displacement-material'
      );
      expect(displacementWarnings).toHaveLength(1);
      expect(displacementWarnings[0]).toMatchObject({
        backend: 'pt-webgl2',
        phase: 'mutation',
        method: 'updatePrimitive',
        details: {
          fields: ['displacementBias', 'displacementScale'],
          primitiveIds: ['tri'],
          primitiveFields: [{
            primitiveId: 'tri',
            fields: ['displacementBias', 'displacementScale'],
          }],
        },
      });
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('displacementBias') && m.includes('displacementScale')
      )).toBe(true);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild'
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('accepts scalar and mapped anisotropy without unsupported-field warnings (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            anisotropy: 0.8,
            anisotropyRotation: 0.5,
            anisotropyMap: { handle: { id: 'aniso' } },
          },
        }],
      };
      e.setScene(scene);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-material-fields' &&
        Array.isArray(w.details?.fields) &&
        (
          w.details.fields.includes('anisotropy') ||
          w.details.fields.includes('anisotropyRotation') ||
          w.details.fields.includes('anisotropyMap')
        ),
      )).toBe(false);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('anisotropy') && m.includes('not rendered'),
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('accepts thicknessMap as an approximate volume field without unsupported-field warnings (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            transmission: 1,
            attenuationDistance: 2,
            attenuationColor: [0.8, 0.9, 1.0],
            thickness: 0.25,
            thicknessMap: { handle: { id: 'thickness' } },
          },
        }],
      };
      e.setScene(scene);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-material-fields' &&
        Array.isArray(w.details?.fields) &&
        (
          w.details.fields.includes('thickness') ||
          w.details.fields.includes('thicknessMap')
        ),
      )).toBe(false);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('thicknessMap') && m.includes('not rendered'),
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('does not warn unsupported-material-fields for layered front/back normal fields (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            frontLayer: {
              transmission: [0.8, 0.9, 1.0],
              roughness: 0.2,
              normalMap: { handle: { id: 'front-normal' } },
              normalScale: 0.75,
            },
            backLayer: {
              transmission: [1.0, 0.9, 0.8],
              roughness: 0.3,
              normalMap: { handle: { id: 'back-normal' } },
              normalScale: 0.5,
            },
          },
        }],
      };
      e.setScene(scene);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-material-fields' &&
        Array.isArray(w.details?.fields) &&
        (
          w.details.fields.includes('frontLayer.normalMap') ||
          w.details.fields.includes('frontLayer.normalScale') ||
          w.details.fields.includes('backLayer.normalMap') ||
          w.details.fields.includes('backLayer.normalScale')
        ),
      )).toBe(false);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('frontLayer.normalMap') &&
        m.includes('backLayer.normalMap') &&
        m.includes('not rendered'),
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('does NOT emit the unsupported-material-fields warning for a plain supported material (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      expect(structured.some((w) => w.code === 'pt-webgl2.unsupported-material-fields')).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('renderFrame skips before a scene, then renders + accumulates one sample per call', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.renderFrame(frame(16)).kind).toBe('skipped'); // no scene yet
    e.setScene(triScene());
    const f1 = e.renderFrame(frame(16));
    expect(f1.kind).toBe('rendered');
    expect(f1.samplesAccumulated).toBe(1);
    expect(f1.isConverged).toBe(false);
    if (f1.kind === 'rendered') expect(f1.primaryRadiance).toBeDefined();
    const f2 = e.renderFrame(frame(16));
    expect(f2.samplesAccumulated).toBe(2);
  });

  it('setSize controls render-target dimensions and resets accumulation only on size changes', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.setSize).toBe('function');
    e.setScene(triScene());

    e.setSize!(32, 48);
    const first = e.renderFrame(frame(16));
    expect(first.kind).toBe('rendered');
    expect(first.samplesAccumulated).toBe(1);
    let captured = await e.captureFrame!();
    expect(captured?.width).toBe(32);
    expect(captured?.height).toBe(48);

    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(2);
    e.setSize!(32, 48);
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(3);

    e.setSize!(16, 20);
    const resized = e.renderFrame(frame(16));
    expect(resized.samplesAccumulated).toBe(1);
    captured = await e.captureFrame!();
    expect(captured?.width).toBe(16);
    expect(captured?.height).toBe(20);

    e.setSize!(0, 20);
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(2);
  });

  it('converges at samplesTarget and stops advancing', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    let out = e.renderFrame(frame(4));
    for (let i = 0; i < 5; i += 1) out = e.renderFrame(frame(4));
    expect(out.samplesAccumulated).toBe(4);
    expect(out.isConverged).toBe(true);
  });

  it('emits pt-spp progress telemetry', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    const seen: Array<{ current: number; target: number }> = [];
    e.onProgress?.((p) => seen.push({ current: p.current, target: p.target }));
    e.renderFrame(frame(16));
    expect(seen.at(-1)).toEqual({ current: 1, target: 16 });
  });

  it('reset() restarts accumulation; pause/resume/dispose drive the state machine', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    e.renderFrame(frame(16));
    e.reset();
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
    e.pause();
    expect(e.state).toBe('paused');
    e.resume();
    expect(e.state).toBe('ready');
    e.dispose();
    expect(e.state).toBe('disposed');
    expect(() => e.setScene(triScene())).toThrow(/disposed/);
    e.dispose(); // idempotent
  });

  it('updatePrimitive material patches update scene textures without rebuilding BVH geometry', async () => {
    const structured: EngineWarning[] = [];
    const e = await createPTEngine_WebGL2({
      ...opts(),
      onWarning: (w) => structured.push(w),
    });
    e.setScene(triScene());
    const beforeGeo = e._debugGeoPack;
    const beforeBvhNodes = beforeGeo?.bvhNodes;
    const beforePositions = beforeGeo?.positions;
    e.renderFrame(frame(16));
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(2);

    e.updatePrimitive?.('tri', { material: { roughness: 0.25 } } as never);
    const afterGeo = e._debugGeoPack;
    expect(afterGeo?.bvhNodes).toBe(beforeBvhNodes);
    expect(afterGeo?.positions).toBe(beforePositions);
    expect(afterGeo?.materials[0]?.roughness).toBe(0.25);
    const scene = e.getScene?.();
    const prim = scene?.primitives[0];
    expect(prim?.kind).toBe('mesh');
    if (prim?.kind === 'mesh') {
      expect(prim.material.roughness).toBe(0.25);
      expect(prim.material.baseColor).toEqual(GREY.baseColor);
    }
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
    expect(structured.some((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild')).toBe(false);
  });

  it('warns once when primitive patches use the scene-texture/BVH rebuild fallback', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }) as unknown as WebGLTexture);
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
      const beforeMaterials = e._debugGeoPack?.materials;

      const moved = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1]);
      const movedAgain = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]);
      e.updatePrimitive?.('tri', { transform: moved } as never);
      const firstRefreshUploads = createTexture.mock.calls.length;
      e.updatePrimitive?.('tri', { transform: movedAgain } as never);

      const fallbackWarnings = structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild');
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.phase).toBe('mutation');
      expect(fallbackWarnings[0]?.method).toBe('updatePrimitive');
      expect(fallbackWarnings[0]?.details).toEqual({
        primitiveId: 'tri',
        fields: ['transform'],
        fallbackReason: 'geometry-bvh-texture-rebuild',
        nativePatchMissing: 'targeted-geometry-bvh-refit',
      });
      expect(firstRefreshUploads - initialTextureUploads).toBe(6);
      expect(createTexture.mock.calls.length - firstRefreshUploads).toBe(6);
      expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
      expect(e._debugGeoPack?.materials).toEqual(beforeMaterials);
      expect(e._debugGeoPack?.positions[0]).toBeCloseTo(2, 6);
      const scene = e.getScene?.();
      const prim = scene?.primitives[0];
      expect(prim?.kind).toBe('mesh');
      if (prim?.kind === 'mesh') {
        expect(Array.from(prim.transform ?? [])).toEqual(Array.from(movedAgain));
      }
      expect(warn.mock.calls.flat().map(String).filter((m) =>
        m.includes('primitive-mutation-fallback-rebuild') ||
        m.includes('updatePrimitive("tri") fields [transform]'),
      )).toHaveLength(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('surfaces nested material texture-map fields when material patches fallback-rebuild', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.45,
          baseColorMap: { handle: { id: 'base-map' } },
        },
      } as never);

      const fallbackWarnings = structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild');
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toEqual({
        primitiveId: 'tri',
        fields: ['material'],
        materialFields: ['baseColorMap', 'roughness'],
        materialTextureFields: ['baseColorMap'],
        fallbackReason: 'texture-map-material-patch',
        nativePatchMissing: 'targeted-material-atlas-texture-update',
      });
    } finally {
      warn.mockRestore();
    }
  });

  it('keeps vertex-color patches on the full upload path because material slots encode color use', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }) as unknown as WebGLTexture);
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;

      e.updatePrimitive?.('tri', {
        colors: new Float32Array([
          1, 0, 0, 1,
          0, 1, 0, 1,
          0, 0, 1, 1,
          1, 1, 1, 1,
        ]),
      } as never);

      const fallbackWarnings = structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild');
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toEqual({
        primitiveId: 'tri',
        fields: ['colors'],
        fallbackReason: 'primitive-scene-texture-repack',
        nativePatchMissing: 'targeted-primitive-layout-or-analytic-update',
        fullUploadFields: ['colors'],
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(8);
    } finally {
      warn.mockRestore();
    }
  });

  it('warns once when primitive add/remove use the scene-texture/BVH rebuild fallback', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }) as unknown as WebGLTexture);
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;

      e.addPrimitive?.(tri('extra', 2));
      const afterAddUploads = createTexture.mock.calls.length;
      e.removePrimitive?.('extra');
      const afterRemoveUploads = createTexture.mock.calls.length;
      // A second add of the same id is the same fallback signature and should not
      // spam animation/streaming hosts that repeatedly rebuild the same slot.
      e.addPrimitive?.(tri('extra', 2));
      const afterSecondAddUploads = createTexture.mock.calls.length;
      e.removePrimitive?.('extra');
      const afterSecondRemoveUploads = createTexture.mock.calls.length;
      e.addPrimitive?.({
        ...tri('extra', 2),
        material: {
          ...GREY,
          baseColorMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Uint8Array([255, 255, 255, 255]),
              __vitrum_hint__: { channels: 4 },
            },
          },
        },
      } as never);
      const afterTexturedAddUploads = createTexture.mock.calls.length;

      const listWarnings = structured.filter((w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild');
      expect(listWarnings).toHaveLength(2);
      expect(listWarnings.map((w) => w.method)).toEqual(['addPrimitive', 'removePrimitive']);
      expect(listWarnings.map((w) => w.details)).toEqual([
        {
          primitiveId: 'extra',
          operation: 'addPrimitive',
          fallbackReason: 'primitive-list-texture-refresh',
          nativePatchMissing: 'targeted-primitive-list-splice',
        },
        {
          primitiveId: 'extra',
          operation: 'removePrimitive',
          fallbackReason: 'primitive-list-texture-refresh',
          nativePatchMissing: 'targeted-primitive-list-splice',
        },
      ]);
      expect(afterAddUploads - initialTextureUploads).toBe(7);
      expect(afterRemoveUploads - afterAddUploads).toBe(7);
      expect(afterSecondAddUploads - afterRemoveUploads).toBe(7);
      expect(afterSecondRemoveUploads - afterSecondAddUploads).toBe(7);
      expect(afterTexturedAddUploads - afterSecondRemoveUploads).toBe(8);
      expect(warn.mock.calls.flat().map(String).filter((m) =>
        m.includes('primitive-list-fallback-rebuild') ||
        m.includes('geometry/material/atlas/BVH texture pack'),
      )).toHaveLength(2);
    } finally {
      warn.mockRestore();
    }
  });

  it('tessellates analytic addPrimitive through the primitive-list texture refresh path', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }) as unknown as WebGLTexture);
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const sphere: AnalyticPrimitive = {
        kind: 'analytic',
        id: 'sphere-added',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: GREY,
      };

      e.addPrimitive?.(sphere);

      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(7);
      const added = e.getScene?.()?.primitives.find((p) => p.id === 'sphere-added');
      expect(added?.kind).toBe('mesh');
      if (added?.kind !== 'mesh') {
        throw new Error('expected analytic addPrimitive to commit the generated mesh fallback');
      }
      expect(added.indices?.length).toBeGreaterThan(0);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.primitive-list-fallback-rebuild' &&
        w.details?.fallbackReason === 'primitive-list-texture-refresh',
      )).toBe(true);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.scene-upload-warning' &&
        w.method === 'addPrimitive' &&
        String(w.details?.warning).includes('tessellated to a generated MeshPrimitive fallback') &&
        w.details?.operation === 'primitive-list-texture-refresh',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('updateEmitter and updateEnvironment patch scene textures without rebuilding BVH geometry', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(sceneWithEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;

    e.updateEmitter?.('point-a', { intensity: 4 });
    expect(e.getScene?.()?.emitters[0]?.intensity).toBe(4);
    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);

    e.updateEnvironment?.(hdriScene().environment);
    const scene = e.getScene?.();
    expect(scene?.environment.kind).toBe('hdri');
    expect(e._debugSceneTex?.envMap).toBe(true);
    expect(e._debugSceneTex?.envTotalSum).toBeGreaterThan(0);
    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
  });

  it('accepts directional angularDiameter on setScene and updateEmitter without unsupported-field warnings', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });

      e.setScene(sceneWithSoftDirectionalEmitter());
      expect(e.getScene?.()?.emitters[0]).toMatchObject({ id: 'sun', angularDiameter: 0.01 });
      expect(structured.find((w) => w.code === 'pt-webgl2.unconsumed-directional-angular-diameter')).toBeUndefined();
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('soft-sun angular spread is ignored'),
      )).toBe(false);

      structured.length = 0;
      warn.mockClear();
      e.updateEmitter?.('sun', { angularDiameter: 0.02 } as never);

      expect(e.getScene?.()?.emitters[0]).toMatchObject({ id: 'sun', angularDiameter: 0.02 });
      expect(structured.find((w) => w.code === 'pt-webgl2.unconsumed-directional-angular-diameter')).toBeUndefined();
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('soft-sun angular spread is ignored'),
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('updateEmitter mesh-area patches folded material and mesh-light textures without rebuilding BVH geometry', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(sceneWithMeshAreaEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const beforePositions = e._debugGeoPack?.positions;
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 1, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(2);

    e.updateEmitter?.('panel-light', { color: [0.25, 0.5, 1], intensity: 4 });
    const scene = e.getScene?.();
    expect(scene?.emitters[0]?.kind).toBe('mesh-area');
    if (scene?.emitters[0]?.kind === 'mesh-area') {
      expect(scene.emitters[0].color).toEqual([0.25, 0.5, 1]);
      expect(scene.emitters[0].intensity).toBe(4);
    }

    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).toBe(beforePositions);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([0.25, 0.5, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(4);
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
  });

  it('updatePrimitive material fast path preserves mesh-area folded emissive radiance', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(sceneWithMeshAreaEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const beforePositions = e._debugGeoPack?.positions;
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 1, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(2);

    e.updatePrimitive?.('panel', { material: { roughness: 0.25 } } as never);

    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).toBe(beforePositions);
    expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.25);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 1, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(2);
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
    const scene = e.getScene?.();
    const prim = scene?.primitives[0];
    expect(prim?.kind).toBe('mesh');
    if (prim?.kind === 'mesh') {
      expect(prim.material.roughness).toBe(0.25);
      expect(prim.material.emissive).toBeUndefined();
      expect(prim.material.emissiveIntensity).toBeUndefined();
    }
  });

  it('updatePrimitive material fast path repacks implicit emissive mesh lights', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene({ ...triScene(), primitives: [tri('panel', 0)] });
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const beforePositions = e._debugGeoPack?.positions;
    expect(e._debugSceneTex?.meshLightCount).toBe(0);
    expect(e._debugSceneTex?.totalEmissiveArea).toBe(0);

    e.updatePrimitive?.('panel', {
      material: {
        emissive: [1, 0.5, 0.25],
        emissiveIntensity: 3,
      },
    } as never);

    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).toBe(beforePositions);
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 0.5, 0.25]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(3);

    e.updatePrimitive?.('panel', {
      material: {
        emissive: [0, 0, 0],
        emissiveIntensity: 1,
      },
    } as never);

    expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).toBe(beforePositions);
    expect(e._debugSceneTex?.meshLightCount).toBe(0);
    expect(e._debugSceneTex?.totalEmissiveArea).toBe(0);
  });

  // Contract-honesty: EngineOptions.denoiser must not be silently ignored.
  // Unsupported non-null non-'none' values warn once; oidn-final is a real
  // final-pass path and requires explicit host model config.
  it("denoiser: 'none' and absent denoiser are both silent", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await createPTEngine_WebGL2({ ...opts(), denoiser: 'none' });
      await createPTEngine_WebGL2(opts()); // absent
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('denoiser'),
      );
      expect(denoiserWarns).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'oidn-final' requires model config and is not reported as unsupported", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      await expect(createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'oidn-final',
        onWarning: (w) => structured.push(w),
      })).rejects.toThrow(/oidn: \{ modelUrl \}/);
      await createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'oidn-final',
        oidn: { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
        oidnBridgeLoader: async () => ({
          denoiseFinal: async (inputs) => new Float32Array(inputs.color.length),
        }),
        onWarning: (w) => structured.push(w),
      });
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('unsupported-denoiser'),
      );
      expect(denoiserWarns).toHaveLength(0);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-denoiser',
      )).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'svgf-real' emits exactly one console.warn naming the value", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await createPTEngine_WebGL2({ ...opts(), denoiser: 'svgf-real' });
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('denoiser'),
      );
      expect(denoiserWarns).toHaveLength(1);
      expect(String(denoiserWarns[0]![0])).toContain('svgf-real');
      expect(String(denoiserWarns[0]![0])).toContain('pt-webgl2');
    } finally {
      warn.mockRestore();
    }
  });

  it('exposes onError subscription (item 28 — GPU error surface)', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.onError).toBe('function');
    const errors: unknown[] = [];
    const unsub = e.onError!((err) => errors.push(err));
    expect(typeof unsub).toBe('function');
    // Unsubscribe should not throw.
    expect(() => unsub()).not.toThrow();
    // After unsubscribe, further unsub calls are idempotent.
    expect(() => unsub()).not.toThrow();
    e.dispose();
  });

  it('exposes onWarning subscription (ENGINE-01 warning surface)', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.onWarning).toBe('function');
    const warnings: EngineWarning[] = [];
    const unsub = e.onWarning!((warning) => warnings.push(warning));
    expect(typeof unsub).toBe('function');
    expect(() => unsub()).not.toThrow();
    e.dispose();
  });

  it('addPrimitive and removePrimitive rebuild, validate ids, and allow an empty scene', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const e = await createPTEngine_WebGL2(opts());
      e.setScene({ primitives: [tri('a', 0)], emitters: [], environment: { kind: 'none' } });

      e.addPrimitive?.(tri('b', 2));
      expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['a', 'b']);
      expect(() => e.addPrimitive?.(tri('b', 4))).toThrow(/already exists/);

      e.removePrimitive?.('a');
      expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['b']);
      expect(() => e.removePrimitive?.('missing')).toThrow(/not found/);

      e.removePrimitive?.('b');
      expect(e.getScene?.()?.primitives).toEqual([]);
      expect(e.renderFrame(frame(16)).kind).toBe('rendered');
      expect(warn.mock.calls.flat().map(String).filter((m) =>
        m.includes('geometry/material/atlas/BVH texture pack'),
      )).toHaveLength(3);
    } finally {
      warn.mockRestore();
    }
  });
});
