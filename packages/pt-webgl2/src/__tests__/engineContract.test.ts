import { describe, expect, it, vi } from 'vitest';
import type {
  AnalyticPrimitive,
  EngineWarning,
  FrameInput,
  MaterialSpec,
  MeshPrimitive,
  Scene,
} from '@vitrum/core';
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

const WHITE_TEX = {
  width: 1,
  height: 1,
  data: new Uint8Array([255, 255, 255, 255]),
  __vitrum_hint__: { channels: 4 },
};

const ROUGHNESS_TEX = {
  width: 1,
  height: 1,
  data: new Uint8Array([128]),
  __vitrum_hint__: { channels: 1 },
};

const METALLIC_TEX = {
  width: 1,
  height: 1,
  data: new Uint8Array([64]),
  __vitrum_hint__: { channels: 1 },
};

const NORMAL_TEX = {
  width: 1,
  height: 1,
  data: new Uint8Array([128, 128, 255, 255]),
  __vitrum_hint__: { channels: 4 },
};

const AO_TEX = {
  width: 1,
  height: 1,
  data: new Uint8Array([255]),
  __vitrum_hint__: { channels: 1 },
};

function texturedTri(id: string, x: number): MeshPrimitive {
  return {
    ...tri(id, x),
    material: {
      ...GREY,
      baseColorMap: { handle: WHITE_TEX },
    },
  };
}

function multiMapTri(id: string, x: number): MeshPrimitive {
  return {
    ...tri(id, x),
    material: {
      ...GREY,
      baseColorMap: { handle: WHITE_TEX },
      roughnessMap: { handle: ROUGHNESS_TEX },
      metallicMap: { handle: METALLIC_TEX },
      normalMap: { handle: NORMAL_TEX },
      aoMap: { handle: AO_TEX },
    },
  };
}

function triListScene(count: number): Scene {
  return {
    primitives: Array.from({ length: count }, (_, i) => tri(`tri-${i}`, i * 2)),
    emitters: [],
    environment: { kind: 'none' },
  };
}

function texturedTriListScene(count: number): Scene {
  return {
    primitives: Array.from({ length: count }, (_, i) => texturedTri(`tri-${i}`, i * 2)),
    emitters: [],
    environment: { kind: 'none' },
  };
}

function oneTexturedTriListScene(untexturedCount: number): Scene {
  return {
    primitives: [
      ...Array.from({ length: untexturedCount }, (_, i) => tri(`tri-${i}`, i * 2)),
      texturedTri(`tri-${untexturedCount}`, untexturedCount * 2),
    ],
    emitters: [],
    environment: { kind: 'none' },
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

function hdriSceneWithPixels(width: number, height: number, scale: number): Scene {
  const data = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    data[i * 4 + 0] = scale * (i + 1);
    data[i * 4 + 1] = scale * 0.5 * (i + 1);
    data[i * 4 + 2] = scale * 0.25 * (i + 1);
    data[i * 4 + 3] = 1;
  }
  return {
    primitives: [tri('tri', 0)],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: { width, height, data },
      intensity: 1,
    },
  };
}

function opts(): PTEngineWebGL2Options {
  return { device: createMockGl() };
}

function frame(spp: number, width = 64, height = 64): FrameInput {
  // an identity-ish view + a finite invertible projection (packFrameParams inverts both)
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width, height, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

describe('PTEngineWebGL2 — contract conformance + accumulation orchestration', () => {
  it('factory returns an engine in state "ready"; rejects a non-WebGL2 device', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.state).toBe('ready');
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(
      /WebGL2RenderingContext/,
    );
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
    expect(c.supportDetails?.mutations.transform).toBe('native');
    expect(c.supportDetails?.mutations.positions).toBe('native');
    expect(c.supportDetails?.mutations.topology).toBe('fallback-rebuild');
    expect(c.supportDetails?.mutations.resize).toBe('native');
    expect(c.supportDetails?.denoisers).toEqual({
      none: 'native',
      auto: 'native',
      atrous: 'unsupported',
      'atrous-variance': 'unsupported',
      'svgf-real': 'unsupported',
      bmfr: 'unsupported',
      'oidn-final': 'native',
      neural: 'unsupported',
    });
    expect(c.supportDetails?.causticStrategies?.bdpt).toEqual(
      expect.objectContaining({
        mode: 'native',
        volumeScattering: 'native',
        emitterKinds: expect.objectContaining({
          directional: 'native',
          point: 'native',
          spot: 'native',
          'rect-area': 'native',
          'disc-area': 'native',
          'mesh-area': 'native',
          environment: 'native',
        }),
      }),
    );
    expect(c.activeFeatures).toEqual(new Set());
    expect(c.inverseRendering).toEqual({
      methods: {
        'finite-difference': 'native',
        'path-replay': 'unsupported',
      },
    });
  });

  it('reports only the resolved selected path-tracing features', async () => {
    const engine = await createPTEngine_WebGL2({
      ...opts(),
      causticStrategy: 'bdpt',
      spectral: true,
      sampling: 'sobol',
    });
    expect(engine.capabilities.activeFeatures).toEqual(
      new Set(['pt-webgl2-bdpt', 'pt-webgl2-sobol-sampling', 'pt-webgl2-spectral']),
    );
    engine.dispose();
  });

  it('selects named BDPT and rejects unsupported caustic strategies/options', async () => {
    const named = await createPTEngine_WebGL2({ ...opts(), causticStrategy: 'bdpt' });
    expect(named.capabilities.causticStrategy).toBe('bdpt');
    expect(named.capabilities.activeFeatures).toContain('pt-webgl2-bdpt');
    named.dispose();

    for (const legacy of [
      { causticStrategy: 'none' },
      { causticStrategy: 'manifold-nee' },
      { causticStrategy: 'photon-map' },
    ]) {
      await expect(
        createPTEngine_WebGL2({
          ...opts(),
          ...legacy,
        } as unknown as PTEngineWebGL2Options),
      ).rejects.toThrow(/causticStrategy must be one of/);
    }
    await expect(
      createPTEngine_WebGL2({
        ...opts(),
        causticOptions: { mneeMaxChainLength: 3 },
      } as unknown as PTEngineWebGL2Options),
    ).rejects.toThrow(/causticOptions are not accepted/);
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

  it('rejects unsupported material extensions before setScene allocates textures', async () => {
    const record = new Map<string, unknown>();
    const e = await createPTEngine_WebGL2({ device: createMockGl(record) });
    const before = (record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0;
    const base = triScene();
    const primitive = base.primitives[0]!;
    const invalid: Scene = {
      ...base,
      primitives: [
        {
          ...primitive,
          material: {
            ...primitive.material,
            extensions: { unsupportedVendorPayload: true },
          },
        },
      ],
    };

    expect(() => e.setScene(invalid)).toThrow(
      /primitive "tri" material\.extensions has unsupported key\(s\): unsupportedVendorPayload/,
    );
    const after = (record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0;
    expect(after).toBe(before);
    expect(e.getScene?.()).toBeNull();
  });

  it('setScene retains authored analytics while rendering a generated mesh representation', async () => {
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
      expect(converted?.kind).toBe('analytic');
      expect(e._debugGeoPack?.triangleCount).toBeGreaterThan(0);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.scene-upload-warning' &&
            String(w.details?.warning).includes(
              'canonical generated MeshPrimitive representation',
            ),
        ),
      ).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects unreadable authored material textures without publishing a partial scene', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      e.setScene(base);
      const retainedScene = e.getScene?.();
      const initialCreates = createTexture.mock.calls.length;
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: { id: 'unreadable-map' } },
            },
          },
        ],
      };

      expect(() => e.setScene(scene)).toThrow(
        /authored material texture during setScene is not CPU-readable/,
      );
      expect(e.getScene?.()).toBe(retainedScene);
      expect(createTexture.mock.calls.length - initialCreates).toBe(0);
      expect(texImage2D.mock.calls.length - initialImage2D).toBe(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBe(0);
      expect(structured.some((w) => w.code === 'pt-webgl2.texture-unreadable')).toBe(false);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects an unreadable HDRI without publishing a partial scene', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      e.setScene(base);
      const retainedScene = e.getScene?.();
      const initialCreates = createTexture.mock.calls.length;
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      expect(() =>
        e.setScene({
          ...base,
          environment: { kind: 'hdri', hdri: { mock: true }, intensity: 1 },
        }),
      ).toThrow(/authored HDRI during setScene is not CPU-readable/);

      expect(e.getScene?.()).toBe(retainedScene);
      expect(createTexture.mock.calls.length - initialCreates).toBe(0);
      expect(texImage2D.mock.calls.length - initialImage2D).toBe(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBe(0);
      expect(structured.some((w) => w.code === 'pt-webgl2.hdri-unreadable')).toBe(false);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('accepts authored sampler policies at the engine surface without approximation warnings', async () => {
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
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: {
                handle: WHITE_TEX,
                magFilter: 'linear',
                minFilter: 'nearest',
                mipFilter: 'linear',
              },
            },
          },
        ],
      };

      e.setScene(scene);
      e.setScene(scene);

      const samplerWarnings = structured.filter((w) => w.code.includes('sampler-policy'));
      expect(samplerWarnings).toHaveLength(0);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter((m) => m.includes('texture sampler policy')),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects unreadable HDRI updates without changing the retained environment', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const retainedScene = e.getScene?.();
      structured.length = 0;
      warn.mockClear();

      expect(() =>
        e.updateEnvironment?.({
          kind: 'hdri',
          hdri: { mock: true },
          intensity: 1,
        }),
      ).toThrow(/authored HDRI during updateEnvironment is not CPU-readable/);

      expect(e.getScene?.()).toBe(retainedScene);
      expect(structured.some((w) => w.code === 'pt-webgl2.hdri-unreadable')).toBe(false);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects unreadable displacement maps without replacing the retained scene', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const displacementMap = { handle: { id: 'height' }, mipFilter: 'none' as const };
      Object.defineProperty(displacementMap, Symbol('vitrum.gltf.textureRefSource'), {
        value: { path: 'materials[0].extensions.VITRUM_displacement.displacementTexture' },
      });
      const scene: Scene = {
        ...base,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              displacementMap,
              displacementScale: 0.2,
              displacementBias: -0.1,
            },
          },
        ],
      };
      e.setScene(base);
      const retainedScene = e.getScene?.();

      expect(() => e.setScene(scene)).toThrow(
        /materials\[0\]\.extensions\.VITRUM_displacement\.displacementTexture handle is not CPU-readable/,
      );
      expect(e.getScene?.()).toBe(retainedScene);
      expect(structured.some((w) => w.code === 'pt-webgl2.vertex-displacement-warning')).toBe(
        false,
      );
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('falls back to a scene repack when scalar displacement fields are patched', async () => {
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
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings.length).toBeGreaterThan(0);
      expect(fallbackWarnings[0]).toMatchObject({
        backend: 'pt-webgl2',
        phase: 'mutation',
        method: 'updatePrimitive',
        details: {
          primitiveId: 'tri',
          fields: ['material'],
          materialFields: ['displacementBias', 'displacementScale', 'roughness'],
          displacementFields: ['displacementBias', 'displacementScale'],
          fallbackReason: 'displacement-geometry-repack',
          nativePatchMissing: 'targeted-displacement-geometry-update',
        },
      });
      expect(structured.some((w) => w.code === 'pt-webgl2.unsupported-displacement-material')).toBe(
        false,
      );
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
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              anisotropy: 0.8,
              anisotropyRotation: 0.5,
              anisotropyMap: { handle: NORMAL_TEX },
            },
          },
        ],
      };
      e.setScene(scene);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.unsupported-material-fields' &&
            Array.isArray(w.details?.fields) &&
            (w.details.fields.includes('anisotropy') ||
              w.details.fields.includes('anisotropyRotation') ||
              w.details.fields.includes('anisotropyMap')),
        ),
      ).toBe(false);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('anisotropy') && m.includes('not rendered')),
      ).toBe(false);
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
      const prim: AnalyticPrimitive = {
        kind: 'analytic',
        id: 'sphere-volume',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: GREY,
      };
      const scene: Scene = {
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              transmission: 1,
              attenuationDistance: 2,
              attenuationColor: [0.8, 0.9, 1.0],
              thickness: 0.25,
              thicknessMap: { handle: ROUGHNESS_TEX },
            },
          },
        ],
        emitters: [],
        environment: { kind: 'none' },
      };
      e.setScene(scene);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.unsupported-material-fields' &&
            Array.isArray(w.details?.fields) &&
            (w.details.fields.includes('thickness') || w.details.fields.includes('thicknessMap')),
        ),
      ).toBe(false);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('thicknessMap') && m.includes('not rendered')),
      ).toBe(false);
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
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              frontLayer: {
                transmission: [0.8, 0.9, 1.0],
                roughness: 0.2,
                normalMap: { handle: NORMAL_TEX },
                normalScale: 0.75,
              },
              backLayer: {
                transmission: [1.0, 0.9, 0.8],
                roughness: 0.3,
                normalMap: { handle: NORMAL_TEX },
                normalScale: 0.5,
              },
            },
          },
        ],
      };
      e.setScene(scene);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.unsupported-material-fields' &&
            Array.isArray(w.details?.fields) &&
            (w.details.fields.includes('frontLayer.normalMap') ||
              w.details.fields.includes('frontLayer.normalScale') ||
              w.details.fields.includes('backLayer.normalMap') ||
              w.details.fields.includes('backLayer.normalScale')),
        ),
      ).toBe(false);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some(
            (m) =>
              m.includes('frontLayer.normalMap') &&
              m.includes('backLayer.normalMap') &&
              m.includes('not rendered'),
          ),
      ).toBe(false);
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
      expect(structured.some((w) => w.code === 'pt-webgl2.unsupported-material-fields')).toBe(
        false,
      );
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

  it('setScene invalidates accumulated samples for the next render', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(2);

    e.setScene(triListScene(1));
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
  });

  it('setSize controls render-target dimensions and resets accumulation only on size changes', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.setSize).toBe('function');
    e.setScene(triScene());

    e.setSize!(32, 48);
    const first = e.renderFrame(frame(16, 32, 48));
    expect(first.kind).toBe('rendered');
    expect(first.samplesAccumulated).toBe(1);
    let captured = await e.captureFrame!();
    expect(captured?.width).toBe(32);
    expect(captured?.height).toBe(48);

    expect(e.renderFrame(frame(16, 32, 48)).samplesAccumulated).toBe(2);
    e.setSize!(32, 48);
    expect(e.renderFrame(frame(16, 32, 48)).samplesAccumulated).toBe(3);

    e.setSize!(16, 20);
    const resized = e.renderFrame(frame(16, 16, 20));
    expect(resized.samplesAccumulated).toBe(1);
    captured = await e.captureFrame!();
    expect(captured?.width).toBe(16);
    expect(captured?.height).toBe(20);

    expect(() => e.setSize!(0, 20)).toThrow(/positive safe integer/);
    expect(e.renderFrame(frame(16, 16, 20)).samplesAccumulated).toBe(2);
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

    e.updatePrimitive?.('tri', { material: { roughness: 0.25 } });
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
    expect(structured.some((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild')).toBe(
      false,
    );
  });

  it('updatePrimitive castShadow patches update the material lane without rebuilding BVH geometry', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
      const beforePositions = e._debugGeoPack?.positions;

      e.updatePrimitive?.('tri', { castShadow: false });
      e.updatePrimitive?.('tri', { castShadow: true, material: { roughness: 0.2 } });

      expect(e._debugGeoPack?.bvhNodes).toBe(beforeBvhNodes);
      expect(e._debugGeoPack?.positions).toBe(beforePositions);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.2);
      const scene = e.getScene?.();
      const prim = scene?.primitives[0];
      expect(prim?.kind).toBe('mesh');
      if (prim?.kind === 'mesh') {
        expect(prim.castShadow).toBe(true);
        expect(prim.material.roughness).toBe(0.2);
      }
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild'),
      ).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(2);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('updatePrimitive("tri") fields [castShadow]'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('updates same-topology primitive geometry with staged textures and no fallback warnings', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;
      const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
      const beforeMaterials = e._debugGeoPack?.materials;

      const moved = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1]);
      const movedAgain = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]);
      e.updatePrimitive?.('tri', { transform: moved } as never);
      const firstRefreshUploads = createTexture.mock.calls.length;
      const firstSubImage2D = texSubImage2D.mock.calls.length;
      const firstSubImage3D = texSubImage3D.mock.calls.length;
      e.updatePrimitive?.('tri', { transform: movedAgain } as never);

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(0);
      expect(firstRefreshUploads - initialTextureUploads).toBe(3);
      expect(createTexture.mock.calls.length - firstRefreshUploads).toBe(3);
      expect(firstSubImage2D - initialSubImage2D).toBe(0);
      expect(firstSubImage3D - initialSubImage3D).toBe(0);
      expect(texSubImage2D.mock.calls.length - firstSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - firstSubImage3D).toBe(0);
      expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
      expect(e._debugGeoPack?.materials).toEqual(beforeMaterials);
      expect(e._debugGeoPack?.positions[0]).toBeCloseTo(2, 6);
      const scene = e.getScene?.();
      const prim = scene?.primitives[0];
      expect(prim?.kind).toBe('mesh');
      if (prim?.kind === 'mesh') {
        expect(Array.from(prim.transform ?? [])).toEqual(Array.from(movedAgain));
      }
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('updatePrimitive("tri") fields [transform]'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('rewrites same-dimension primitive topology through staged scene textures', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;

      const nextIndices = new Uint32Array([0, 1, 2, 2, 3, 0]);
      e.updatePrimitive?.('tri', { indices: nextIndices });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(6);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBe(0);
      const scene = e.getScene?.();
      const prim = scene?.primitives[0];
      expect(prim?.kind).toBe('mesh');
      if (prim?.kind === 'mesh') {
        expect(Array.from(prim.indices ?? [])).toEqual(Array.from(nextIndices));
      }
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('targeted-primitive-geometry-splice'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('respecifies resident scene textures for dimension-changing primitive topology patches', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const deleteTexture = vi.fn();
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const previousTextures = createTexture.mock.results
        .slice(0, initialTextureUploads)
        .map((result) => result.value);
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      const nextPositions = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0, 0, 2, 0]);
      const nextNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
      const nextUvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 1.5]);
      const nextIndices = new Uint32Array([0, 2, 1, 2, 0, 3, 3, 2, 4]);

      e.updatePrimitive?.('tri', {
        positions: nextPositions,
        normals: nextNormals,
        uvs: nextUvs,
        indices: nextIndices,
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri',
        fields: ['indices', 'normals', 'positions', 'uvs'],
        fallbackReason: 'geometry-bvh-texture-rebuild',
        nativePatchMissing: 'targeted-primitive-geometry-splice',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      for (const texture of previousTextures) {
        expect(deleteTexture).toHaveBeenCalledWith(texture);
      }
      const scene = e.getScene?.();
      const prim = scene?.primitives[0];
      expect(prim?.kind).toBe('mesh');
      if (prim?.kind === 'mesh') {
        expect(prim.positions.length).toBe(nextPositions.length);
        expect(Array.from(prim.indices ?? [])).toEqual(Array.from(nextIndices));
      }
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('targeted-primitive-geometry-splice'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds for the first readable material texture map', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
      const handle = { image: { data: new Float32Array([0.2, 0.4, 0.6, 1]), width: 1, height: 1 } };

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.45,
          baseColorMap: { handle },
        },
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri',
        fields: ['material'],
        fallbackReason: 'texture-map-material-patch',
        nativePatchMissing: 'targeted-material-atlas-texture-update',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(1);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.45);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(handle);
      expect(structured.some((w) => w.code === 'pt-webgl2.texture-unreadable')).toBe(false);
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.material-atlas-texture-refresh'),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds both atlas classes for same-dimension map insertions', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    const texStorage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    try {
      const baseMap = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const roughnessMap = {
        image: { data: new Float32Array([0.25, 0.25, 0.25, 1]), width: 1, height: 1 },
      };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: baseMap },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      expect(texStorage3D.mock.calls.some((call) => call[3] === 1 && call[5] === 1)).toBe(true);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const initialStorage3D = texStorage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.4,
          baseColorMap: { handle: baseMap },
          roughnessMap: { handle: roughnessMap },
        },
      });

      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild'),
      ).toHaveLength(1);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      expect(texStorage3D.mock.calls.length - initialStorage3D).toBeGreaterThan(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBeGreaterThan(0);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(baseMap);
      expect(e._debugGeoPack?.materials[0]?.roughnessMap?.handle).toBe(roughnessMap);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.4);
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.material-atlas-texture-refresh'),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('uploads linear HDR emissive maps through the dedicated RGBA16F atlas', async () => {
    const gl = createMockGl();
    const texStorage3D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    const baseScene = triScene();
    const primitive = baseScene.primitives[0];
    if (primitive?.kind !== 'mesh') throw new Error('expected mesh fixture');
    const hdrHandle = {
      width: 1,
      height: 1,
      data: new Float32Array([8, 4, 2, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const scene: Scene = {
      ...baseScene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          emissive: [1, 1, 1],
          emissiveIntensity: 1,
          emissiveMap: { handle: hdrHandle },
        },
      }],
    };
    const engine = await createPTEngine_WebGL2({ device: gl });

    engine.setScene(scene);

    expect(texStorage3D.mock.calls.some((call) => call[2] === gl.RGBA16F)).toBe(true);
    const hdrUpload = texSubImage3D.mock.calls.find((call) => call[9] === gl.HALF_FLOAT);
    expect(hdrUpload?.[10]).toBeInstanceOf(Uint16Array);
    expect((hdrUpload?.[10] as Uint16Array | undefined)?.[0]).toBe(0x4800);
  });

  it('keeps the previous scene published when a staged HDR-atlas upload fails', async () => {
    const gl = createMockGl();
    let nextTextureId = 0;
    let rejectHdrStorage = false;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const deleteTexture = vi.fn();
    const texStorage3D = vi.fn((...args: unknown[]) => {
      if (rejectHdrStorage && args[2] === gl.RGBA16F) {
        throw new Error('synthetic RGBA16F allocation failure');
      }
    });
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    const engine = await createPTEngine_WebGL2({ device: gl, onWarning: () => {} });
    engine.setScene(triScene());
    const retainedScene = engine.getScene?.();
    const firstCandidateTextureId = nextTextureId;
    rejectHdrStorage = true;
    const hdrHandle = {
      width: 1,
      height: 1,
      data: new Float32Array([8, 4, 2, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };

    expect(() => engine.updatePrimitive?.('tri', {
      material: {
        emissive: [1, 1, 1],
        emissiveMap: { handle: hdrHandle },
      },
    } as never)).toThrow(/synthetic RGBA16F allocation failure/);

    expect(engine.getScene?.()).toBe(retainedScene);
    expect(engine._debugGeoPack?.materials[0]?.emissiveMap).toBeUndefined();
    expect(deleteTexture).toHaveBeenCalled();
    expect(deleteTexture.mock.calls.every(
      ([texture]) => (texture as { id: number }).id >= firstCandidateTextureId,
    )).toBe(true);
  });

  it('transactionally rebuilds when a material atlas changes dimensions', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    const texStorage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    try {
      const baseMap = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const largerMap = {
        image: {
          data: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]),
          width: 2,
          height: 2,
        },
      };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: baseMap },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      expect(texStorage3D.mock.calls.some((call) => call[3] === 1 && call[5] === 1)).toBe(true);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const initialStorage3D = texStorage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.5,
          baseColorMap: { handle: largerMap },
        },
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri',
        fields: ['material'],
        fallbackReason: 'texture-map-material-patch',
        nativePatchMissing: 'targeted-material-atlas-texture-update',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      expect(texStorage3D.mock.calls.length - initialStorage3D).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(largerMap);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.5);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when material atlas membership expands', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl({ maxArrayLayers: 5 });
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    const texStorage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    try {
      const baseMap = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const roughnessMap = {
        image: { data: new Float32Array([0.25, 0.25, 0.25, 1]), width: 1, height: 1 },
      };
      const metallicMap = {
        image: { data: new Float32Array([0.75, 0.75, 0.75, 1]), width: 1, height: 1 },
      };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: baseMap },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      expect(texStorage3D.mock.calls.some((call) => call[3] === 1 && call[5] === 1)).toBe(true);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const initialStorage3D = texStorage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        material: {
          baseColorMap: { handle: baseMap },
          roughnessMap: { handle: roughnessMap },
          metallicMap: { handle: metallicMap },
        },
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri',
        fields: ['material'],
        fallbackReason: 'texture-map-material-patch',
        nativePatchMissing: 'targeted-material-atlas-texture-update',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      expect(texStorage3D.mock.calls.length - initialStorage3D).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
      expect(e._debugGeoPack?.materials[0]?.roughnessMap?.handle).toBe(roughnessMap);
      expect(e._debugGeoPack?.materials[0]?.metallicMap?.handle).toBe(metallicMap);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when material atlas storage compacts', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    const texStorage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    (gl as unknown as { texStorage3D: typeof texStorage3D }).texStorage3D = texStorage3D;
    try {
      const baseMap = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const roughnessMap = {
        image: { data: new Float32Array([0.25, 0.25, 0.25, 1]), width: 1, height: 1 },
      };
      const metallicMap = {
        image: { data: new Float32Array([0.75, 0.75, 0.75, 1]), width: 1, height: 1 },
      };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: baseMap },
              roughnessMap: { handle: roughnessMap },
              metallicMap: { handle: metallicMap },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      expect(texStorage3D.mock.calls.some((call) => call[3] === 1 && call[5] === 3)).toBe(true);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const initialStorage3D = texStorage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        material: {
          roughnessMap: undefined,
          metallicMap: undefined,
        },
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(1);
      expect(fallbackWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri',
        fields: ['material'],
        fallbackReason: 'texture-map-material-patch',
        nativePatchMissing: 'targeted-material-atlas-texture-update',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      expect(texStorage3D.mock.calls.length - initialStorage3D).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(baseMap);
      expect(e._debugGeoPack?.materials[0]?.roughnessMap).toBeUndefined();
      expect(e._debugGeoPack?.materials[0]?.metallicMap).toBeUndefined();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects an unreadable replacement map without mutating the resident atlas or scene', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const baseMap = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const unreadableMap = { label: 'unreadable' };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            material: {
              ...prim.material,
              baseColorMap: { handle: baseMap },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const retainedScene = e.getScene?.();

      expect(() =>
        e.updatePrimitive?.('tri', {
          material: {
            baseColorMap: { handle: unreadableMap },
          },
        }),
      ).toThrow(/authored material texture during updatePrimitive is not CPU-readable/);

      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild'),
      ).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBe(0);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(structured.filter((w) => w.code === 'pt-webgl2.texture-unreadable')).toHaveLength(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
      expect(e.getScene?.()).toBe(retainedScene);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(baseMap);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('texture-map-material-patch'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('updates existing-atlas material texture-map patches in place', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    try {
      const handle = { image: { data: new Float32Array([1, 0, 0, 1]), width: 1, height: 1 } };
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            uv1: new Float32Array(8),
            material: {
              ...prim.material,
              baseColorMap: { handle },
            },
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        material: {
          roughness: 0.35,
          baseColorMap: { handle, texCoord: 1, wrapS: 'clamp-to-edge' },
        },
      } as never);

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(1);
      const materialRowUploads = texSubImage2D.mock.calls.slice(initialSubImage2D);
      expect(materialRowUploads).toHaveLength(0);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.35);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.handle).toBe(handle);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.texCoord).toBe(1);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap?.wrapS).toBe('clamp-to-edge');
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('texture-map-material-patch'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects an unreadable new material map without publishing the scalar patch', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const opaqueHandle = { id: 'opaque-base-map' };
      const retainedScene = e.getScene?.();

      expect(() =>
        e.updatePrimitive?.('tri', {
          material: {
            roughness: 0.5,
            baseColorMap: { handle: opaqueHandle },
          },
        }),
      ).toThrow(/authored material texture during updatePrimitive is not CPU-readable/);

      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild'),
      ).toHaveLength(0);
      const textureWarnings = structured.filter((w) => w.code === 'pt-webgl2.texture-unreadable');
      expect(textureWarnings).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBe(0);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(e.getScene?.()).toBe(retainedScene);
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(1);
      expect(e._debugGeoPack?.materials[0]?.baseColorMap).toBeUndefined();
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-mutation-fallback-rebuild') ||
              m.includes('texture-map-material-patch'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('updates vertex-color material flags and attributes with staged texture writes', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        colors: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]),
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(4);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBe(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('does not repack the material texture when a vertex-color patch keeps the same material flag set', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const baseScene = triScene();
      const prim = baseScene.primitives[0];
      if (prim?.kind !== 'mesh') throw new Error('expected mesh fixture');
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...prim,
            colors: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
          },
        ],
      };
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(scene);
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;

      e.updatePrimitive?.('tri', {
        colors: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]),
      });

      const fallbackWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-mutation-fallback-rebuild',
      );
      expect(fallbackWarnings).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(3);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBe(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('rewrites dimension-stable primitive list mutations through staged scene textures', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;

      e.addPrimitive?.(tri('tri-5', 10));

      expect(e.getScene?.()?.primitives.map((p) => String(p.id))).toEqual([
        'tri-0',
        'tri-1',
        'tri-2',
        'tri-3',
        'tri-4',
        'tri-5',
      ]);
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild'),
      ).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBe(7);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBe(0);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-list-fallback-rebuild') ||
              m.includes('targeted-primitive-list-splice'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when a primitive-list mutation creates the first mesh-light texture', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;

      e.addPrimitive?.({
        ...tri('tri-5', 10),
        material: {
          ...GREY,
          emissive: [1, 0.5, 0.25],
          emissiveIntensity: 2,
        },
      });

      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(1);
      expect(listWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri-5',
        fallbackReason: 'primitive-list-scene-repack',
        nativePatchMissing: 'targeted-primitive-list-splice',
      });
      expect(e._debugSceneTex?.meshLightCount).toBe(2);
      expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
      expect(e._debugSceneTex?.totalEmissivePower).toBeGreaterThan(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThanOrEqual(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds atlas-backed primitive-list mutations', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(texturedTriListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialSubImage2D = texSubImage2D.mock.calls.length;
      const initialSubImage3D = texSubImage3D.mock.calls.length;

      e.addPrimitive?.(texturedTri('tri-5', 10));

      expect(e.getScene?.()?.primitives.map((p) => String(p.id))).toEqual([
        'tri-0',
        'tri-1',
        'tri-2',
        'tri-3',
        'tri-4',
        'tri-5',
      ]);
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild'),
      ).toHaveLength(1);
      expect(
        structured.filter((w) => w.code === 'pt-webgl2.material-atlas-texture-refresh'),
      ).toHaveLength(0);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
      expect(texSubImage3D.mock.calls.length - initialSubImage3D).toBeGreaterThan(0);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-list-fallback-rebuild') ||
              m.includes('targeted-primitive-list-splice') ||
              m.includes('material-atlas-texture-refresh'),
          ),
      ).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when a primitive-list mutation creates the first atlas', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      e.addPrimitive?.(texturedTri('tri-5', 10));

      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(1);
      expect(listWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri-5',
        fallbackReason: 'primitive-list-scene-repack',
        nativePatchMissing: 'targeted-primitive-list-splice',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when a primitive-list mutation changes atlas capacity', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(texturedTriListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      e.addPrimitive?.(multiMapTri('tri-5', 10));

      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(1);
      expect(listWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri-5',
        fallbackReason: 'primitive-list-scene-repack',
        nativePatchMissing: 'targeted-primitive-list-splice',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally swaps scene textures during dimension-changing primitive list fallbacks', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const deleteTexture = vi.fn();
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(texturedTriListScene(1));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialTextureDeletes = deleteTexture.mock.calls.length;
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      e.addPrimitive?.(multiMapTri('tri-extra', 2));

      expect(e.getScene?.()?.primitives.map((p) => String(p.id))).toEqual(['tri-0', 'tri-extra']);
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(deleteTexture.mock.calls.length - initialTextureDeletes).toBeGreaterThan(0);
      expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(1);
      expect(listWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri-extra',
        fallbackReason: 'primitive-list-scene-repack',
        nativePatchMissing: 'targeted-primitive-list-splice',
      });
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds when a primitive-list mutation removes the atlas', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const deleteTexture = vi.fn();
    const texSubImage2D = vi.fn();
    const texSubImage3D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texSubImage3D: typeof texSubImage3D }).texSubImage3D = texSubImage3D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(oneTexturedTriListScene(5));
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialTextureDeletes = deleteTexture.mock.calls.length;

      e.removePrimitive?.('tri-5');

      expect(e.getScene?.()?.primitives.map((p) => String(p.id))).toEqual([
        'tri-0',
        'tri-1',
        'tri-2',
        'tri-3',
        'tri-4',
      ]);
      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(1);
      expect(listWarnings[0]?.details).toMatchObject({
        primitiveId: 'tri-5',
        fallbackReason: 'primitive-list-scene-repack',
        nativePatchMissing: 'targeted-primitive-list-splice',
      });
      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(deleteTexture.mock.calls.length - initialTextureDeletes).toBeGreaterThan(0);
      const atlasWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.material-atlas-texture-refresh',
      );
      expect(atlasWarnings).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('warns once when primitive add/remove use the scene-texture/BVH rebuild fallback', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;

      e.addPrimitive?.(tri('extra', 2));
      const afterAddUploads = createTexture.mock.calls.length;
      const afterAddImage2D = texImage2D.mock.calls.length;
      const afterAddImage3D = texImage3D.mock.calls.length;
      e.removePrimitive?.('extra');
      const afterRemoveUploads = createTexture.mock.calls.length;
      const afterRemoveImage2D = texImage2D.mock.calls.length;
      const afterRemoveImage3D = texImage3D.mock.calls.length;
      // A second add of the same id is the same fallback signature and should not
      // spam animation/streaming hosts that repeatedly rebuild the same slot.
      e.addPrimitive?.(tri('extra', 2));
      const afterSecondAddUploads = createTexture.mock.calls.length;
      const afterSecondAddImage2D = texImage2D.mock.calls.length;
      const afterSecondAddImage3D = texImage3D.mock.calls.length;
      e.removePrimitive?.('extra');
      const afterSecondRemoveUploads = createTexture.mock.calls.length;
      const afterSecondRemoveImage2D = texImage2D.mock.calls.length;
      const afterSecondRemoveImage3D = texImage3D.mock.calls.length;
      e.addPrimitive?.({
        ...tri('extra', 2),
        material: {
          ...GREY,
          baseColorMap: { handle: WHITE_TEX },
        },
      });
      const afterTexturedAddUploads = createTexture.mock.calls.length;
      const afterTexturedAddImage2D = texImage2D.mock.calls.length;
      const afterTexturedAddImage3D = texImage3D.mock.calls.length;

      const listWarnings = structured.filter(
        (w) => w.code === 'pt-webgl2.primitive-list-fallback-rebuild',
      );
      expect(listWarnings).toHaveLength(2);
      expect(listWarnings.map((w) => w.method)).toEqual(['addPrimitive', 'removePrimitive']);
      expect(listWarnings.map((w) => w.details)).toEqual([
        {
          primitiveId: 'extra',
          operation: 'addPrimitive',
          fallbackReason: 'primitive-list-scene-repack',
          nativePatchMissing: 'targeted-primitive-list-splice',
        },
        {
          primitiveId: 'extra',
          operation: 'removePrimitive',
          fallbackReason: 'primitive-list-scene-repack',
          nativePatchMissing: 'targeted-primitive-list-splice',
        },
      ]);
      expect(afterAddUploads - initialTextureUploads).toBeGreaterThan(0);
      expect(afterRemoveUploads - afterAddUploads).toBeGreaterThan(0);
      expect(afterSecondAddUploads - afterRemoveUploads).toBeGreaterThan(0);
      expect(afterSecondRemoveUploads - afterSecondAddUploads).toBeGreaterThan(0);
      expect(afterTexturedAddUploads - afterSecondRemoveUploads).toBeGreaterThan(0);
      expect(afterAddImage2D - initialImage2D).toBeGreaterThan(0);
      expect(afterRemoveImage2D - afterAddImage2D).toBeGreaterThan(0);
      expect(afterSecondAddImage2D - afterRemoveImage2D).toBeGreaterThan(0);
      expect(afterSecondRemoveImage2D - afterSecondAddImage2D).toBeGreaterThan(0);
      expect(afterTexturedAddImage2D - afterSecondRemoveImage2D).toBeGreaterThan(0);
      expect(afterAddImage3D - initialImage3D).toBeGreaterThan(0);
      expect(afterRemoveImage3D - afterAddImage3D).toBeGreaterThan(0);
      expect(afterSecondAddImage3D - afterRemoveImage3D).toBeGreaterThan(0);
      expect(afterSecondRemoveImage3D - afterSecondAddImage3D).toBeGreaterThan(0);
      expect(afterTexturedAddImage3D - afterSecondRemoveImage3D).toBeGreaterThan(0);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter(
            (m) =>
              m.includes('primitive-list-fallback-rebuild') || m.includes('scene-texture/BVH pack'),
          ),
      ).toHaveLength(2);
    } finally {
      warn.mockRestore();
    }
  });

  it('retains analytic addPrimitive while rebuilding its represented mesh transaction', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texImage2D = vi.fn();
    const texImage3D = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { texImage3D: typeof texImage3D }).texImage3D = texImage3D;
    try {
      const e = await createPTEngine_WebGL2({
        device: gl,
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      const initialTextureUploads = createTexture.mock.calls.length;
      const initialImage2D = texImage2D.mock.calls.length;
      const initialImage3D = texImage3D.mock.calls.length;
      const sphere: AnalyticPrimitive = {
        kind: 'analytic',
        id: 'sphere-added',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: GREY,
      };

      e.addPrimitive?.(sphere);

      expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
      expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
      expect(texImage3D.mock.calls.length - initialImage3D).toBeGreaterThan(0);
      const added = e.getScene?.()?.primitives.find((p) => p.id === 'sphere-added');
      expect(added?.kind).toBe('analytic');
      expect(e._debugGeoPack?.triangleCount).toBeGreaterThan(2);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.primitive-list-fallback-rebuild' &&
            w.details?.fallbackReason === 'primitive-list-scene-repack',
        ),
      ).toBe(true);
      expect(
        structured.some(
          (w) =>
            w.code === 'pt-webgl2.scene-upload-warning' &&
            String(w.details?.warning).includes(
              'canonical generated MeshPrimitive representation',
            ),
        ),
      ).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('transactionally rebuilds global proposal lanes for emitter and environment updates', async () => {
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texImage2D = vi.fn();
    const deleteTexture = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    const e = await createPTEngine_WebGL2({ device: gl });
    e.setScene(sceneWithEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const initialTextureUploads = createTexture.mock.calls.length;
    const initialSubImage2D = texSubImage2D.mock.calls.length;
    const initialImage2D = texImage2D.mock.calls.length;
    const initialDeletes = deleteTexture.mock.calls.length;

    e.updateEmitter?.('point-a', { intensity: 4 });
    expect(e.getScene?.()?.emitters[0]?.intensity).toBe(4);
    expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
    expect(deleteTexture.mock.calls.length - initialDeletes).toBeGreaterThan(0);
    expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
    expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
    const afterEmitterBvhNodes = e._debugGeoPack?.bvhNodes;
    const afterEmitterTextureUploads = createTexture.mock.calls.length;
    const afterEmitterDeletes = deleteTexture.mock.calls.length;

    e.updateEnvironment?.(hdriScene().environment);
    const scene = e.getScene?.();
    expect(scene?.environment.kind).toBe('hdri');
    expect(e._debugSceneTex?.envMap).toBe(true);
    expect(e._debugSceneTex?.envTotalSum).toBeGreaterThan(0);
    expect(e._debugGeoPack?.bvhNodes).not.toBe(afterEmitterBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(createTexture.mock.calls.length - afterEmitterTextureUploads).toBeGreaterThan(0);
    expect(deleteTexture.mock.calls.length - afterEmitterDeletes).toBeGreaterThan(0);
  });

  it('transactionally rebuilds proposal lanes for same-size and resized HDRIs', async () => {
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texImage2D = vi.fn();
    const deleteTexture = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;

    const e = await createPTEngine_WebGL2({ device: gl });
    e.setScene(hdriSceneWithPixels(1, 1, 0.25));
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const initialTextureUploads = createTexture.mock.calls.length;
    const initialSubImage2D = texSubImage2D.mock.calls.length;
    const initialImage2D = texImage2D.mock.calls.length;
    const initialDeletes = deleteTexture.mock.calls.length;

    e.updateEnvironment?.(hdriSceneWithPixels(1, 1, 0.5).environment);
    expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(e._debugSceneTex?.envWidth).toBe(1);
    expect(e._debugSceneTex?.envHeight).toBe(1);
    expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
    expect(deleteTexture.mock.calls.length - initialDeletes).toBeGreaterThan(0);
    expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
    expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);

    const afterSameSizeSubImage2D = texSubImage2D.mock.calls.length;
    const afterSameSizeImage2D = texImage2D.mock.calls.length;
    const afterSameSizeBvhNodes = e._debugGeoPack?.bvhNodes;
    const afterSameSizeTextureUploads = createTexture.mock.calls.length;
    const afterSameSizeDeletes = deleteTexture.mock.calls.length;
    e.updateEnvironment?.(hdriSceneWithPixels(2, 2, 0.125).environment);
    expect(e._debugGeoPack?.bvhNodes).not.toBe(afterSameSizeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(e._debugSceneTex?.envWidth).toBe(2);
    expect(e._debugSceneTex?.envHeight).toBe(2);
    expect(createTexture.mock.calls.length - afterSameSizeTextureUploads).toBeGreaterThan(0);
    expect(deleteTexture.mock.calls.length - afterSameSizeDeletes).toBeGreaterThan(0);
    expect(texImage2D.mock.calls.length - afterSameSizeImage2D).toBeGreaterThan(0);
    expect(texSubImage2D.mock.calls.length - afterSameSizeSubImage2D).toBe(0);
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
      expect(
        structured.find((w) => w.code === 'pt-webgl2.unconsumed-directional-angular-diameter'),
      ).toBeUndefined();
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('soft-sun angular spread is ignored')),
      ).toBe(false);

      structured.length = 0;
      warn.mockClear();
      e.updateEmitter?.('sun', { angularDiameter: 0.02 });

      expect(e.getScene?.()?.emitters[0]).toMatchObject({ id: 'sun', angularDiameter: 0.02 });
      expect(
        structured.find((w) => w.code === 'pt-webgl2.unconsumed-directional-angular-diameter'),
      ).toBeUndefined();
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('soft-sun angular spread is ignored')),
      ).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('updateEmitter mesh-area patches rebuild the globally coupled proposal transaction', async () => {
    const gl = createMockGl();
    let nextTextureId = 0;
    const createTexture = vi.fn(() => ({ id: nextTextureId++ }));
    const texSubImage2D = vi.fn();
    const texImage2D = vi.fn();
    const deleteTexture = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { texSubImage2D: typeof texSubImage2D }).texSubImage2D = texSubImage2D;
    (gl as unknown as { texImage2D: typeof texImage2D }).texImage2D = texImage2D;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    const e = await createPTEngine_WebGL2({ device: gl });
    e.setScene(sceneWithMeshAreaEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const beforePositions = e._debugGeoPack?.positions;
    const initialTextureUploads = createTexture.mock.calls.length;
    const initialSubImage2D = texSubImage2D.mock.calls.length;
    const initialImage2D = texImage2D.mock.calls.length;
    const initialDeletes = deleteTexture.mock.calls.length;
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

    expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).not.toBe(beforePositions);
    expect(e._debugGeoPack?.positions).toStrictEqual(beforePositions);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([0.25, 0.5, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(4);
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
    expect(createTexture.mock.calls.length - initialTextureUploads).toBeGreaterThan(0);
    expect(deleteTexture.mock.calls.length - initialDeletes).toBeGreaterThan(0);
    expect(texImage2D.mock.calls.length - initialImage2D).toBeGreaterThan(0);
    expect(texSubImage2D.mock.calls.length - initialSubImage2D).toBe(0);
  });

  it('updatePrimitive preserves folded emissive radiance through the coupled proposal transaction', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(sceneWithMeshAreaEmitter());
    const beforeBvhNodes = e._debugGeoPack?.bvhNodes;
    const beforePositions = e._debugGeoPack?.positions;
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 1, 1]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(2);

    e.updatePrimitive?.('panel', { material: { roughness: 0.25 } });

    expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).not.toBe(beforePositions);
    expect(e._debugGeoPack?.positions).toStrictEqual(beforePositions);
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

  it('updatePrimitive validates only the patched material and never reads an accepted sibling', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene({
      primitives: [
        {
          ...tri('target', 0),
          material: {
            ...GREY,
            baseColor: [0.4, 0.5, 0.6],
          },
        },
        tri('untouched', 2),
      ],
      emitters: [],
      environment: { kind: 'none' },
    });
    const untouched = e.getScene?.()?.primitives[1];
    expect(untouched?.id).toBe('untouched');
    const acceptedMaterial = untouched!.material;
    Object.defineProperty(untouched!, 'material', {
      configurable: true,
      get: () => {
        throw new Error('untouched sibling material was traversed');
      },
    });

    try {
      expect(() =>
        e.updatePrimitive?.('target', {
          material: { roughness: 0.2 },
        }),
      ).not.toThrow();
      expect(e._debugGeoPack?.materials[0]?.roughness).toBe(0.2);
    } finally {
      Object.defineProperty(untouched!, 'material', {
        configurable: true,
        value: acceptedMaterial,
      });
      e.dispose();
    }
  });

  it('updatePrimitive repacks implicit emitters through complete proposal transactions', async () => {
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

    expect(e._debugGeoPack?.bvhNodes).not.toBe(beforeBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(beforeBvhNodes);
    expect(e._debugGeoPack?.positions).not.toBe(beforePositions);
    expect(e._debugGeoPack?.positions).toStrictEqual(beforePositions);
    expect(e._debugSceneTex?.meshLightCount).toBe(2);
    expect(e._debugSceneTex?.totalEmissiveArea).toBeCloseTo(2, 6);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 0.5, 0.25]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(3);

    const afterActivationBvhNodes = e._debugGeoPack?.bvhNodes;
    const afterActivationPositions = e._debugGeoPack?.positions;
    e.updatePrimitive?.('panel', {
      material: {
        emissive: [0, 0, 0],
        emissiveIntensity: 1,
      },
    } as never);

    expect(e._debugGeoPack?.bvhNodes).not.toBe(afterActivationBvhNodes);
    expect(e._debugGeoPack?.bvhNodes).toStrictEqual(afterActivationBvhNodes);
    expect(e._debugGeoPack?.positions).not.toBe(afterActivationPositions);
    expect(e._debugGeoPack?.positions).toStrictEqual(afterActivationPositions);
    expect(e._debugSceneTex?.meshLightCount).toBe(0);
    expect(e._debugSceneTex?.totalEmissiveArea).toBe(0);
  });

  it('accepts skipEmitter through setScene and keeps emission camera-visible without implicit NEE', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene({
      primitives: [{
        ...tri('panel', 0),
        material: {
          ...GREY,
          emissive: [1, 0.5, 0.25],
          emissiveIntensity: 3,
          extensions: { skipEmitter: true },
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    });

    expect(e._debugSceneTex?.meshLightCount).toBe(0);
    expect(e._debugSceneTex?.totalEmissivePower).toBe(0);
    expect(e._debugGeoPack?.materials[0]?.emissive).toEqual([1, 0.5, 0.25]);
    expect(e._debugGeoPack?.materials[0]?.emissiveIntensity).toBe(3);
    e.dispose();
  });

  // Contract honesty: unsupported denoisers fail construction instead of
  // silently disappearing or degrading to another estimator.
  it("denoiser: 'none' and absent denoiser are both silent", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await createPTEngine_WebGL2({ ...opts(), denoiser: 'none' });
      await createPTEngine_WebGL2(opts()); // absent
      const denoiserWarns = warn.mock.calls.filter((args) => String(args[0]).includes('denoiser'));
      expect(denoiserWarns).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'oidn-final' default-resolves a model URL and is not reported as unsupported", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const withoutHostUrl = await createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'oidn-final',
        onWarning: (w) => structured.push(w),
      });
      withoutHostUrl.dispose();
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
      expect(structured.some((w) => w.code === 'pt-webgl2.unsupported-denoiser')).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'auto' resolves to OIDN via the default model URL", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      await createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'auto',
        onWarning: (w) => structured.push(w),
      });
      expect(structured).toEqual([
        expect.objectContaining({
          code: 'pt-webgl2.denoiser-auto-resolved',
          details: expect.objectContaining({
            requested: 'auto',
            resolved: 'oidn-final',
            reason: 'default-oidn-model-url',
            packageProvidesProductionWeights: false,
          }),
        }),
      ]);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('unsupported-denoiser')),
      ).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'auto' resolves to OIDN when host model config exists", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      await createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'auto',
        oidn: { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
        oidnBridgeLoader: async () => ({
          denoiseFinal: async (inputs) => new Float32Array(inputs.color.length),
        }),
        onWarning: (w) => structured.push(w),
      });
      expect(structured).toEqual([
        expect.objectContaining({
          code: 'pt-webgl2.denoiser-auto-resolved',
          details: expect.objectContaining({
            requested: 'auto',
            resolved: 'oidn-final',
            reason: 'host-oidn-model-url',
            packageProvidesProductionWeights: false,
          }),
        }),
      ]);
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .some((m) => m.includes('unsupported-denoiser')),
      ).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it.each(['atrous', 'atrous-variance', 'svgf-real', 'bmfr', 'neural'] as const)(
    "denoiser: '%s' is rejected instead of degrading",
    async (denoiser) => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      try {
        await expect(createPTEngine_WebGL2({ ...opts(), denoiser: denoiser as never })).rejects.toThrow(
          /denoiser must be one of/,
        );
        expect(warn).not.toHaveBeenCalled();
      } finally {
        warn.mockRestore();
      }
    },
  );

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
      expect(
        warn.mock.calls
          .flat()
          .map(String)
          .filter((m) => m.includes('scene-texture/BVH pack')),
      ).toHaveLength(3);
    } finally {
      warn.mockRestore();
    }
  });
});
