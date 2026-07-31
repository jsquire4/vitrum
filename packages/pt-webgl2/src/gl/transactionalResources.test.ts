import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { buildSceneTextures } from '../scene/uploadSceneTextures.js';
import { createMockGl } from '../__tests__/mockGl.js';
import {
  createNeeCandidateTarget,
  createProgressiveTarget,
  createRenderTarget,
} from './framebuffer.js';
import { GlResources, programGraphKey } from './glResources.js';
import { DEFAULT_TRACE_FEATURES, featureDefines } from '../featureTypes.js';
import { FLOAT16_HALF_MIN_SUBNORMAL } from '../scene/halfFloat.js';

function triangleScene(id = 'triangle'): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id,
      positions: new Float32Array([
        -1, -1, 0,
        1, -1, 0,
        0, 1, 0,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      uvs: new Float32Array([
        0, 0,
        1, 0,
        0.5, 1,
      ]),
      indices: new Uint32Array([0, 1, 2]),
      material: { baseColor: [0.8, 0.8, 0.8], roughness: 1, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgl2 transactional framebuffer allocation', () => {
  it('retains compiler-visible shader dimensions in the relink key', () => {
    const key = programGraphKey(DEFAULT_TRACE_FEATURES);
    for (const changed of [
      { bdpt: true },
      { dof: true },
      { fog: true },
      { randomType: 1 as const },
      { basicMaterials: true, mappedRichMaterials: false },
    ]) {
      expect(programGraphKey({ ...DEFAULT_TRACE_FEATURES, ...changed })).not.toBe(key);
    }
  });

  it('does not retain fixed or unreachable fork flags as relink dimensions', () => {
    for (const field of [
      'mis',
      'russianRoulette',
      'backgroundMap',
      'debugMode',
      'stainedGlassPerturbation',
      'pathStepLimit',
    ]) {
      expect(DEFAULT_TRACE_FEATURES).not.toHaveProperty(field);
    }
    const defines = featureDefines(DEFAULT_TRACE_FEATURES);
    expect(defines.FEATURE_MIS).toBe(1);
    expect(defines.FEATURE_RUSSIAN_ROULETTE).toBe(1);
    expect(defines).not.toHaveProperty('FEATURE_BACKGROUND_MAP');
    expect(defines).not.toHaveProperty('FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION');
    expect(defines).not.toHaveProperty('DEBUG_MODE');
    expect(defines).not.toHaveProperty('RANDOM_TYPE');
  });

  it('keeps a parallel-compiling pass graph private until every program is ready', () => {
    const gl = createMockGl();
    const completionStatus = 0x91b1;
    let complete = false;
    const originalGetExtension = gl.getExtension.bind(gl);
    const originalGetParameter = gl.getProgramParameter.bind(gl);
    (gl as unknown as { getExtension(name: string): unknown }).getExtension = (name) =>
      name === 'KHR_parallel_shader_compile'
        ? { COMPLETION_STATUS_KHR: completionStatus }
        : originalGetExtension(name);
    (gl as unknown as { getProgramParameter(program: WebGLProgram, pname: number): unknown })
      .getProgramParameter = (program, pname) =>
        pname === completionStatus ? complete : originalGetParameter(program, pname);

    const resources = new GlResources(gl, false);
    expect(resources.ensureProgram(DEFAULT_TRACE_FEATURES)).toBe(false);
    expect(resources.ptProgram).toBeNull();

    complete = true;
    expect(resources.ensureProgram(DEFAULT_TRACE_FEATURES)).toBe(true);
    expect(resources.ptProgram).not.toBeNull();
    resources.dispose();
  });

  it('rejects insufficient candidate attachments before allocating an FBO', () => {
    const gl = createMockGl();
    const originalGetParameter = gl.getParameter.bind(gl);
    const createFramebuffer = vi.fn(() => ({ id: 'must-not-allocate' } as never));
    (gl as unknown as { createFramebuffer: typeof createFramebuffer }).createFramebuffer =
      createFramebuffer;
    (gl as unknown as { getParameter(pname: number): unknown }).getParameter = (pname) =>
      pname === gl.MAX_DRAW_BUFFERS || pname === gl.MAX_COLOR_ATTACHMENTS
        ? 3
        : originalGetParameter(pname);

    expect(() => createNeeCandidateTarget(gl, 8, 8)).toThrow(/requires four RGBA32F/);
    expect(createFramebuffer).not.toHaveBeenCalled();
  });

  it('releases prior candidate textures when a later attachment allocation fails', () => {
    const gl = createMockGl();
    const framebuffer = { id: 'candidate-fbo' } as unknown as WebGLFramebuffer;
    const textures = [
      { id: 'candidate-0' },
      { id: 'candidate-1' },
    ] as unknown as WebGLTexture[];
    const createTexture = vi
      .fn<[], WebGLTexture | null>()
      .mockReturnValueOnce(textures[0]!)
      .mockReturnValueOnce(textures[1]!)
      .mockReturnValueOnce(null);
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    (gl as unknown as { createFramebuffer(): WebGLFramebuffer }).createFramebuffer =
      () => framebuffer;
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { deleteFramebuffer: typeof deleteFramebuffer }).deleteFramebuffer =
      deleteFramebuffer;

    expect(() => createNeeCandidateTarget(gl, 8, 8)).toThrow(/render-target texture/);
    expect(deleteTexture.mock.calls.map(([texture]) => texture)).toEqual(textures);
    expect(deleteFramebuffer).toHaveBeenCalledWith(framebuffer);
  });

  it('releases all four candidate textures when the FBO is incomplete', () => {
    const gl = createMockGl();
    const framebuffer = { id: 'candidate-fbo' } as unknown as WebGLFramebuffer;
    const textures = Array.from(
      { length: 4 },
      (_, id) => ({ id: `candidate-${id}` }),
    );
    const createTexture = vi.fn<[], WebGLTexture | null>();
    for (const texture of textures) createTexture.mockReturnValueOnce(texture);
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    (gl as unknown as { createFramebuffer(): WebGLFramebuffer }).createFramebuffer =
      () => framebuffer;
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { checkFramebufferStatus(): number }).checkFramebufferStatus =
      () => gl.FRAMEBUFFER_UNSUPPORTED;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { deleteFramebuffer: typeof deleteFramebuffer }).deleteFramebuffer =
      deleteFramebuffer;

    expect(() => createNeeCandidateTarget(gl, 8, 8)).toThrow(/FRAMEBUFFER_UNSUPPORTED/);
    expect(deleteTexture.mock.calls.map(([texture]) => texture)).toEqual(textures);
    expect(deleteFramebuffer).toHaveBeenCalledWith(framebuffer);
  });

  it('rolls back the FBO and prior MRT textures when an auxiliary allocation fails', () => {
    const gl = createMockGl();
    const handles = [{ id: 'color' }, { id: 'normal' }];
    const createTexture = vi.fn(() => handles.shift() ?? null);
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { deleteFramebuffer: typeof deleteFramebuffer }).deleteFramebuffer = deleteFramebuffer;

    expect(() => createRenderTarget(gl, 32, 16, true)).toThrow(/render-target texture/);
    expect(deleteTexture.mock.calls.map(([texture]) => texture)).toEqual([
      { id: 'color' },
      { id: 'normal' },
    ]);
    expect(deleteFramebuffer).toHaveBeenCalledOnce();
  });

  it('allocates the compact progressive formats and shares one auxiliary pair', () => {
    const gl = createMockGl();
    const texImage2D = vi.fn(gl.texImage2D.bind(gl));
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    Object.assign(gl, { texImage2D, deleteTexture, deleteFramebuffer });

    const target = createProgressiveTarget(gl, 32, 16, true);
    expect(texImage2D.mock.calls.map((call) => [call[2], call[7]])).toEqual([
      [gl.RGBA32F, gl.FLOAT],
      [gl.RGBA32F, gl.FLOAT],
      [gl.RGBA32F, gl.FLOAT],
      [gl.RGBA16F, gl.HALF_FLOAT],
    ]);
    expect(target.fbos).toHaveLength(2);
    expect(target.colors).toHaveLength(2);
    expect(target.normalDepth).not.toBeNull();
    expect(target.albedo).not.toBeNull();

    target.destroy();
    expect(deleteFramebuffer).toHaveBeenCalledTimes(2);
    expect(deleteTexture).toHaveBeenCalledTimes(4);
  });

  it('rolls back both progressive FBOs and every prior texture on allocation failure', () => {
    const gl = createMockGl();
    const framebuffers = [
      { id: 'progressive-fbo-0' },
      { id: 'progressive-fbo-1' },
    ] as unknown as WebGLFramebuffer[];
    const textures = [
      { id: 'progressive-color-0' },
      { id: 'progressive-color-1' },
      { id: 'progressive-normal' },
    ] as unknown as WebGLTexture[];
    const createFramebuffer = vi
      .fn<[], WebGLFramebuffer | null>()
      .mockReturnValueOnce(framebuffers[0]!)
      .mockReturnValueOnce(framebuffers[1]!);
    const createTexture = vi
      .fn<[], WebGLTexture | null>()
      .mockReturnValueOnce(textures[0]!)
      .mockReturnValueOnce(textures[1]!)
      .mockReturnValueOnce(textures[2]!)
      .mockReturnValueOnce(null);
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    Object.assign(gl, {
      createFramebuffer,
      createTexture,
      deleteTexture,
      deleteFramebuffer,
    });

    expect(() => createProgressiveTarget(gl, 32, 16, true)).toThrow(
      /progressive albedo texture/,
    );
    expect(deleteTexture.mock.calls.map(([texture]) => texture)).toEqual(textures);
    expect(deleteFramebuffer.mock.calls.map(([fbo]) => fbo)).toEqual(framebuffers);
  });

  it('rejects an incomplete framebuffer and releases every candidate resource', () => {
    const gl = createMockGl();
    const texture = { id: 'candidate' } as unknown as WebGLTexture;
    const framebuffer = { id: 'candidate-fbo' } as unknown as WebGLFramebuffer;
    const deleteTexture = vi.fn();
    const deleteFramebuffer = vi.fn();
    (gl as unknown as { createTexture(): WebGLTexture }).createTexture = () => texture;
    (gl as unknown as { createFramebuffer(): WebGLFramebuffer }).createFramebuffer = () => framebuffer;
    (gl as unknown as { checkFramebufferStatus(): number }).checkFramebufferStatus =
      () => gl.FRAMEBUFFER_UNSUPPORTED;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;
    (gl as unknown as { deleteFramebuffer: typeof deleteFramebuffer }).deleteFramebuffer = deleteFramebuffer;

    expect(() => createRenderTarget(gl, 8, 8, false)).toThrow(/FRAMEBUFFER_UNSUPPORTED/);
    expect(deleteTexture).toHaveBeenCalledWith(texture);
    expect(deleteFramebuffer).toHaveBeenCalledWith(framebuffer);
  });

  it('keeps the last complete accumulation and present targets when present replacement is incomplete', () => {
    const gl = createMockGl();
    const resources = new GlResources(gl, false);
    resources.ensureAccumResources(8, 8);
    const previousResult = resources.resultTexture();
    const previousDims = resources.accumDims;

    let statusCall = 0;
    (gl as unknown as { checkFramebufferStatus(): number }).checkFramebufferStatus = () => {
      statusCall += 1;
      return statusCall === 1 ? gl.FRAMEBUFFER_COMPLETE : gl.FRAMEBUFFER_UNSUPPORTED;
    };

    expect(() => resources.ensureAccumResources(16, 16)).toThrow(/FRAMEBUFFER_UNSUPPORTED/);
    expect(resources.accumDims).toEqual(previousDims);
    expect(resources.resultTexture()).toBe(previousResult);
    resources.dispose();
  });
});

describe('pt-webgl2 transactional scene texture upload', () => {
  it('keeps CPU and resident scene state unpublished when a fast mutation upload fails', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(triangleScene('transactional-fast-path'));
    const beforeBvh = engine._debugGeoPack?.bvhNodes;
    const beforeRoughness = (
      engine.getScene?.()?.primitives[0] as { material?: { roughness?: number } } | undefined
    )?.material?.roughness;
    const originalDeleteTexture = gl.deleteTexture.bind(gl);
    const deleteTexture = vi.fn((texture: WebGLTexture | null) =>
      originalDeleteTexture(texture));
    let errorRead = 0;
    Object.assign(gl, {
      deleteTexture,
      getError: () => {
        errorRead += 1;
        return errorRead === 2 ? gl.OUT_OF_MEMORY : gl.NO_ERROR;
      },
    });

    expect(() =>
      engine.updatePrimitive?.('transactional-fast-path', {
        material: { roughness: 0.25 },
      }),
    ).toThrow(/scene materials.*OUT_OF_MEMORY/);
    expect(engine._debugGeoPack?.bvhNodes).toBe(beforeBvh);
    expect(engine._debugGeoPack?.materials[0]?.roughness).toBe(beforeRoughness);
    const retained = engine.getScene?.()?.primitives[0];
    expect(retained?.kind).toBe('mesh');
    if (retained?.kind === 'mesh') {
      expect(retained.material.roughness).toBe(beforeRoughness);
    }
    // allocGlTexture owns and retires the failed candidate. No resident handle
    // is exposed to the mutation publisher.
    expect(deleteTexture).toHaveBeenCalledTimes(1);

    Object.assign(gl, { getError: () => gl.NO_ERROR });
    expect(() =>
      engine.updatePrimitive?.('transactional-fast-path', {
        material: { roughness: 0.25 },
      }),
    ).not.toThrow();
    expect(engine._debugGeoPack?.materials[0]?.roughness).toBe(0.25);
    engine.dispose();
  });

  it('rejects an opaque emissive map before allocating any GL texture', () => {
    const gl = createMockGl();
    const originalCreateTexture = gl.createTexture.bind(gl);
    const createTexture = vi.fn(() => originalCreateTexture());
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const scene = triangleScene('opaque-emissive');
    const primitive = scene.primitives[0];
    if (primitive?.kind !== 'mesh') throw new Error('test fixture must be a mesh');
    const rejected: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          emissive: [1, 1, 1],
          emissiveIntensity: 1,
          emissiveMap: { handle: { id: 'gpu-only-texture' } },
        },
      }],
    };
    const capabilities = {
      backend: 'pt-webgl2',
      sceneSupport: {
        primitiveKinds: ['mesh'],
        emitterKinds: [],
        environmentKinds: ['none'],
      },
    } as never;

    expect(() => buildSceneTextures(gl, rejected, capabilities)).toThrow(
      /emissiveMap without complete CPU-readable texels/,
    );
    expect(createTexture).not.toHaveBeenCalled();
  });

  it.each([
    [
      'overflowing',
      Number.MAX_VALUE,
      /envMapIntensity.*(?:representable as float32|overflows WebGL float32 storage)/,
    ],
    [
      'positive-underflowing',
      Number.MIN_VALUE,
      /envMapIntensity.*(?:must not underflow to zero as float32|underflows WebGL float32 storage)/,
    ],
  ])('rejects a %s envMapIntensity before initial scene allocation', (
    _label,
    envMapIntensity,
    message,
  ) => {
    const gl = createMockGl();
    const originalCreateTexture = gl.createTexture.bind(gl);
    const createTexture = vi.fn(() => originalCreateTexture());
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const scene = triangleScene('invalid-environment-scale');
    const primitive = scene.primitives[0];
    if (primitive?.kind !== 'mesh') throw new Error('test fixture must be a mesh');
    const rejected: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          envMapIntensity,
        },
      }],
    };
    const capabilities = {
      backend: 'pt-webgl2',
      sceneSupport: {
        primitiveKinds: ['mesh'],
        emitterKinds: [],
        environmentKinds: ['none'],
      },
    } as never;

    expect(() => buildSceneTextures(gl, rejected, capabilities)).toThrow(message);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it.each([
    [
      'a mapped level-0 value that underflows RGBA16F',
      {
        emissive: [1, 1, 1],
        emissiveIntensity: 1,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Float32Array([FLOAT16_HALF_MIN_SUBNORMAL, 0, 0, 1]),
            __vitrum_hint__: {
              channels: 4,
              dataType: 'float32',
              colorSpace: 'linear',
            },
          },
        },
      },
      /underflows to \+0 in RGBA16F storage/,
    ],
    [
      'a positive emissive operand that underflows its RGBA32F material slot',
      {
        emissive: [2 ** -150, 0, 0],
        emissiveIntensity: 2 ** 100,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([128, 0, 0, 255]),
            __vitrum_hint__: {
              channels: 4,
              dataType: 'uint8',
            },
          },
        },
      },
      /emissive\[0\] underflows material RGBA32F storage/,
    ],
    [
      'negative mapped outgoing radiance',
      {
        emissive: [1, 1, 1],
        emissiveIntensity: 1,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Float32Array([-(2 ** -24), 1, 1, 1]),
            __vitrum_hint__: {
              channels: 4,
              dataType: 'float32',
              colorSpace: 'linear',
            },
          },
        },
      },
      /outgoing-radiance RGB value .* must be non-negative/,
    ],
  ])('rejects %s before allocating any GL texture', (_label, materialPatch, message) => {
    const gl = createMockGl();
    const originalCreateTexture = gl.createTexture.bind(gl);
    const createTexture = vi.fn(() => originalCreateTexture());
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    const scene = triangleScene('precision-reject');
    const primitive = scene.primitives[0];
    if (primitive?.kind !== 'mesh') throw new Error('test fixture must be a mesh');
    const rejected: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          ...materialPatch,
        } as never,
      }],
    };
    const capabilities = {
      backend: 'pt-webgl2',
      sceneSupport: {
        primitiveKinds: ['mesh'],
        emitterKinds: [],
        environmentKinds: ['none'],
      },
    } as never;

    expect(() => buildSceneTextures(gl, rejected, capabilities)).toThrow(message);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('releases every successful allocation when the final attribute-array allocation fails', () => {
    const gl = createMockGl();
    const created: WebGLTexture[] = [];
    let attempt = 0;
    const createTexture = vi.fn(() => {
      attempt += 1;
      if (attempt === 8) return null;
      const texture = { id: attempt } as unknown as WebGLTexture;
      created.push(texture);
      return texture;
    });
    const deleteTexture = vi.fn();
    (gl as unknown as { createTexture: typeof createTexture }).createTexture = createTexture;
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;

    const capabilities = {
      backend: 'pt-webgl2',
      sceneSupport: {
        primitiveKinds: ['mesh'],
        emitterKinds: [],
        environmentKinds: ['none'],
      },
    } as never;
    expect(() => buildSceneTextures(gl, triangleScene(), capabilities)).toThrow(
      /failed to create vertex attributes texture/,
    );
    expect(new Set(deleteTexture.mock.calls.map(([texture]) => texture))).toEqual(new Set(created));
  });

  it('preserves the prior scene and size after failed scene and resize replacements', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(triangleScene('original'));
    engine.setSize?.(8, 8);
    engine.renderFrame({
      frameIndex: 0,
      frameSeed: 1,
      viewport: { width: 8, height: 8, devicePixelRatio: 1 },
      viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
      projMatrix: asMat4(new Float32Array([
        1.5, 0, 0, 0,
        0, 1.5, 0, 0,
        0, 0, -1.002, -1,
        0, 0, -0.2, 0,
      ])),
      quality: { samplesTarget: 1 },
    });

    const originalCreateTexture = gl.createTexture.bind(gl);
    (gl as unknown as { createTexture(): WebGLTexture | null }).createTexture = () => null;
    expect(() => engine.setScene(triangleScene('replacement'))).toThrow();
    expect(engine.getScene?.()?.primitives[0]?.id).toBe('original');
    expect(() => engine.setSize?.(16, 16)).toThrow();
    (gl as unknown as { createTexture(): WebGLTexture | null }).createTexture = originalCreateTexture;

    const capture = await engine.captureFrame?.();
    expect(capture).toMatchObject({ width: 8, height: 8 });
    engine.dispose();
  });

  it('preserves the active scene when a storage-changing mutation replacement fails', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    try {
      engine.setScene(triangleScene('original'));
      const beforeBvh = engine._debugGeoPack?.bvhNodes;
      const handle = {
        image: {
          data: new Float32Array([1, 0.5, 0.25, 1]),
          width: 1,
          height: 1,
        },
      };
      const originalCreateTexture = gl.createTexture.bind(gl);
      (gl as unknown as { createTexture(): WebGLTexture | null }).createTexture = () => null;

      expect(() => engine.updatePrimitive?.('original', {
        material: { baseColorMap: { handle } },
      })).toThrow();
      const failedPrimitive = engine.getScene?.()?.primitives[0];
      expect(failedPrimitive?.kind).toBe('mesh');
      if (failedPrimitive?.kind === 'mesh') {
        expect(failedPrimitive.material.baseColorMap).toBeUndefined();
      }
      expect(engine._debugGeoPack?.bvhNodes).toBe(beforeBvh);

      (gl as unknown as { createTexture(): WebGLTexture | null }).createTexture =
        originalCreateTexture;
      engine.updatePrimitive?.('original', {
        material: { baseColorMap: { handle } },
      });
      const committedPrimitive = engine.getScene?.()?.primitives[0];
      expect(committedPrimitive?.kind).toBe('mesh');
      if (committedPrimitive?.kind === 'mesh') {
        expect(committedPrimitive.material.baseColorMap?.handle).toBe(handle);
      }
      expect(engine._debugGeoPack?.bvhNodes).not.toBe(beforeBvh);
    } finally {
      engine.dispose();
      warn.mockRestore();
    }
  });

  it('reaches a terminal disposed state even when a cleanup callback throws', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(triangleScene());
    const cleanupError = new Error('injected delete failure');
    const deleteTexture = vi.fn(() => {
      throw cleanupError;
    });
    (gl as unknown as { deleteTexture: typeof deleteTexture }).deleteTexture = deleteTexture;

    let disposalError: unknown;
    try {
      engine.dispose();
    } catch (error) {
      disposalError = error;
    }
    expect(disposalError).toBeInstanceOf(AggregateError);
    expect((disposalError as AggregateError).errors).toContain(cleanupError);
    expect(engine.state).toBe('disposed');
    expect(engine.getScene?.()).toBeNull();
    expect(deleteTexture.mock.calls.length).toBeGreaterThan(1);
    expect(() => engine.dispose()).not.toThrow();
  });
});
