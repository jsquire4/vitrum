import { describe, expect, it, vi } from 'vitest';
import { asMat4, type FrameInput, type Scene } from '@vitrum/core';
import {
  AUX_ALLOCATION_BYTES_PER_PIXEL,
  AUX_RENDER_TARGET_BYTES_PER_PIXEL,
  BASE_RENDER_TARGET_BYTES_PER_PIXEL,
  DEFAULT_RENDER_TARGET_BUDGET_BYTES,
  DENOISED_RENDER_TARGET_BYTES_PER_PIXEL,
  createPTEngine_WebGL2,
  estimateWebGl2AllocationBytes,
  estimateWebGl2ResidentBytes,
  estimateWebGl2RenderTargetBytes,
} from '../index.js';
import {
  NEE_CANDIDATE_BYTES_PER_PIXEL,
  estimateWebGl2NeeCandidateBytes,
  selectWebGl2NeeCandidateRows,
} from '../gl/renderTargetBudget.js';
import { createMockGl } from './mockGl.js';

function scene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'triangle',
      positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: { baseColor: [0.8, 0.8, 0.8], roughness: 1, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function frame(): FrameInput {
  const view = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  const projection = asMat4(new Float32Array([
    1.5, 0, 0, 0,
    0, 1.5, 0, 0,
    0, 0, -1.002, -1,
    0, 0, -0.2, 0,
  ]));
  return {
    viewMatrix: view,
    projMatrix: projection,
    viewport: { width: 64, height: 64, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: 1 },
  };
}

function installAllocationSpies(gl: WebGL2RenderingContext) {
  const createTexture = vi.fn(gl.createTexture.bind(gl));
  const createFramebuffer = vi.fn(gl.createFramebuffer.bind(gl));
  const createShader = vi.fn(gl.createShader.bind(gl));
  Object.assign(gl, { createTexture, createFramebuffer, createShader });
  return { createTexture, createFramebuffer, createShader };
}

describe('pt-webgl2 strict frame and size boundary', () => {
  it('accepts an omitted legacy camera position and derives it from the view matrix', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    expect(engine.renderFrame(frame()).kind).toBe('skipped');
    engine.dispose();
  });

  it('accepts a coherent legacy camera position', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    const coherent = {
      ...frame(),
      viewMatrix: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -2, 1,
      ])),
      cameraPosition: [0, 0, 2] as const,
    };
    expect(engine.renderFrame(coherent).kind).toBe('skipped');
    engine.dispose();
  });

  it('rejects a mismatched legacy camera position before GL work', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();

    expect(() => engine.renderFrame({
      ...frame(),
      cameraPosition: [0, 0, 2],
    })).toThrow(/cameraPosition disagrees with inverse\(viewMatrix\)/);
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('rejects malformed viewport/index/seed values before GL allocation or compilation', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(scene());
    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();

    const valid = frame();
    const symbolQuality = Object.assign(
      { samplesTarget: 1 },
      { [Symbol('unsupported-quality-key')]: true },
    );
    const cases: ReadonlyArray<readonly [FrameInput, RegExp]> = [
      [[] as never, /input must be a non-array object/],
      [{ ...valid, viewport: undefined } as never, /viewport must be a non-array object/],
      [{ ...valid, viewport: [] } as never, /viewport must be a non-array object/],
      [{ ...valid, viewport: { width: 0, height: 64, devicePixelRatio: 1 } }, /width.*positive safe integer/],
      [{ ...valid, viewport: { width: 1.5, height: 64, devicePixelRatio: 1 } }, /width.*positive safe integer/],
      [{ ...valid, viewport: { width: 64, height: 64, devicePixelRatio: 0 } }, /devicePixelRatio.*finite and > 0/],
      [{ ...valid, frameIndex: -1 }, /frameIndex.*unsigned 32-bit integer/],
      [{ ...valid, frameIndex: 0.5 }, /frameIndex.*unsigned 32-bit integer/],
      [{ ...valid, frameIndex: 0x1_0000_0000 }, /frameIndex.*unsigned 32-bit integer/],
      [{ ...valid, frameSeed: -1 }, /frameSeed.*unsigned 32-bit integer/],
      [{ ...valid, frameSeed: 0x1_0000_0000 }, /frameSeed.*unsigned 32-bit integer/],
      [{ ...valid, viewMatrix: new Float32Array(15) as never }, /viewMatrix.*exactly 16/],
      [{ ...valid, projMatrix: new Float32Array(16).fill(Number.NaN) as never }, /projMatrix\[0\].*finite/],
      [{ ...valid, prevViewMatrix: new Float32Array(15) as never }, /prevViewMatrix.*exactly 16/],
      [{ ...valid, prevProjMatrix: new Float32Array(16).fill(Number.POSITIVE_INFINITY) as never }, /prevProjMatrix\[0\].*finite/],
      [{ ...valid, cameraPosition: [0, Number.NaN, 2] }, /cameraPosition\[1\].*finite/],
      [{ ...valid, quality: null } as never, /quality must be a non-array object/],
      [{ ...valid, quality: [] } as never, /quality must be a non-array object/],
      [{ ...valid, quality: { samplesTargte: 1 } } as never, /unsupported field "samplesTargte"/],
      [{ ...valid, quality: symbolQuality }, /unsupported symbol keys/],
      [{ ...valid, quality: { samplesTarget: 0 } }, /quality.samplesTarget.*positive safe integer/],
      [{ ...valid, quality: { bounces: 1.5 } }, /quality.bounces.*positive safe integer/],
      [{ ...valid, quality: { resolutionFactor: 0 } }, /quality.resolutionFactor.*\(0, 1\]/],
      [{ ...valid, quality: { resolutionFactor: 1.01 } }, /quality.resolutionFactor.*\(0, 1\]/],
      [{ ...valid, quality: { filteredGlossyFactor: -0.01 } }, /quality.filteredGlossyFactor.*\[0, 1\]/],
      [{ ...valid, quality: { filteredGlossyFactor: 1.01 } }, /quality.filteredGlossyFactor.*\[0, 1\]/],
      [{ ...valid, quality: { filteredGlossyFactor: Number.MIN_VALUE } }, /quality.filteredGlossyFactor underflows WebGL float32 storage/],
      [{ ...valid, quality: { exposure: -1 } }, /quality.exposure.*>= 0/],
      [{ ...valid, quality: { exposure: Number.NaN } }, /quality.exposure.*finite/],
      [{ ...valid, quality: { exposure: 1e300 } }, /quality.exposure overflows WebGL float32 storage/],
      [{ ...valid, quality: { exposure: Number.MIN_VALUE } }, /quality.exposure underflows WebGL float32 storage/],
      [{ ...valid, quality: { tonemap: 'filmic' } } as never, /quality.tonemap is unsupported/],
      [{ ...valid, quality: { outputColorSpace: 'display-p3' } } as never, /quality.outputColorSpace is unsupported/],
    ];
    for (const [input, pattern] of cases) {
      expect(() => engine.renderFrame(input)).toThrow(pattern);
    }
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('preflights finite singular matrices before programs, targets, or retained accumulation mutate', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(scene());
    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    const singularProjection = new Float32Array(16);
    const invalidColdFrame: FrameInput = {
      ...frame(),
      projMatrix: singularProjection as never,
      viewport: { width: 32, height: 24, devicePixelRatio: 1 },
      quality: { samplesTarget: 3 },
    };
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();

    expect(() => engine.renderFrame(invalidColdFrame)).toThrow(
      /singular view\/projection matrix/,
    );
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    expect(await engine.captureFrame?.()).toBeNull();

    const validFrame = {
      ...frame(),
      quality: { samplesTarget: 3 },
    };
    expect(engine.renderFrame(validFrame)).toMatchObject({
      kind: 'rendered',
      samplesAccumulated: 1,
    });
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();

    expect(() => engine.renderFrame({
      ...invalidColdFrame,
      frameIndex: 1,
    })).toThrow(/singular view\/projection matrix/);
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    expect(await engine.captureFrame?.()).toMatchObject({
      width: 64,
      height: 64,
    });
    expect(engine.renderFrame({
      ...validFrame,
      frameIndex: 1,
    })).toMatchObject({
      kind: 'rendered',
      samplesAccumulated: 2,
    });
    engine.dispose();
  });

  it('rejects inverse-view camera separation before program or target work', async () => {
    const farScene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'far-triangle',
        positions: new Float32Array([
          -3e38, -1, -1,
          -3e38, 1, -1,
          -3e38, 0, 1,
        ]),
        normals: new Float32Array([
          1, 0, 0,
          1, 0, 0,
          1, 0, 0,
        ]),
        indices: new Uint32Array([0, 1, 2]),
        material: { baseColor: [0.8, 0.8, 0.8], roughness: 1, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(farScene);
    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();
    const positiveEye = Math.fround(3e38);

    expect(() => engine.renderFrame({
      ...frame(),
      viewMatrix: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -positiveEye, 0, 0, 1,
      ])),
      cameraPosition: [positiveEye, 0, 0],
    })).toThrow(
      /world-space camera origin|camera-to-transport separation|camera-to-transport distance/,
    );
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('setSize requires positive safe-integer physical dimensions', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const setSize = engine.setSize?.bind(engine);
    expect(setSize).toBeDefined();
    for (const [width, height] of [
      [0, 1],
      [1, -1],
      [Number.NaN, 1],
      [1.25, 1],
      [Number.MAX_SAFE_INTEGER + 1, 1],
    ] as const) {
      expect(() => setSize?.(width, height)).toThrow(/positive safe integer/);
    }
    engine.dispose();
  });

  it('setSize allocates the requested render targets before the first frame', async () => {
    const gl = createMockGl();
    const { createTexture } = installAllocationSpies(gl);
    const engine = await createPTEngine_WebGL2({ device: gl });
    const texturesBeforeSize = createTexture.mock.calls.length;

    engine.setSize?.(48, 40);

    expect(createTexture.mock.calls.length).toBeGreaterThan(texturesBeforeSize);
    expect(await engine.captureFrame?.()).toMatchObject({
      width: 48,
      height: 40,
    });
    engine.dispose();
  });

  it('honors every frame viewport after an eager setSize call', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());

    expect(engine.renderFrame(frame()).kind).toBe('rendered');
    expect(await engine.captureFrame?.()).toMatchObject({ width: 64, height: 64 });

    engine.setSize?.(48, 40);
    expect(await engine.captureFrame?.()).toMatchObject({ width: 48, height: 40 });

    const firstFrameResize = {
      ...frame(),
      viewport: { width: 32, height: 24, devicePixelRatio: 1 },
      frameIndex: 1,
    };
    expect(engine.renderFrame(firstFrameResize)).toMatchObject({
      kind: 'rendered',
      samplesAccumulated: 1,
    });
    expect(await engine.captureFrame?.()).toMatchObject({ width: 32, height: 24 });

    const secondFrameResize = {
      ...firstFrameResize,
      viewport: { width: 20, height: 12, devicePixelRatio: 1 },
      frameIndex: 2,
    };
    expect(engine.renderFrame(secondFrameResize)).toMatchObject({
      kind: 'rendered',
      samplesAccumulated: 1,
    });
    expect(await engine.captureFrame?.()).toMatchObject({ width: 20, height: 12 });
    engine.dispose();
  });

  it('rejects malformed constructor payloads before any capability probe', async () => {
    await expect(createPTEngine_WebGL2(null as never)).rejects.toThrow(/options must be a non-array object/);
    await expect(createPTEngine_WebGL2([] as never)).rejects.toThrow(/options must be a non-array object/);

    const cases: ReadonlyArray<readonly [Record<PropertyKey, unknown>, RegExp]> = [
      [{ maxBouncs: 4 }, /unsupported field "maxBouncs"/],
      [{ [Symbol('option')]: true }, /unsupported symbol keys/],
      [{ onWarning: 1 }, /onWarning must be a function/],
      [{ extensions: [] }, /extensions must be a non-array object/],
      [{ spectral: 1 }, /spectral must be a boolean/],
      [{ bdpt: 'yes' }, /bdpt must be a boolean/],
      [{ debug: 'yes' }, /debug must be a boolean/],
      [{ traceTier: 'maximum' }, /traceTier must be one of/],
      [{ sampling: 'stratified' }, /sampling must be one of/],
      [{ cameraType: 'fisheye' }, /cameraType must be one of/],
      [{ denoiser: 'svgf-real' }, /denoiser must be one of/],
      [{ maxSamplesPerPixel: 1.5 }, /maxSamplesPerPixel must be a positive safe integer/],
      [{ materialLodDepth: 1.5 }, /materialLodDepth must be an integer/],
      [{ backgroundAlpha: -0.1 }, /backgroundAlpha must be in \[0, 1\]/],
      [{ backgroundAlpha: Number.NaN }, /backgroundAlpha must be finite/],
      [{ backgroundAlpha: Number.MIN_VALUE }, /backgroundAlpha underflows WebGL float32 storage/],
      [{ backgroundBlur: Number.POSITIVE_INFINITY }, /backgroundBlur must be finite/],
      [{ backgroundBlur: 1e300 }, /backgroundBlur overflows WebGL float32 storage/],
      [{ backgroundBlur: Number.MIN_VALUE }, /backgroundBlur underflows WebGL float32 storage/],
      [{ bdptOptions: [] }, /bdptOptions must be a non-array object/],
      [{ bdptOptions: { maxLightBounce: 2 } }, /bdptOptions contains unsupported field/],
      [{ dof: { focusDistance: 1, bokehSize: 1, typo: true } }, /dof contains unsupported field/],
      [{ dof: { focusDistance: 0, bokehSize: 1 } }, /dof.focusDistance must be > 0/],
      [{ dof: { focusDistance: 1e300, bokehSize: 1 } }, /dof.focusDistance overflows WebGL float32 storage/],
      [{ dof: { focusDistance: Number.MIN_VALUE, bokehSize: 1 } }, /dof.focusDistance underflows WebGL float32 storage/],
      [{ dof: { focusDistance: 1, bokehSize: -1 } }, /dof.bokehSize must be >= 0/],
      [{ dof: { focusDistance: 1, bokehSize: 1e300 } }, /dof.bokehSize overflows WebGL float32 storage/],
      [{ dof: { focusDistance: 1, bokehSize: Number.MIN_VALUE } }, /dof.bokehSize underflows WebGL float32 storage/],
      [{ dof: { focusDistance: 1, bokehSize: 1, apertureBlades: 2 } }, /dof.apertureBlades/],
      [{ dof: { focusDistance: 1, bokehSize: 1, apertureRotation: 1e300 } }, /dof.apertureRotation overflows WebGL float32 storage/],
      [{ dof: { focusDistance: 1, bokehSize: 1, anamorphicRatio: 0 } }, /dof.anamorphicRatio must be > 0/],
      [{ dof: { focusDistance: 1, bokehSize: 1, anamorphicRatio: 1e300 } }, /dof.anamorphicRatio overflows WebGL float32 storage/],
      [{ dof: { focusDistance: 1, bokehSize: 1, anamorphicRatio: Number.MIN_VALUE } }, /dof.anamorphicRatio underflows WebGL float32 storage/],
      [{ cameraType: 'equirectangular', dof: { focusDistance: 1, bokehSize: 1 } }, /active dof is unsupported.*equirectangular/],
      [{ oidn: { modelUrl: '   ' } }, /oidn.modelUrl must be a non-empty string/],
      [{ oidn: { modelUrl: '/model.onnx', executionProviders: [] } }, /executionProviders must be a non-empty array/],
      [{ oidn: { modelUrl: '/model.onnx', executionProviders: ['cpu'] } }, /executionProviders\[0\] is unsupported/],
      [{ oidn: { modelUrl: '/model.onnx', typo: true } }, /oidn contains unsupported field/],
      [{ oidnBridgeLoader: true }, /oidnBridgeLoader must be a function/],
      [{ oidnReadbackFn: true }, /oidnReadbackFn must be a function/],
      [{ denoiser: 'oidn-final' }, /oidn: \{ modelUrl \}/],
    ];

    for (const [overrides, pattern] of cases) {
      const gl = createMockGl();
      const getExtension = vi.fn(gl.getExtension.bind(gl));
      Object.assign(gl, { getExtension });
      await expect(createPTEngine_WebGL2({ device: gl, ...overrides })).rejects.toThrow(pattern);
      expect(getExtension).not.toHaveBeenCalled();
    }
  });

  it('accepts active DOF at both positive-float32 anamorphic extremes', async () => {
    for (const anamorphicRatio of [
      Math.fround(2 ** -149),
      Math.fround(3.4028234663852886e38),
    ]) {
      const engine = await createPTEngine_WebGL2({
        device: createMockGl(),
        dof: {
          focusDistance: 1,
          bokehSize: 1,
          anamorphicRatio,
        },
      });
      engine.dispose();
    }
  });

  it('accepts a zero-aperture DOF object for an equirectangular camera', async () => {
    const engine = await createPTEngine_WebGL2({
      device: createMockGl(),
      cameraType: 'equirectangular',
      dof: {
        focusDistance: 1,
        bokehSize: 0,
      },
    });
    engine.dispose();
  });

  it('rejects accessor-backed constructor options without invoking getters', async () => {
    const gl = createMockGl();
    const getExtension = vi.fn(gl.getExtension.bind(gl));
    Object.assign(gl, { getExtension });
    const spectralGetter = vi.fn(() => true);
    const options = { device: gl } as Record<string, unknown>;
    Object.defineProperty(options, 'spectral', {
      enumerable: true,
      get: spectralGetter,
    });

    await expect(
      createPTEngine_WebGL2(options as never),
    ).rejects.toThrow(/spectral must be an enumerable own data property/);
    expect(spectralGetter).not.toHaveBeenCalled();
    expect(getExtension).not.toHaveBeenCalled();
  });
});

describe('pt-webgl2 render-target budget', () => {
  it('estimates the exact compact 40/64-byte persistent layouts and tiled scratch', () => {
    expect(estimateWebGl2RenderTargetBytes(1920, 1080, false)).toBe(
      1920 * 1080 * BASE_RENDER_TARGET_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2RenderTargetBytes(1920, 1080, true)).toBe(
      1920 * 1080 * AUX_RENDER_TARGET_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2RenderTargetBytes(3840, 2160, true)).toBe(530_841_600);
    expect(estimateWebGl2AllocationBytes(1920, 1080, true)).toBe(
      1920 * 1080 * AUX_ALLOCATION_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2ResidentBytes(1920, 1080, true, {
      candidateRows: 1080,
      denoised: true,
    })).toBe(
      1920 * 1080 *
        (
          AUX_RENDER_TARGET_BYTES_PER_PIXEL +
          NEE_CANDIDATE_BYTES_PER_PIXEL +
          DENOISED_RENDER_TARGET_BYTES_PER_PIXEL
        ),
    );
    expect(estimateWebGl2NeeCandidateBytes(3840, 24)).toBe(5_898_240);
    expect(() =>
      estimateWebGl2RenderTargetBytes(Number.MAX_SAFE_INTEGER, 2, true),
    ).toThrow(/overflows Number.MAX_SAFE_INTEGER/);
  });

  it('rejects resize replacement peak before allocation and preserves the old frame', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({
      device: gl,
      maxRenderTargetBytes: 500_000,
    });
    engine.setScene(scene());
    const first = engine.renderFrame(frame());
    expect(first.kind).toBe('rendered');
    const previous = await engine.captureFrame?.();
    expect(previous).toMatchObject({ width: 64, height: 64 });

    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();
    expect(() => engine.setSize?.(64, 63)).toThrow(/replacement peak/);
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    const retained = await engine.captureFrame?.();
    expect(retained).toMatchObject({ width: 64, height: 64 });
    engine.dispose();
  });

  it('rejects an over-budget frame before targets or programs are allocated', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({
      device: gl,
      maxRenderTargetBytes:
        64 * 64 * BASE_RENDER_TARGET_BYTES_PER_PIXEL +
        64 * NEE_CANDIDATE_BYTES_PER_PIXEL -
        1,
    });
    engine.setScene(scene());
    const { createTexture, createFramebuffer, createShader } = installAllocationSpies(gl);
    createTexture.mockClear();
    createFramebuffer.mockClear();
    createShader.mockClear();

    expect(() => engine.renderFrame(frame())).toThrow(/exceeding maxRenderTargetBytes/);
    expect(createTexture).not.toHaveBeenCalled();
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createShader).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('fits default-budget 4K full tier with a 24-row candidate tile', async () => {
    expect(3840 * 2160 * AUX_RENDER_TARGET_BYTES_PER_PIXEL).toBeLessThan(
      DEFAULT_RENDER_TARGET_BUDGET_BYTES,
    );
    expect(
      selectWebGl2NeeCandidateRows(
        3840,
        2160,
        true,
        DEFAULT_RENDER_TARGET_BUDGET_BYTES,
      ),
    ).toBe(24);
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    expect(() => engine.setSize?.(3840, 2160)).not.toThrow();
    engine.dispose();
  });

  it('executes 4K NEE in the selected row tiles with full-frame logical origins', async () => {
    const record = new Map<string, unknown>();
    const gl = createMockGl(record);
    const drawArrays = vi.fn();
    const originalUniform2f = gl.uniform2f.bind(gl);
    const uniform2f = vi.fn((
      location: WebGLUniformLocation | null,
      x: number,
      y: number,
    ) => originalUniform2f(location, x, y));
    Object.assign(gl, { drawArrays, uniform2f });
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(scene());

    const rendered = engine.renderFrame({
      ...frame(),
      viewport: { width: 3840, height: 2160, devicePixelRatio: 1 },
    });
    expect(rendered.kind).toBe('rendered');
    // main + (candidate + resolve) × 90 tiles + present
    expect(drawArrays).toHaveBeenCalledTimes(182);
    const tileOrigins = uniform2f.mock.calls
      .filter(([location]) =>
        location != null &&
        typeof location === 'object' &&
        '__u' in location &&
        (location as unknown as { __u: string }).__u === 'uTileOrigin')
      .map(([, x, y]) => [x, y]);
    expect(tileOrigins.filter(([, y]) => y === 2136)).toEqual([
      [0, 2136],
      [0, 2136],
    ]);
    engine.dispose();
  });

  it('rejects invalid explicit render-target budgets', async () => {
    const gl = createMockGl();
    await expect(createPTEngine_WebGL2({ device: gl, maxRenderTargetBytes: 0 })).rejects.toThrow(
      /maxRenderTargetBytes must be a positive safe integer/,
    );
  });
});
