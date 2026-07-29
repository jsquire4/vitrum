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
  const identity = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  return {
    viewMatrix: identity,
    projMatrix: identity,
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
      [{ ...valid, quality: { exposure: -1 } }, /quality.exposure.*>= 0/],
      [{ ...valid, quality: { exposure: Number.NaN } }, /quality.exposure.*finite/],
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
      [{ traceTier: 'maximum' }, /traceTier must be one of/],
      [{ sampling: 'stratified' }, /sampling must be one of/],
      [{ cameraType: 'fisheye' }, /cameraType must be one of/],
      [{ denoiser: 'svgf-real' }, /denoiser must be one of/],
      [{ maxSamplesPerPixel: 1.5 }, /maxSamplesPerPixel must be a positive safe integer/],
      [{ materialLodDepth: 1.5 }, /materialLodDepth must be an integer/],
      [{ backgroundAlpha: -0.1 }, /backgroundAlpha must be in \[0, 1\]/],
      [{ backgroundAlpha: Number.NaN }, /backgroundAlpha must be finite/],
      [{ backgroundBlur: Number.POSITIVE_INFINITY }, /backgroundBlur must be finite/],
      [{ bdptOptions: [] }, /bdptOptions must be a non-array object/],
      [{ bdptOptions: { maxLightBounce: 2 } }, /bdptOptions contains unsupported field/],
      [{ dof: { focusDistance: 1, bokehSize: 1, typo: true } }, /dof contains unsupported field/],
      [{ dof: { focusDistance: 0, bokehSize: 1 } }, /dof.focusDistance must be > 0/],
      [{ dof: { focusDistance: 1, bokehSize: -1 } }, /dof.bokehSize must be >= 0/],
      [{ dof: { focusDistance: 1, bokehSize: 1, apertureBlades: 2 } }, /dof.apertureBlades/],
      [{ dof: { focusDistance: 1, bokehSize: 1, anamorphicRatio: 0 } }, /dof.anamorphicRatio must be > 0/],
      [{ cameraType: 'equirectangular', dof: { focusDistance: 1, bokehSize: 1 } }, /dof is unsupported.*equirectangular/],
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
  it('estimates the exact 128/160-byte core target layouts with safe arithmetic', () => {
    expect(estimateWebGl2RenderTargetBytes(1920, 1080, false)).toBe(
      1920 * 1080 * BASE_RENDER_TARGET_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2RenderTargetBytes(1920, 1080, true)).toBe(
      1920 * 1080 * AUX_RENDER_TARGET_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2RenderTargetBytes(3840, 2160, true)).toBe(1_327_104_000);
    expect(estimateWebGl2AllocationBytes(1920, 1080, true)).toBe(
      1920 * 1080 * AUX_ALLOCATION_BYTES_PER_PIXEL,
    );
    expect(estimateWebGl2ResidentBytes(1920, 1080, true, {
      blend: true,
      denoised: true,
    })).toBe(
      1920 * 1080 * (AUX_RENDER_TARGET_BYTES_PER_PIXEL + DENOISED_RENDER_TARGET_BYTES_PER_PIXEL),
    );
    expect(() =>
      estimateWebGl2RenderTargetBytes(Number.MAX_SAFE_INTEGER, 2, true),
    ).toThrow(/overflows Number.MAX_SAFE_INTEGER/);
  });

  it('rejects resize replacement peak before allocation and preserves the old frame', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({
      device: gl,
      maxRenderTargetBytes: 700_000,
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
      maxRenderTargetBytes: 64 * 64 * BASE_RENDER_TARGET_BYTES_PER_PIXEL - 1,
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

  it('blocks default-budget 4K full-tier targets and invalid explicit budgets', async () => {
    expect(3840 * 2160 * AUX_RENDER_TARGET_BYTES_PER_PIXEL).toBeGreaterThan(
      DEFAULT_RENDER_TARGET_BUDGET_BYTES,
    );
    const gl = createMockGl();
    await expect(createPTEngine_WebGL2({ device: gl, maxRenderTargetBytes: 0 })).rejects.toThrow(
      /maxRenderTargetBytes must be a positive safe integer/,
    );
  });
});
