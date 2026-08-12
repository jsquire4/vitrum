// H54 — VitrumCanvas React component mount + lifecycle tests.
//
// Verifies:
//   (a) A <canvas> element is rendered into the DOM when VitrumCanvas mounts.
//   (b) The attach path (attachVitrum) is invoked with the supplied scene + camera.
//   (c) Unmounting the component disposes the engine handle.
//
// Uses happy-dom (manually, mirroring attachVitrumLoop.test.ts) to provide a
// DOM environment, and React 18's createRoot to exercise the real component
// lifecycle (including Strict-Mode re-mount teardown / re-attach).
//
// esbuild jsx transform: vitest.config.ts adds `esbuild: { jsx: 'automatic' }`
// so .tsx files use the react-jsx runtime without further per-file config.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Window } from 'happy-dom';
import type { Scene, ScenePrimitive, ScenePrimitivePatch } from '@vitrum/core';
import type { GltfAssetResult, GltfForEngineResult, GltfJson } from '@vitrum/gltf-adapter';
import type { EngineWithBackendId } from '../src/createEngine.js';
import type { GltfProgressiveEngineResult } from '../src/gltf.js';
import type { CameraLike } from '../src/lifecycle/vanilla.js';
import type { AttachVitrumHandle } from '../src/lifecycle/vanilla.js';
import {
  attachGltfResourceOwner,
  DecodedImageHandleOwner,
} from '../../gltf-adapter/src/importResourceBudget.js';

// ── DOM setup / teardown ─────────────────────────────────────────────────────
// Mirror the pattern from attachVitrumLoop.test.ts: inject happy-dom globals
// so React (createRoot) and VitrumCanvas (which reads document / window) see a
// real-enough DOM. We do NOT set the vitest `environment` directive because the
// non-tsx test files in this package run without a DOM.

let happyWindow: Window;
const savedGlobals: Record<string, unknown> = {};
const DOM_GLOBALS = [
  'window',
  'document',
  'navigator',
  'requestAnimationFrame',
  'cancelAnimationFrame',
  'ResizeObserver',
] as const;

beforeEach(() => {
  happyWindow = new Window({ url: 'http://localhost/' });
  savedGlobals.window = (globalThis as Record<string, unknown>).window;
  savedGlobals.document = (globalThis as Record<string, unknown>).document;
  savedGlobals.navigator = (globalThis as Record<string, unknown>).navigator;
  savedGlobals.requestAnimationFrame = (
    globalThis as Record<string, unknown>
  ).requestAnimationFrame;
  savedGlobals.cancelAnimationFrame = (globalThis as Record<string, unknown>).cancelAnimationFrame;
  savedGlobals.ResizeObserver = (globalThis as Record<string, unknown>).ResizeObserver;

  (globalThis as Record<string, unknown>).window = happyWindow;
  (globalThis as Record<string, unknown>).document = happyWindow.document;
  Object.defineProperty(globalThis, 'navigator', {
    value: happyWindow.navigator,
    configurable: true,
    writable: true,
  });
  (globalThis as Record<string, unknown>).requestAnimationFrame =
    happyWindow.requestAnimationFrame.bind(happyWindow);
  (globalThis as Record<string, unknown>).cancelAnimationFrame =
    happyWindow.cancelAnimationFrame.bind(happyWindow);
  (globalThis as Record<string, unknown>).ResizeObserver = happyWindow.ResizeObserver;
});

afterEach(() => {
  for (const key of DOM_GLOBALS) {
    if (key === 'navigator') {
      Object.defineProperty(globalThis, 'navigator', {
        value: savedGlobals.navigator,
        configurable: true,
        writable: true,
      });
    } else {
      (globalThis as Record<string, unknown>)[key] = savedGlobals[key];
    }
  }
  happyWindow.close();
  vi.restoreAllMocks();
});

// ── Shared scene / camera fixtures ───────────────────────────────────────────

const SCENE: Scene = {
  primitives: [],
  emitters: [],
  environment: { kind: 'none' as const },
};

// Minimal CameraLike that satisfies the VitrumCanvas prop type.
// VitrumCanvas passes this to attachVitrum which reads the matrix properties.
const CAMERA: CameraLike = {
  updateMatrixWorld: vi.fn(),
  matrixWorldInverse: {
    elements: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
  },
  projectionMatrix: {
    elements: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
  },
  position: { x: 0, y: 0, z: 0 },
};

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function makeInlineTriangleGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    },
    buffers: new Map([[0, positions]]),
  };
}

function makeMockEngine(): EngineWithBackendId {
  return {
    backendId: 'pt-webgl2',
    state: 'ready',
    capabilities: {
      supportsIncrementalScene: false,
      supportsAddRemovePrimitive: false,
      supportsAuxBuffers: false,
      accumulates: false,
      supportsProgressiveSeedSource: false,
      supportsAccumulatorSeed: false,
      maxSamplesPerPixel: 1,
      maxBounces: 1,
      supportedAnalyticShapes: new Set(),
      supportedEmitterKinds: new Set(),
      presentationMode: 'offscreen-texture',
      causticStrategy: 'none',
    },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => ({ kind: 'skipped', samplesAccumulated: 0, isConverged: false })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  } as unknown as EngineWithBackendId;
}

function makeMockGltfAsset(gltf: GltfJson, scene: Scene): GltfAssetResult {
  return {
    gltf,
    sceneIndex: 0,
    scene,
    warnings: [],
    diagnostics: [],
    animations: [],
    animationTargets: {},
    featureReport: {},
    backendCompatibility: [],
    recommendedBackend: { backend: 'pt-webgl2' },
    textureDecodeReport: {
      mapCount: 0,
      uniqueHandleCount: 0,
      rawImageCount: 0,
      opaqueHandleCount: 0,
      cpuReadableCount: 0,
      rawImageRefs: [],
      entries: [],
    },
  } as unknown as GltfAssetResult;
}

function attachMockImageOwner(
  asset: GltfAssetResult,
  handle: object,
): GltfAssetResult {
  const owner = new DecodedImageHandleOwner();
  owner.track(handle);
  return attachGltfResourceOwner(asset, owner);
}

function makeMockGltfForEngineResult(
  asset: GltfAssetResult,
  engine: EngineWithBackendId,
  controller: { attachEngine: ReturnType<typeof vi.fn>; warnings: readonly string[] },
): GltfForEngineResult<EngineWithBackendId> {
  return {
    asset,
    backend: engine.backendId === 'progressive' ? 'pt-webgpu' : engine.backendId,
    profileId: engine.backendProfileId
      ?? (engine.backendId === 'progressive' ? 'pt-webgpu' : engine.backendId),
    engine,
    controller: controller as unknown as GltfForEngineResult<EngineWithBackendId>['controller'],
    attached: true,
    textureDecodeReport: asset.textureDecodeReport,
    decodedTextureCount: 0,
    unchangedTextureCount: 0,
    textureDecodeDiagnostics: [],
    textureDecodeWarnings: [],
    warnings: [],
    diagnostics: asset.diagnostics,
  };
}

function makeMockProgressiveGltfResult(
  asset: GltfAssetResult,
  coordinator: Record<string, unknown>,
): GltfProgressiveEngineResult {
  const realtime = makeMockEngine();
  const converged = makeMockEngine();
  const progressiveCoordinator = { getScene: vi.fn(() => asset.scene), ...coordinator };

  return {
    asset,
    backend: 'pt-webgpu',
    profileId: 'pt-webgpu',
    engine: {
      coordinator: progressiveCoordinator,
      realtime: {
        ...realtime,
        backendId: undefined,
        capabilities: {
          ...realtime.capabilities,
          presentationMode: 'swapchain-required',
        },
      },
      converged: {
        ...converged,
        backendId: undefined,
        capabilities: {
          ...converged.capabilities,
          presentationMode: 'offscreen-texture',
        },
      },
      dispose: vi.fn(),
    } as unknown as GltfProgressiveEngineResult['engine'],
    controller: {
      attachEngine: vi.fn(),
      advance: vi.fn(),
      warnings: [],
    } as unknown as GltfProgressiveEngineResult['controller'],
    attached: true,
    textureDecodeReport: asset.textureDecodeReport,
    decodedTextureCount: 0,
    unchangedTextureCount: 0,
    textureDecodeDiagnostics: [],
    textureDecodeWarnings: [],
    warnings: [],
    diagnostics: asset.diagnostics,
  };
}

// ── Helper: build a mock AttachVitrumHandle ───────────────────────────────────

function makeMockHandle(disposeSpy = vi.fn()): AttachVitrumHandle {
  return {
    dispose: disposeSpy,
    // AttachVitrumHandle only requires dispose; other optional methods omitted.
  } as unknown as AttachVitrumHandle;
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('VitrumCanvas — mount / attach / dispose', () => {
  it('(a) renders a <canvas> element into the container', async () => {
    // Lazily import modules AFTER happy-dom globals are in place so React and
    // VitrumCanvas see the injected document/window.
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    // Mock attachVitrum so VitrumCanvas does not attempt a real GPU engine.
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);

    // Mount — React renders synchronously up to the first effect boundary.
    await new Promise<void>((resolve) => {
      root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA }));
      // happy-dom fires microtasks synchronously; await a tick so effects run.
      happyWindow.happyDOM.waitUntilComplete().then(resolve).catch(resolve);
    });

    // (a) A <canvas> must be present in the DOM.
    const canvas = container.querySelector('canvas');
    expect(canvas).not.toBeNull();

    root.unmount();
    // Suppress unused-variable warning.
    void attachSpy;
  });

  it('(b) attachVitrum is called with the correct scene and camera on mount', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA }));
    await happyWindow.happyDOM.waitUntilComplete();

    // (b) attachVitrum must have been invoked at least once.
    // React 18 Strict Mode fires effects twice in development; in test mode
    // (non-Strict) it fires exactly once. We assert at least one call.
    expect(attachSpy).toHaveBeenCalled();

    // The call must have been made with an options object containing the canvas,
    // the correct scene reference, and the correct camera reference.
    const [callArgs] = attachSpy.mock.calls;
    expect(callArgs).toBeDefined();
    const opts = callArgs![0];
    expect(opts.scene).toBe(SCENE);
    expect(opts.camera).toBe(CAMERA);
    expect(opts.canvas).toBeInstanceOf(Object); // HTMLCanvasElement (happy-dom)

    root.unmount();
  });

  it('(c) unmounting the component calls dispose on the engine handle', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const disposeSpy = vi.fn();
    const handle = makeMockHandle(disposeSpy);

    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(handle);

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA }));

    // Wait for the async attachVitrum promise to resolve so the handle is wired.
    await happyWindow.happyDOM.waitUntilComplete();

    // (c) Unmount — the component's useEffect cleanup must call handle.dispose().
    root.unmount();
    // Give the unmount effect a tick to propagate.
    await new Promise<void>((r) => setTimeout(r, 0));

    expect(disposeSpy).toHaveBeenCalled();
  });

  it('recreates the engine when advanced backend options change identity', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const firstDispose = vi.fn();
    const secondDispose = vi.fn();
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi
      .spyOn(vanillaModule, 'attachVitrum')
      .mockResolvedValueOnce(makeMockHandle(firstDispose))
      .mockResolvedValueOnce(makeMockHandle(secondDispose));

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    const advancedA = { denoiser: 'atrous-variance' as const };
    const advancedB = { denoiser: 'svgf-real' as const };

    root.render(
      React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA, advanced: advancedA }),
    );
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(attachSpy).toHaveBeenCalledTimes(1);
    expect(attachSpy.mock.calls[0]![0].advanced).toBe(advancedA);

    root.render(
      React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA, advanced: advancedB }),
    );
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(firstDispose).toHaveBeenCalled();
    expect(attachSpy).toHaveBeenCalledTimes(2);
    expect(attachSpy.mock.calls[1]![0].advanced).toBe(advancedB);

    root.unmount();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(secondDispose).toHaveBeenCalled();
  });

  it('loads the gltf prop through the engine bridge and forwards the prepared engine/controller to attachVitrum', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());
    const gltfModule = await import('../src/gltf.js');

    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const engine = makeMockEngine();
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const bridgeResult = makeMockGltfForEngineResult(asset, engine, controller);
    const loadSpy = vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(bridgeResult);
    const decodePixels = vi.fn();
    const advanced = { denoiser: 'atrous-variance' as const };
    const advancedByBackend = {
      'pt-webgl2': { maxBounces: 8 },
    };
    const onGltfLoaded = vi.fn();
    const onWarning = vi.fn((_warning: unknown) => {
      throw new Error('host warning callback failed');
    });
    const onAdapterProfile = vi.fn((_profile: unknown) => {
      throw new Error('host adapter-profile callback failed');
    });

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        gltf,
        gltfOptions: {
          buffers,
          compatibilityMode: 'reject-degraded',
          decodeTextures: true,
          decodePixels,
        },
        camera: CAMERA,
        prefer: 'quality',
        advanced,
        advancedBackend: 'pt-webgl2',
        advancedByBackend,
        experimentalPreset: 'spectral-bdpt',
        debug: true,
        gltfPlayback: { loop: false },
        onGltfLoaded,
        onWarning,
        onAdapterProfile,
      }),
    );
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await happyWindow.happyDOM.waitUntilComplete();

    expect(onGltfLoaded).toHaveBeenCalledTimes(1);
    expect(onGltfLoaded).toHaveBeenCalledWith(asset, bridgeResult);
    expect(loadSpy).toHaveBeenCalledWith(
      gltf,
      expect.objectContaining({
        buffers,
        compatibilityMode: 'reject-degraded',
        decodeTextures: true,
        decodePixels,
        attachScene: false,
        engineOptions: expect.objectContaining({
          prefer: 'quality',
          advanced,
          advancedBackend: 'pt-webgl2',
          advancedByBackend,
          experimentalPreset: 'spectral-bdpt',
          debug: true,
        }),
      }),
    );
    const bridgeOptions = loadSpy.mock.calls[0]![1]!;
    expect(bridgeOptions.engineOptions?.canvas).toBeInstanceOf(Object);
    const warning = {
      code: 'vitrum-canvas.test-warning',
      backend: 'createEngine',
      phase: 'construction',
      method: 'VitrumCanvas.test',
      message: 'test warning',
    };
    const profile = {
      hasWebGPU: true,
      hasWebGL2: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'full',
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu-full',
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      maxStorageBuffersPerStage: 64,
      maxStorageTexturesPerStage: 8,
      limits: {},
    };
    expect(() => bridgeOptions.engineOptions?.onWarning?.(warning)).not.toThrow();
    expect(() => bridgeOptions.engineOptions?.onAdapterProfile?.(profile as never)).not.toThrow();
    expect(attachSpy).toHaveBeenCalledTimes(1);
    const opts = attachSpy.mock.calls[0]![0];
    expect(opts.scene).toBe(importedScene);
    expect(opts.gltfAsset).toEqual({
      recommendedBackend: { backend: 'pt-webgl2' },
    });
    expect(opts.gltfAsset).not.toHaveProperty('gltf');
    expect(opts.gltfAsset).not.toHaveProperty('scene');
    expect(opts.gltfAsset).not.toHaveProperty('textureDecodeReport');
    expect(opts.engine).toBe(engine);
    expect(opts.sceneController).toBe(controller);
    expect(opts.sceneControllerPlayback).toEqual({ loop: false });
    expect(opts.gltfAsset?.recommendedBackend?.backend).toBe('pt-webgl2');
    expect(opts.prefer).toBe('quality');
    expect(opts.advanced).toBe(advanced);
    expect(opts.advancedBackend).toBe('pt-webgl2');
    expect(opts.advancedByBackend).toBe(advancedByBackend);
    expect(opts.experimentalPreset).toBe('spectral-bdpt');
    expect(opts.debug).toBe(true);
    expect(() => opts.onWarning?.(warning)).not.toThrow();
    expect(() => opts.onAdapterProfile?.(profile as never)).not.toThrow();
    expect(onWarning).toHaveBeenCalledTimes(2);
    expect(onAdapterProfile).toHaveBeenCalledTimes(2);

    root.unmount();
  });

  it('defaults glTF clip playback on and can omit the host camera', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const gltfModule = await import('../src/gltf.js');
    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const engine = makeMockEngine();
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());
    vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(
      makeMockGltfForEngineResult(asset, engine, controller),
    );

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { gltf, gltfOptions: { buffers } }));
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await happyWindow.happyDOM.waitUntilComplete();

    expect(attachSpy).toHaveBeenCalledTimes(1);
    const opts = attachSpy.mock.calls[0]![0];
    expect(opts.sceneController).toBe(controller);
    expect(opts.sceneControllerPlayback).toBe(true);
    expect(opts.camera).toBeUndefined();
    root.unmount();
  });

  it('keeps imported image handles alive through mount and releases them after attach disposal on unmount', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const gltfModule = await import('../src/gltf.js');
    const { gltf } = makeInlineTriangleGltf();
    const close = vi.fn();
    const attachDispose = vi.fn();
    const engine = makeMockEngine();
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = attachMockImageOwner(
      makeMockGltfAsset(gltf, SCENE),
      { close },
    );
    const result = makeMockGltfForEngineResult(asset, engine, controller);
    vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(result);
    vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(
      makeMockHandle(attachDispose),
    );

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { gltf, camera: CAMERA }));
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(close).not.toHaveBeenCalled();
    root.unmount();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(attachDispose).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(attachDispose.mock.invocationCallOrder[0])
      .toBeLessThan(close.mock.invocationCallOrder[0]!);
  });

  it('releases imported image handles when attach disposal throws during unmount', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const gltfModule = await import('../src/gltf.js');
    const { gltf } = makeInlineTriangleGltf();
    const close = vi.fn();
    const attachDispose = vi.fn((): void => {
      throw new Error('dispose failed');
    });
    const engine = makeMockEngine();
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = attachMockImageOwner(
      makeMockGltfAsset(gltf, SCENE),
      { close },
    );
    vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(
      makeMockGltfForEngineResult(asset, engine, controller),
    );
    vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(
      makeMockHandle(attachDispose),
    );

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { gltf, camera: CAMERA }));
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(close).not.toHaveBeenCalled();
    expect(() => root.unmount()).not.toThrow();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(attachDispose).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(attachDispose.mock.invocationCallOrder[0])
      .toBeLessThan(close.mock.invocationCallOrder[0]!);
  });

  it('defers imported-handle release until an in-flight attach resolves and disposes', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const gltfModule = await import('../src/gltf.js');
    const { gltf } = makeInlineTriangleGltf();
    const close = vi.fn();
    const attachDispose = vi.fn();
    const pendingAttach = deferred<AttachVitrumHandle>();
    const engine = makeMockEngine();
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = attachMockImageOwner(
      makeMockGltfAsset(gltf, SCENE),
      { close },
    );
    vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(
      makeMockGltfForEngineResult(asset, engine, controller),
    );
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum')
      .mockReturnValue(pendingAttach.promise);

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { gltf, camera: CAMERA }));
    await vi.waitFor(() => expect(attachSpy).toHaveBeenCalledOnce());

    root.unmount();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(close).not.toHaveBeenCalled();
    expect(attachDispose).not.toHaveBeenCalled();

    pendingAttach.resolve(makeMockHandle(attachDispose));
    await vi.waitFor(() => expect(attachDispose).toHaveBeenCalledOnce());
    expect(close).toHaveBeenCalledOnce();
    expect(attachDispose.mock.invocationCallOrder[0])
      .toBeLessThan(close.mock.invocationCallOrder[0]!);
  });

  it('serializes structural reattach generations so one canvas never has concurrent owners', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const firstPending = deferred<AttachVitrumHandle>();
    const firstDispose = vi.fn();
    const secondDispose = vi.fn();
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum')
      .mockReturnValueOnce(firstPending.promise)
      .mockResolvedValueOnce(makeMockHandle(secondDispose));
    const nextScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    };

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA }));
    await vi.waitFor(() => expect(attachSpy).toHaveBeenCalledOnce());

    root.render(React.createElement(VitrumCanvas, { scene: nextScene, camera: CAMERA }));
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(attachSpy).toHaveBeenCalledOnce();

    firstPending.resolve(makeMockHandle(firstDispose));
    await vi.waitFor(() => expect(attachSpy).toHaveBeenCalledTimes(2));
    expect(firstDispose).toHaveBeenCalledOnce();
    expect(firstDispose.mock.invocationCallOrder[0])
      .toBeLessThan(attachSpy.mock.invocationCallOrder[1]!);
    expect(attachSpy.mock.calls[1]![0].scene).toBe(nextScene);

    root.unmount();
    await vi.waitFor(() => expect(secondDispose).toHaveBeenCalledOnce());
  });

  it('disposes the pending glTF engine before releasing image handles when attach rejects', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const gltfModule = await import('../src/gltf.js');
    const { gltf } = makeInlineTriangleGltf();
    const close = vi.fn();
    const engine = makeMockEngine();
    const engineDispose = engine.dispose as unknown as ReturnType<typeof vi.fn>;
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = attachMockImageOwner(
      makeMockGltfAsset(gltf, SCENE),
      { close },
    );
    const result = makeMockGltfForEngineResult(asset, engine, controller);
    vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(result);
    const failure = new Error('attach failed');
    vi.spyOn(vanillaModule, 'attachVitrum').mockRejectedValue(failure);
    vi.spyOn(console, 'error').mockImplementation(() => {});

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );
    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, { gltf, camera: CAMERA }));
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(engineDispose).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(engineDispose.mock.invocationCallOrder[0])
      .toBeLessThan(close.mock.invocationCallOrder[0]!);

    root.unmount();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(close).toHaveBeenCalledOnce();
  });

  it('loads progressive glTF and adapts the coordinator to attachVitrum without double-driving the controller', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());
    const gltfModule = await import('../src/gltf.js');

    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const frameOutput = { kind: 'skipped', samplesAccumulated: 0, isConverged: false } as const;
    const coordinator = {
      getScene: vi.fn(() => importedScene),
      setScene: vi.fn(),
      updatePrimitive: vi.fn(),
      addPrimitive: vi.fn(),
      removePrimitive: vi.fn(),
      reset: vi.fn(),
      frame: vi.fn(() => ({ phase: 'realtime', output: frameOutput })),
    };
    const progressiveResult = makeMockProgressiveGltfResult(asset, coordinator);
    const loadProgressiveSpy = vi
      .spyOn(gltfModule, 'loadGltfWithProgressiveEngine')
      .mockResolvedValue(progressiveResult);
    const loadSingleSpy = vi.spyOn(gltfModule, 'loadGltfWithEngine');
    const onGltfLoaded = vi.fn();

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        gltf,
        gltfOptions: { buffers },
        gltfProgressive: true,
        gltfProgressiveOptions: {
          seedFromRealtime: false,
          stillFramesBeforeHandoff: 2,
        },
        gltfPlayback: { loop: false },
        camera: CAMERA,
        debug: true,
        onGltfLoaded,
      }),
    );
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await happyWindow.happyDOM.waitUntilComplete();

    expect(loadSingleSpy).not.toHaveBeenCalled();
    expect(loadProgressiveSpy).toHaveBeenCalledWith(
      gltf,
      expect.objectContaining({
        buffers,
        engineOptions: expect.objectContaining({
          seedFromRealtime: false,
          stillFramesBeforeHandoff: 2,
          controllerLoop: false,
          debug: true,
        }),
      }),
    );
    expect(onGltfLoaded).toHaveBeenCalledWith(asset, progressiveResult);
    expect(attachSpy).toHaveBeenCalledTimes(1);
    const opts = attachSpy.mock.calls[0]![0];
    expect(opts.scene).toBe(importedScene);
    expect(opts.gltfAsset).toEqual({
      recommendedBackend: { backend: 'pt-webgl2' },
    });
    expect(opts.gltfAsset).not.toHaveProperty('gltf');
    expect(opts.gltfAsset).not.toHaveProperty('scene');
    expect(opts.gltfAsset).not.toHaveProperty('textureDecodeReport');
    expect(opts.engine?.backendId).toBe('progressive');
    expect(opts.sceneController).toBeUndefined();
    expect(opts.sceneControllerPlayback).toBeUndefined();

    const engine = opts.engine!;
    expect(engine.getScene?.()).toBe(importedScene);
    expect(coordinator.getScene).toHaveBeenCalledTimes(2);

    const replacementScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    engine.setScene(replacementScene);
    expect(coordinator.setScene).toHaveBeenCalledWith(replacementScene);

    const primitivePatch: ScenePrimitivePatch = {
      material: { baseColor: [0.25, 0.5, 0.75], roughness: 1, metallic: 0 },
    };
    engine.updatePrimitive?.('gltf-prim-0', primitivePatch);
    expect(coordinator.updatePrimitive).toHaveBeenCalledWith('gltf-prim-0', primitivePatch);

    const addedPrimitive: ScenePrimitive = {
      kind: 'mesh',
      id: 'added',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    };
    engine.addPrimitive?.(addedPrimitive);
    expect(coordinator.addPrimitive).toHaveBeenCalledWith(addedPrimitive);

    engine.removePrimitive?.('added');
    expect(coordinator.removePrimitive).toHaveBeenCalledWith('added');

    const realtimeReset = progressiveResult.engine.realtime.reset as unknown as ReturnType<
      typeof vi.fn
    >;
    const convergedReset = progressiveResult.engine.converged.reset as unknown as ReturnType<
      typeof vi.fn
    >;
    engine.reset();
    expect(realtimeReset).toHaveBeenCalledTimes(1);
    expect(convergedReset).toHaveBeenCalledTimes(1);
    expect(coordinator.reset).toHaveBeenCalledTimes(1);

    const output = engine.renderFrame({} as Parameters<typeof engine.renderFrame>[0]);
    expect(output).toBe(frameOutput);
    expect(coordinator.frame).toHaveBeenCalledTimes(1);

    root.unmount();
  });

  it('passes a progressive recreate factory that rebuilds the coordinator with the current scene', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const gltfModule = await import('../src/gltf.js');
    const progressiveModule = await import('../src/createProgressiveEngine.js');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());
    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const currentScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const initialCoordinator = {
      getScene: vi.fn(() => importedScene),
      setScene: vi.fn(),
      updatePrimitive: vi.fn(),
      addPrimitive: vi.fn(),
      removePrimitive: vi.fn(),
      reset: vi.fn(),
      frame: vi.fn(() => ({
        phase: 'realtime',
        output: { kind: 'skipped', samplesAccumulated: 0, isConverged: false },
      })),
    };
    const recreatedCoordinator = {
      getScene: vi.fn(() => currentScene),
      setScene: vi.fn(),
      updatePrimitive: vi.fn(),
      addPrimitive: vi.fn(),
      removePrimitive: vi.fn(),
      reset: vi.fn(),
      frame: vi.fn(() => ({
        phase: 'realtime',
        output: { kind: 'skipped', samplesAccumulated: 0, isConverged: false },
      })),
    };
    const progressiveResult = makeMockProgressiveGltfResult(asset, initialCoordinator);
    const recreatedHandle = makeMockProgressiveGltfResult(asset, recreatedCoordinator).engine;
    vi.spyOn(gltfModule, 'loadGltfWithProgressiveEngine').mockResolvedValue(progressiveResult);
    const createProgressiveSpy = vi
      .spyOn(progressiveModule, 'createProgressiveEngine')
      .mockResolvedValue(recreatedHandle);

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        gltf,
        gltfOptions: { buffers },
        gltfProgressive: true,
        gltfProgressiveOptions: {
          seedFromRealtime: false,
          stillFramesBeforeHandoff: 2,
        },
        gltfPlayback: { loop: false },
        camera: CAMERA,
        debug: true,
      }),
    );
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await happyWindow.happyDOM.waitUntilComplete();

    const opts = attachSpy.mock.calls[0]![0];
    expect(typeof opts.recreateEngine).toBe('function');

    const recreatedEngine = await opts.recreateEngine!({
      scene: currentScene,
      previousBackendId: 'pt-webgpu',
      reason: 'device-lost',
      attempt: 1,
    });

    expect(createProgressiveSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        canvas: expect.any(Object),
        scene: currentScene,
        controller: progressiveResult.controller,
        seedFromRealtime: false,
        stillFramesBeforeHandoff: 2,
        controllerLoop: false,
        debug: true,
      }),
    );
    expect(progressiveResult.controller.attachEngine).toHaveBeenCalledWith(
      recreatedHandle.coordinator,
      { setScene: false },
    );
    expect(recreatedEngine.backendId).toBe('progressive');
    recreatedEngine.setScene(currentScene);
    expect(recreatedCoordinator.setScene).toHaveBeenCalledWith(currentScene);

    root.unmount();
  });

  it('disposes a loaded glTF engine when unmounted before attachVitrum takes ownership', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum').mockResolvedValue(makeMockHandle());
    const gltfModule = await import('../src/gltf.js');

    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const dispose = vi.fn();
    const engine = { ...makeMockEngine(), dispose } as EngineWithBackendId;
    const controller = { attachEngine: vi.fn(), advance: vi.fn(), warnings: [] };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const bridgeResult = makeMockGltfForEngineResult(asset, engine, controller);
    const loadDeferred = deferred<GltfForEngineResult<EngineWithBackendId>>();
    const loadSpy = vi
      .spyOn(gltfModule, 'loadGltfWithEngine')
      .mockReturnValue(loadDeferred.promise);

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        gltf,
        gltfOptions: { buffers },
        camera: CAMERA,
      }),
    );

    await vi.waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    root.unmount();
    loadDeferred.resolve(bridgeResult);

    await vi.waitFor(() => expect(dispose).toHaveBeenCalledTimes(1));
    expect(attachSpy).not.toHaveBeenCalled();
  });

  it('disposes a progressive glTF handle when attachVitrum rejects before ownership is established', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const attachError = new Error('attach failed after progressive load');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    vi.spyOn(vanillaModule, 'attachVitrum').mockRejectedValue(attachError);
    const gltfModule = await import('../src/gltf.js');
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const onAttachError = vi.fn();

    const { gltf, buffers } = makeInlineTriangleGltf();
    const importedScene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const coordinator = {
      setScene: vi.fn(),
      updatePrimitive: vi.fn(),
      addPrimitive: vi.fn(),
      removePrimitive: vi.fn(),
      reset: vi.fn(),
      frame: vi.fn(),
    };
    const progressiveResult = makeMockProgressiveGltfResult(asset, coordinator);
    const progressiveDispose = progressiveResult.engine.dispose as unknown as ReturnType<
      typeof vi.fn
    >;
    vi.spyOn(gltfModule, 'loadGltfWithProgressiveEngine').mockResolvedValue(progressiveResult);

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        gltf,
        gltfOptions: { buffers },
        gltfProgressive: true,
        camera: CAMERA,
        onAttachError,
      }),
    );

    await vi.waitFor(() => expect(onAttachError).toHaveBeenCalledWith(attachError));
    expect(progressiveDispose).toHaveBeenCalledTimes(1);
    expect(consoleError).toHaveBeenCalledWith('[VitrumCanvas] attachVitrum failed:', attachError);

    root.unmount();
  });

  it('aborts glTF loading on unmount when AbortSignal.any is unavailable', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const gltfModule = await import('../src/gltf.js');
    const { gltf, buffers } = makeInlineTriangleGltf();
    const externalAbort = new AbortController();
    const originalAny = Object.getOwnPropertyDescriptor(globalThis.AbortSignal, 'any');
    let capturedSignal: AbortSignal | undefined;

    Object.defineProperty(globalThis.AbortSignal, 'any', {
      value: undefined,
      configurable: true,
    });

    try {
      vi.spyOn(gltfModule, 'loadGltfWithEngine').mockImplementation((_input, options) => {
        capturedSignal = options?.signal;
        return new Promise(() => {}) as ReturnType<typeof gltfModule.loadGltfWithEngine>;
      });

      const container = happyWindow.document.createElement('div') as unknown as Element;
      happyWindow.document.body.appendChild(
        container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
      );

      const root = createRoot(container);
      root.render(
        React.createElement(VitrumCanvas, {
          gltf,
          gltfOptions: {
            buffers,
            signal: externalAbort.signal,
          },
          camera: CAMERA,
        }),
      );

      await vi.waitFor(() => expect(capturedSignal).toBeDefined());
      expect(capturedSignal?.aborted).toBe(false);

      root.unmount();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));

      expect(externalAbort.signal.aborted).toBe(false);
      expect(capturedSignal?.aborted).toBe(true);
    } finally {
      if (originalAny != null) {
        Object.defineProperty(globalThis.AbortSignal, 'any', originalAny);
      } else {
        Reflect.deleteProperty(globalThis.AbortSignal, 'any');
      }
    }
  });

  it('reports initial attach failures through guarded React callbacks', async () => {
    const { createRoot } = await import('react-dom/client');
    const React = await import('react');
    const { VitrumCanvas } = await import('../src/react/VitrumCanvas.js');

    const attachError = new Error('attach failed');
    const vanillaModule = await import('../src/lifecycle/vanilla.js');
    vi.spyOn(vanillaModule, 'attachVitrum').mockRejectedValue(attachError);
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const onAttachError = vi.fn(() => {
      throw new Error('host attach callback failed');
    });
    const onError = vi.fn(() => {
      throw new Error('host structured callback failed');
    });

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(
      container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0],
    );

    const root = createRoot(container);
    root.render(
      React.createElement(VitrumCanvas, {
        scene: SCENE,
        camera: CAMERA,
        onAttachError,
        onError,
      }),
    );

    await vi.waitFor(() => expect(onAttachError).toHaveBeenCalledWith(attachError));
    expect(onError).toHaveBeenCalledWith(attachError, {
      phase: 'attach:initial',
      recoverable: false,
    });
    expect(consoleError).toHaveBeenCalledWith('[VitrumCanvas] attachVitrum failed:', attachError);

    root.unmount();
  });
});
