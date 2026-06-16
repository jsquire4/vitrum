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
import type { Scene } from '@vitrum/core';
import type { GltfAssetResult, GltfForEngineResult, GltfJson } from '@vitrum/gltf-adapter';
import type { EngineWithBackendId } from '../src/createEngine.js';
import type { CameraLike } from '../src/lifecycle/vanilla.js';
import type { AttachVitrumHandle } from '../src/lifecycle/vanilla.js';

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
  savedGlobals.requestAnimationFrame = (globalThis as Record<string, unknown>).requestAnimationFrame;
  savedGlobals.cancelAnimationFrame = (globalThis as Record<string, unknown>).cancelAnimationFrame;
  savedGlobals.ResizeObserver = (globalThis as Record<string, unknown>).ResizeObserver;

  (globalThis as Record<string, unknown>).window = happyWindow;
  (globalThis as Record<string, unknown>).document = happyWindow.document;
  Object.defineProperty(globalThis, 'navigator', {
    value: happyWindow.navigator,
    configurable: true,
    writable: true,
  });
  (globalThis as Record<string, unknown>).requestAnimationFrame = happyWindow.requestAnimationFrame.bind(happyWindow);
  (globalThis as Record<string, unknown>).cancelAnimationFrame = happyWindow.cancelAnimationFrame.bind(happyWindow);
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
  matrixWorldInverse: { elements: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) },
  projectionMatrix: { elements: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) },
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
    capabilities: { presentationMode: 'offscreen-texture' },
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

function makeMockGltfForEngineResult(
  asset: GltfAssetResult,
  engine: EngineWithBackendId,
  controller: { attachEngine: ReturnType<typeof vi.fn>; warnings: readonly string[] },
): GltfForEngineResult<EngineWithBackendId> {
  return {
    asset,
    backend: engine.backendId,
    profileId: engine.backendId,
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

// ── Helper: build a mock AttachVitrumHandle ───────────────────────────────────

function makeMockHandle(disposeSpy = vi.fn()): AttachVitrumHandle {
  return {
    dispose: disposeSpy,
    // AttachVitrumHandle only requires dispose; other optional methods omitted.
  } as unknown as AttachVitrumHandle;
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
    happyWindow.document.body.appendChild(container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0]);

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
    happyWindow.document.body.appendChild(container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0]);

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
    happyWindow.document.body.appendChild(container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0]);

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
    const attachSpy = vi.spyOn(vanillaModule, 'attachVitrum')
      .mockResolvedValueOnce(makeMockHandle(firstDispose))
      .mockResolvedValueOnce(makeMockHandle(secondDispose));

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0]);

    const root = createRoot(container);
    const advancedA = { denoiser: 'atrous-variance' as const };
    const advancedB = { denoiser: 'svgf-real' as const };

    root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA, advanced: advancedA }));
    await happyWindow.happyDOM.waitUntilComplete();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(attachSpy).toHaveBeenCalledTimes(1);
    expect(attachSpy.mock.calls[0]![0].advanced).toBe(advancedA);

    root.render(React.createElement(VitrumCanvas, { scene: SCENE, camera: CAMERA, advanced: advancedB }));
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
    const controller = { attachEngine: vi.fn(), warnings: [] };
    const asset = makeMockGltfAsset(gltf, importedScene);
    const bridgeResult = makeMockGltfForEngineResult(asset, engine, controller);
    const loadSpy = vi.spyOn(gltfModule, 'loadGltfWithEngine').mockResolvedValue(bridgeResult);
    const decodePixels = vi.fn();
    const advanced = { denoiser: 'atrous-variance' as const };
    const onGltfLoaded = vi.fn();

    const container = happyWindow.document.createElement('div') as unknown as Element;
    happyWindow.document.body.appendChild(container as unknown as Parameters<typeof happyWindow.document.body.appendChild>[0]);

    const root = createRoot(container);
    root.render(React.createElement(VitrumCanvas, {
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
      debug: true,
      onGltfLoaded,
    }));
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
          debug: true,
        }),
      }),
    );
    const bridgeOptions = loadSpy.mock.calls[0]![1]!;
    expect(bridgeOptions.engineOptions?.canvas).toBeInstanceOf(Object);
    expect(attachSpy).toHaveBeenCalledTimes(1);
    const opts = attachSpy.mock.calls[0]![0];
    expect(opts.scene).toBe(importedScene);
    expect(opts.gltfAsset).toBe(asset);
    expect(opts.engine).toBe(engine);
    expect(opts.sceneController).toBe(controller);
    expect(opts.gltfAsset?.recommendedBackend?.backend).toBe('pt-webgl2');
    expect(opts.prefer).toBe('quality');
    expect(opts.advanced).toBe(advanced);
    expect(opts.debug).toBe(true);

    root.unmount();
  });
});
