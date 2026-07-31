/**
 * pt-webgl2-direct — host acquires a WebGL2 context and drives createPTEngine_WebGL2.
 *
 * Demonstrates the backend-direct usage pattern: the host owns the
 * WebGL2RenderingContext lifecycle and passes it to the factory. The engine
 * allocates GL resources against it but never destroys the context.
 *
 * The factory signature:
 *   createPTEngine_WebGL2(opts: PTEngineWebGL2Options): Promise<Engine & PTEngineWebGL2Surface>
 *
 * Required opts: device (WebGL2RenderingContext).
 * Optional: maxBounces, maxSamplesPerPixel, spectral, bdpt, bdptOptions.
 *
 * Capture protocol: sets VITRUM_CAPTURE_READY after targetSpp samples.
 *
 * API sharp edges observed while writing this example:
 * - The engine renders to the canvas's WebGL2 context directly. The host must
 *   NOT call gl.clear() or bind framebuffers between renderFrame() calls —
 *   the engine owns the full GL state per frame.
 * - FrameInput.viewport must match the actual canvas pixel dimensions or
 *   rendering will be clipped / stretched. Use canvas.width / canvas.height
 *   (backing store size), not clientWidth / clientHeight (CSS size).
 * - There is no built-in ResizeObserver; the host must resize the canvas
 *   and call renderFrame with the updated viewport. The engine does not cache
 *   the previous viewport size.
 * - FrameInput.frameSeed is required for correct MC sampling. Omitting it
 *   produces correlated samples (the same sample per frame).
 * - FrameStats (frameTimeMs, spp) are available via engine.onFrame, NOT in
 *   the FrameOutput returned by renderFrame. renderFrame returns
 *   samplesAccumulated and isConverged only.
 */

import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl2';
import type { FrameStats, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  syncCanvasToDisplaySize,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp  = Number(params.get('vitrumSpp'))    || 128;
const maxBounces = Number(params.get('vitrumBounces')) || 8;
const spectral = params.get('vitrumSpectral') === '1';
const bdpt = params.get('vitrumBdpt') === '1';
const bdptMaxLightBouncesParam = params.get('vitrumBdptMaxLightBounces');
const bdptMaxLightBounces =
  bdptMaxLightBouncesParam == null ? undefined : Number(bdptMaxLightBouncesParam);
if (
  bdptMaxLightBounces !== undefined &&
  (!Number.isInteger(bdptMaxLightBounces) ||
    bdptMaxLightBounces < 1 ||
    bdptMaxLightBounces > 8)
) {
  throw new RangeError('vitrumBdptMaxLightBounces must be an integer from 1 through 8');
}
if (!bdpt && bdptMaxLightBounces !== undefined) {
  throw new RangeError('vitrumBdptMaxLightBounces requires vitrumBdpt=1');
}
const sampling = params.get('vitrumSampling') === 'sobol' ? 'sobol' : 'pcg';
const fidelityScenario = params.get('vitrumScenario') ?? 'cornell';
const captureMode = params.get('vitrumCaptureMode') === '1';
const traceProbeStage = Math.max(0, Math.min(33, Number(params.get('vitrumProbeStage')) || 0));
const frameSeedOffset = Number(params.get('vitrumFrameSeedOffset')) >>> 0;

// ── Canvas + WebGL2 context ───────────────────────────────────────────────────
const canvas   = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const sppLabel = document.getElementById('spp') as HTMLDivElement;
const initialViewport = syncCanvasToDisplaySize(canvas);

// HOST owns the WebGL2RenderingContext — pass it to the factory, never destroy it.
// The guard throws at module scope; the non-null assertion satisfies TypeScript
// for uses inside async function closures where control-flow narrowing lapse.
const glOrNull = canvas.getContext('webgl2');
if (glOrNull == null) {
  sppLabel.textContent = 'WebGL2 unavailable';
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR =
    'WebGL2 is not available in this browser';
  throw new Error('[pt-webgl2-direct] WebGL2 is not available in this browser.');
}
const gl: WebGL2RenderingContext = glOrNull;
const captureDiagnostics = {
  contextCreated: true,
  driverPreflight: null as null | {
    framebufferComplete: boolean;
    glError: number;
    checksum: number;
    floatFramebufferComplete: boolean;
    floatGlError: number;
    floatChecksum: number;
  },
  engineCreated: false,
  sceneUploaded: false,
  firstRenderedSpp: null as number | null,
  firstNonzeroSpp: null as number | null,
  finalSpp: 0,
  lastRenderKind: null as null | string,
  lastGlError: null as null | number,
  lastReadbackChecksum: null as null | number,
  captureFramebufferComplete: false,
  gpuFenceStatus: null as null | string,
  gpuFenceLatencyMs: null as null | number,
  readbackPhase: null as null | string,
  traceProbeStage,
  renderFrameLatencyMs: null as null | number,
};
(globalThis as Record<string, unknown>).VITRUM_CAPTURE_DIAGNOSTICS = captureDiagnostics;
if (captureMode) {
  console.debug('[pt-webgl2-direct] capture milestone: context-created');
  captureDiagnostics.driverPreflight = runDriverPreflight(gl);
  console.debug(
    `[pt-webgl2-direct] capture milestone: driver-preflight ${JSON.stringify(captureDiagnostics.driverPreflight)}`,
  );
}

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
const viewMatrix = asMat4(createAxisAlignedView(CORNELL_CAMERA_POSITION));
const projMatrix = asMat4(
  createPerspectiveProjection(initialViewport.width, initialViewport.height),
);

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createFidelityScene(fidelityScenario);

// ── Engine ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const engine = await createPTEngine_WebGL2({
    device: gl,
    maxBounces,
    maxSamplesPerPixel: targetSpp,
    spectral,
    bdpt,
    ...(bdpt && bdptMaxLightBounces !== undefined
      ? { bdptOptions: { maxLightBounces: bdptMaxLightBounces } }
      : {}),
    sampling,
    extensions: {
      'pt-webgl2.validationTraceProbeStage': traceProbeStage,
    },
  });
  captureDiagnostics.engineCreated = true;
  if (captureMode) console.debug('[pt-webgl2-direct] capture milestone: engine-created');

  try {
    await engine.setScene(scene);
    engine.setSize?.(initialViewport.width, initialViewport.height);
  } catch (error) {
    try { engine.dispose(); } catch { /* initialization rollback is best-effort */ }
    throw error;
  }
  captureDiagnostics.sceneUploaded = true;
  if (captureMode) console.debug('[pt-webgl2-direct] capture milestone: scene-uploaded');

  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_FRAME = async (
    colorSpace: 'linear' | 'output' = 'output',
  ) => {
    if (typeof engine.captureFrame !== 'function') return null;
    const frame = await engine.captureFrame({ colorSpace });
    if (frame == null) return null;
    return {
      width: frame.width,
      height: frame.height,
      rgba: Array.from(frame.rgba),
    };
  };
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_TELEMETRY = {
    backend: 'pt-webgl2',
    scenario: fidelityScenario,
    spectral,
    bdpt,
    bdptMaxLightBounces: bdpt ? (bdptMaxLightBounces ?? null) : null,
    sampling,
    targetSpp,
    maxBounces,
    frameSeedOffset,
  };

  let frameIndex       = 0;
  let captureSignalled = false;
  let disposed = false;
  let rafHandle = 0;
  let firstNonzeroSpp: number | null = null;

  const dispose = (): void => {
    if (disposed) return;
    disposed = true;
    if (rafHandle !== 0) cancelAnimationFrame(rafHandle);
    engine.dispose();
  };
  (globalThis as Record<string, unknown>).VITRUM_DISPOSE = dispose;

  const reportTickError = (error: unknown): void => {
    const message = String(error instanceof Error ? error.stack ?? error.message : error);
    console.error('[pt-webgl2-direct] render loop failed:', error);
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = message;
    dispose();
  };

  // FrameStats (frameTimeMs, spp) come from the engine.onFrame subscription,
  // not from the renderFrame() return value.
  engine.onFrame?.((stats: FrameStats) => {
    const spp = stats.spp ?? 0;
    sppLabel.textContent = `spp: ${spp}`;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
  });

  async function tick(): Promise<void> {
    if (disposed) return;
    // Sync backing store to CSS size (host responsibility in direct mode).
    const viewport = syncCanvasToDisplaySize(canvas);
    const { width, height, devicePixelRatio } = viewport;
    writePerspectiveProjection(projMatrix, width, height);
    if (viewport.resized) engine.setSize?.(width, height);

    if (captureMode) {
      console.debug(
        `[pt-webgl2-direct] renderFrame start frame=${frameIndex} glError=${gl.getError()}`,
      );
    }
    const renderFrameStarted = performance.now();
    const output = engine.renderFrame({
      viewMatrix,
      projMatrix,
      cameraPosition: CORNELL_CAMERA_POSITION,
      viewport: { width, height, devicePixelRatio },
      quality: { samplesTarget: targetSpp, bounces: maxBounces },
      frameIndex,
      frameSeed:
        (frameIndex * 1664525 + 1013904223 + frameSeedOffset) >>> 0,
    });
    captureDiagnostics.renderFrameLatencyMs = performance.now() - renderFrameStarted;
    captureDiagnostics.lastRenderKind = output.kind;
    captureDiagnostics.lastGlError = gl.getError();
    if (captureMode) {
      console.debug(
        `[pt-webgl2-direct] renderFrame return kind=${output.kind} ` +
        `samples=${output.samplesAccumulated} glError=${captureDiagnostics.lastGlError} ` +
        `cpuMs=${captureDiagnostics.renderFrameLatencyMs.toFixed(1)}`,
      );
    }

    if (output.kind === 'rendered') {
      frameIndex++;
      const spp = output.samplesAccumulated;
      captureDiagnostics.firstRenderedSpp ??= spp;
      captureDiagnostics.finalSpp = spp;

      // The proof harness must distinguish an engine that merely advances its
      // sample counter from one that produces finite radiance. In capture mode,
      // read the tiny proof target after each rendered sample and record the
      // first non-black accumulation frame. Normal example usage pays no such
      // readback cost.
      if (captureMode && firstNonzeroSpp == null) {
        const fenceStarted = performance.now();
        const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        captureDiagnostics.gpuFenceStatus = sync == null ? 'allocation-failed' : 'submitted';
        captureDiagnostics.readbackPhase = 'before-captureFrame';
        gl.flush();
        console.debug(
          `[pt-webgl2-direct] probe=${traceProbeStage} GPU fence submitted; readback starting`,
        );
        const frame = await engine.captureFrame?.({
          colorSpace: traceProbeStage === 8 ? 'output' : 'linear',
        });
        captureDiagnostics.readbackPhase = 'after-captureFrame';
        captureDiagnostics.gpuFenceLatencyMs = performance.now() - fenceStarted;
        if (sync != null) {
          captureDiagnostics.gpuFenceStatus =
            gl.getSyncParameter(sync, gl.SYNC_STATUS) === gl.SIGNALED
              ? 'signaled-after-readback'
              : 'unsignaled-after-readback';
          gl.deleteSync(sync);
        }
        console.debug(
          `[pt-webgl2-direct] probe=${traceProbeStage} readback returned after ` +
          `${captureDiagnostics.gpuFenceLatencyMs.toFixed(1)}ms`,
        );
        if (frame != null) {
          let luminanceSum = 0;
          let finite = true;
          let checksum = 0;
          for (let i = 0; i < frame.rgba.length; i += 4) {
            const r = frame.rgba[i] ?? Number.NaN;
            const g = frame.rgba[i + 1] ?? Number.NaN;
            const b = frame.rgba[i + 2] ?? Number.NaN;
            finite = finite && Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b);
            luminanceSum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
            checksum += (i + 1) * r + (i + 2) * g + (i + 3) * b;
          }
          captureDiagnostics.lastReadbackChecksum = checksum;
          // captureFrame now throws before readPixels unless the selected FBO
          // reports FRAMEBUFFER_COMPLETE; reaching this line proves that check.
          captureDiagnostics.captureFramebufferComplete = true;
          captureDiagnostics.lastGlError = gl.getError();
          if (finite && luminanceSum / Math.max(frame.rgba.length / 4, 1) > 1e-5) {
            firstNonzeroSpp = spp;
            captureDiagnostics.firstNonzeroSpp = spp;
          }
        }
      }

      if (!captureSignalled && spp >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    }

    // Keep the host loop alive after convergence: renderFrame's converged
    // fast-out is cheap, while a later resize still needs a new backing store,
    // projection, and accumulation reset.
    if (!disposed) {
      rafHandle = requestAnimationFrame(() => { void tick().catch(reportTickError); });
    }
  }

  rafHandle = requestAnimationFrame(() => { void tick().catch(reportTickError); });
}

function runDriverPreflight(context: WebGL2RenderingContext): {
  framebufferComplete: boolean;
  glError: number;
  checksum: number;
  floatFramebufferComplete: boolean;
  floatGlError: number;
  floatChecksum: number;
} {
  const compile = (type: number, source: string): WebGLShader => {
    const shader = context.createShader(type);
    if (shader == null) throw new Error('WebGL2 preflight could not allocate a shader');
    context.shaderSource(shader, source);
    context.compileShader(shader);
    if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) {
      throw new Error(`WebGL2 preflight shader compile failed: ${context.getShaderInfoLog(shader) ?? ''}`);
    }
    return shader;
  };
  const vertex = compile(context.VERTEX_SHADER, `#version 300 es
    void main() {
      vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
      gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }`);
  const fragment = compile(context.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    out vec4 color;
    void main() { color = vec4(0.25, 0.5, 0.75, 1.0); }`);
  const program = context.createProgram();
  if (program == null) throw new Error('WebGL2 preflight could not allocate a program');
  context.attachShader(program, vertex);
  context.attachShader(program, fragment);
  context.linkProgram(program);
  if (!context.getProgramParameter(program, context.LINK_STATUS)) {
    throw new Error(`WebGL2 preflight program link failed: ${context.getProgramInfoLog(program) ?? ''}`);
  }
  const texture = context.createTexture();
  const framebuffer = context.createFramebuffer();
  const vao = context.createVertexArray();
  if (texture == null || framebuffer == null || vao == null) {
    throw new Error('WebGL2 preflight resource allocation failed');
  }
  context.bindTexture(context.TEXTURE_2D, texture);
  context.texImage2D(
    context.TEXTURE_2D, 0, context.RGBA8, 1, 1, 0,
    context.RGBA, context.UNSIGNED_BYTE, null,
  );
  context.bindFramebuffer(context.FRAMEBUFFER, framebuffer);
  context.framebufferTexture2D(
    context.FRAMEBUFFER, context.COLOR_ATTACHMENT0,
    context.TEXTURE_2D, texture, 0,
  );
  const framebufferComplete =
    context.checkFramebufferStatus(context.FRAMEBUFFER) === context.FRAMEBUFFER_COMPLETE;
  context.bindVertexArray(vao);
  context.viewport(0, 0, 1, 1);
  context.useProgram(program);
  context.drawArrays(context.TRIANGLES, 0, 3);
  const pixel = new Uint8Array(4);
  context.readPixels(0, 0, 1, 1, context.RGBA, context.UNSIGNED_BYTE, pixel);
  context.finish();
  const glError = context.getError();
  const checksum = pixel[0]! + pixel[1]! * 257 + pixel[2]! * 65537 + pixel[3]! * 16777213;
  context.bindFramebuffer(context.FRAMEBUFFER, null);
  context.bindVertexArray(null);
  context.useProgram(null);
  context.deleteFramebuffer(framebuffer);
  context.deleteTexture(texture);
  context.deleteVertexArray(vao);
  context.deleteProgram(program);
  context.deleteShader(vertex);
  context.deleteShader(fragment);

  const floatTexture = context.createTexture();
  const floatFramebuffer = context.createFramebuffer();
  if (floatTexture == null || floatFramebuffer == null) {
    throw new Error('WebGL2 float preflight resource allocation failed');
  }
  context.getExtension('EXT_color_buffer_float');
  context.bindTexture(context.TEXTURE_2D, floatTexture);
  context.texImage2D(
    context.TEXTURE_2D, 0, context.RGBA32F, 1, 1, 0,
    context.RGBA, context.FLOAT, null,
  );
  context.bindFramebuffer(context.FRAMEBUFFER, floatFramebuffer);
  context.framebufferTexture2D(
    context.FRAMEBUFFER, context.COLOR_ATTACHMENT0,
    context.TEXTURE_2D, floatTexture, 0,
  );
  const floatFramebufferComplete =
    context.checkFramebufferStatus(context.FRAMEBUFFER) === context.FRAMEBUFFER_COMPLETE;
  context.clearColor(0.25, 0.5, 0.75, 1);
  context.clear(context.COLOR_BUFFER_BIT);
  const floatPixel = new Float32Array(4);
  context.readPixels(0, 0, 1, 1, context.RGBA, context.FLOAT, floatPixel);
  const floatGlError = context.getError();
  const floatChecksum =
    floatPixel[0]! + 3 * floatPixel[1]! + 5 * floatPixel[2]! + 7 * floatPixel[3]!;
  context.bindFramebuffer(context.FRAMEBUFFER, null);
  context.deleteFramebuffer(floatFramebuffer);
  context.deleteTexture(floatTexture);
  return {
    framebufferComplete,
    glError,
    checksum,
    floatFramebufferComplete,
    floatGlError,
    floatChecksum,
  };
}

function createFidelityScene(id: string): Scene {
  if (id === 'smoke-sky') {
    return {
      primitives: [],
      emitters: [],
      environment: {
        kind: 'procedural-sky',
        sunDirection: [0.22792115, 0.91168461, 0.34188173],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.005,
        mieDirectionalG: 0.8,
        intensity: 1,
      },
    };
  }
  if (id === 'smoke-triangle') {
    return {
      primitives: [{
        kind: 'mesh',
        id: 'smoke-triangle',
        positions: new Float32Array([-1, 0, 0, 1, 0, 0, 0, 2, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material: {
          baseColor: [0.7, 0.4, 0.2],
          roughness: 0.6,
          metallic: 0,
        },
      }],
      emitters: [{
        kind: 'point',
        id: 'smoke-point',
        position: [0, 2.5, 2],
        color: [1, 1, 1],
        intensity: 10,
        decay: 2,
      }],
      environment: {
        kind: 'procedural-sky',
        sunDirection: [0.22792115, 0.91168461, 0.34188173],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.005,
        mieDirectionalG: 0.8,
        intensity: 0.2,
      },
    };
  }
  if (id === 'bsdf-oracle') {
    return {
      primitives: [{
        kind: 'mesh',
        id: 'bsdf-oracle-screen',
        positions: new Float32Array([
          -20, -20, 0,
           20, -20, 0,
            0,  20, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material: {
          baseColor: [0.72, 0.41, 0.19],
          roughness: 0.45,
          metallic: 0.25,
          transmission: 0.5,
          ior: 1.52,
          clearcoat: 0.5,
          clearcoatRoughness: 0.3,
          sheen: 0.5,
          sheenColor: [0.8, 0.3, 0.12],
          sheenRoughness: 0.4,
          iridescence: 0.5,
          iridescenceIor: 1.4,
          iridescenceThicknessRange: [100, 700],
          thinFilmStack: {
            incidentIor: 1,
            angleDependent: true,
            layers: [
              { ior: 1.38, thicknessNm: 115 },
              { ior: 2.1, thicknessNm: 78 },
            ],
          },
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
  }
  const base = createCornellScene();
  const material = fidelityPanelMaterial(id);
  if (material == null) return base;
  return {
    ...base,
    primitives: [...base.primitives, fidelityPanel(material)],
  };
}

function fidelityPanelMaterial(id: string): MaterialSpec | null {
  const glass: MaterialSpec = {
    baseColor: [1, 1, 1],
    roughness: 0,
    metallic: 0,
    transmission: 1,
    ior: 1.52,
    attenuationColor: [0.72, 0.9, 0.58],
    attenuationDistance: 1,
    thickness: 0.35,
  };
  switch (id) {
    case 'mapped-pbr':
    case 'mapped-rich': {
      const baseColorHandle = {
        width: 2,
        height: 2,
        data: new Uint8Array([
          255, 48, 32, 255,
          32, 192, 255, 255,
          32, 192, 255, 255,
          255, 48, 32, 255,
        ]),
      };
      const ormHandle = {
        width: 2,
        height: 2,
        data: new Uint8Array([
          255, 48, 224, 255,
          255, 224, 32, 255,
          255, 224, 32, 255,
          255, 48, 224, 255,
        ]),
      };
      const normalHandle = {
        width: 2,
        height: 2,
        data: new Uint8Array([
          128, 128, 255, 255,
          160, 128, 250, 255,
          128, 160, 250, 255,
          128, 128, 255, 255,
        ]),
      };
      const mappedPbr: MaterialSpec = {
        baseColor: [0.9, 0.9, 0.9],
        roughness: 0.8,
        metallic: 0.7,
        baseColorMap: { handle: baseColorHandle },
        roughnessMap: { handle: ormHandle },
        metallicMap: { handle: ormHandle },
        normalMap: { handle: normalHandle },
        normalScale: 0.6,
      };
      if (id === 'mapped-pbr') return mappedPbr;
      return {
        ...mappedPbr,
        transmission: 0.72,
        transmissionMap: { handle: ormHandle },
        ior: 1.52,
        attenuationColor: [0.7, 0.88, 0.96],
        attenuationDistance: 1.4,
        thickness: 0.35,
        thicknessMap: { handle: ormHandle },
        bumpMap: { handle: ormHandle },
        bumpScale: 0.08,
        scatteringCoefficientRGB: [0.025, 0.015, 0.008],
        scatteringAnisotropy: 0.3,
        clearcoat: 0.5,
        clearcoatRoughness: 0.22,
        clearcoatMap: { handle: ormHandle },
        clearcoatRoughnessMap: { handle: ormHandle },
        clearcoatNormalMap: { handle: normalHandle },
        clearcoatNormalScale: 0.45,
        sheen: 0.22,
        sheenColor: [0.8, 0.3, 0.12],
        sheenRoughness: 0.4,
        sheenColorMap: { handle: baseColorHandle },
        sheenRoughnessMap: { handle: ormHandle },
        iridescence: 0.35,
        iridescenceIor: 1.4,
        iridescenceThicknessRange: [110, 620],
        iridescenceMap: { handle: ormHandle },
        iridescenceThicknessMap: { handle: ormHandle },
        specularIntensity: 0.85,
        specularColor: [1, 0.92, 0.84],
        specularColorMap: { handle: baseColorHandle },
        specularIntensityMap: { handle: ormHandle },
        anisotropy: 0.3,
        anisotropyRotation: 0.4,
        anisotropyMap: { handle: normalHandle },
        frontLayer: {
          transmission: [0.92, 0.82, 0.72],
          roughness: 0.18,
          normalMap: { handle: normalHandle },
          normalScale: 0.5,
        },
        backLayer: {
          transmission: [0.75, 0.85, 0.95],
          roughness: 0.28,
        },
        thinFilmStack: {
          incidentIor: 1,
          angleDependent: true,
          layers: [
            { ior: 1.38, thicknessNm: 115 },
            { ior: 2.1, extinctionCoefficient: 0.01, thicknessNm: 78 },
          ],
        },
      };
    }
    case 'beer-flat':
      return glass;
    case 'beer-curve':
      return {
        ...glass,
        spectralAttenuation: {
          wavelengthStart: 380,
          wavelengthEnd: 700,
          values: new Float32Array([0.05, 0.08, 0.18, 1.4, 1.25, 0.2, 0.08, 0.05]),
        },
      };
    case 'thinfilm-off':
      return glass;
    case 'thinfilm-on':
      return {
        ...glass,
        thinFilmStack: {
          incidentIor: 1,
          angleDependent: true,
          layers: [
            { ior: 1.38, thicknessNm: 115 },
            { ior: 2.1, thicknessNm: 78 },
          ],
        },
      };
    case 'dispersion-low':
      return { ...glass, dispersionAbbeNumber: 20 };
    case 'dispersion-high':
      return { ...glass, dispersionAbbeNumber: 90 };
    default:
      return null;
  }
}

function fidelityPanel(material: MaterialSpec): MeshPrimitive {
  type Point3 = readonly [number, number, number];
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const faceUv = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [1, 1],
    [0, 1],
  ] as const;
  const triangleOrder = [0, 1, 2, 0, 2, 3] as const;
  const addFace = (
    corners: readonly [Point3, Point3, Point3, Point3],
    normal: Point3,
  ): void => {
    const ab: Point3 = [
      corners[1][0] - corners[0][0],
      corners[1][1] - corners[0][1],
      corners[1][2] - corners[0][2],
    ];
    const ac: Point3 = [
      corners[2][0] - corners[0][0],
      corners[2][1] - corners[0][1],
      corners[2][2] - corners[0][2],
    ];
    const windingDotNormal =
      (ab[1] * ac[2] - ab[2] * ac[1]) * normal[0] +
      (ab[2] * ac[0] - ab[0] * ac[2]) * normal[1] +
      (ab[0] * ac[1] - ab[1] * ac[0]) * normal[2];
    if (!(windingDotNormal > 0)) {
      throw new Error('pt-webgl2 fidelity slab face winding must match its outward normal');
    }
    for (const [vertex, cornerIndex] of triangleOrder.entries()) {
      const point = corners[cornerIndex];
      positions.push(point[0], point[1], point[2]);
      normals.push(normal[0], normal[1], normal[2]);
      const uv = faceUv[vertex]!;
      uvs.push(uv[0], uv[1]);
    }
  };

  const x0 = -0.58;
  const x1 = 0.58;
  const y0 = 0.28;
  const y1 = 1.48;
  // Match the authored transport distance. The camera sees the +Z face;
  // transmitted paths leave through the -Z face, which exercises exit-boundary
  // Beer–Lambert attenuation as well as thin-film and dispersion.
  const slabThickness = material.thickness ?? 0.22;
  if (!(Number.isFinite(slabThickness) && slabThickness > 0)) {
    throw new Error('pt-webgl2 fidelity slab requires a positive finite thickness');
  }
  const zFront = slabThickness * 0.5;
  const zBack = -zFront;
  addFace(
    [[x0, y0, zFront], [x1, y0, zFront], [x1, y1, zFront], [x0, y1, zFront]],
    [0, 0, 1],
  );
  addFace(
    [[x0, y0, zBack], [x0, y1, zBack], [x1, y1, zBack], [x1, y0, zBack]],
    [0, 0, -1],
  );
  addFace(
    [[x0, y0, zBack], [x0, y0, zFront], [x0, y1, zFront], [x0, y1, zBack]],
    [-1, 0, 0],
  );
  addFace(
    [[x1, y0, zBack], [x1, y1, zBack], [x1, y1, zFront], [x1, y0, zFront]],
    [1, 0, 0],
  );
  addFace(
    [[x0, y0, zBack], [x1, y0, zBack], [x1, y0, zFront], [x0, y0, zFront]],
    [0, -1, 0],
  );
  addFace(
    [[x0, y1, zBack], [x0, y1, zFront], [x1, y1, zFront], [x1, y1, zBack]],
    [0, 1, 0],
  );

  return {
    kind: 'mesh',
    id: 'fidelity-panel',
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    material,
    transform: asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])),
  };
}

main().catch((err: unknown) => {
  console.error('[pt-webgl2-direct] fatal:', err);
  sppLabel.textContent = `error: ${String(err)}`;
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
    err instanceof Error ? err.stack ?? err.message : err,
  );
});
