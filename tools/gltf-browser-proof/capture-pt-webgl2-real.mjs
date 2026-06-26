#!/usr/bin/env node
// Captures real glTF assets through the browser pt-webgl2 one-call path.

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import { gltfBrowserProofStatusExitCode } from './status-exit-code.mjs';

const CHECKER_PATH = 'tools/gltf-browser-proof/check-status.mjs';
const CAPTURE_HARNESS_PATH = 'tools/gltf-browser-proof/capture-pt-webgl2-real.mjs';
const MANIFEST_PATH = 'tools/reference-renders/gltf-real-browser-pt-webgl2/manifest.json';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const exampleDir = resolve(repoRoot, 'examples/gltf-viewer');
const updateGolden = process.argv.includes('--update-golden') || process.argv.includes('--update-goldens');
const selectedAssetIds = readMultiFlag('--asset');
const width = Number(process.env.VITRUM_WIDTH ?? '64');
const height = Number(process.env.VITRUM_HEIGHT ?? '64');
const spp = Number(process.env.VITRUM_SPP ?? '1');
const port = Number(process.env.VITRUM_GLTF_BROWSER_PORT ?? '5187');
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? '120000');
const screenshotTimeoutMs = Number(process.env.VITRUM_SCREENSHOT_TIMEOUT_MS ?? '15000');
const dataUrlTimeoutMs = Number(process.env.VITRUM_DATA_URL_TIMEOUT_MS ?? '15000');
const statusPath = resolveStatusPath(process.env.VITRUM_GLTF_BROWSER_STATUS_PATH);
const browserExtraArgs = parseEnvArgs(process.env.VITRUM_CHROMIUM_EXTRA_ARGS);
const pauseBeforeEngineCapture = process.env.VITRUM_PAUSE_BEFORE_CAPTURE == null
  ? true
  : parseBooleanEnv(process.env.VITRUM_PAUSE_BEFORE_CAPTURE);
const engineCaptureMode = parseEngineCaptureMode(process.env.VITRUM_ENGINE_CAPTURE_MODE);
const DEFAULT_GOLDEN_THRESHOLDS = { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 };
// Prevent browser readback failures from becoming white/black "successful" goldens.
const structureThresholds = {
  minLumaRange: 12,
  minUniqueColorCount: 16,
  minNonDominantFraction: 0.05,
};
const REAL_BROWSER_ASSETS = [
  {
    assetId: 'box-textured-glb',
    kind: 'textured-glb',
    goldenPath: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png',
    minTextures: 1,
    requiredExtensions: [],
    requiredHooks: [],
    thresholds: DEFAULT_GOLDEN_THRESHOLDS,
  },
  {
    assetId: 'cesium-milk-truck-draco',
    kind: 'draco',
    goldenPath: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png',
    minTextures: 0,
    requiredExtensions: ['KHR_draco_mesh_compression'],
    requiredHooks: ['draco'],
    thresholds: DEFAULT_GOLDEN_THRESHOLDS,
  },
  {
    assetId: 'meshopt-cube-real',
    kind: 'meshopt',
    goldenPath: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png',
    minTextures: 0,
    requiredExtensions: ['KHR_meshopt_compression'],
    requiredHooks: ['meshopt'],
    thresholds: DEFAULT_GOLDEN_THRESHOLDS,
  },
];
let activeBrowser = null;
let captureStep = 'not-started';
let lastTelemetry = null;
let lastConsole = [];
let lastCaptureAttempts = [];
let lastPageDiagnostics = null;
let hostReadbackProbe = null;

const server = spawn(
  process.execPath,
  [
    resolve(repoRoot, 'node_modules/vite/bin/vite.js'),
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--strictPort',
  ],
  {
    cwd: exampleDir,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  },
);

let serverLog = '';
server.stdout.on('data', (chunk) => { serverLog += chunk.toString(); });
server.stderr.on('data', (chunk) => { serverLog += chunk.toString(); });

let finalExitCode = 0;
try {
  await waitForServer(port, timeoutMs);
  hostReadbackProbe = await preflightBrowserReadbackProbe();
  const assets = selectedAssets();
  const results = [];
  for (const asset of assets) {
    lastTelemetry = null;
    lastConsole = [];
    lastCaptureAttempts = [];
    lastPageDiagnostics = null;
    captureStep = 'not-started';
    try {
      results.push(await withTimeout(capture(asset), timeoutMs, 'browser capture timed out'));
    } catch (error) {
      await closeActiveBrowser();
      results.push(hostBlockedStatus(asset, error));
    }
  }
  const summary = summarize(results);
  await writeStatus(summary);
  finalExitCode = gltfBrowserProofStatusExitCode(summary.verdict);
} catch (error) {
  await closeActiveBrowser();
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'FAIL',
    backend: 'pt-webgl2',
    step: captureStep,
    error: String(error?.stack ?? error),
    reason: {
      code: 'gltf-browser-proof-setup-failed',
      message:
        'The browser real-glTF capture harness failed before a per-asset page proved capture readiness; this cannot count as HOST-BLOCKED proof.',
    },
    telemetry: lastTelemetry,
    console: lastConsole.slice(-80),
    hostReadbackProbe,
    serverLog: serverLog.slice(-4000),
  };
  await writeStatus(status);
  finalExitCode = gltfBrowserProofStatusExitCode(status.verdict);
} finally {
  await stopServer();
}
process.exit(finalExitCode);

function readMultiFlag(name) {
  const values = [];
  for (let i = 0; i < process.argv.length; i += 1) {
    const arg = process.argv[i];
    if (arg === name && process.argv[i + 1]) values.push(process.argv[i + 1]);
    if (arg?.startsWith(`${name}=`)) values.push(arg.slice(name.length + 1));
  }
  return values.flatMap((value) => value.split(',').map((item) => item.trim()).filter(Boolean));
}

function parseEnvArgs(rawValue) {
  if (rawValue == null || rawValue.trim().length === 0) return [];
  return rawValue.split(/[\s,]+/u).map((arg) => arg.trim()).filter(Boolean);
}

function parseBooleanEnv(rawValue) {
  if (rawValue == null || rawValue.length === 0) return false;
  const normalized = rawValue.toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes';
}

function parseEngineCaptureMode(rawValue) {
  const normalized = String(rawValue ?? 'engine-first').trim().toLowerCase();
  if (normalized === 'first' || normalized === 'engine-first') return 'engine-first';
  if (normalized === 'fallback' || normalized === 'engine-fallback') return 'engine-fallback';
  if (normalized === 'canvas-only' || normalized === 'canvas-only-no-engine') return 'canvas-only';
  if (normalized === 'canvas-first' || normalized === 'browser-first') return 'canvas-first';
  if (normalized === 'off' || normalized === 'disabled' || normalized === 'none') return 'canvas-only';
  return 'canvas-first';
}

async function preflightBrowserReadbackProbe() {
  const startedAt = new Date().toISOString();
  const probeTimeout = Math.max(1000, Math.min(timeoutMs, 10000));
  let browser = null;
  try {
    captureStep = 'browser-readback-preflight';
    const { chromium } = await import('playwright');
    browser = await chromium.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        ...browserExtraArgs,
      ],
    });
    const browserVersion = typeof browser.version === 'function' ? browser.version() : null;
    activeBrowser = browser;
    const page = await browser.newPage({ viewport: { width: 16, height: 16 } });
    const result = await withTimeout(
      page.evaluate(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 2;
        canvas.height = 2;
        document.body.appendChild(canvas);
        const gl = canvas.getContext('webgl2', {
          alpha: false,
          depth: false,
          stencil: false,
          antialias: false,
          preserveDrawingBuffer: true,
        });
        if (gl == null) {
          return {
            webgl2: false,
            status: 'FAIL',
            reason: 'webgl2-context-unavailable',
          };
        }
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const renderer = debugInfo == null ? null : String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
        const vendor = debugInfo == null ? null : String(gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
        const extensions = gl.getSupportedExtensions() ?? [];
        gl.viewport(0, 0, 2, 2);
        gl.clearColor(0.25, 0.5, 0.75, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        const u8 = new Uint8Array(4);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, u8);
        const unsignedByteReadback = {
          status: gl.getError() === gl.NO_ERROR ? 'PASS' : 'FAIL',
          rgba: Array.from(u8),
        };
        let dataUrl = { status: 'FAIL', length: 0, prefix: '' };
        try {
          const url = canvas.toDataURL('image/png');
          dataUrl = {
            status: url.startsWith('data:image/png;base64,') ? 'PASS' : 'FAIL',
            length: url.length,
            prefix: url.slice(0, 22),
          };
        } catch (error) {
          dataUrl = { status: 'FAIL', length: 0, prefix: '', error: String(error?.message ?? error) };
        }
        let floatReadback = { status: 'SKIPPED', reason: 'EXT_color_buffer_float-unavailable' };
        if (gl.getExtension('EXT_color_buffer_float') != null) {
          const tex = gl.createTexture();
          const fbo = gl.createFramebuffer();
          if (tex != null && fbo != null) {
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            const framebufferStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (framebufferStatus === gl.FRAMEBUFFER_COMPLETE) {
              gl.clearColor(0.125, 0.25, 0.5, 1.0);
              gl.clear(gl.COLOR_BUFFER_BIT);
              const f32 = new Float32Array(4);
              gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, f32);
              floatReadback = {
                status: gl.getError() === gl.NO_ERROR ? 'PASS' : 'FAIL',
                rgba: Array.from(f32),
              };
            } else {
              floatReadback = {
                status: 'FAIL',
                reason: `framebuffer-incomplete:${framebufferStatus}`,
              };
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteFramebuffer(fbo);
            gl.deleteTexture(tex);
          } else {
            floatReadback = { status: 'FAIL', reason: 'texture-or-framebuffer-allocation-failed' };
          }
        }
        return {
          webgl2: true,
          status:
            unsignedByteReadback.status === 'PASS' &&
            dataUrl.status === 'PASS' &&
            (floatReadback.status === 'PASS' || floatReadback.status === 'SKIPPED')
              ? 'PASS'
              : 'FAIL',
          renderer,
          vendor,
          version: String(gl.getParameter(gl.VERSION)),
          shadingLanguageVersion: String(gl.getParameter(gl.SHADING_LANGUAGE_VERSION)),
          extensions,
          unsignedByteReadback,
          floatReadback,
          dataUrl,
        };
      }),
      probeTimeout,
      'browser readback preflight timed out',
    );
    return {
      generatedAt: startedAt,
      finishedAt: new Date().toISOString(),
      harness: 'gltf-browser-proof:host-readback-preflight',
      browserVersion,
      timeoutMs: probeTimeout,
      ...result,
    };
  } catch (error) {
    return {
      generatedAt: startedAt,
      finishedAt: new Date().toISOString(),
      harness: 'gltf-browser-proof:host-readback-preflight',
      status: 'FAIL',
      timeoutMs: probeTimeout,
      error: String(error?.stack ?? error),
    };
  } finally {
    if (browser != null) {
      await Promise.race([
        browser.close().catch(() => undefined),
        new Promise((resolvePromise) => setTimeout(resolvePromise, 2000)),
      ]);
      if (activeBrowser === browser) activeBrowser = null;
    }
  }
}

function selectedAssets() {
  if (selectedAssetIds.length === 0) return REAL_BROWSER_ASSETS;
  const selected = REAL_BROWSER_ASSETS.filter((asset) => selectedAssetIds.includes(asset.assetId));
  if (selected.length !== selectedAssetIds.length) {
    const known = new Set(REAL_BROWSER_ASSETS.map((asset) => asset.assetId));
    const missing = selectedAssetIds.filter((assetId) => !known.has(assetId));
    throw new Error(`unknown --asset id(s): ${missing.join(', ')}`);
  }
  return selected;
}

async function capture(asset) {
  captureStep = 'import-playwright';
  const { chromium } = await import('playwright');
  captureStep = 'launch-browser';
  const browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      ...browserExtraArgs,
    ],
  });
  activeBrowser = browser;
  try {
    captureStep = 'new-page';
    const page = await browser.newPage({ viewport: { width, height } });
    const consoleLines = [];
    page.on('console', (msg) => consoleLines.push(`${msg.type()}: ${msg.text()}`));
    page.on('pageerror', (err) => consoleLines.push(`pageerror: ${String(err)}`));
    const url = new URL(`http://127.0.0.1:${port}/`);
    url.searchParams.set('vitrumGltfAsset', asset.assetId);
    url.searchParams.set('vitrumBackend', 'pt-webgl2');
    url.searchParams.set('vitrumSpp', String(spp));

    captureStep = 'goto';
    await page.goto(url.toString(), { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    captureStep = 'wait-for-capture-ready';
    const ready = await page.waitForFunction(
      () => globalThis.VITRUM_CAPTURE_READY === true || globalThis.VITRUM_CAPTURE_ERROR != null,
      null,
      { timeout: timeoutMs },
    ).then(() => true, () => false);
    captureStep = 'read-telemetry';
    const telemetry = await page.evaluate(() => ({
      ready: globalThis.VITRUM_CAPTURE_READY === true,
      error: globalThis.VITRUM_CAPTURE_ERROR ?? null,
      telemetry: globalThis.VITRUM_CAPTURE_TELEMETRY ?? null,
    }));
    const summarizedTelemetry = summarizeTelemetry(telemetry.telemetry);
    lastTelemetry = summarizedTelemetry;
    lastConsole = consoleLines.slice(-80);
    captureStep = 'page-pre-capture-diagnostics';
    const pageDiagnostics = await snapshotPageDiagnostics(page, 'pre-capture');
    lastPageDiagnostics = pageDiagnostics;
    if (!ready || telemetry.error != null || telemetry.ready !== true) {
      return {
        generatedAt: new Date().toISOString(),
        harness: 'gltf-browser-proof:pt-webgl2-real',
        verdict: 'FAIL',
        assetId: asset.assetId,
        kind: asset.kind,
        backend: 'pt-webgl2',
        error: telemetry.error ?? 'capture did not become ready',
        pageDiagnostics,
        telemetry: summarizedTelemetry,
        console: consoleLines.slice(-80),
      };
    }

    const goldenPath = resolve(repoRoot, asset.goldenPath);
    await mkdir(dirname(goldenPath), { recursive: true });
    const capture = await captureCanvasPng(page);
    const pngBytes = capture.bytes;
    const png = PNG.sync.read(pngBytes);
    const luminance = meanLuminance(png.data);
    const structure = imageStructureMetrics(png.data);
    if (!captureLooksInformative(structure)) {
      captureStep = 'classify-result';
      return {
        generatedAt: new Date().toISOString(),
        harness: 'gltf-browser-proof:pt-webgl2-real',
        verdict: 'FAIL',
        command: `node tools/gltf-browser-proof/capture-pt-webgl2-real.mjs --asset ${asset.assetId}`,
        updateGolden,
        assetId: asset.assetId,
        kind: asset.kind,
        backend: 'pt-webgl2',
        captureMode: engineCaptureMode,
        width: png.width,
        height: png.height,
        samplesPerPixel: spp,
        captureMethod: capture.method,
        captureAttempts: capture.attempts,
        luminance,
        structure,
        error: `capture is visually uninformative: ${structureFailureReason(structure)}`,
        pageDiagnostics,
        telemetry: summarizedTelemetry,
        console: consoleLines.slice(-80),
      };
    }
    const compare = await compareOrUpdate(pngBytes, png, goldenPath, asset.thresholds);
    captureStep = 'classify-result';
    const pass =
      summarizedTelemetry?.backend === 'pt-webgl2' &&
      summarizedTelemetry?.assetId === asset.assetId &&
      summarizedTelemetry?.realAssetReady === true &&
      requiredExtensionsPresent(asset, summarizedTelemetry) &&
      requiredHooksPresent(asset, summarizedTelemetry) &&
      luminance > 0.005 &&
      captureLooksInformative(structure) &&
      compare.pass === true;
    return {
      generatedAt: new Date().toISOString(),
      harness: 'gltf-browser-proof:pt-webgl2-real',
      verdict: pass ? 'PASS' : 'FAIL',
      command: `node tools/gltf-browser-proof/capture-pt-webgl2-real.mjs --asset ${asset.assetId}`,
      updateGolden,
      assetId: asset.assetId,
      kind: asset.kind,
      backend: 'pt-webgl2',
      captureMode: engineCaptureMode,
      width: png.width,
      height: png.height,
      samplesPerPixel: spp,
      captureMethod: capture.method,
      captureAttempts: capture.attempts,
      luminance,
      structure,
      pageDiagnostics,
      telemetry: summarizedTelemetry,
      golden: compare,
      console: consoleLines.slice(-80),
    };
  } finally {
    await browser.close();
    if (activeBrowser === browser) activeBrowser = null;
  }
}

async function captureCanvasPng(page) {
  const canvas = page.locator('canvas').first();
  const timeout = Math.max(1000, Math.min(timeoutMs, screenshotTimeoutMs));
  let engineError = null;
  lastCaptureAttempts = [];
  if (engineCaptureMode === 'engine-first') {
    try {
      captureStep = 'engine-captureFrame-output';
      return await captureAttempt(
        'engine-captureFrame-output',
        (attempt) => captureEngineFramePng(page, timeout, attempt),
      );
    } catch (error) {
      engineError = error;
      if (isEngineReadbackHostBlock(error)) {
        const failedAttempt = lastCaptureAttempts.at(-1);
        if (failedAttempt?.method === 'engine-captureFrame-output') {
          failedAttempt.hostBlockHint = 'engine-readback';
        }
        // A timed-out WebGL readback can leave the page main thread wedged in
        // ReadPixels. Browser screenshot/data-URL fallbacks on the same page then
        // only add misleading locator timeouts, so preserve the precise blocker.
        throw error;
      }
    }
  }

  const canvasPaused = pauseBeforeEngineCapture
    ? await pauseExampleRenderingForCanvasCapture(page, timeout)
    : false;
  try {
    let clipError = null;
    try {
      captureStep = 'page-canvas-clip-screenshot';
      return await captureAttempt(
        'page-canvas-clip-screenshot',
        () => withTimeout(pageCanvasClipScreenshot(page, timeout), timeout + 1000, 'page clipped screenshot timed out'),
      );
    } catch (error) {
      clipError = error;
    }

    let screenshotError = null;
    try {
      captureStep = 'canvas-screenshot';
      return await captureAttempt(
        'playwright-screenshot',
        () => withTimeout(canvas.screenshot({ type: 'png', timeout }), timeout + 1000, 'canvas element screenshot timed out'),
      );
    } catch (error) {
      screenshotError = error;
    }

    if (engineCaptureMode === 'engine-fallback') {
      try {
        captureStep = 'engine-captureFrame-output';
        return await captureAttempt(
          'engine-captureFrame-output',
          (attempt) => captureEngineFramePng(page, timeout, attempt),
        );
      } catch (fallbackEngineError) {
        engineError = fallbackEngineError;
      }
    }

    try {
      captureStep = 'canvas-data-url';
      return await captureAttempt('canvas-data-url', () => captureCanvasDataUrlPng(page, engineError, screenshotError, clipError));
    } catch (dataUrlError) {
      throw dataUrlError;
    }
  } finally {
    if (canvasPaused) await resumeExampleRendering(page, 1000);
  }
}

async function snapshotPageDiagnostics(page, phase) {
  try {
    return await page.evaluate((label) => {
      const canvas = document.querySelector('canvas');
      const rect = canvas instanceof HTMLCanvasElement ? canvas.getBoundingClientRect() : null;
      return {
        phase: label,
        url: location.href,
        ready: globalThis.VITRUM_CAPTURE_READY === true,
        captureError: globalThis.VITRUM_CAPTURE_ERROR ?? null,
        captureFrameInstalled: typeof globalThis.VITRUM_CAPTURE_FRAME === 'function',
        captureFrameType: typeof globalThis.VITRUM_CAPTURE_FRAME,
        capturePaused: globalThis.VITRUM_CAPTURE_PAUSED === true,
        canvasPresent: canvas instanceof HTMLCanvasElement,
        canvasId: canvas instanceof HTMLCanvasElement ? canvas.id : null,
        canvasWidth: canvas instanceof HTMLCanvasElement ? canvas.width : null,
        canvasHeight: canvas instanceof HTMLCanvasElement ? canvas.height : null,
        canvasClientWidth: canvas instanceof HTMLCanvasElement ? canvas.clientWidth : null,
        canvasClientHeight: canvas instanceof HTMLCanvasElement ? canvas.clientHeight : null,
        canvasRect: rect == null
          ? null
          : {
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height,
            },
        devicePixelRatio: globalThis.devicePixelRatio ?? null,
      };
    }, phase);
  } catch (error) {
    return {
      phase,
      error: String(error?.message ?? error),
    };
  }
}

function isEngineReadbackHostBlock(error) {
  const message = String(error?.message ?? error);
  return (
    message.includes('engine captureFrame fallback timed out') ||
    message.includes('pausing example render loop before capture timed out')
  );
}

async function captureAttempt(method, run) {
  const attempt = {
    method,
    status: 'started',
    step: captureStep,
    startedAt: new Date().toISOString(),
  };
  lastCaptureAttempts.push(attempt);
  try {
    const result = await run(attempt);
    const bytes = result?.bytes ?? result;
    attempt.status = 'succeeded';
    attempt.finishedAt = new Date().toISOString();
    return {
      method: result?.method ?? method,
      bytes,
      attempts: snapshotCaptureAttempts(),
    };
  } catch (error) {
    attempt.status = 'failed';
    attempt.error = String(error?.message ?? error);
    attempt.finishedAt = new Date().toISOString();
    throw error;
  }
}

async function pageCanvasClipScreenshot(page, timeout) {
  const box = await page.locator('canvas').first().boundingBox({ timeout });
  if (box == null) throw new Error('canvas bounding box unavailable for clipped page screenshot');
  return page.screenshot({
    type: 'png',
    timeout,
    clip: {
      x: Math.max(0, box.x),
      y: Math.max(0, box.y),
      width: Math.max(1, box.width),
      height: Math.max(1, box.height),
    },
  });
}

async function captureEngineFramePng(page, timeout, attempt = null) {
  if (attempt != null) {
    attempt.pauseBeforeCapture = pauseBeforeEngineCapture;
    attempt.pauseProtocol = 'VITRUM_CAPTURE_PAUSED';
  }
  if (pauseBeforeEngineCapture) await pauseExampleRendering(page, timeout);
  if (attempt != null) {
    attempt.pausedAtCaptureStart = await page.evaluate(() => globalThis.VITRUM_CAPTURE_PAUSED === true)
      .catch(() => false);
  }
  let frame;
  try {
    frame = await withTimeout(
      page.evaluate(async () => {
        const capture = globalThis.VITRUM_CAPTURE_FRAME;
        if (typeof capture !== 'function') {
          throw new Error('VITRUM_CAPTURE_FRAME is not installed by the example page');
        }
        return capture('output');
      }),
      timeout,
      'engine captureFrame fallback timed out',
    );
  } finally {
    if (pauseBeforeEngineCapture) await resumeExampleRendering(page, 1000);
  }
  if (frame == null || typeof frame !== 'object') throw new Error('engine captureFrame returned no frame');
  const width = Number(frame.width);
  const height = Number(frame.height);
  const rgba = frame.rgba;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error(`engine captureFrame returned invalid dimensions ${width}x${height}`);
  }
  if (!Array.isArray(rgba) || rgba.length !== width * height * 4) {
    throw new Error(`engine captureFrame returned invalid rgba payload length ${Array.isArray(rgba) ? rgba.length : typeof rgba}`);
  }
  const png = new PNG({ width, height });
  for (let i = 0; i < rgba.length; i += 1) {
    const value = Number(rgba[i]);
    png.data[i] = floatToByte(value);
  }
  return PNG.sync.write(png);
}

async function pauseExampleRendering(page, timeout) {
  await withTimeout(
    page.evaluate(async () => {
      globalThis.VITRUM_CAPTURE_PAUSED = true;
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }),
    timeout,
    'pausing example render loop before capture timed out',
  );
}

async function resumeExampleRendering(page, timeout) {
  await withTimeout(
    page.evaluate(() => {
      globalThis.VITRUM_CAPTURE_PAUSED = false;
    }),
    timeout,
    'resuming example render loop after capture timed out',
  ).catch(() => undefined);
}

async function pauseExampleRenderingForCanvasCapture(page, timeout) {
  try {
    await pauseExampleRendering(page, timeout);
    return true;
  } catch {
    return false;
  }
}

async function captureCanvasDataUrlPng(page, engineError, screenshotError, clipError) {
  captureStep = 'canvas-data-url';
  const dataUrlTimeout = Math.max(1000, Math.min(timeoutMs, dataUrlTimeoutMs));
  const dataUrl = await withTimeout(
    page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!(canvas instanceof HTMLCanvasElement)) {
        throw new Error('no HTMLCanvasElement is present for capture');
      }
      return canvas.toDataURL('image/png');
    }),
    dataUrlTimeout,
    'canvas PNG data URL fallback timed out',
  ).catch((fallbackError) => {
    const enginePart = engineError == null
      ? 'engine captureFrame fallback was not attempted'
      : `engine captureFrame fallback failed (${engineError instanceof Error ? engineError.message : String(engineError)})`;
    const screenshotPart = screenshotError == null
      ? 'Playwright canvas screenshot was not attempted'
      : `Playwright canvas screenshot failed (${screenshotError instanceof Error ? screenshotError.message : String(screenshotError)})`;
    const clipPart = clipError == null
      ? 'page clipped screenshot was not attempted'
      : `page clipped screenshot failed (${clipError instanceof Error ? clipError.message : String(clipError)})`;
    throw new Error(
      `${enginePart}; ` +
        `${screenshotPart}; ` +
        `${clipPart}; ` +
        `canvas PNG data URL fallback failed (${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)})`,
    );
  });
  return {
    method: 'canvas-data-url',
    bytes: pngBytesFromDataUrl(dataUrl),
  };
}

function summarizeTelemetry(telemetry) {
  if (telemetry == null || typeof telemetry !== 'object') return null;
  return {
    assetId: telemetry.assetId,
    backend: telemetry.backend,
    profileId: telemetry.profileId,
    primitiveCount: telemetry.primitiveCount,
    extensionsUsed: telemetry.extensionsUsed ?? [],
    extensionsRequired: telemetry.extensionsRequired ?? [],
    browserDecodeHooks: telemetry.browserDecodeHooks ?? {},
    textureDecodeReport: {
      mapCount: telemetry.textureDecodeReport?.mapCount ?? 0,
      uniqueHandleCount: telemetry.textureDecodeReport?.uniqueHandleCount ?? 0,
      rawImageCount: telemetry.textureDecodeReport?.rawImageCount ?? 0,
      opaqueHandleCount: telemetry.textureDecodeReport?.opaqueHandleCount ?? 0,
      cpuReadableCount: telemetry.textureDecodeReport?.cpuReadableCount ?? 0,
    },
    warningCount: Array.isArray(telemetry.warnings) ? telemetry.warnings.length : 0,
    diagnosticCount: Array.isArray(telemetry.diagnostics) ? telemetry.diagnostics.length : 0,
    realAssetReady: telemetry.realAssetReady === true,
  };
}

function requiredExtensionsPresent(asset, telemetry) {
  const used = telemetry?.extensionsUsed ?? [];
  return asset.requiredExtensions.every((ext) => used.includes(ext));
}

function requiredHooksPresent(asset, telemetry) {
  const hooks = telemetry?.browserDecodeHooks ?? {};
  return asset.requiredHooks.every((hook) => hooks[hook] === true);
}

function hostBlockedStatus(asset, error) {
  const hostBlock = classifyHostBlock(error, lastCaptureAttempts);
  return {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    assetId: asset.assetId,
    kind: asset.kind,
    backend: 'pt-webgl2',
    captureMode: engineCaptureMode,
    step: captureStep,
    error: String(error?.stack ?? error),
    hostBlockClass: hostBlock.class,
    hostBlockMethods: hostBlock.methods,
    hostBlockReason: hostBlock.reason,
    captureAttempts: snapshotCaptureAttempts(),
    pageDiagnostics: lastPageDiagnostics,
    hostReadbackProbe,
    telemetry: lastTelemetry,
    console: lastConsole.slice(-80),
    serverLog: serverLog.slice(-4000),
  };
}

function classifyHostBlock(error, attempts) {
  const failedAttempts = attempts.filter((attempt) => attempt.status === 'failed' || attempt.status === 'started');
  const methods = Array.from(new Set(failedAttempts.map((attempt) => attempt.method)));
  const fragments = [
    String(error?.message ?? error ?? ''),
    ...failedAttempts.map((attempt) => `${attempt.method}: ${attempt.error ?? ''} ${attempt.hostBlockHint ?? ''}`),
  ].join('\n');
  const hasEngineReadback = failedAttempts.some((attempt) =>
    attempt.method === 'engine-captureFrame-output' &&
    (
      attempt.hostBlockHint === 'engine-readback' ||
      String(attempt.error ?? '').includes('engine captureFrame fallback timed out')
    )
  );
  const hasCanvasScreenshot = failedAttempts.some((attempt) =>
    attempt.method === 'playwright-screenshot' ||
    attempt.method === 'page-canvas-clip-screenshot'
  );
  const hasCanvasDataUrl = failedAttempts.some((attempt) => attempt.method === 'canvas-data-url');
  const timedOut = /timed out|Timeout/i.test(fragments);

  if (hasEngineReadback && !hasCanvasScreenshot && !hasCanvasDataUrl) {
    return {
      class: 'engine-readback-timeout',
      methods,
      reason: 'pt-webgl2 engine.captureFrame readPixels timed out before browser fallback could run safely',
    };
  }
  if (hasEngineReadback && (hasCanvasScreenshot || hasCanvasDataUrl)) {
    return {
      class: 'multi-readback-timeout',
      methods,
      reason: 'engine captureFrame and browser canvas readback paths did not return pixels on this host',
    };
  }
  if ((hasCanvasScreenshot || hasCanvasDataUrl) && timedOut) {
    return {
      class: 'browser-canvas-readback-timeout',
      methods,
      reason: 'browser canvas screenshot/data-url readback timed out after the real glTF page became capture-ready',
    };
  }
  return {
    class: 'host-readback-blocked',
    methods,
    reason: 'browser host could not return pixels after the real glTF page became capture-ready',
  };
}

function snapshotCaptureAttempts() {
  return lastCaptureAttempts.map((attempt) => ({ ...attempt }));
}

function summarize(results) {
  const pass = results.every((result) => result.verdict === 'PASS');
  const hostBlocked = results.every((result) => result.verdict === 'PASS' || result.verdict === 'HOST-BLOCKED');
  const hostBlockClasses = Array.from(new Set(
    results
      .filter((result) => result.verdict === 'HOST-BLOCKED')
      .map((result) => result.hostBlockClass)
      .filter(Boolean),
  ));
  return {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: pass ? 'PASS' : hostBlocked ? 'HOST-BLOCKED' : 'FAIL',
    backend: 'pt-webgl2',
    ...(hostBlockClasses.length > 0 ? { hostBlockClasses } : {}),
    captureMode: engineCaptureMode,
    hostReadbackProbe,
    assets: results,
    assetCount: results.length,
  };
}

async function compareOrUpdate(pngBytes, png, goldenPath, thresholds) {
  if (updateGolden) {
    await writeFile(goldenPath, pngBytes);
    return { pass: true, updated: true, path: relative(goldenPath), rmse: 0, meanAbs: 0, maxAbs: 0, thresholds };
  }
  let baseline;
  try {
    baseline = PNG.sync.read(await readFile(goldenPath));
  } catch (error) {
    return { pass: false, path: relative(goldenPath), error: `missing/unreadable golden PNG: ${error.message}`, thresholds };
  }
  if (baseline.width !== png.width || baseline.height !== png.height) {
    return {
      pass: false,
      path: relative(goldenPath),
      error: `golden PNG size ${baseline.width}x${baseline.height} does not match capture ${png.width}x${png.height}`,
      thresholds,
    };
  }
  const metrics = comparePixels(png.data, baseline.data);
  return {
    pass: metrics.rmse <= thresholds.maxRmse && metrics.meanAbs <= thresholds.maxMeanAbs && metrics.maxAbs <= thresholds.maxAbs,
    path: relative(goldenPath),
    ...metrics,
    thresholds,
  };
}

async function waitForServer(serverPort, timeout) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (server.exitCode != null) throw new Error(`vite exited early with code ${server.exitCode}: ${serverLog}`);
    try {
      const response = await fetch(`http://127.0.0.1:${serverPort}/`);
      if (response.ok) return;
    } catch {
      // keep polling
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error(`vite did not become ready within ${timeout}ms: ${serverLog}`);
}

async function stopServer() {
  if (server.exitCode != null || server.signalCode != null) return;
  server.kill('SIGTERM');
  await new Promise((resolvePromise) => {
    const timer = setTimeout(() => {
      if (server.exitCode == null && server.signalCode == null) server.kill('SIGKILL');
      resolvePromise(undefined);
    }, 1500);
    server.once('exit', () => {
      clearTimeout(timer);
      resolvePromise(undefined);
    });
  });
}

async function closeActiveBrowser() {
  if (activeBrowser == null) return;
  const browser = activeBrowser;
  activeBrowser = null;
  await Promise.race([
    browser.close().catch(() => undefined),
    new Promise((resolvePromise) => setTimeout(resolvePromise, 2000)),
  ]);
}

async function withTimeout(promise, ms, message) {
  let timer;
  const guarded = Promise.resolve(promise);
  guarded.catch(() => undefined);
  try {
    return await Promise.race([
      guarded,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function writeStatus(status) {
  const statusWithProvenance = {
    ...status,
    provenance: await buildStatusProvenance(),
  };
  await mkdir(dirname(statusPath), { recursive: true });
  await writeFile(statusPath, `${JSON.stringify(statusWithProvenance, null, 2)}\n`);
}

async function buildStatusProvenance() {
  return {
    schema: 'vitrum.gltf-browser-proof.status-provenance.v1',
    checkerPath: CHECKER_PATH,
    checkerSha256: await sha256RepoPath(CHECKER_PATH),
    captureHarnessPath: CAPTURE_HARNESS_PATH,
    captureHarnessSha256: await sha256RepoPath(CAPTURE_HARNESS_PATH),
    manifestPath: MANIFEST_PATH,
    manifestSha256: await sha256RepoPath(MANIFEST_PATH),
    goldenFiles: await Promise.all(REAL_BROWSER_ASSETS.map(async (asset) => ({
      assetId: asset.assetId,
      ...(await repoFileState(asset.goldenPath)),
    }))),
  };
}

async function sha256RepoPath(path) {
  const bytes = await readFile(resolve(repoRoot, path));
  return createHash('sha256').update(bytes).digest('hex');
}

async function repoFileState(path) {
  try {
    const sha256 = await sha256RepoPath(path);
    return { path, exists: true, sha256 };
  } catch (error) {
    if (error && error.code === 'ENOENT') return { path, exists: false, sha256: null };
    throw error;
  }
}

function comparePixels(candidate, baseline) {
  let sumSq = 0;
  let sumAbs = 0;
  let maxAbs = 0;
  for (let i = 0; i < candidate.length; i += 1) {
    const delta = candidate[i] - baseline[i];
    const abs = Math.abs(delta);
    sumSq += delta * delta;
    sumAbs += abs;
    maxAbs = Math.max(maxAbs, abs);
  }
  return {
    rmse: Math.sqrt(sumSq / candidate.length),
    meanAbs: sumAbs / candidate.length,
    maxAbs,
  };
}

function meanLuminance(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += 0.2126 * (pixels[i] / 255) + 0.7152 * (pixels[i + 1] / 255) + 0.0722 * (pixels[i + 2] / 255);
  }
  return sum / (pixels.length / 4);
}

function imageStructureMetrics(pixels) {
  let minLuma = 255;
  let maxLuma = 0;
  let dominantColorCount = 0;
  const colorCounts = new Map();
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const a = pixels[i + 3];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    minLuma = Math.min(minLuma, luma);
    maxLuma = Math.max(maxLuma, luma);
    const key = `${r},${g},${b},${a}`;
    const count = (colorCounts.get(key) ?? 0) + 1;
    colorCounts.set(key, count);
    dominantColorCount = Math.max(dominantColorCount, count);
  }
  const pixelCount = pixels.length / 4;
  const dominantColorFraction = pixelCount > 0 ? dominantColorCount / pixelCount : 1;
  return {
    thresholds: structureThresholds,
    minLuma,
    maxLuma,
    lumaRange: maxLuma - minLuma,
    uniqueColorCount: colorCounts.size,
    dominantColorFraction,
    nonDominantFraction: 1 - dominantColorFraction,
  };
}

function captureLooksInformative(structure) {
  return (
    structure.lumaRange >= structureThresholds.minLumaRange &&
    structure.uniqueColorCount >= structureThresholds.minUniqueColorCount &&
    structure.nonDominantFraction >= structureThresholds.minNonDominantFraction
  );
}

function structureFailureReason(structure) {
  const reasons = [];
  if (structure.lumaRange < structureThresholds.minLumaRange) {
    reasons.push(`lumaRange ${structure.lumaRange.toFixed(3)} < ${structureThresholds.minLumaRange}`);
  }
  if (structure.uniqueColorCount < structureThresholds.minUniqueColorCount) {
    reasons.push(`uniqueColorCount ${structure.uniqueColorCount} < ${structureThresholds.minUniqueColorCount}`);
  }
  if (structure.nonDominantFraction < structureThresholds.minNonDominantFraction) {
    reasons.push(`nonDominantFraction ${structure.nonDominantFraction.toFixed(4)} < ${structureThresholds.minNonDominantFraction}`);
  }
  return reasons.join('; ');
}

function relative(path) {
  return path.startsWith(`${repoRoot}/`) ? path.slice(repoRoot.length + 1) : path;
}

function pngBytesFromDataUrl(dataUrl) {
  if (typeof dataUrl !== 'string') throw new Error(`canvas PNG data URL must be a string, got ${typeof dataUrl}`);
  const match = /^data:image\/png;base64,(.+)$/i.exec(dataUrl);
  if (match == null) throw new Error('canvas PNG data URL fallback did not return image/png data');
  return Buffer.from(match[1], 'base64');
}

function floatToByte(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

function resolveStatusPath(rawPath) {
  if (rawPath == null || rawPath.length === 0) return resolve(scriptDir, 'pt-webgl2-real-status.json');
  return resolve(repoRoot, rawPath);
}
