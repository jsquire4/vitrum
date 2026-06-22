#!/usr/bin/env node
// Captures real glTF assets through the browser pt-webgl2 one-call path.

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

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
const thresholds = { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 };
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
  },
  {
    assetId: 'cesium-milk-truck-draco',
    kind: 'draco',
    goldenPath: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png',
    minTextures: 0,
    requiredExtensions: ['KHR_draco_mesh_compression'],
    requiredHooks: ['draco'],
  },
  {
    assetId: 'meshopt-cube-real',
    kind: 'meshopt',
    goldenPath: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png',
    minTextures: 0,
    requiredExtensions: ['KHR_meshopt_compression'],
    requiredHooks: ['meshopt'],
  },
];
let activeBrowser = null;
let captureStep = 'not-started';
let lastTelemetry = null;
let lastConsole = [];

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
  const assets = selectedAssets();
  const results = [];
  for (const asset of assets) {
    lastTelemetry = null;
    lastConsole = [];
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
  finalExitCode = statusExitCode(summary.verdict);
} catch (error) {
  await closeActiveBrowser();
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    step: captureStep,
    error: String(error?.stack ?? error),
    telemetry: lastTelemetry,
    console: lastConsole.slice(-80),
    serverLog: serverLog.slice(-4000),
  };
  await writeStatus(status);
  finalExitCode = 2;
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
    if (!ready || telemetry.error != null || telemetry.ready !== true) {
      return {
        generatedAt: new Date().toISOString(),
        harness: 'gltf-browser-proof:pt-webgl2-real',
        verdict: 'FAIL',
        assetId: asset.assetId,
        kind: asset.kind,
        backend: 'pt-webgl2',
        error: telemetry.error ?? 'capture did not become ready',
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
        width: png.width,
        height: png.height,
        samplesPerPixel: spp,
        captureMethod: capture.method,
        luminance,
        structure,
        error: `capture is visually uninformative: ${structureFailureReason(structure)}`,
        telemetry: summarizedTelemetry,
        console: consoleLines.slice(-80),
      };
    }
    const compare = await compareOrUpdate(pngBytes, png, goldenPath);
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
      width: png.width,
      height: png.height,
      samplesPerPixel: spp,
      captureMethod: capture.method,
      luminance,
      structure,
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
  try {
    captureStep = 'engine-captureFrame-output';
    return {
      method: 'engine-captureFrame-output',
      bytes: await captureEngineFramePng(page, timeout),
    };
  } catch (error) {
    try {
      captureStep = 'canvas-screenshot';
      return {
        method: 'playwright-screenshot',
        bytes: await canvas.screenshot({ type: 'png', timeout }),
      };
    } catch (screenshotError) {
      try {
        captureStep = 'page-canvas-clip-screenshot';
        return {
          method: 'page-canvas-clip-screenshot',
          bytes: await pageCanvasClipScreenshot(page, timeout),
        };
      } catch (clipError) {
        captureStep = 'canvas-data-url';
        return await captureCanvasDataUrlPng(page, error, screenshotError, clipError);
      }
    }
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

async function captureEngineFramePng(page, timeout) {
  await pauseExampleRendering(page, timeout);
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
    await resumeExampleRendering(page, 1000);
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
    throw new Error(
      `engine captureFrame fallback failed (${engineError instanceof Error ? engineError.message : String(engineError)}); ` +
        `Playwright canvas screenshot failed (${screenshotError instanceof Error ? screenshotError.message : String(screenshotError)}); ` +
        `page clipped screenshot failed (${clipError instanceof Error ? clipError.message : String(clipError)}); ` +
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
  return {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    assetId: asset.assetId,
    kind: asset.kind,
    backend: 'pt-webgl2',
    step: captureStep,
    error: String(error?.stack ?? error),
    telemetry: lastTelemetry,
    console: lastConsole.slice(-80),
    serverLog: serverLog.slice(-4000),
  };
}

function summarize(results) {
  const pass = results.every((result) => result.verdict === 'PASS');
  const hostBlocked = results.every((result) => result.verdict === 'PASS' || result.verdict === 'HOST-BLOCKED');
  return {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: pass ? 'PASS' : hostBlocked ? 'HOST-BLOCKED' : 'FAIL',
    backend: 'pt-webgl2',
    assets: results,
    assetCount: results.length,
  };
}

function statusExitCode(verdict) {
  if (verdict === 'PASS') return 0;
  if (verdict === 'HOST-BLOCKED') return 2;
  return 1;
}

async function compareOrUpdate(pngBytes, png, goldenPath) {
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
  await mkdir(dirname(statusPath), { recursive: true });
  await writeFile(statusPath, `${JSON.stringify(status, null, 2)}\n`);
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
