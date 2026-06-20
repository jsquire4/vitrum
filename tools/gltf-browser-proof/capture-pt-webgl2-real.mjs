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
const statusPath = resolve(scriptDir, 'pt-webgl2-real-status.json');
const thresholds = { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 };
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
  if (summary.verdict !== 'PASS') process.exit(summary.verdict === 'HOST-BLOCKED' ? 2 : 1);
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
  process.exit(2);
} finally {
  await stopServer();
}

function readMultiFlag(name) {
  const values = [];
  for (let i = 0; i < process.argv.length; i += 1) {
    const arg = process.argv[i];
    if (arg === name && process.argv[i + 1]) values.push(process.argv[i + 1]);
    if (arg?.startsWith(`${name}=`)) values.push(arg.slice(name.length + 1));
  }
  return values.flatMap((value) => value.split(',').map((item) => item.trim()).filter(Boolean));
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

    captureStep = 'canvas-screenshot';
    const goldenPath = resolve(repoRoot, asset.goldenPath);
    await mkdir(dirname(goldenPath), { recursive: true });
    const canvas = page.locator('canvas').first();
    const pngBytes = await canvas.screenshot({ type: 'png', timeout: timeoutMs });
    const png = PNG.sync.read(pngBytes);
    const luminance = meanLuminance(png.data);
    const compare = await compareOrUpdate(pngBytes, png, goldenPath);
    captureStep = 'classify-result';
    const pass =
      summarizedTelemetry?.backend === 'pt-webgl2' &&
      summarizedTelemetry?.assetId === asset.assetId &&
      summarizedTelemetry?.realAssetReady === true &&
      requiredExtensionsPresent(asset, summarizedTelemetry) &&
      requiredHooksPresent(asset, summarizedTelemetry) &&
      luminance > 0.005 &&
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
      luminance,
      telemetry: summarizedTelemetry,
      golden: compare,
      console: consoleLines.slice(-80),
    };
  } finally {
    await browser.close();
    if (activeBrowser === browser) activeBrowser = null;
  }
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
  try {
    return await Promise.race([
      promise,
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

function relative(path) {
  return path.startsWith(`${repoRoot}/`) ? path.slice(repoRoot.length + 1) : path;
}
