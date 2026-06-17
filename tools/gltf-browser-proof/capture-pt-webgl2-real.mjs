#!/usr/bin/env node
// Captures the real BoxTextured glTF through the browser pt-webgl2 one-call path.

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const exampleDir = resolve(repoRoot, 'examples/gltf-viewer');
const updateGolden = process.argv.includes('--update-golden') || process.argv.includes('--update-goldens');
const width = Number(process.env.VITRUM_WIDTH ?? '64');
const height = Number(process.env.VITRUM_HEIGHT ?? '64');
const spp = Number(process.env.VITRUM_SPP ?? '1');
const port = Number(process.env.VITRUM_GLTF_BROWSER_PORT ?? '5187');
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? '120000');
const statusPath = resolve(scriptDir, 'pt-webgl2-real-status.json');
const goldenPath = resolve(repoRoot, 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png');
const thresholds = { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 };
let activeBrowser = null;
let captureStep = 'not-started';

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
  const result = await withTimeout(capture(), timeoutMs, 'browser capture timed out');
  await writeStatus(result);
  if (result.verdict !== 'PASS') process.exit(result.verdict === 'HOST-BLOCKED' ? 2 : 1);
} catch (error) {
  await closeActiveBrowser();
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    assetId: 'box-textured-glb',
    backend: 'pt-webgl2',
    step: captureStep,
    error: String(error?.stack ?? error),
    serverLog: serverLog.slice(-4000),
  };
  await writeStatus(status);
  process.exit(2);
} finally {
  await stopServer();
}

async function capture() {
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
    url.searchParams.set('vitrumGltfAsset', 'box-textured-glb');
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
    if (!ready || telemetry.error != null || telemetry.ready !== true) {
      return {
        generatedAt: new Date().toISOString(),
        harness: 'gltf-browser-proof:pt-webgl2-real',
        verdict: 'FAIL',
        assetId: 'box-textured-glb',
        backend: 'pt-webgl2',
        error: telemetry.error ?? 'capture did not become ready',
        telemetry: telemetry.telemetry,
        console: consoleLines.slice(-80),
      };
    }

    captureStep = 'canvas-readback';
    await mkdir(dirname(goldenPath), { recursive: true });
    const dataUrl = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!(canvas instanceof HTMLCanvasElement)) throw new Error('capture canvas not found');
      return canvas.toDataURL('image/png');
    });
    const pngBytes = Buffer.from(dataUrl.replace(/^data:image\/png;base64,/, ''), 'base64');
    const png = PNG.sync.read(pngBytes);
    const luminance = meanLuminance(png.data);
    const compare = await compareOrUpdate(pngBytes, png);
    captureStep = 'classify-result';
    const pass =
      telemetry.telemetry?.backend === 'pt-webgl2' &&
      telemetry.telemetry?.assetId === 'box-textured-glb' &&
      telemetry.telemetry?.realAssetReady === true &&
      luminance > 0.005 &&
      compare.pass === true;
    return {
      generatedAt: new Date().toISOString(),
      harness: 'gltf-browser-proof:pt-webgl2-real',
      verdict: pass ? 'PASS' : 'FAIL',
      command: 'node tools/gltf-browser-proof/capture-pt-webgl2-real.mjs',
      updateGolden,
      assetId: 'box-textured-glb',
      backend: 'pt-webgl2',
      width: png.width,
      height: png.height,
      samplesPerPixel: spp,
      luminance,
      telemetry: telemetry.telemetry,
      golden: compare,
      console: consoleLines.slice(-80),
    };
  } finally {
    await browser.close();
    if (activeBrowser === browser) activeBrowser = null;
  }
}

async function compareOrUpdate(pngBytes, png) {
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
