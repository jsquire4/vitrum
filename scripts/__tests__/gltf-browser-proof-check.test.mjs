import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const checkScript = join(repoRoot, 'tools', 'gltf-browser-proof', 'check-status.mjs');
const captureScript = join(repoRoot, 'tools', 'gltf-browser-proof', 'capture-pt-webgl2-real.mjs');
const statusExitCodeHelper = join(repoRoot, 'tools', 'gltf-browser-proof', 'status-exit-code.mjs');
const packageJsonPath = join(repoRoot, 'package.json');

test('gltf browser capture harness fail-closes host-blocked and failed statuses', async () => {
  const { gltfBrowserProofStatusExitCode } = await import(pathToFileURL(statusExitCodeHelper).href);

  assert.equal(gltfBrowserProofStatusExitCode('PASS'), 0);
  assert.equal(gltfBrowserProofStatusExitCode('HOST-BLOCKED'), 2);
  assert.equal(gltfBrowserProofStatusExitCode('FAIL'), 1);
  assert.equal(gltfBrowserProofStatusExitCode('UNKNOWN'), 1);
});

test('gltf browser proof checker validates PASS rows inside HOST-BLOCKED summaries', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'mixed-host-blocked.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['multi-readback-timeout'],
    assets: [
      {
        generatedAt: '2026-06-22T00:00:00.000Z',
        harness: 'gltf-browser-proof:pt-webgl2-real',
        verdict: 'PASS',
        assetId: 'box-textured-glb',
        kind: 'textured-glb',
        backend: 'pt-webgl2',
        captureMethod: 'engine-captureFrame-output',
        luminance: 1,
        telemetry: telemetryFor('box-textured-glb', {
          textureDecodeReport: { mapCount: 1 },
        }),
        golden: {
          pass: true,
          path: 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png',
          thresholds: { maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
        },
      },
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /box-textured-glb: PASS status must include visual-structure metrics/);
});

test('gltf browser capture harness orders engine and browser readbacks by mode', async () => {
  const source = await readFile(captureScript, 'utf8');
  assert.match(source, /VITRUM_ENGINE_CAPTURE_MODE/);
  assert.match(source, /preflightBrowserReadbackProbe/);
  assert.match(source, /hostReadbackProbe/);
  assert.match(source, /unsignedByteReadback/);
  assert.match(source, /floatReadback/);
  assert.match(source, /WEBGL_debug_renderer_info/);
  assert.match(source, /String\(rawValue \?\? 'engine-first'\)/);
  assert.match(source, /normalized === 'canvas-only'/);
  assert.match(source, /normalized === 'canvas-first'/);
  assert.match(source, /isEngineReadbackHostBlock\(error\)/);
  assert.match(source, /hostBlockHint = 'engine-readback'/);
  assert.match(source, /snapshotPageDiagnostics\(page, 'pre-capture'\)/);
  assert.match(source, /attempt\.pauseBeforeCapture = pauseBeforeEngineCapture/);
  assert.match(source, /attempt\.pausedAtCaptureStart = await page\.evaluate/);
  assert.match(source, /attempt\.pauseProtocol = 'VITRUM_CAPTURE_PAUSED'/);

  const captureFnStart = source.indexOf('async function captureCanvasPng(page)');
  assert.notEqual(captureFnStart, -1);
  const captureFn = source.slice(captureFnStart, source.indexOf('\nasync function pageCanvasClipScreenshot', captureFnStart));
  assert.match(captureFn, /engineCaptureMode === 'engine-first'/);
  assert.match(captureFn, /engineCaptureMode === 'engine-fallback'/);
  assert.match(captureFn, /pauseExampleRenderingForCanvasCapture\(page, timeout\)/);
  assert.match(captureFn, /let clipError = null/);
  assert.match(captureFn, /let screenshotError = null/);
  assert.match(captureFn, /captureStep = 'canvas-screenshot'/);
  assert.match(captureFn, /captureStep = 'page-canvas-clip-screenshot'/);
  assert.match(captureFn, /page clipped screenshot timed out/);
  assert.match(captureFn, /canvas element screenshot timed out/);
  assert.doesNotMatch(captureFn, /isBrowserReadbackHostBlock/);
  assert.match(captureFn, /isEngineReadbackHostBlock\(error\)[\s\S]*throw error/);

  const firstEngineReadback = captureFn.indexOf("captureStep = 'engine-captureFrame-output'");
  const canvasPause = captureFn.indexOf('pauseExampleRenderingForCanvasCapture(page, timeout)');
  const firstDataUrlReadback = captureFn.indexOf("captureStep = 'canvas-data-url'");
  const firstClipScreenshot = captureFn.indexOf("captureStep = 'page-canvas-clip-screenshot'");
  const firstScreenshot = captureFn.indexOf("captureStep = 'canvas-screenshot'");
  const fallbackReadback = captureFn.indexOf("engineCaptureMode === 'engine-fallback'");
  const finalDataUrlReadback = captureFn.lastIndexOf("captureStep = 'canvas-data-url'");

  assert.ok(firstEngineReadback > 0);
  assert.ok(canvasPause > firstEngineReadback);
  assert.ok(firstClipScreenshot > canvasPause);
  assert.ok(firstScreenshot > firstClipScreenshot);
  assert.ok(fallbackReadback > firstScreenshot);
  assert.ok(firstDataUrlReadback > fallbackReadback);
  assert.equal(firstDataUrlReadback, finalDataUrlReadback);
});

test('gltf browser proof checker validates optional host readback preflight probes', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-with-probe.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    captureMode: 'engine-first',
    hostBlockClasses: ['multi-readback-timeout'],
    hostReadbackProbe: validHostReadbackProbe(),
    assets: [
      {
        ...hostBlockedRow('box-textured-glb', 'textured-glb', {
          textureDecodeReport: { mapCount: 1 },
        }),
        hostReadbackProbe: validHostReadbackProbe(),
      },
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.equal(result.status, 0);
});

test('gltf browser capture harness fails closed before per-asset readiness proof', async () => {
  const source = await readFile(captureScript, 'utf8');
  assert.match(source, /gltf-browser-proof-setup-failed/);
  assert.match(source, /failed before a per-asset page proved capture readiness/);

  const setupCatchStart = source.indexOf('} catch (error) {');
  assert.notEqual(setupCatchStart, -1);
  const setupCatch = source.slice(setupCatchStart, source.indexOf('\n} finally {', setupCatchStart));
  assert.match(setupCatch, /verdict: 'FAIL'/);
  assert.doesNotMatch(setupCatch, /verdict: 'HOST-BLOCKED'/);
});

test('gltf browser proof package scripts check both committed host-block artifacts', async () => {
  const pkg = JSON.parse(await readFile(packageJsonPath, 'utf8'));
  const check = pkg.scripts?.['gltf-browser-proof-check'] ?? '';
  const required = pkg.scripts?.['gltf-browser-proof-check:required'] ?? '';

  assert.match(check, /tools\/gltf-browser-proof\/check-status\.mjs/);
  assert.match(check, /pt-webgl2-real-canvas-first-status\.json/);
  assert.match(required, /tools\/gltf-browser-proof\/check-status\.mjs --require-pass/);
  assert.match(required, /pt-webgl2-real-canvas-first-status\.json --require-pass/);
});

test('gltf browser proof checker requires structured host-blocked capture attempts', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-missing-attempts.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['multi-readback-timeout'],
    assets: [
      {
        ...hostBlockedRow('box-textured-glb', 'textured-glb', {
          textureDecodeReport: { mapCount: 1 },
        }),
        captureAttempts: undefined,
      },
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /box-textured-glb: HOST-BLOCKED status must include captureAttempts\[\]/);
});

test('gltf browser proof checker requires engine capture attempts to prove paused readback', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-unpaused-engine-attempt.json');
  const row = hostBlockedRow('box-textured-glb', 'textured-glb', {
    textureDecodeReport: { mapCount: 1 },
  });
  row.captureAttempts = row.captureAttempts.map((attempt) =>
    attempt.method === 'engine-captureFrame-output'
      ? { ...attempt, pausedAtCaptureStart: false }
      : attempt
  );
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['multi-readback-timeout'],
    assets: [
      row,
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /box-textured-glb: engine-captureFrame-output attempt must prove pausedAtCaptureStart:true/);
});

test('gltf browser proof checker requires pre-capture page diagnostics for host blocks', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-missing-diagnostics.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['multi-readback-timeout'],
    assets: [
      {
        ...hostBlockedRow('box-textured-glb', 'textured-glb', {
          textureDecodeReport: { mapCount: 1 },
        }),
        pageDiagnostics: undefined,
      },
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /box-textured-glb: HOST-BLOCKED status must include pageDiagnostics/);
});

test('gltf browser proof checker accepts canvas-first host blocks without engine attempts', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-canvas-first.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['browser-canvas-readback-timeout'],
    assets: [
      canvasFirstHostBlockedRow('box-textured-glb', 'textured-glb', {
        textureDecodeReport: { mapCount: 1 },
      }),
      canvasFirstHostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      canvasFirstHostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.equal(result.status, 0);
});

test('gltf browser proof checker requires structured host-block classification', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-gltf-browser-proof-'));
  const statusPath = join(dir, 'host-blocked-missing-classification.json');
  await writeFile(statusPath, `${JSON.stringify({
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    backend: 'pt-webgl2',
    hostBlockClasses: ['multi-readback-timeout'],
    assets: [
      {
        ...hostBlockedRow('box-textured-glb', 'textured-glb', {
          textureDecodeReport: { mapCount: 1 },
        }),
        hostBlockClass: undefined,
      },
      hostBlockedRow('cesium-milk-truck-draco', 'draco', {
        extensionsUsed: ['KHR_draco_mesh_compression'],
        extensionsRequired: ['KHR_draco_mesh_compression'],
        browserDecodeHooks: { requested: ['draco'], draco: true, meshopt: false },
      }),
      hostBlockedRow('meshopt-cube-real', 'meshopt', {
        extensionsUsed: ['KHR_meshopt_compression'],
        extensionsRequired: ['KHR_meshopt_compression'],
        browserDecodeHooks: { requested: ['meshopt'], draco: false, meshopt: true },
      }),
    ],
    assetCount: 3,
  }, null, 2)}\n`);

  const result = await runChecker(['--status', pathToFileURL(statusPath).href]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /box-textured-glb: HOST-BLOCKED status must include a recognized hostBlockClass/);
});

function hostBlockedRow(assetId, kind, overrides = {}) {
  return {
    generatedAt: '2026-06-22T00:00:00.000Z',
    harness: 'gltf-browser-proof:pt-webgl2-real',
    verdict: 'HOST-BLOCKED',
    assetId,
    kind,
    backend: 'pt-webgl2',
    captureMode: 'engine-first',
    step: 'canvas-data-url',
    error: 'canvas PNG data URL fallback failed',
    hostBlockClass: 'multi-readback-timeout',
    hostBlockMethods: [
      'playwright-screenshot',
      'page-canvas-clip-screenshot',
      'engine-captureFrame-output',
      'canvas-data-url',
    ],
    hostBlockReason: 'engine captureFrame and browser canvas readback paths did not return pixels on this host',
    captureAttempts: [
      {
        method: 'playwright-screenshot',
        status: 'failed',
        step: 'canvas-screenshot',
        error: 'canvas screenshot failed',
      },
      {
        method: 'page-canvas-clip-screenshot',
        status: 'failed',
        step: 'page-canvas-clip-screenshot',
        error: 'page clipped screenshot failed',
      },
      {
        method: 'engine-captureFrame-output',
        status: 'failed',
        step: 'engine-captureFrame-output',
        error: 'engine captureFrame fallback timed out',
        pauseBeforeCapture: true,
        pausedAtCaptureStart: true,
        pauseProtocol: 'VITRUM_CAPTURE_PAUSED',
      },
      {
        method: 'canvas-data-url',
        status: 'failed',
        step: 'canvas-data-url',
        error: 'canvas PNG data URL fallback failed',
      },
    ],
    pageDiagnostics: pageDiagnosticsFor(assetId),
    telemetry: telemetryFor(assetId, overrides),
    console: [],
    serverLog: '',
  };
}

function canvasFirstHostBlockedRow(assetId, kind, overrides = {}) {
  return {
    ...hostBlockedRow(assetId, kind, overrides),
    captureMode: 'canvas-first',
    step: 'canvas-data-url',
    error: 'canvas PNG data URL fallback failed',
    hostBlockClass: 'browser-canvas-readback-timeout',
    hostBlockMethods: [
      'page-canvas-clip-screenshot',
      'playwright-screenshot',
      'canvas-data-url',
    ],
    hostBlockReason: 'browser canvas screenshot/data-url readback timed out after the real glTF page became capture-ready',
    captureAttempts: [
      {
        method: 'page-canvas-clip-screenshot',
        status: 'failed',
        step: 'page-canvas-clip-screenshot',
        error: 'page.screenshot: Timeout 15000ms exceeded',
      },
      {
        method: 'playwright-screenshot',
        status: 'failed',
        step: 'canvas-screenshot',
        error: 'locator.screenshot: Timeout 15000ms exceeded',
      },
      {
        method: 'canvas-data-url',
        status: 'failed',
        step: 'canvas-data-url',
        error: 'canvas PNG data URL fallback failed',
      },
    ],
  };
}

function pageDiagnosticsFor(assetId) {
  return {
    phase: 'pre-capture',
    url: `http://127.0.0.1:5187/?vitrumGltfAsset=${assetId}&vitrumBackend=pt-webgl2&vitrumSpp=1`,
    ready: true,
    captureError: null,
    captureFrameInstalled: true,
    captureFrameType: 'function',
    capturePaused: false,
    canvasPresent: true,
    canvasId: 'vitrum-canvas',
    canvasWidth: 64,
    canvasHeight: 64,
    canvasClientWidth: 64,
    canvasClientHeight: 64,
    canvasRect: { x: 0, y: 0, width: 64, height: 64 },
    devicePixelRatio: 1,
  };
}

function validHostReadbackProbe() {
  return {
    generatedAt: '2026-06-22T00:00:00.000Z',
    finishedAt: '2026-06-22T00:00:01.000Z',
    harness: 'gltf-browser-proof:host-readback-preflight',
    status: 'PASS',
    browserVersion: '123.0.0.0',
    timeoutMs: 10000,
    webgl2: true,
    renderer: 'test renderer',
    vendor: 'test vendor',
    version: 'WebGL 2.0',
    shadingLanguageVersion: 'WebGL GLSL ES 3.00',
    extensions: ['EXT_color_buffer_float'],
    unsignedByteReadback: {
      status: 'PASS',
      rgba: [64, 128, 191, 255],
    },
    floatReadback: {
      status: 'PASS',
      rgba: [0.125, 0.25, 0.5, 1],
    },
    dataUrl: {
      status: 'PASS',
      length: 42,
      prefix: 'data:image/png;base64,',
    },
  };
}

function telemetryFor(assetId, overrides = {}) {
  return {
    assetId,
    backend: 'pt-webgl2',
    realAssetReady: true,
    extensionsUsed: overrides.extensionsUsed ?? [],
    extensionsRequired: overrides.extensionsRequired ?? [],
    browserDecodeHooks: overrides.browserDecodeHooks ?? { requested: [], draco: false, meshopt: false },
    textureDecodeReport: {
      mapCount: overrides.textureDecodeReport?.mapCount ?? 0,
    },
  };
}

function runChecker(args) {
  return new Promise((resolveResult) => {
    const child = spawn('deno', ['run', '--sloppy-imports', '--allow-read', checkScript, ...args], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status, signal) => {
      resolveResult({ status, signal, stdout, stderr });
    });
  });
}
