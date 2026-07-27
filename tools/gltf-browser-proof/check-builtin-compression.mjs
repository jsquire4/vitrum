#!/usr/bin/env node
// Builds and executes the package-owned Draco + meshopt decoders in Chromium.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const harnessRoot = resolve(scriptDir, 'builtin-compression');
const tempRoot = await mkdtemp(join(tmpdir(), 'vitrum-gltf-compression-browser-'));
const distDir = join(tempRoot, 'dist');

try {
  const build = await run(
    process.execPath,
    [
      resolve(repoRoot, 'node_modules/vite/bin/vite.js'),
      'build',
      harnessRoot,
      '--outDir',
      distDir,
      '--emptyOutDir',
    ],
    repoRoot,
  );
  const buildLog = `${build.stdout}\n${build.stderr}`;
  if (
    /Module ["'](?:node:)?(?:fs|path)(?:\/promises)?["'] has been externalized/iu.test(buildLog)
  ) {
    throw new Error(`Browser build externalized a Node fs/path module:\n${buildLog}`);
  }

  const emitted = await listFiles(distDir);
  const wasmFiles = emitted.filter((file) => extname(file) === '.wasm');
  if (wasmFiles.length !== 1) {
    throw new Error(`Expected exactly one emitted Draco WASM asset; found ${wasmFiles.length}.`);
  }

  const jsFiles = emitted.filter((file) => extname(file) === '.js');
  const jsSources = await Promise.all(
    jsFiles.map(async (file) => ({
      file,
      source: await readFile(file, 'utf8'),
    })),
  );
  const forbiddenMeshoptRuntimeSymbols = [
    'MeshoptEncoder',
    'encodeVertexBuffer',
    'MeshoptSimplifier',
    'MeshoptClusterizer',
  ];
  for (const symbol of forbiddenMeshoptRuntimeSymbols) {
    if (jsSources.some(({ source }) => source.includes(symbol))) {
      throw new Error(
        `Browser output unexpectedly contains meshoptimizer tooling symbol ${symbol}.`,
      );
    }
  }
  const meshoptChunks = jsSources.filter(
    ({ file, source }) =>
      basename(file).startsWith('meshopt_decoder-') &&
      source.includes('decodeGltfBuffer') &&
      source.includes('MeshoptDecoder'),
  );
  if (meshoptChunks.length !== 1) {
    throw new Error(`Expected one decoder-only meshopt chunk; found ${meshoptChunks.length}.`);
  }

  const server = await serveDirectory(distDir);
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    const page = await browser.newPage();
    const consoleErrors = [];
    const pageErrors = [];
    const responses = [];
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => pageErrors.push(String(error?.stack ?? error)));
    page.on('response', (response) => {
      responses.push({ url: response.url(), status: response.status() });
    });

    await page.goto(server.url, { waitUntil: 'networkidle' });
    await page.waitForFunction(
      () => globalThis.__VITRUM_BUILTIN_COMPRESSION_PROOF__?.status != null,
      undefined,
      { timeout: 30_000 },
    );
    const result = await page.evaluate(() => globalThis.__VITRUM_BUILTIN_COMPRESSION_PROOF__);
    if (pageErrors.length > 0 || consoleErrors.length > 0) {
      throw new Error(
        `Browser emitted errors: ${JSON.stringify({ pageErrors, consoleErrors }, null, 2)}`,
      );
    }
    if (result?.status !== 'PASS' || result.hostHooksPresent !== false) {
      throw new Error(`Browser decode proof failed: ${JSON.stringify(result, null, 2)}`);
    }

    assertArrayClose(result.draco?.positions, [0, 0, 0, 1, 0, 0, 0, 1, 0], 1e-6, 'Draco positions');
    assertArrayClose(result.draco?.normals, [0, 0, 1, 0, 0, 1, 0, 0, 1], 1e-6, 'Draco normals');
    assertArrayClose(result.draco?.indices, [0, 1, 2], 0, 'Draco indices');
    assertArrayClose(
      result.meshopt?.positions,
      [0, 0, 0, 1, 0, 0, 0, 1, 0],
      1e-6,
      'meshopt positions',
    );
    assertArrayClose(result.meshopt?.indices, [0, 1, 2], 0, 'meshopt indices');
    assertArrayClose(
      result.meshopt?.colors,
      [1, 0, 0.5, 1, 0, 1, 0.25, 1, 0.125, 0.375, 0.875, 1],
      2 / 255,
      'meshopt colors',
    );

    const successfulResponseNames = new Set(
      responses
        .filter(({ status }) => status >= 200 && status < 300)
        .map(({ url }) => basename(new URL(url).pathname)),
    );
    const wasmName = basename(wasmFiles[0]);
    const meshoptChunkName = basename(meshoptChunks[0].file);
    if (!successfulResponseNames.has(wasmName)) {
      throw new Error(`Chromium did not fetch emitted Draco WASM ${wasmName}.`);
    }
    if (!successfulResponseNames.has(meshoptChunkName)) {
      throw new Error(`Chromium did not fetch decoder-only meshopt chunk ${meshoptChunkName}.`);
    }

    console.log(
      JSON.stringify(
        {
          verdict: 'PASS',
          harness: 'gltf-browser-proof:builtin-compression',
          wasmAsset: wasmName,
          meshoptDecoderChunk: meshoptChunkName,
          hostDecoderHooks: false,
          dracoIndices: result.draco.indices,
          meshoptIndices: result.meshopt.indices,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser?.close();
    await server.close();
  }
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}

function run(command, args, cwd) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolvePromise({ stdout, stderr });
      else reject(new Error(`${command} exited ${code}:\n${stdout}\n${stderr}`));
    });
  });
}

async function listFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await listFiles(path)));
    else if (entry.isFile()) files.push(path);
  }
  return files;
}

async function serveDirectory(directory) {
  const root = normalize(resolve(directory));
  const server = createServer(async (request, response) => {
    try {
      const requestPath = decodeURIComponent(
        new URL(request.url ?? '/', 'http://localhost').pathname,
      );
      const relative = requestPath === '/' ? 'index.html' : requestPath.slice(1);
      const file = normalize(resolve(root, relative));
      if (file !== root && !file.startsWith(`${root}${sep}`)) {
        response.writeHead(403).end('Forbidden');
        return;
      }
      const bytes = await readFile(file);
      response.writeHead(200, { 'content-type': mimeType(file) });
      response.end(bytes);
    } catch {
      response.writeHead(404).end('Not found');
    }
  });
  await new Promise((resolvePromise, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolvePromise);
  });
  const address = server.address();
  if (address == null || typeof address === 'string')
    throw new Error('Could not resolve proof server port.');
  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () =>
      new Promise((resolvePromise, reject) => {
        server.close((error) => (error ? reject(error) : resolvePromise()));
      }),
  };
}

function mimeType(file) {
  switch (extname(file)) {
    case '.html':
      return 'text/html; charset=utf-8';
    case '.js':
      return 'text/javascript; charset=utf-8';
    case '.wasm':
      return 'application/wasm';
    default:
      return 'application/octet-stream';
  }
}

function assertArrayClose(actual, expected, tolerance, label) {
  if (!Array.isArray(actual) || actual.length !== expected.length) {
    throw new Error(`${label} length mismatch: ${JSON.stringify(actual)}.`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (Math.abs(actual[index] - expected[index]) > tolerance) {
      throw new Error(
        `${label}[${index}] was ${actual[index]}, expected ${expected[index]} ± ${tolerance}.`,
      );
    }
  }
}
