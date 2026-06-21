import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const validateGpuScript = join(repoRoot, 'scripts', 'validate-gpu.mjs');

async function makeFakeWslGpu(runnerSource) {
  const dir = await mkdtemp(join(tmpdir(), 'vitrum-validate-gpu-'));
  await mkdir(join(dir, 'scripts'), { recursive: true });
  await writeFile(join(dir, 'package.json'), '{"name":"fake-wsl-gpu","type":"module"}\n');
  await writeFile(join(dir, 'scripts', 't1-smoke.mjs'), runnerSource);
  return dir;
}

function runValidate(args, wslGpuDir) {
  return new Promise((resolveResult) => {
    const child = spawn(process.execPath, [validateGpuScript, ...args], {
      cwd: repoRoot,
      env: {
        ...process.env,
        WSL_GPU_DIR: wslGpuDir,
        VITRUM_VALIDATE_GPU_TIMEOUT_MS: '',
      },
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

test('validate-gpu smoke passes through a successful runner', async () => {
  const fakeWslGpu = await makeFakeWslGpu(`
    console.error('[fake-wsl-gpu] smoke ok');
    process.exit(0);
  `);

  const result = await runValidate(['--smoke', '--timeout-ms=500'], fakeWslGpu);

  assert.equal(result.status, 0);
  assert.match(result.stderr, /invoking T1 smoke/);
  assert.match(result.stderr, /smoke ok/);
  assert.doesNotMatch(result.stderr, /timed out/);
});

test('validate-gpu smoke default timeout accommodates current dzn T1 runtime', async () => {
  const fakeWslGpu = await makeFakeWslGpu(`
    console.error('[fake-wsl-gpu] smoke ok');
    process.exit(0);
  `);

  const result = await runValidate(['--smoke', '--warn-only'], fakeWslGpu);

  assert.equal(result.status, 0);
  assert.match(result.stderr, /runner timeout: 300000ms \(warn-only\)/);
  assert.match(result.stderr, /smoke ok/);
});

test('validate-gpu smoke timeout exits nonzero outside warn-only mode', async () => {
  const fakeWslGpu = await makeFakeWslGpu(`
    console.error('[fake-wsl-gpu] hanging');
    setInterval(() => {}, 1000);
  `);

  const result = await runValidate(['--smoke', '--timeout-ms=50'], fakeWslGpu);

  assert.equal(result.status, 124);
  assert.match(result.stderr, /runner timed out after 50ms/);
  assert.match(result.stderr, /wsl-gpu runner timed out after 50ms/);
});

test('validate-gpu warn-only timeout exits zero and reports the warning', async () => {
  const fakeWslGpu = await makeFakeWslGpu(`
    console.error('[fake-wsl-gpu] hanging');
    setInterval(() => {}, 1000);
  `);

  const result = await runValidate(['--smoke', '--warn-only', '--timeout-ms=50'], fakeWslGpu);

  assert.equal(result.status, 0);
  assert.match(result.stderr, /runner timed out after 50ms/);
  assert.match(result.stderr, /warn-only — push NOT blocked/);
});
