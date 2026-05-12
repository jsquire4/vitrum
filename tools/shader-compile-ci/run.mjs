#!/usr/bin/env node
// Shader-compile CI guard.
//
// Loads each cornell-box scenario in headless Chromium with the smallest
// possible workload (64×64, SPP=1, 1 bounce), waits for VITRUM_CAPTURE_READY
// (which is set after the first sample completes — i.e. the shader compiled
// AND linked AND ran AND wrote to the canvas), and asserts the page produced
// no shader-error / WebGL-error console entries and no uncaught exceptions.
//
// Catches: a) the fork pulling in a uniform name that PhysicalPathTracingMaterial
// no longer declares, b) WGSL/GLSL syntax regressions, c) bind-group-layout
// mismatches that fail at compile rather than at draw, d) accidental opt-in
// to a WebGPU optional feature (e.g. r16float storage), and e) any uncaught
// JS that fires during scene/material setup.
//
// Usage:  npm run shader-compile-ci  (from repo root or from this dir)
// Env:    VITRUM_SHADER_CI_PORT  (default 5174, picked to avoid colliding
//                                  with a developer's running 5173 vite)
//         VITRUM_SHADER_CI_TIMEOUT_MS  (per-scenario; default 60000)
//         VITRUM_SHADER_CI_HEADFUL=1   (debug: run with a visible browser)
//         VITRUM_SHADER_CI_SCENARIOS=cornell-glass,cornell-spectral
//                                       (debug: run only the listed scenarios)

import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const cornellBoxDir = resolve(repoRoot, 'examples', 'cornell-box');

// Canonical scenario list. Mirrors the alias map in cornell-box/src/main.ts.
// Each scenario hits a different fork shader-define permutation
// (transmission, dispersion, clearcoat, subsurface, etc.), so loading all
// of them validates the union of compile paths the library actually ships.
const ALL_SCENARIOS = [
  'cornell-box',
  'cornell-glass',
  'cornell-caustic',
  'cornell-spectral',
  'cornell-layered',
  'cornell-sss',
  'cornell-parity',
];

const PORT = Number(process.env.VITRUM_SHADER_CI_PORT ?? '5174');
const PER_SCENARIO_TIMEOUT_MS = Number(process.env.VITRUM_SHADER_CI_TIMEOUT_MS ?? '60000');
const HEADFUL = process.env.VITRUM_SHADER_CI_HEADFUL === '1';
const SCENARIOS = (() => {
  const env = process.env.VITRUM_SHADER_CI_SCENARIOS;
  if (env == null || env.trim().length === 0) return ALL_SCENARIOS;
  const requested = env.split(',').map((s) => s.trim()).filter((s) => s.length > 0);
  const unknown = requested.filter((s) => !ALL_SCENARIOS.includes(s));
  if (unknown.length > 0) {
    console.error(`shader-compile-ci: unknown scenarios in VITRUM_SHADER_CI_SCENARIOS: ${unknown.join(', ')}`);
    console.error(`                    valid: ${ALL_SCENARIOS.join(', ')}`);
    process.exit(2);
  }
  return requested;
})();

// Patterns that indicate a real shader / WebGL / WebGPU compile or link
// failure. We deliberately do NOT match generic "warning" / "deprecated"
// messages — those are noise from third-party libs and not regressions.
//
// three.js writes shader-compile failures via console.error with a header
// like "THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false".
// Browsers also surface "WebGL: INVALID_OPERATION" and "WebGL: ERROR" when
// a draw uses a program that didn't link. WGSL/WebGPU compile failures
// surface as "Tint WGSL reader failure" or "Internal compiler error".
const SHADER_ERROR_PATTERNS = [
  /THREE\.WebGL(?:Program|Shader)[^\n]*Shader Error/i,
  /Shader compilation failed/i,
  /Shader linking failed/i,
  /Failed to (?:compile|link) (?:the )?(?:shader|program)/i,
  /WebGL: INVALID/i,
  /WebGL: ERROR/i,
  /GL_INVALID/i,
  /Tint WGSL reader/i,
  /Internal compiler error/i,
  /createShaderModule.*error/i,
];

function makeUrl(scenarioId) {
  const u = new URL(`http://127.0.0.1:${PORT}/`);
  u.searchParams.set('vitrumScenario', scenarioId);
  u.searchParams.set('vitrumSeed', '12345');
  u.searchParams.set('vitrumWidth', '64');
  u.searchParams.set('vitrumHeight', '64');
  u.searchParams.set('vitrumBounces', '1');
  u.searchParams.set('vitrumSpp', '1');
  u.searchParams.set('vitrumSpf', '1');
  u.searchParams.set('vitrumAutoStart', '1');
  return u.toString();
}

async function waitForVite(timeoutMs) {
  const url = `http://127.0.0.1:${PORT}/`;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url, { method: 'GET' });
      if (r.ok) return true;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

function startVite() {
  const child = spawn(
    'npx',
    ['vite', '--port', String(PORT), '--host', '127.0.0.1', '--strictPort'],
    {
      cwd: cornellBoxDir,
      env: { ...process.env, FORCE_COLOR: '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  let viteOutput = '';
  child.stdout.on('data', (b) => { viteOutput += b.toString(); });
  child.stderr.on('data', (b) => { viteOutput += b.toString(); });
  child.on('exit', (code, signal) => {
    if (code != null && code !== 0) {
      console.error(`vite exited prematurely (code ${code}, signal ${signal}). Last output:\n${viteOutput.slice(-1000)}`);
    }
  });
  return { child, getOutput: () => viteOutput };
}

async function runScenario(browser, scenarioId) {
  const ctx = await browser.newContext({ viewport: { width: 200, height: 200 } });
  const page = await ctx.newPage();

  const errors = [];
  const consoleLog = [];
  page.on('console', (msg) => {
    const text = msg.text();
    consoleLog.push(`[${msg.type()}] ${text}`);
    if (msg.type() === 'error' || msg.type() === 'warning') {
      for (const pat of SHADER_ERROR_PATTERNS) {
        if (pat.test(text)) {
          errors.push({ kind: 'console', pattern: pat.source, text });
          break;
        }
      }
    }
  });
  page.on('pageerror', (err) => {
    errors.push({ kind: 'pageerror', text: err.message ?? String(err) });
  });

  let ready = false;
  let timedOut = false;
  try {
    await page.goto(makeUrl(scenarioId), { waitUntil: 'load', timeout: PER_SCENARIO_TIMEOUT_MS });
    await page
      .waitForFunction(() => globalThis.VITRUM_CAPTURE_READY === true, null, {
        timeout: PER_SCENARIO_TIMEOUT_MS,
      });
    ready = true;
  } catch (err) {
    timedOut = true;
    errors.push({
      kind: 'timeout',
      text: `did not reach VITRUM_CAPTURE_READY within ${PER_SCENARIO_TIMEOUT_MS} ms: ${err?.message ?? err}`,
    });
  } finally {
    await ctx.close();
  }

  return { scenarioId, ready, timedOut, errors, consoleTail: consoleLog.slice(-30) };
}

async function main() {
  const startedAt = Date.now();
  console.log(`shader-compile-ci: starting vite on :${PORT} (cornell-box)`);
  const { child: vite, getOutput } = startVite();

  process.on('SIGINT', () => { try { vite.kill('SIGTERM'); } catch {} process.exit(130); });
  process.on('SIGTERM', () => { try { vite.kill('SIGTERM'); } catch {} process.exit(143); });

  const viteUp = await waitForVite(30000);
  if (!viteUp) {
    console.error('shader-compile-ci: vite never came up. Last output:');
    console.error(getOutput().slice(-1500));
    try { vite.kill('SIGTERM'); } catch {}
    process.exit(2);
  }
  console.log(`shader-compile-ci: vite ready, launching headless Chromium (${SCENARIOS.length} scenario(s))`);

  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch (err) {
    console.error('shader-compile-ci: playwright is not installed in this workspace.');
    console.error('                   try: npm install --workspace=@vitrum/shader-compile-ci');
    console.error(String(err));
    try { vite.kill('SIGTERM'); } catch {}
    process.exit(3);
  }

  const browser = await chromium.launch({
    headless: !HEADFUL,
    args: [
      '--disable-dev-shm-usage',
      // Force WebGL2 software rasterisation when no GPU is available.
      // Modern Chromium falls back to SwiftShader automatically in CI, but
      // explicit is safer across runner images.
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
    ],
  });

  const results = [];
  try {
    for (const scenarioId of SCENARIOS) {
      const t0 = Date.now();
      const result = await runScenario(browser, scenarioId);
      const elapsed = Date.now() - t0;
      const status = result.errors.length === 0 && result.ready
        ? `✓ OK (${elapsed} ms)`
        : `✗ FAIL (${elapsed} ms)`;
      console.log(`  ${scenarioId.padEnd(18)} ${status}`);
      results.push({ ...result, elapsedMs: elapsed });
    }
  } finally {
    await browser.close().catch(() => {});
    try { vite.kill('SIGTERM'); } catch {}
  }

  const failed = results.filter((r) => r.errors.length > 0 || !r.ready);
  if (failed.length > 0) {
    console.error('');
    console.error(`shader-compile-ci: ${failed.length}/${results.length} scenario(s) failed`);
    for (const r of failed) {
      console.error(`\nscenario: ${r.scenarioId}`);
      console.error(`  ready: ${r.ready}, timedOut: ${r.timedOut}, errors: ${r.errors.length}`);
      for (const e of r.errors) {
        console.error(`  - [${e.kind}] ${e.text}`);
      }
      if (r.consoleTail.length > 0) {
        console.error('  recent console output:');
        for (const line of r.consoleTail) {
          console.error(`    ${line}`);
        }
      }
    }
    process.exit(1);
  }

  console.log(`\nshader-compile-ci: all ${results.length} scenario(s) compiled cleanly in ${Date.now() - startedAt} ms`);
  process.exit(0);
}

main().catch((err) => {
  console.error('shader-compile-ci: unexpected failure');
  console.error(err);
  process.exit(2);
});
