/**
 * Node port of `scripts/capture-cornell-suite.sh` for Windows-host Playwright.
 *
 * Usage:
 *   node capture-cornell-scenarios.mjs --out <dir> [--only layered,parity] [--bdpt] [--quick]
 *
 * Env:
 *   VITRUM_CORNELL_SKIP_VITE   1 — do not start vite (use VITRUM_CAPTURE_URL base)
 *   VITRUM_CAPTURE_URL         base URL (optional ?vitrumBdpt=1…)
 *   VITRUM_CORNELL_DEV_PORT    default 5173
 *   VITRUM_BDPT_MIN_PNG_BYTES  fail capture when output PNG is smaller (default 0 = off)
 */

import { access, mkdir, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const adapter = resolve(here, 'capture-adapter-playwright.mjs');

const args = process.argv.slice(2);
let outDir = resolve(repoRoot, `tools/reference-renders/post-sweep-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`);
let scenarios = ['glass', 'caustic', 'spectral', 'layered', 'sss', 'parity'];
let bdpt = false;
let width = 1280;
let height = 720;
let spp = 512;
let bounces = 8;
let seed = 12345;
let timeoutMs = 120_000;

for (let i = 0; i < args.length; i += 1) {
  const a = args[i];
  if (a === '--out' && args[i + 1]) {
    outDir = resolve(repoRoot, args[++i]);
  } else if (a === '--only' && args[i + 1]) {
    scenarios = args[++i].split(',').map((s) => s.trim()).filter(Boolean);
  } else if (a === '--bdpt') {
    bdpt = true;
  } else if (a === '--quick') {
    width = 512;
    height = 512;
    spp = 64;
    bounces = 4;
  } else if (a === '--help' || a === '-h') {
    console.log('Usage: node capture-cornell-scenarios.mjs --out <dir> [--only a,b] [--bdpt] [--quick]');
    process.exit(0);
  }
}

const cornellPort = process.env.VITRUM_CORNELL_DEV_PORT ?? '5173';
let baseUrl = process.env.VITRUM_CAPTURE_URL ?? `http://127.0.0.1:${cornellPort}/`;

if (bdpt) {
  spp *= 2;
  const u = new URL(baseUrl);
  if (!u.searchParams.has('vitrumBdpt')) u.searchParams.set('vitrumBdpt', '1');
  if (!u.searchParams.has('vitrumSpf')) u.searchParams.set('vitrumSpf', '16');
  baseUrl = u.toString();
}

const minPngBytes = Number(process.env.VITRUM_BDPT_MIN_PNG_BYTES ?? 0);

function runAdapter(env) {
  return new Promise((resolveRun) => {
    const child = spawn(process.execPath, [adapter], {
      cwd: repoRoot,
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    child.stdout?.on('data', (c) => {
      stdout += String(c);
    });
    child.stderr?.on('data', (c) => {
      stdout += String(c);
    });
    child.on('close', (code) => {
      resolveRun({ code: code ?? 1, stdout: stdout.trim() });
    });
  });
}

await mkdir(outDir, { recursive: true });

const summary = [];
let failed = false;

for (const short of scenarios) {
  const scenarioId = `cornell-${short}`;
  const outPng = resolve(
    outDir,
    bdpt ? `${scenarioId}-bdpt.png` : `${scenarioId}.png`,
  );
  console.log(`[capture-cornell] → ${scenarioId} (${width}x${height}, ${spp} spp, ${bounces} bounces)`);
  const started = Date.now();
  const { code, stdout } = await runAdapter({
    VITRUM_OUTPUT_PNG: outPng,
    VITRUM_SCENARIO_ID: scenarioId,
    VITRUM_SEED: String(seed),
    VITRUM_WIDTH: String(width),
    VITRUM_HEIGHT: String(height),
    VITRUM_BOUNCES: String(bounces),
    VITRUM_SPP: String(spp),
    VITRUM_CAPTURE_TIMEOUT_MS: String(timeoutMs),
    VITRUM_CAPTURE_URL: baseUrl,
  });
  const elapsed = Math.round((Date.now() - started) / 1000);
  let ok = code === 0;
  let size = 0;
  try {
    await access(outPng);
    size = (await stat(outPng)).size;
    if (minPngBytes > 0 && size < minPngBytes) {
      console.warn(
        `[capture-cornell] ${scenarioId}: PNG only ${size} bytes (< ${minPngBytes})`,
      );
      ok = false;
    }
  } catch {
    ok = false;
  }
  if (!ok) failed = true;
  const mark = ok ? '✓' : '✗';
  summary.push(`${mark} ${scenarioId}  (${elapsed}s, ${size} bytes)  → ${outPng}`);
  if (!ok) summary.push(`    telemetry: ${stdout.split('\n').pop() ?? stdout}`);
}

console.log(`\n[capture-cornell] Done. Output dir: ${outDir}`);
for (const line of summary) console.log(`  ${line}`);
process.exit(failed ? 1 : 0);
