#!/usr/bin/env node
/**
 * Wrapper for the walkaround-hybrid radiometric A/B harness.
 *
 * The harness itself runs under Deno native WebGPU. In the current WSL
 * validation host, Deno 2.8.1 can panic inside wgpu-hal before the harness is
 * able to write a normal verdict. This wrapper keeps that failure explicit and
 * machine-readable instead of leaving future runs as an unclassified crash.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'walkaround-ab-host-status.json');
const resultPath = resolve(scriptDir, 'walkaround-ab-results.json');
const DEFAULT_TIMEOUT_MS = 180_000;

function parseTimeoutMs(raw) {
  if (raw == null || raw === '') return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.max(1, Math.trunc(parsed));
}

const denoArgs = [
  'run',
  '--unstable-webgpu',
  '--sloppy-imports',
  '--allow-read',
  '--allow-env',
  '--allow-write',
  'tools/radiometric-ab/walkaround-ab.mjs',
];
const timeoutMs = parseTimeoutMs(process.env.VITRUM_WALKAROUND_AB_TIMEOUT_MS);
const selectedCases = process.env.VITRUM_WALKAROUND_AB_CASES ?? null;

const result = spawnSync('deno', denoArgs, {
  cwd: repoRoot,
  env: process.env,
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
  timeout: timeoutMs,
  killSignal: 'SIGTERM',
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

const timedOut = result.error?.code === 'ETIMEDOUT';
if (timedOut) {
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    timeoutMs,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    reason: {
      code: 'walkaround-ab-timeout',
      message:
        'The walkaround radiometric A/B harness exceeded the host timeout before producing a verdict.',
    },
    preservedResultFile: 'tools/radiometric-ab/walkaround-ab-results.json',
    nextSteps: [
      'Re-run with VITRUM_WALKAROUND_AB_TIMEOUT_MS set higher if the host is merely slow.',
      'Run the same harness in the browser/real-adapter validation lane if native Deno remains blocked.',
      'Do not promote GRIS, rich-material GI, glass, or glossy walkaround rows from this timed-out run.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[walkaround-ab] HOST-BLOCKED timeout status written to ${statusPath}`);
  process.exit(2);
}

if (result.error) {
  throw result.error;
}
if (result.status === 0) {
  let walkaroundResults = null;
  try {
    walkaroundResults = JSON.parse(readFileSync(resultPath, 'utf8'));
  } catch {
    // The Deno harness is expected to write this file before exiting 0; keep
    // the success status truthful even if a future harness changes that shape.
  }
  const caseVerdicts = walkaroundResults == null
    ? {}
    : Object.fromEntries(Object.entries(walkaroundResults).map(([key, value]) => [
      key,
      value?.verdict ?? 'UNKNOWN',
    ]));
  const partial = Object.values(caseVerdicts).some((verdict) =>
    verdict === 'PASS-PARTIAL' ||
    verdict === 'PASS-WEAK' ||
    verdict === 'FINDING' ||
    verdict === 'SMALL' ||
    verdict === 'MODERATE' ||
    verdict === 'SIGNIFICANT'
  );
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: partial ? 'PASS-PARTIAL' : 'PASS',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    timeoutMs,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    resultFile: 'tools/radiometric-ab/walkaround-ab-results.json',
    caseVerdicts,
    reason: partial
      ? {
        code: 'walkaround-ab-partial-proof',
        message:
          'The native WebGPU harness ran to completion, but at least one case is only a partial/weak proof.',
      }
      : {
        code: 'walkaround-ab-complete',
        message: 'The native WebGPU harness ran to completion and all checked cases met their full proof thresholds.',
      },
    nextSteps: partial
      ? [
        'Do not promote GRIS, rich-material GI, glass, or glossy walkaround rows from this partial proof alone.',
        'Use higher-SPP, HDR, browser/real-adapter, or case-specific reference captures before promotion.',
      ]
      : [],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  process.exit(0);
}

const combined = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
const knownDenoWgpuPanic =
  combined.includes('Deno has panicked') &&
  combined.includes('wgpu-hal-28.0.0/src/gles/command.rs:771:21');

if (knownDenoWgpuPanic) {
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    timeoutMs,
    reason: {
      code: 'deno-wgpu-hal-gles-index-oob',
      location: 'wgpu-hal-28.0.0/src/gles/command.rs:771:21',
      message:
        'Deno native WebGPU panicked before the walkaround radiometric A/B harness could produce a verdict.',
    },
    preservedResultFile: 'tools/radiometric-ab/walkaround-ab-results.json',
    nextSteps: [
      'Run the same harness in the browser/real-adapter validation lane.',
      'Re-run this wrapper after the Deno native WebGPU panic is fixed or the host path changes.',
      'Do not promote GRIS, rich-material GI, glass, or glossy walkaround rows from this blocked native-Deno run.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[walkaround-ab] HOST-BLOCKED status written to ${statusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);
