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
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  walkaroundResultProvenance,
  walkaroundStatusProvenance,
} from './resultProvenance.mjs';
import {
  validateWalkaroundRunResult,
  WalkaroundRunValidationError,
} from './walkaroundRunValidation.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const repoRootUrl = pathToFileURL(`${repoRoot}/`).href;
const DEFAULT_TIMEOUT_MS = 180_000;
const args = new Set(process.argv.slice(2));
const glossySpp64 = args.has('--glossy-spp64');
const allSpp64 = args.has('--all-spp64');
if (glossySpp64 && allSpp64) {
  console.error('[walkaround-ab] choose only one of --glossy-spp64 or --all-spp64.');
  process.exit(2);
}
const statusPath = resolveFromRepo(
  process.env.VITRUM_WALKAROUND_AB_STATUS_PATH,
  glossySpp64
    ? 'tools/radiometric-ab/walkaround-ab-glossy-spp64-status.json'
    : allSpp64
      ? 'tools/radiometric-ab/walkaround-ab-all-spp64-status.json'
      : 'tools/radiometric-ab/walkaround-ab-host-status.json',
);
const resultPath = resolveFromRepo(
  process.env.VITRUM_WALKAROUND_AB_OUTPUT_PATH,
  glossySpp64
    ? 'tools/radiometric-ab/walkaround-ab-glossy-spp64.json'
    : allSpp64
      ? 'tools/radiometric-ab/walkaround-ab-all-spp64.json'
      : 'tools/radiometric-ab/walkaround-ab-results.json',
);
const WALKAROUND_AB_CASE_IDS = ['a8', 'sun', 'glass', 'glossy'];

function resolveFromRepo(raw, fallbackRelative) {
  if (raw == null || raw === '') return resolve(repoRoot, fallbackRelative);
  return resolve(repoRoot, raw);
}

function repoRelative(path) {
  return relative(repoRoot, path).replaceAll('\\', '/');
}

function captureResultFileState() {
  try {
    const text = readFileSync(resultPath, 'utf8');
    const stat = statSync(resultPath);
    return { text, mtimeMs: stat.mtimeMs, size: stat.size };
  } catch {
    return null;
  }
}

function parseTimeoutMs(raw) {
  if (raw == null || raw === '') return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.max(1, Math.trunc(parsed));
}

function expectedCaseIdsForStatus() {
  if (selectedCases == null || selectedCases === '') return WALKAROUND_AB_CASE_IDS;
  return String(selectedCases)
    .split(',')
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);
}

function failClosedResultStatus(generatedAt, code, message) {
  const status = {
    generatedAt,
    harness: 'walkaround-ab',
    verdict: 'FAIL',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    timeoutMs,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    renderConfig,
    exitStatus: result.status,
    signal: result.signal,
    preservedResultFile: resultFile,
    reason: {
      code,
      message,
    },
    nextSteps: [
      'Re-run the walkaround radiometric A/B wrapper so it can write a complete result artifact.',
      'Do not treat this malformed capture as a regression result.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[walkaround-ab] FAIL status written to ${statusPath}: ${message}`);
  process.exit(1);
}

async function stampWalkaroundResultProvenance(payload) {
  payload.provenance = await walkaroundResultProvenance(repoRootUrl, resultFile);
  writeFileSync(resultPath, `${JSON.stringify(payload, null, 2)}\n`);
  return payload;
}

function readWalkaroundResultsForStatus(generatedAt) {
  const afterState = captureResultFileState();
  if (afterState == null) {
    failClosedResultStatus(
      generatedAt,
      'walkaround-ab-missing-result',
      'The native WebGPU harness exited 0 but did not write a readable result artifact.',
    );
  }
  let payload;
  try {
    payload = JSON.parse(afterState.text);
  } catch (err) {
    failClosedResultStatus(
      generatedAt,
      'walkaround-ab-invalid-result',
      `The native WebGPU harness exited 0 but wrote invalid result JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  try {
    return {
      payload,
      ...validateWalkaroundRunResult(payload, {
        expectedCaseIds: expectedCaseIdsForStatus(),
        beforeState: resultStateBeforeRun,
        afterState,
        startedAtMs: runStartedAtMs,
      }),
    };
  } catch (err) {
    failClosedResultStatus(
      generatedAt,
      err instanceof WalkaroundRunValidationError
        ? err.code
        : 'walkaround-ab-invalid-result',
      err instanceof Error ? err.message : String(err),
    );
  }
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
const selectedCases = process.env.VITRUM_WALKAROUND_AB_CASES ?? (glossySpp64 ? 'glossy' : null);
const resultFile = repoRelative(resultPath);
const renderConfig = {
  width: process.env.VITRUM_WALKAROUND_AB_WIDTH ?? '128',
  height: process.env.VITRUM_WALKAROUND_AB_HEIGHT ?? '128',
  spp: process.env.VITRUM_WALKAROUND_AB_SPP ?? (glossySpp64 || allSpp64 ? '64' : '16'),
  qualityProfile: process.env.VITRUM_WALKAROUND_AB_PROFILE ?? (glossySpp64 ? 'glossy-spp64' : allSpp64 ? 'all-spp64' : null),
};
const denoEnv = {
  ...process.env,
  VITRUM_WALKAROUND_AB_OUTPUT_PATH: resultPath,
  VITRUM_WALKAROUND_AB_WIDTH: renderConfig.width,
  VITRUM_WALKAROUND_AB_HEIGHT: renderConfig.height,
  VITRUM_WALKAROUND_AB_SPP: renderConfig.spp,
};
if (selectedCases != null) denoEnv.VITRUM_WALKAROUND_AB_CASES = selectedCases;
if (renderConfig.qualityProfile != null) denoEnv.VITRUM_WALKAROUND_AB_PROFILE = renderConfig.qualityProfile;

const resultStateBeforeRun = captureResultFileState();
const runStartedAtMs = Date.now();
const result = spawnSync('deno', denoArgs, {
  cwd: repoRoot,
  env: denoEnv,
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
    provenance: await walkaroundStatusProvenance(repoRootUrl, repoRelative(statusPath), resultFile),
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    timeoutMs,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    renderConfig,
    exitStatus: result.status,
    signal: result.signal,
    reason: {
      code: 'walkaround-ab-timeout',
      message:
        'The walkaround radiometric A/B harness exceeded the host timeout before producing a verdict.',
    },
    preservedResultFile: resultFile,
    nextSteps: [
      'Re-run with VITRUM_WALKAROUND_AB_TIMEOUT_MS set higher if the host is merely slow.',
      'Run the same harness in the browser/real-adapter validation lane if native Deno remains blocked.',
      'Do not treat this timed-out run as a regression result.',
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
  const generatedAt = new Date().toISOString();
  const validated = readWalkaroundResultsForStatus(generatedAt);
  await stampWalkaroundResultProvenance(validated.payload);
  const { caseVerdicts, partialCaseIds, partial } = validated;
  const status = {
    provenance: await walkaroundStatusProvenance(repoRootUrl, repoRelative(statusPath), resultFile),
    generatedAt,
    harness: 'walkaround-ab',
    verdict: partial ? 'PASS-PARTIAL' : 'PASS',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    timeoutMs,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    renderConfig,
    exitStatus: result.status,
    signal: result.signal,
    resultFile,
    caseVerdicts,
    reason: partial
      ? {
        code: 'walkaround-ab-partial-regression-result',
        message:
          'The native WebGPU harness ran to completion, but at least one case did not meet every regression threshold.',
      }
      : {
        code: 'walkaround-ab-complete',
        message: 'The native WebGPU harness ran to completion and all checked cases met their full regression thresholds.',
      },
    nextSteps: partial
      ? [
        `Investigate the cases that did not meet every threshold (${partialCaseIds.join(', ')}).`,
        'Use higher-SPP, HDR, browser/real-adapter, or case-specific reference captures to isolate the regression.',
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
    provenance: await walkaroundStatusProvenance(repoRootUrl, repoRelative(statusPath), resultFile),
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    selectedCases,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    renderConfig,
    exitStatus: result.status,
    signal: result.signal,
    timeoutMs,
    reason: {
      code: 'deno-wgpu-hal-gles-index-oob',
      location: 'wgpu-hal-28.0.0/src/gles/command.rs:771:21',
      message:
        'Deno native WebGPU panicked before the walkaround radiometric A/B harness could produce a verdict.',
    },
    preservedResultFile: resultFile,
    nextSteps: [
      'Run the same harness in the browser/real-adapter validation lane.',
      'Re-run this wrapper after the Deno native WebGPU panic is fixed or the host path changes.',
      'Do not treat this blocked native-Deno run as a regression result.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[walkaround-ab] HOST-BLOCKED status written to ${statusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);
