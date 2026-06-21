#!/usr/bin/env node
/**
 * Wrapper for pt-webgpu radiometric A/B recaptures.
 *
 * The underlying A/B scripts require a native-Deno WebGPU adapter with the
 * pt-webgpu full-tier storage limits. Some WSL hosts can run the Node/DZN
 * behavioral gates but still expose only a low-limit adapter to native Deno.
 * This wrapper records that host boundary as machine-readable evidence instead
 * of leaving repeated stack traces as the only status.
 */
import { spawnSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'pt-ab-host-status.json');
const DEFAULT_TIMEOUT_MS = 900_000;

const CASES = {
  sppm: {
    script: 'tools/radiometric-ab/ab-sppm.mjs',
    resultFile: 'tools/radiometric-ab/results-sppm.json',
  },
  bdpt: {
    script: 'tools/radiometric-ab/ab-bdpt.mjs',
    resultFile: 'tools/radiometric-ab/results-bdpt.json',
  },
  'restir-pt': {
    script: 'tools/radiometric-ab/ab-restir-pt.mjs',
    resultFile: 'tools/radiometric-ab/results-restir-pt.json',
  },
  sobol: {
    script: 'tools/radiometric-ab/ab-sobol.mjs',
    resultFile: 'tools/radiometric-ab/results-sobol.json',
  },
};

function parseTimeoutMs(raw) {
  if (raw == null || raw === '') return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.max(1, Math.trunc(parsed));
}

function selectedCaseIds() {
  const raw = process.env.VITRUM_PT_RADIOMETRIC_AB_CASES ?? process.argv.slice(2).join(',');
  if (raw == null || raw.trim() === '') return Object.keys(CASES);
  const ids = raw.split(',').map((part) => part.trim()).filter(Boolean);
  const invalid = ids.filter((id) => CASES[id] == null);
  if (invalid.length > 0) {
    throw new Error(`Unknown pt radiometric A/B case(s): ${invalid.join(', ')}`);
  }
  return ids;
}

function classifyHostBoundary(result, output, timeoutMs) {
  if (result.error?.code === 'ETIMEDOUT') {
    return {
      code: 'pt-radiometric-ab-timeout',
      message: `The pt radiometric A/B case exceeded the host timeout (${timeoutMs}ms).`,
    };
  }
  if (output.includes('traceTier=full requested but adapter reports')) {
    return {
      code: 'pt-radiometric-full-tier-unavailable',
      message:
        'Native Deno WebGPU resolved a low-limit adapter; pt-webgpu full-tier radiometric recapture cannot run on this host.',
    };
  }
  if (output.includes('No WebGPU adapter')) {
    return {
      code: 'pt-radiometric-no-adapter',
      message: 'Native Deno WebGPU did not expose an adapter for the pt radiometric A/B harness.',
    };
  }
  if (output.includes('Deno has panicked')) {
    return {
      code: 'pt-radiometric-deno-wgpu-panic',
      message: 'Native Deno WebGPU panicked before the pt radiometric A/B harness produced a verdict.',
    };
  }
  return null;
}

const timeoutMs = parseTimeoutMs(process.env.VITRUM_PT_RADIOMETRIC_AB_TIMEOUT_MS);
const selected = selectedCaseIds();
const caseStatuses = [];

for (const id of selected) {
  const entry = CASES[id];
  const denoArgs = [
    'run',
    '--unstable-webgpu',
    '--sloppy-imports',
    '--allow-read',
    '--allow-env',
    '--allow-write',
    entry.script,
  ];
  console.log(`=== pt radiometric A/B wrapper: ${id} ===`);
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

  const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  const hostBoundary = classifyHostBoundary(result, output, timeoutMs);
  const status = result.status === 0
    ? 'PASS'
    : hostBoundary != null
      ? 'HOST-BLOCKED'
      : 'FAIL';
  caseStatuses.push({
    id,
    status,
    script: entry.script,
    resultFile: entry.resultFile,
    exitStatus: result.status,
    signal: result.signal,
    reason: hostBoundary,
  });
}

const hasFail = caseStatuses.some((entry) => entry.status === 'FAIL');
const hasPass = caseStatuses.some((entry) => entry.status === 'PASS');
const hasBlocked = caseStatuses.some((entry) => entry.status === 'HOST-BLOCKED');
const verdict = hasFail ? 'FAIL' : hasBlocked ? (hasPass ? 'PASS-PARTIAL' : 'HOST-BLOCKED') : 'PASS';

const status = {
  generatedAt: new Date().toISOString(),
  harness: 'pt-radiometric-ab',
  verdict,
  command: 'deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write tools/radiometric-ab/ab-*.mjs',
  selectedCases: selected,
  timeoutMs,
  icd: process.env.VK_ICD_FILENAMES ?? null,
  cases: caseStatuses,
  preservedResultFiles: Object.values(CASES).map((entry) => entry.resultFile),
  reason: hasFail
    ? {
      code: 'pt-radiometric-ab-failed',
      message: 'At least one pt radiometric A/B case exited nonzero for a reason other than a known host boundary.',
    }
    : hasBlocked
      ? {
        code: hasPass ? 'pt-radiometric-ab-partial-host-blocked' : 'pt-radiometric-ab-host-blocked',
        message: 'At least one pt radiometric A/B recapture could not run on this native-Deno host.',
      }
      : {
        code: 'pt-radiometric-ab-complete',
        message: 'All selected pt radiometric A/B recaptures completed on this host.',
      },
  nextSteps: hasBlocked
    ? [
      'Run these A/B scripts on a native-Deno full-tier adapter or browser/real-adapter validation lane.',
      'Do not promote SPPM, BDPT multi-vertex, ReSTIR-PT specialty, or Sobol default rows from this host-blocked recapture.',
      'Use the preserved result JSONs only as committed historical evidence until a full recapture replaces them.',
    ]
    : [],
};

writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);

if (hasFail) process.exit(1);
if (hasBlocked) process.exit(2);
process.exit(0);
