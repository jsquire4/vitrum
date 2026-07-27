#!/usr/bin/env node
/**
 * Wrapper for the stable pt-webgpu radiometric A/B suite.
 *
 * Each case must exit zero and replace its result JSON with a PASS artifact.
 * Adapter absence is recorded as HOST-BLOCKED in a local, gitignored status.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  ptRadiometricStatusProvenance,
  radiometricResultProvenance,
} from './resultProvenance.mjs';
import { RADIOMETRIC_AB_PROOFS } from './proofs.mjs';
import { validateRadiometricResult } from './resultValidation.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const repoRootUrl = pathToFileURL(`${repoRoot}/`).href;
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
const ALL_CASE_IDS = Object.keys(CASES);
const PROOFS_BY_ID = new Map(RADIOMETRIC_AB_PROOFS.map((proof) => [proof.id, proof]));

function parseTimeoutMs(raw) {
  if (raw == null || raw === '') return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.max(1, Math.trunc(parsed));
}

function selectedCaseIds() {
  const raw = process.env.VITRUM_PT_RADIOMETRIC_AB_CASES ?? process.argv.slice(2).join(',');
  if (raw == null || raw.trim() === '') return ALL_CASE_IDS;
  const ids = raw.split(',').map((part) => part.trim()).filter(Boolean);
  const invalid = ids.filter((id) => CASES[id] == null);
  if (invalid.length > 0) {
    throw new Error(`Unknown pt radiometric A/B case(s): ${invalid.join(', ')}`);
  }
  if (new Set(ids).size !== ids.length) {
    throw new Error('Duplicate pt radiometric A/B case ids are not allowed.');
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
      message: 'The native Deno adapter does not expose the required pt-webgpu full-tier limits.',
    };
  }
  if (output.includes('No WebGPU adapter')) {
    return {
      code: 'pt-radiometric-no-adapter',
      message: 'Native Deno WebGPU did not expose an adapter.',
    };
  }
  if (output.includes('Deno has panicked')) {
    return {
      code: 'pt-radiometric-deno-wgpu-panic',
      message: 'Native Deno WebGPU panicked before producing a verdict.',
    };
  }
  return null;
}

function resultArtifactSnapshot(resultFile) {
  try {
    const stat = statSync(resolve(repoRoot, resultFile));
    return { exists: stat.isFile(), mtimeMs: stat.mtimeMs };
  } catch {
    return { exists: false, mtimeMs: -Infinity };
  }
}

async function resultArtifactProblem(id, resultFile, before, captureStartedAtMs) {
  let payload;
  try {
    payload = readJson(resultFile);
  } catch (error) {
    return {
      code: 'pt-radiometric-ab-missing-result',
      message:
        `${id} exited 0 but did not write readable result JSON: ` +
        `${error instanceof Error ? error.message : String(error)}`,
    };
  }
  if (payload == null || typeof payload !== 'object' || Array.isArray(payload)) {
    return {
      code: 'pt-radiometric-ab-invalid-result',
      message: `${id} exited 0 but wrote a non-object result artifact.`,
    };
  }
  const proof = PROOFS_BY_ID.get(id);
  if (proof == null) {
    return {
      code: 'pt-radiometric-ab-invalid-result',
      message: `${id} has no pinned proof contract.`,
    };
  }
  try {
    const expectedProvenance = await radiometricResultProvenance(
      pathToFileURL(resolve(repoRoot, proof.scriptPath)).href,
      proof.scriptPath,
      proof.resultPath,
      {
        repoRootImportMetaUrl: repoRootUrl,
        sourceRoots: proof.sourceRoots,
      },
    );
    const validation = validateRadiometricResult(proof, payload, {
      expectedProvenance,
    });
    const after = resultArtifactSnapshot(resultFile);
    if (
      !after.exists ||
      (before.exists && after.mtimeMs <= before.mtimeMs) ||
      after.mtimeMs < captureStartedAtMs - 2_000 ||
      validation.capturedAtMs < captureStartedAtMs - 5_000 ||
      validation.capturedAtMs > Date.now() + 5_000
    ) {
      return {
        code: 'pt-radiometric-ab-stale-result',
        message: `${id} exited 0 but did not produce a fresh result artifact for this run.`,
      };
    }
  } catch (error) {
    return {
      code: payload.verdict === 'PASS'
        ? 'pt-radiometric-ab-invalid-result'
        : 'pt-radiometric-ab-nonpass-result',
      message:
        `${id} exited 0 but failed strict artifact validation: ` +
        `${error instanceof Error ? error.message : String(error)}`,
    };
  }
  return null;
}

function readJson(relativePath) {
  return JSON.parse(readFileSync(resolve(repoRoot, relativePath), 'utf8'));
}

const timeoutMs = parseTimeoutMs(process.env.VITRUM_PT_RADIOMETRIC_AB_TIMEOUT_MS);
const selected = selectedCaseIds();
const caseStatuses = [];

for (const id of selected) {
  const entry = CASES[id];
  const denoArgs = [
    'run',
    '--config',
    'tools/radiometric-ab/deno.json',
    '--unstable-webgpu',
    '--sloppy-imports',
    '--allow-read',
    '--allow-env',
    '--allow-write',
    entry.script,
  ];
  console.log(`=== pt radiometric A/B wrapper: ${id} ===`);
  const artifactBefore = resultArtifactSnapshot(entry.resultFile);
  const captureStartedAtMs = Date.now();
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
  const artifactProblem =
    result.status === 0
      ? await resultArtifactProblem(
        id,
        entry.resultFile,
        artifactBefore,
        captureStartedAtMs,
      )
      : null;
  const caseStatus =
    result.status === 0 && artifactProblem == null
      ? 'PASS'
      : hostBoundary != null
        ? 'HOST-BLOCKED'
        : 'FAIL';
  caseStatuses.push({
    id,
    status: caseStatus,
    script: entry.script,
    resultFile: entry.resultFile,
    exitStatus: result.status,
    signal: result.signal,
    reason: artifactProblem ?? hostBoundary,
  });
}

const hasFail = caseStatuses.some((entry) => entry.status === 'FAIL');
const hasBlocked = caseStatuses.some((entry) => entry.status === 'HOST-BLOCKED');
const verdict = hasFail ? 'FAIL' : hasBlocked ? 'HOST-BLOCKED' : 'PASS';
const status = {
  generatedAt: new Date().toISOString(),
  harness: 'pt-radiometric-ab',
  verdict,
  command:
    'deno run --config tools/radiometric-ab/deno.json --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write tools/radiometric-ab/ab-*.mjs',
  selectedCases: selected,
  timeoutMs,
  icd: process.env.VK_ICD_FILENAMES ?? null,
  cases: caseStatuses,
  preservedResultFiles: Object.values(CASES).map((entry) => entry.resultFile),
  reason: hasFail
    ? {
        code: 'pt-radiometric-ab-failed',
        message: 'At least one stable pt radiometric A/B case failed.',
      }
    : hasBlocked
      ? {
          code: 'pt-radiometric-ab-host-blocked',
          message: 'The selected recapture could not complete on this native-Deno host.',
        }
      : {
          code: 'pt-radiometric-ab-complete',
          message: 'All selected stable pt radiometric A/B recaptures completed.',
        },
  nextSteps: hasBlocked
    ? ['Rerun the same deterministic suite on a native-Deno adapter that exposes the required limits.']
    : [],
};

status.provenance = await ptRadiometricStatusProvenance(
  repoRootUrl,
  'tools/radiometric-ab/pt-ab-host-status.json',
  status.preservedResultFiles,
);

writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);

if (hasFail) process.exit(1);
if (hasBlocked) process.exit(2);
process.exit(0);
