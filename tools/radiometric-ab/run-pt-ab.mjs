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
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'pt-ab-host-status.json');
const promotionStatusPath = resolve(scriptDir, 'pt-promotion-status.json');
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
const SOURCE_STATUS_PATHS = [
  'tools/radiometric-ab/pt-ab-host-status.json',
  'tools/radiometric-ab/results-sppm.json',
  'tools/radiometric-ab/results-bdpt.json',
  'tools/radiometric-ab/results-restir-pt.json',
  'tools/radiometric-ab/results-restir-pt-specialty.json',
  'tools/radiometric-ab/results-restir-pt-glossy-research.json',
  'tools/radiometric-ab/results-sobol.json',
];

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

function passResultArtifactProblem(id, resultFile) {
  let payload;
  try {
    payload = JSON.parse(readFileSync(resolve(repoRoot, resultFile), 'utf8'));
  } catch (err) {
    return {
      code: 'pt-radiometric-ab-missing-result',
      message: `${id} exited 0 but did not write readable result JSON: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
  if (payload == null || typeof payload !== 'object' || Array.isArray(payload)) {
    return {
      code: 'pt-radiometric-ab-invalid-result',
      message: `${id} exited 0 but wrote a non-object result artifact.`,
    };
  }
  if (typeof payload.verdict !== 'string') {
    return {
      code: 'pt-radiometric-ab-incomplete-result',
      message: `${id} exited 0 but the result artifact is missing a verdict.`,
    };
  }
  return null;
}

function readJson(relativePath) {
  return JSON.parse(readFileSync(resolve(repoRoot, relativePath), 'utf8'));
}

function maxBy(values, getter) {
  if (values.length === 0) return undefined;
  return Math.max(...values.map(getter));
}

function buildPromotionStatus(hostStatus) {
  const sppm = readJson('tools/radiometric-ab/results-sppm.json');
  const bdpt = readJson('tools/radiometric-ab/results-bdpt.json');
  const restirPt = readJson('tools/radiometric-ab/results-restir-pt.json');
  const specialty = readJson('tools/radiometric-ab/results-restir-pt-specialty.json');
  const glossyResearch = readJson('tools/radiometric-ab/results-restir-pt-glossy-research.json');
  const sobol = readJson('tools/radiometric-ab/results-sobol.json');

  const finalSppm = sppm.sppm?.[sppm.sppm.length - 1];
  const bdptControls = bdpt.controls?.byMaxLightBounces ?? [];
  const endpoint = bdptControls.find((entry) => entry.maxLightBounces === 1);
  const firstBdptFinding = bdptControls.find((entry) => entry.maxLightBounces === 2);
  const bdptMultiVertexFinding = bdpt.researchFindings?.bdptMultiVertex ?? {
    defaultReady: bdpt.controls?.multiVertexPromotion?.defaultReady,
    warningCode: 'pt-webgpu.bdpt-multivertex-research-mode',
    currentEstimator: bdpt.controls?.multiVertexPromotion?.currentEstimator,
    blocker: bdpt.controls?.multiVertexPromotion?.blocker,
    requiredEstimator: bdpt.controls?.multiVertexPromotion?.requiredEstimator,
    safeAlternative: bdpt.controls?.multiVertexPromotion?.safeAlternative,
    firstFindingMaxLightBounces: firstBdptFinding?.maxLightBounces,
    firstFindingGlobalRelErr: firstBdptFinding?.globalRelErr,
    evidencePath: 'tools/radiometric-ab/results-bdpt.json',
  };
  const restirPtGlossyFinding = glossyResearch.researchFindings?.restirPtGlossyResearch ?? {
    verdict: glossyResearch.verdict,
    defaultReady: glossyResearch.promotion?.defaultReady,
    warningCode: 'pt-webgpu.restir-pt-glossy-reuse-research-mode',
    blocker: 'glossy-visible-vertex-reuse-outside-diffuse-safe-validation-envelope',
    requiredEvidence: 'glossy-material-furnace-reference-ab-and-browser-real-adapter-recapture',
    globalRelErr: glossyResearch.globalRelErr,
    varRatio: glossyResearch.varRatio,
    evidencePath: 'tools/radiometric-ab/results-restir-pt-glossy-research.json',
  };
  const sobolRatios = (sobol.scenes ?? []).map((scene) => scene.ratios ?? {});
  const sobolDefaultFinding = sobol.researchFindings?.sobolDefault ?? {
    defaultReady: sobol.promotion?.defaultReady,
    evidenceClass: sobol.promotion?.evidenceClass,
    requiredEvidence: sobol.promotion?.requiredEvidence,
    maxGlobalRmseRatio: maxBy(sobolRatios, (ratio) => ratio.globalRmse),
    maxRoiRmseRatio: maxBy(sobolRatios, (ratio) => ratio.roiRmse),
    maxElapsedMsRatio: maxBy(sobolRatios, (ratio) => ratio.elapsedMs),
    evidencePath: 'tools/radiometric-ab/results-sobol.json',
  };

  return {
    generatedAt: hostStatus.generatedAt,
    harness: 'pt-radiometric-promotion-proof',
    verdict: 'PASS-PARTIAL',
    hostStatus: {
      verdict: hostStatus.verdict,
      caseCount: hostStatus.selectedCases.length,
      selectedCases: hostStatus.selectedCases,
    },
    safeDefaultProofs: {
      sppm: {
        verdict: sppm.verdict,
        converging: sppm.converging,
        inBallpark: sppm.inBallpark,
        finalRelErr: finalSppm?.relErr,
      },
      bdptEndpointOnly: {
        verdict: bdpt.verdict,
        endpointOnlyMatchesUni: bdpt.controls?.endpointOnlyMatchesUni,
        maxLightBounces: endpoint?.maxLightBounces,
        globalRelErr: endpoint?.globalRelErr,
        roiRelErr: endpoint?.roiRelErr,
      },
      restirPtDiffuse: {
        verdict: restirPt.verdict,
        meanAgreement: restirPt.meanAgreement,
        varianceNotWorse: restirPt.varianceNotWorse,
        globalRelErr: restirPt.globalRelErr,
        varRatio: restirPt.varRatio,
      },
      restirPtSpecialty: {
        mode: specialty.mode,
        caseCount: specialty.summary?.caseCount,
        maxAbsoluteError: specialty.summary?.maxAbsoluteError,
        maxRelativeError: specialty.summary?.maxRelativeError,
      },
    },
    researchFindings: {
      bdptMultiVertex: {
        defaultReady: bdptMultiVertexFinding.defaultReady,
        warningCode: bdptMultiVertexFinding.warningCode,
        currentEstimator: bdptMultiVertexFinding.currentEstimator,
        blocker: bdptMultiVertexFinding.blocker,
        requiredEstimator: bdptMultiVertexFinding.requiredEstimator,
        safeAlternative: bdptMultiVertexFinding.safeAlternative,
        firstFindingMaxLightBounces: bdptMultiVertexFinding.firstFindingMaxLightBounces,
        firstFindingGlobalRelErr: bdptMultiVertexFinding.firstFindingGlobalRelErr,
        evidencePath: bdptMultiVertexFinding.evidencePath,
      },
      restirPtGlossyResearch: {
        verdict: restirPtGlossyFinding.verdict,
        defaultReady: restirPtGlossyFinding.defaultReady,
        warningCode: restirPtGlossyFinding.warningCode,
        blocker: restirPtGlossyFinding.blocker,
        requiredEvidence: restirPtGlossyFinding.requiredEvidence,
        globalRelErr: restirPtGlossyFinding.globalRelErr,
        varRatio: restirPtGlossyFinding.varRatio,
        evidencePath: restirPtGlossyFinding.evidencePath,
      },
      sobolDefault: {
        defaultReady: sobolDefaultFinding.defaultReady,
        evidenceClass: sobolDefaultFinding.evidenceClass,
        requiredEvidence: sobolDefaultFinding.requiredEvidence,
        maxGlobalRmseRatio: sobolDefaultFinding.maxGlobalRmseRatio,
        maxRoiRmseRatio: sobolDefaultFinding.maxRoiRmseRatio,
        maxElapsedMsRatio: sobolDefaultFinding.maxElapsedMsRatio,
        evidencePath: sobolDefaultFinding.evidencePath,
      },
    },
    sourceStatuses: SOURCE_STATUS_PATHS,
  };
}

function maybeWritePromotionStatus(hostStatus) {
  const selectedAllCases = ALL_CASE_IDS.every((id) => hostStatus.selectedCases.includes(id))
    && hostStatus.selectedCases.length === ALL_CASE_IDS.length;
  if (hostStatus.verdict !== 'PASS' || !selectedAllCases) return;
  const promotionStatus = buildPromotionStatus(hostStatus);
  writeFileSync(promotionStatusPath, `${JSON.stringify(promotionStatus, null, 2)}\n`);
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
  const artifactProblem = result.status === 0
    ? passResultArtifactProblem(id, entry.resultFile)
    : null;
  const status = result.status === 0 && artifactProblem == null
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
    reason: artifactProblem ?? hostBoundary,
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
maybeWritePromotionStatus(status);

if (hasFail) process.exit(1);
if (hasBlocked) process.exit(2);
process.exit(0);
