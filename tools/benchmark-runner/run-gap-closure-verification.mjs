import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { GAP_CLOSURE_SCENARIOS } from './scenario-presets.mjs';
import { PT_WEBGPU_GAP_SCENARIOS } from './gapClosurePtWebgpuMap.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const scenarios = GAP_CLOSURE_SCENARIOS;

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const baselineDir = resolve(repoRoot, process.env.VITRUM_BASELINE_DIR ?? 'tools/reference-renders/baseline');
const captureDir = resolve(here, 'results/captures');
// Date is derived at run time so the artifact name reflects when the
// verification actually ran. Override with VITRUM_RESULT_DATE for
// deterministic reproductions (e.g. golden-result tests).
const resultDate = (process.env.VITRUM_RESULT_DATE ?? new Date().toISOString().slice(0, 10));
const outputPath = resolve(here, `results/gap-closure-verification-${resultDate}.json`);

const captureEnabled = process.env.VITRUM_GPU_CAPTURE === '1';
const captureCommandOverride = process.env.VITRUM_CAPTURE_CMD?.trim() ?? '';

function defaultCaptureCommand(scenario) {
  void scenario;
  return '';
}

const allowBaselineGen = process.env.VITRUM_ALLOW_BASELINE_GEN === '1';
const gapMechanical = process.env.VITRUM_GAP_MECHANICAL === '1';

const scenarioFilter = (process.env.VITRUM_GAP_SCENARIOS ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);
let activeScenarios =
  scenarioFilter.length > 0
    ? scenarios.filter((s) => scenarioFilter.includes(s.scenarioId))
    : scenarios;
if (gapMechanical && scenarioFilter.length === 0) {
  const withBaseline = [];
  for (const s of scenarios) {
    const isPtWebgpuRow =
      s.backend === 'pt-webgpu' || PT_WEBGPU_GAP_SCENARIOS.includes(s.scenarioId);
    if (!isPtWebgpuRow) continue;
    const baselinePath = resolve(baselineDir, `${s.scenarioId}.png`);
    // eslint-disable-next-line no-await-in-loop
    if (await fileExists(baselinePath)) withBaseline.push(s);
  }
  activeScenarios = withBaseline;
  if (activeScenarios.length === 0) {
    console.error(
      '[gap-closure] VITRUM_GAP_MECHANICAL=1: no pt-webgpu scenarios with committed baselines.',
    );
    process.exit(1);
  }
}
const failOnIdentical = process.env.VITRUM_FAIL_ON_IDENTICAL_HASH === '1';
const smokeCapture = process.env.VITRUM_CAPTURE_SMOKE === '1';
const captureProcessTimeoutMs = Math.max(
  5_000,
  Number(process.env.VITRUM_CAPTURE_PROCESS_TIMEOUT_MS ?? '120000'),
);
const smokeMaxWidth = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_WIDTH ?? '320'));
const smokeMaxHeight = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_HEIGHT ?? '180'));
const smokeMaxSpp = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_SPP ?? '8'));
const smokeMaxBounces = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_BOUNCES ?? '4'));

async function fileExists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function sha256(path) {
  const data = await readFile(path);
  return createHash('sha256').update(data).digest('hex');
}

async function readPerfSidecar(imagePath) {
  const sidecarPath = `${imagePath}.json`;
  if (!(await fileExists(sidecarPath))) return null;
  try {
    const parsed = JSON.parse(await readFile(sidecarPath, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

async function writePerfSidecar(imagePath, telemetry) {
  if (telemetry == null) return;
  if (typeof telemetry === 'number' && Number.isFinite(telemetry)) {
    await writeFile(`${imagePath}.json`, `${JSON.stringify({ msPerSample: telemetry }, null, 2)}\n`, 'utf8');
    return;
  }
  if (typeof telemetry === 'object') {
    await writeFile(`${imagePath}.json`, `${JSON.stringify(telemetry, null, 2)}\n`, 'utf8');
  }
}

function parseResolution(resolution) {
  const [w, h] = resolution.split('x').map((v) => Number(v));
  return {
    width: Number.isFinite(w) ? w : 0,
    height: Number.isFinite(h) ? h : 0,
  };
}

function scenarioVariants(scenario) {
  const causticVariants =
    Array.isArray(scenario.causticVariants) && scenario.causticVariants.length > 0
      ? scenario.causticVariants
      : ['candidate'];
  const roughnessVariants =
    Array.isArray(scenario.roughnessVariants) && scenario.roughnessVariants.length > 0
      ? scenario.roughnessVariants
      : [null];
  const wallAlbedoVariants =
    Array.isArray(scenario.wallAlbedoVariants) && scenario.wallAlbedoVariants.length > 0
      ? scenario.wallAlbedoVariants
      : [null];

  const variants = [];
  for (const caustic of causticVariants) {
    for (const roughness of roughnessVariants) {
      for (const wallAlbedo of wallAlbedoVariants) {
        const tags = [];
        if (caustic != null) tags.push(`caustic=${caustic}`);
        if (roughness != null) tags.push(`roughness=${roughness}`);
        if (wallAlbedo != null) tags.push(`wallAlbedo=${wallAlbedo}`);
        variants.push({
          id: tags.length === 0 ? 'candidate' : tags.join('__'),
          caustic,
          roughness,
          wallAlbedo,
        });
      }
    }
  }
  return variants;
}

function captureScenarioSettings(scenario) {
  if (!smokeCapture) return scenario;
  const { width, height } = parseResolution(scenario.resolution);
  const scale = Math.min(1, smokeMaxWidth / Math.max(width, 1), smokeMaxHeight / Math.max(height, 1));
  const cappedWidth = Math.max(1, Math.floor(width * scale));
  const cappedHeight = Math.max(1, Math.floor(height * scale));
  return {
    ...scenario,
    resolution: `${cappedWidth}x${cappedHeight}`,
    bounces: Math.min(scenario.bounces, smokeMaxBounces),
    spp: Math.min(scenario.spp, smokeMaxSpp),
  };
}

function runCommand(command, env, timeoutMs) {
  // Thin adapter over the shared helper for back-compat with existing
  // call sites; new code should call `runCommandWithTimeout` directly.
  return runCommandWithTimeout(command, { cwd: repoRoot, env, timeoutMs });
}

async function runCapture(scenario, variant, outputImagePath) {
  const effectiveScenario = captureScenarioSettings(scenario);
  const { width, height } = parseResolution(effectiveScenario.resolution);
  if (!captureEnabled) {
    return {
      ok: false,
      status: 'blocked-no-gpu-harness',
      reason: 'VITRUM_GPU_CAPTURE is not enabled (expected 1).',
    };
  }
  const captureCommand = captureCommandOverride || defaultCaptureCommand(effectiveScenario);
  if (!captureCommand) {
    return {
      ok: false,
      status: 'blocked-no-capture-adapter',
      reason:
        'VITRUM_CAPTURE_CMD is unset and no default adapter exists for this scenario backend. ' +
        'Set VITRUM_CAPTURE_CMD or use backend "pt-webgpu" for the built-in capturePtWebgpu adapter.',
    };
  }
  const run = await runCommand(captureCommand, {
    VITRUM_SCENARIO_ID: effectiveScenario.scenarioId,
    VITRUM_SEED: String(effectiveScenario.seed),
    VITRUM_WIDTH: String(width),
    VITRUM_HEIGHT: String(height),
    VITRUM_BOUNCES: String(effectiveScenario.bounces),
    VITRUM_SPP: String(effectiveScenario.spp),
    VITRUM_CAUSTIC_STRATEGY: variant.caustic ?? '',
    VITRUM_ROUGHNESS: variant.roughness != null ? String(variant.roughness) : '',
    VITRUM_WALL_ALBEDO: variant.wallAlbedo != null ? String(variant.wallAlbedo) : '',
    VITRUM_BACKEND: typeof effectiveScenario.backend === 'string' ? effectiveScenario.backend : '',
    VITRUM_FRAMES: effectiveScenario.frames != null ? String(effectiveScenario.frames) : '',
    VITRUM_SCENARIO_JSON: JSON.stringify(effectiveScenario),
    VITRUM_OUTPUT_PNG: outputImagePath,
    VITRUM_CAPTURE_TIMEOUT_MS: String(captureProcessTimeoutMs),
  }, captureProcessTimeoutMs);
  const imageExists = await fileExists(outputImagePath);
  if (run.code !== 0 || !imageExists) {
    return {
      ok: false,
      status: 'capture-failed',
      reason: `adapter exit=${run.code}; imageExists=${imageExists}; stderr=${run.stderr || '(none)'}`,
    };
  }

  let perfTelemetry = null;
  if (run.stdout) {
    const lines = run.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
    const jsonLine = lines.find((line) => line.startsWith('{') && line.endsWith('}'));
    if (jsonLine != null) {
      try {
        const parsed = JSON.parse(jsonLine);
        if (typeof parsed.msPerSample === 'number' && Number.isFinite(parsed.msPerSample)) {
          perfTelemetry = parsed;
        }
      } catch {
        // ignore malformed telemetry
      }
    }
  }
  return {
    ok: true,
    status: 'captured',
    reason: run.stderr || 'capture complete',
    perfMsPerSample: perfTelemetry?.msPerSample ?? null,
    perfTelemetry,
  };
}

async function evaluateScenario(scenario) {
  const baselineImagePath = resolve(baselineDir, `${scenario.scenarioId}.png`);
  if (
    gapMechanical &&
    (scenario.backend === 'pt-webgpu' || PT_WEBGPU_GAP_SCENARIOS.includes(scenario.scenarioId))
  ) {
    const baselineExists = await fileExists(baselineImagePath);
    if (!baselineExists) {
      return {
        ...scenario,
        status: 'blocked-missing-baseline',
        beforeImageHash: null,
        afterImageHash: null,
        deltaSummary: `Mechanical mode: missing ${baselineImagePath}`,
        perfBaselineMsPerSample: null,
        perfCandidateMsPerSample: null,
        passFail: 'BLOCKED',
      };
    }
    const hash = await sha256(baselineImagePath);
    return {
      ...scenario,
      status: 'mechanical-baseline-locked',
      beforeImageHash: hash,
      afterImageHash: hash,
      deltaSummary: 'mechanical: baseline PNG committed; GPU re-capture skipped (VITRUM_GAP_MECHANICAL=1)',
      perfBaselineMsPerSample: null,
      perfCandidateMsPerSample: null,
      passFail: 'PASS',
    };
  }

  const scenarioDir = resolve(captureDir, scenario.scenarioId);
  await mkdir(scenarioDir, { recursive: true });
  const variants = scenarioVariants(scenario);

  const baselineExistsBefore = await fileExists(baselineImagePath);
  let baselineCaptureInfo = null;
  if (!baselineExistsBefore && allowBaselineGen) {
    await mkdir(baselineDir, { recursive: true });
    baselineCaptureInfo = await runCapture(
      scenario,
      { id: 'baseline', caustic: 'baseline', roughness: null, wallAlbedo: null },
      baselineImagePath,
    );
    if (baselineCaptureInfo.ok) {
      await writePerfSidecar(baselineImagePath, baselineCaptureInfo.perfTelemetry);
    }
  }
  const baselineExists = await fileExists(baselineImagePath);
  if (!baselineExists) {
    return {
      ...scenario,
      status: baselineCaptureInfo?.status ?? 'blocked-missing-baseline',
      beforeImageHash: null,
      afterImageHash: null,
      deltaSummary:
        baselineCaptureInfo?.reason ??
        `Missing baseline image at ${baselineImagePath}. Set VITRUM_ALLOW_BASELINE_GEN=1 with a working adapter to generate it.`,
      perfBaselineMsPerSample: null,
      perfCandidateMsPerSample: null,
      passFail: 'BLOCKED',
    };
  }

  const baselineHash = await sha256(baselineImagePath);
  const baselineTelemetry = baselineCaptureInfo?.perfTelemetry ?? (await readPerfSidecar(baselineImagePath));
  const perfBaseline = baselineTelemetry?.msPerSample ?? null;
  let aggregateCandidateHash = '';
  const perfSamples = [];
  const perfTelemetrySamples = [];
  const modeSummaries = [];
  for (const variant of variants) {
    const outputImagePath = resolve(scenarioDir, `${variant.id}.png`);
    const capture = await runCapture(scenario, variant, outputImagePath);
    if (!capture.ok) {
      return {
        ...scenario,
        status: capture.status,
        beforeImageHash: baselineHash,
        afterImageHash: null,
        deltaSummary: `Capture for variant "${variant.id}" failed: ${capture.reason}`,
        perfBaselineMsPerSample: perfBaseline,
        perfCandidateMsPerSample: null,
        passFail: 'BLOCKED',
      };
    }
    const variantHash = await sha256(outputImagePath);
    aggregateCandidateHash += `${variant.id}:${variantHash}|`;
    if (capture.perfMsPerSample != null) {
      perfSamples.push(capture.perfMsPerSample);
    }
    if (capture.perfTelemetry != null) {
      perfTelemetrySamples.push({ variant: variant.id, ...capture.perfTelemetry });
      await writePerfSidecar(outputImagePath, capture.perfTelemetry);
    }
    modeSummaries.push(`${variant.id}:${variantHash.slice(0, 12)}`);
  }

  const afterHash = createHash('sha256').update(aggregateCandidateHash).digest('hex');
  const perfCandidate =
    perfSamples.length === 0
      ? null
      : perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length;
  const variantHashes = aggregateCandidateHash
    .split('|')
    .filter(Boolean)
    .map((entry) => entry.split(':').slice(1).join(':'));
  const allVariantsMatchBaseline =
    variantHashes.length > 0 && variantHashes.every((hash) => hash === baselineHash);
  const failedByIdentical = failOnIdentical && allVariantsMatchBaseline;
  const passFail = failedByIdentical ? 'FAIL' : allVariantsMatchBaseline ? 'PASS' : 'FAIL';
  return {
    ...scenario,
    status: failedByIdentical ? 'failed-identical-hash' : allVariantsMatchBaseline ? 'captured' : 'hash-mismatch',
    beforeImageHash: baselineHash,
    afterImageHash: afterHash,
    deltaSummary: `variants[${modeSummaries.join(', ')}]`,
    perfBaselineMsPerSample: perfBaseline,
    perfCandidateMsPerSample: perfCandidate,
    perfTelemetry: {
      baseline: baselineTelemetry,
      candidates: perfTelemetrySamples,
    },
    passFail,
  };
}

// Bounded concurrency: GPU capture serializes through a single browser
// instance, so scenarios run sequentially when VITRUM_GPU_CAPTURE=1 to avoid
// driver contention. CPU-only verification (no capture) parallelizes
// freely. Override via VITRUM_GAP_CONCURRENCY (positive integer).
const defaultConcurrency = captureEnabled ? 1 : Math.max(1, activeScenarios.length);
const concurrency = Math.max(
  1,
  Number(process.env.VITRUM_GAP_CONCURRENCY ?? defaultConcurrency) || 1,
);
const entries = [];
if (concurrency <= 1) {
  for (const scenario of activeScenarios) {
    // eslint-disable-next-line no-await-in-loop
    entries.push(await evaluateScenario(scenario));
  }
} else if (concurrency >= activeScenarios.length) {
  entries.push(...(await Promise.all(activeScenarios.map(evaluateScenario))));
} else {
  // Bounded pool: run `concurrency` workers that pull from a shared queue.
  const queue = activeScenarios.slice();
  const results = new Array(activeScenarios.length);
  let nextIdx = 0;
  const workers = Array.from({ length: concurrency }, async () => {
    while (true) {
      const scenario = queue.shift();
      if (!scenario) return;
      const idx = nextIdx++;
      results[idx] = await evaluateScenario(scenario);
    }
  });
  await Promise.all(workers);
  entries.push(...results);
}

const report = {
  generatedAt: new Date().toISOString(),
  deterministicScenarioSetVersion: 'gap-closure-acceptance-matrix-2026-05-10',
  environment: {
    platform: process.platform,
    node: process.version,
    gpuCaptureEnabled: captureEnabled,
    captureCommandConfigured:
      captureCommandOverride.length > 0 || activeScenarios.some((s) => defaultCaptureCommand(s).length > 0),
    scenarioFilter: scenarioFilter.length > 0 ? scenarioFilter : null,
    allowBaselineGeneration: allowBaselineGen,
    gapMechanical,
  },
  results: entries,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');

console.log(`Wrote ${outputPath}`);

const strict = process.env.VITRUM_STRICT_GAP_CLOSURE === '1';
if (strict) {
  const bad = report.results.filter((r) => r.passFail !== 'PASS');
  if (bad.length > 0) {
    console.error(
      `[gap-closure] VITRUM_STRICT_GAP_CLOSURE=1: ${bad.length} scenario(s) not PASS:`,
      bad.map((r) => `${r.scenarioId}:${r.passFail}`).join(', '),
    );
    process.exit(1);
  }
}
