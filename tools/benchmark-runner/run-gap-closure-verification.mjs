import { createHash } from 'node:crypto';
import { access, copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { GAP_CLOSURE_SCENARIOS } from './scenario-presets.mjs';

const scenarioFilter = (process.env.VITRUM_SCENARIO_FILTER ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
const scenarios = scenarioFilter.length === 0
  ? GAP_CLOSURE_SCENARIOS
  : GAP_CLOSURE_SCENARIOS.filter((s) => scenarioFilter.some((needle) => s.scenarioId.includes(needle)));

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const baselineDir = resolve(repoRoot, process.env.VITRUM_BASELINE_DIR ?? 'tools/reference-renders/baseline');
const captureDir = resolve(here, 'results/captures');
// Date is derived at run time so the artifact name reflects when the
// verification actually ran. Override with VITRUM_RESULT_DATE for
// deterministic reproductions (e.g. golden-result tests).
const resultDate = (process.env.VITRUM_RESULT_DATE ?? new Date().toISOString().slice(0, 10));
const outputPath = resolve(here, `results/gap-closure-verification-${resultDate}.json`);

const captureEnabled = process.env.VITRUM_GPU_CAPTURE === '1';
const captureCommand = process.env.VITRUM_CAPTURE_CMD ?? '';
const allowBaselineGen = process.env.VITRUM_ALLOW_BASELINE_GEN === '1';
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

function stableLabel(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(v).replace('.', '_');
  return String(v);
}

function scenarioVariants(scenario) {
  const axes = [];
  if (Array.isArray(scenario.causticVariants) && scenario.causticVariants.length > 0) {
    axes.push({
      key: 'caustic',
      values: scenario.causticVariants,
      envName: 'VITRUM_CAUSTIC_STRATEGY',
    });
  }
  if (Array.isArray(scenario.roughnessVariants) && scenario.roughnessVariants.length > 0) {
    axes.push({
      key: 'roughness',
      values: scenario.roughnessVariants,
      envName: 'VITRUM_ROUGHNESS',
    });
  }
  if (Array.isArray(scenario.wallAlbedoVariants) && scenario.wallAlbedoVariants.length > 0) {
    axes.push({
      key: 'wall-albedo',
      values: scenario.wallAlbedoVariants,
      envName: 'VITRUM_WALL_ALBEDO',
    });
  }
  if (axes.length === 0) {
    return [{ id: 'candidate', env: {} }];
  }
  let variants = [{ id: 'candidate', env: {} }];
  for (const axis of axes) {
    const next = [];
    for (const base of variants) {
      for (const value of axis.values) {
        next.push({
          id: `${base.id}__${axis.key}-${stableLabel(value)}`,
          env: { ...base.env, [axis.envName]: String(value) },
        });
      }
    }
    variants = next;
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

function withIfDefined(target, key, value) {
  if (value !== undefined && value !== null) target[key] = String(value);
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
  if (!captureCommand) {
    return {
      ok: false,
      status: 'blocked-no-capture-adapter',
      reason:
        'VITRUM_CAPTURE_CMD is unset. Provide a deterministic capture adapter command that writes VITRUM_OUTPUT_PNG.',
    };
  }
  const env = {
    VITRUM_SCENARIO_ID: effectiveScenario.scenarioId,
    VITRUM_SEED: String(effectiveScenario.seed),
    VITRUM_WIDTH: String(width),
    VITRUM_HEIGHT: String(height),
    VITRUM_BOUNCES: String(effectiveScenario.bounces),
    VITRUM_SPP: String(effectiveScenario.spp),
    VITRUM_OUTPUT_PNG: outputImagePath,
    ...variant.env,
  };
  withIfDefined(env, 'VITRUM_BACKEND', effectiveScenario.backend);
  withIfDefined(env, 'VITRUM_FRAMES', effectiveScenario.frames);
  withIfDefined(env, 'VITRUM_ENVIRONMENT_MODE', effectiveScenario.environmentMode);
  withIfDefined(env, 'VITRUM_GI_MODE', effectiveScenario.giMode);
  withIfDefined(env, 'VITRUM_SCENE_VARIANT', effectiveScenario.sceneVariant);
  withIfDefined(env, 'VITRUM_CAMERA_ELEVATION_DEG', effectiveScenario.cameraElevationDeg);
  withIfDefined(env, 'VITRUM_RECT_AREA_LIGHT_COUNT', effectiveScenario.rectAreaLightCount);
  withIfDefined(env, 'VITRUM_GLANCING_ANGLE_DEG', effectiveScenario.glancingAngleDeg);
  withIfDefined(env, 'VITRUM_FLOOR_VARIANT', effectiveScenario.floorVariant);
  if (Array.isArray(effectiveScenario.instanceScale) && effectiveScenario.instanceScale.length === 3) {
    env.VITRUM_INSTANCE_SCALE = effectiveScenario.instanceScale.join(',');
  }
  if (effectiveScenario.glassDimensions != null) {
    withIfDefined(env, 'VITRUM_GLASS_WIDTH', effectiveScenario.glassDimensions.width);
    withIfDefined(env, 'VITRUM_GLASS_HEIGHT', effectiveScenario.glassDimensions.height);
    withIfDefined(env, 'VITRUM_GLASS_THICKNESS', effectiveScenario.glassDimensions.thickness);
  }
  const run = await runCommand(captureCommand, env, captureProcessTimeoutMs);
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
  console.log(`[gap-closure] scenario start: ${scenario.scenarioId}`);
  const scenarioDir = resolve(captureDir, scenario.scenarioId);
  await mkdir(scenarioDir, { recursive: true });
  const variants = scenarioVariants(scenario);
  const baselineCaptureInfos = [];
  const perfSamples = [];
  const perfTelemetrySamples = [];
  const modeSummaries = [];
  const beforeHashes = {};
  const afterHashes = {};
  const mismatches = [];
  for (const variant of variants) {
    const outputImagePath = resolve(scenarioDir, `${variant.id}.png`);
    const baselineVariantPath = resolve(baselineDir, `${scenario.scenarioId}-${variant.id}.png`);
    const baselineDefaultPath = resolve(baselineDir, `${scenario.scenarioId}.png`);
    let baselineImagePath = baselineVariantPath;
    let baselineExistsBefore = await fileExists(baselineImagePath);
    if (!baselineExistsBefore) {
      baselineImagePath = baselineDefaultPath;
      baselineExistsBefore = await fileExists(baselineImagePath);
    }
    let baselineCaptureInfo = null;
    if (!baselineExistsBefore && allowBaselineGen) {
      await mkdir(baselineDir, { recursive: true });
      baselineCaptureInfo = await runCapture(scenario, variant, baselineVariantPath);
      if (baselineCaptureInfo.ok) {
        await writePerfSidecar(baselineVariantPath, baselineCaptureInfo.perfTelemetry);
        // Keep legacy single-file layout for non-axis scenarios while also
        // writing the canonical variant-aware filename.
        if (variant.id === 'candidate') {
          await copyFile(baselineVariantPath, baselineDefaultPath);
          await writePerfSidecar(
            baselineDefaultPath,
            baselineCaptureInfo.perfTelemetry,
          );
        }
        baselineImagePath = baselineVariantPath;
      }
    }
    if (!(await fileExists(baselineImagePath))) {
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
    beforeHashes[variant.id] = baselineHash;
    const baselineTelemetry = baselineCaptureInfo?.perfTelemetry ?? (await readPerfSidecar(baselineImagePath));
    if (baselineTelemetry != null) baselineCaptureInfos.push(baselineTelemetry);

    const capture = await runCapture(scenario, variant, outputImagePath);
    if (!capture.ok) {
      return {
        ...scenario,
        status: capture.status,
        beforeImageHash: beforeHashes,
        afterImageHash: null,
        deltaSummary: `Capture for variant "${variant.id}" failed: ${capture.reason}`,
        perfBaselineMsPerSample: null,
        perfCandidateMsPerSample: null,
        passFail: 'BLOCKED',
      };
    }
    const variantHash = await sha256(outputImagePath);
    afterHashes[variant.id] = variantHash;
    const identical = baselineHash === variantHash;
    if (!identical) mismatches.push(`${variant.id}: baseline=${baselineHash.slice(0, 12)} candidate=${variantHash.slice(0, 12)}`);
    if (capture.perfMsPerSample != null) {
      perfSamples.push(capture.perfMsPerSample);
    }
    if (capture.perfTelemetry != null) {
      perfTelemetrySamples.push({ variant: variant.id, ...capture.perfTelemetry });
      await writePerfSidecar(outputImagePath, capture.perfTelemetry);
    }
    modeSummaries.push(`${variant.id}:${variantHash.slice(0, 12)}`);
    console.log(`[gap-closure] ${scenario.scenarioId}/${variant.id} captured`);
  }

  const beforeHash = createHash('sha256').update(JSON.stringify(beforeHashes)).digest('hex');
  const afterHash = createHash('sha256').update(JSON.stringify(afterHashes)).digest('hex');
  const perfBaselineValues = baselineCaptureInfos
    .map((t) => t?.msPerSample)
    .filter((x) => typeof x === 'number' && Number.isFinite(x));
  const perfBaseline = perfBaselineValues.length === 0
    ? null
    : perfBaselineValues.reduce((a, b) => a + b, 0) / perfBaselineValues.length;
  const perfCandidate =
    perfSamples.length === 0
      ? null
      : perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length;
  const hasMismatch = mismatches.length > 0;
  const failedByHash = (failOnIdentical && !hasMismatch) || (!failOnIdentical && hasMismatch);
  return {
    ...scenario,
    status: failedByHash ? (hasMismatch ? 'failed-baseline-mismatch' : 'failed-identical-hash') : 'captured',
    beforeImageHash: beforeHashes,
    afterImageHash: afterHashes,
    deltaSummary: hasMismatch
      ? `mismatch[${mismatches.join('; ')}] variants[${modeSummaries.join(', ')}]`
      : `match variants[${modeSummaries.join(', ')}]`,
    perfBaselineMsPerSample: perfBaseline,
    perfCandidateMsPerSample: perfCandidate,
    perfTelemetry: {
      baseline: baselineCaptureInfos,
      candidates: perfTelemetrySamples,
    },
    passFail: failedByHash ? 'FAIL' : 'PASS',
  };
}

// Bounded concurrency: GPU capture serializes through a single browser
// instance, so scenarios run sequentially when VITRUM_GPU_CAPTURE=1 to avoid
// driver contention. CPU-only verification (no capture) parallelizes
// freely. Override via VITRUM_GAP_CONCURRENCY (positive integer).
const defaultConcurrency = captureEnabled ? 1 : Math.max(1, scenarios.length);
const concurrency = Math.max(
  1,
  Number(process.env.VITRUM_GAP_CONCURRENCY ?? defaultConcurrency) || 1,
);
const entries = [];
if (concurrency <= 1) {
  for (const scenario of scenarios) {
    // eslint-disable-next-line no-await-in-loop
    entries.push(await evaluateScenario(scenario));
  }
} else if (concurrency >= scenarios.length) {
  entries.push(...(await Promise.all(scenarios.map(evaluateScenario))));
} else {
  // Bounded pool: run `concurrency` workers that pull from a shared queue.
  const queue = scenarios.slice();
  const results = new Array(scenarios.length);
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
    captureCommandConfigured: captureCommand.length > 0,
    allowBaselineGeneration: allowBaselineGen,
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
