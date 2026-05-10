import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const scenarios = [
  { scenarioId: 'rfe03-layered-front-back', seed: 1337, resolution: '1280x720', bounces: 8, spp: 512 },
  { scenarioId: 'rfe07-11-sss-mixed-panels', seed: 2027, resolution: '1280x720', bounces: 8, spp: 512 },
  { scenarioId: 'rfe08-13-spectral-payload', seed: 4242, resolution: '1280x720', bounces: 10, spp: 1024 },
  { scenarioId: 'rfe14-thinfilm-angle-shift', seed: 9001, resolution: '1280x720', bounces: 10, spp: 1024 },
  { scenarioId: 'rfe09-bridge-global-cmf', seed: 31415, resolution: '1024x1024', bounces: 8, spp: 256 },
  { scenarioId: 'rfe05-caustic-strategy', seed: 27182, resolution: '1280x720', bounces: 10, spp: 1024 },
  { scenarioId: 'ptwgpu-parity-material-fields', seed: 777, resolution: '1280x720', bounces: 8, spp: 512 },
];

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const baselineDir = resolve(repoRoot, process.env.VITRUM_BASELINE_DIR ?? 'tools/reference-renders/baseline');
const captureDir = resolve(here, 'results/captures');
const outputPath = resolve(here, 'results/gap-closure-verification-2026-05-10.json');

const captureEnabled = process.env.VITRUM_GPU_CAPTURE === '1';
const captureCommand = process.env.VITRUM_CAPTURE_CMD ?? '';
const allowBaselineGen = process.env.VITRUM_ALLOW_BASELINE_GEN === '1';
const failOnIdentical = process.env.VITRUM_FAIL_ON_IDENTICAL_HASH === '1';

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

function parseResolution(resolution) {
  const [w, h] = resolution.split('x').map((v) => Number(v));
  return {
    width: Number.isFinite(w) ? w : 0,
    height: Number.isFinite(h) ? h : 0,
  };
}

function scenarioVariants(scenario) {
  if (scenario.scenarioId === 'rfe05-caustic-strategy') {
    return ['none', 'manifold-nee', 'photon-map'];
  }
  return ['candidate'];
}

function runCommand(command, env) {
  return new Promise((resolveResult) => {
    const child = spawn(command, {
      cwd: repoRoot,
      env: { ...process.env, ...env },
      shell: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    child.on('close', (code) => {
      resolveResult({
        code: code ?? -1,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
      });
    });
  });
}

async function runCapture(scenario, variant, outputImagePath) {
  const { width, height } = parseResolution(scenario.resolution);
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
  const run = await runCommand(captureCommand, {
    VITRUM_SCENARIO_ID: scenario.scenarioId,
    VITRUM_SEED: String(scenario.seed),
    VITRUM_WIDTH: String(width),
    VITRUM_HEIGHT: String(height),
    VITRUM_BOUNCES: String(scenario.bounces),
    VITRUM_SPP: String(scenario.spp),
    VITRUM_CAUSTIC_STRATEGY: variant,
    VITRUM_OUTPUT_PNG: outputImagePath,
  });
  const imageExists = await fileExists(outputImagePath);
  if (run.code !== 0 || !imageExists) {
    return {
      ok: false,
      status: 'capture-failed',
      reason: `adapter exit=${run.code}; imageExists=${imageExists}; stderr=${run.stderr || '(none)'}`,
    };
  }

  let perfMsPerSample = null;
  if (run.stdout) {
    const lines = run.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
    const jsonLine = lines.find((line) => line.startsWith('{') && line.endsWith('}'));
    if (jsonLine != null) {
      try {
        const parsed = JSON.parse(jsonLine);
        if (typeof parsed.msPerSample === 'number' && Number.isFinite(parsed.msPerSample)) {
          perfMsPerSample = parsed.msPerSample;
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
    perfMsPerSample,
  };
}

async function evaluateScenario(scenario) {
  const scenarioDir = resolve(captureDir, scenario.scenarioId);
  await mkdir(scenarioDir, { recursive: true });
  const variants = scenarioVariants(scenario);

  const baselineImagePath = resolve(baselineDir, `${scenario.scenarioId}.png`);
  const baselineExistsBefore = await fileExists(baselineImagePath);
  let baselineCaptureInfo = null;
  if (!baselineExistsBefore && allowBaselineGen) {
    await mkdir(baselineDir, { recursive: true });
    baselineCaptureInfo = await runCapture(scenario, 'baseline', baselineImagePath);
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
  let aggregateCandidateHash = '';
  const perfSamples = [];
  const modeSummaries = [];
  for (const variant of variants) {
    const outputImagePath = resolve(scenarioDir, `${variant}.png`);
    const capture = await runCapture(scenario, variant, outputImagePath);
    if (!capture.ok) {
      return {
        ...scenario,
        status: capture.status,
        beforeImageHash: baselineHash,
        afterImageHash: null,
        deltaSummary: `Capture for variant "${variant}" failed: ${capture.reason}`,
        perfBaselineMsPerSample: null,
        perfCandidateMsPerSample: null,
        passFail: 'BLOCKED',
      };
    }
    const variantHash = await sha256(outputImagePath);
    aggregateCandidateHash += `${variant}:${variantHash}|`;
    if (capture.perfMsPerSample != null) {
      perfSamples.push(capture.perfMsPerSample);
    }
    modeSummaries.push(`${variant}:${variantHash.slice(0, 12)}`);
  }

  const afterHash = createHash('sha256').update(aggregateCandidateHash).digest('hex');
  const perfCandidate =
    perfSamples.length === 0
      ? null
      : perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length;
  const identical = baselineHash === afterHash;
  const failedByHash = failOnIdentical && identical;
  return {
    ...scenario,
    status: failedByHash ? 'failed-identical-hash' : 'captured',
    beforeImageHash: baselineHash,
    afterImageHash: afterHash,
    deltaSummary: `variants[${modeSummaries.join(', ')}]`,
    perfBaselineMsPerSample: null,
    perfCandidateMsPerSample: perfCandidate,
    passFail: failedByHash ? 'FAIL' : 'PASS',
  };
}

const entries = [];
for (const scenario of scenarios) {
  // eslint-disable-next-line no-await-in-loop
  entries.push(await evaluateScenario(scenario));
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
