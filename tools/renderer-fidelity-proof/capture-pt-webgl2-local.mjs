#!/usr/bin/env node
/**
 * Deterministic local-browser fidelity capture for the native pt-webgl2 backend.
 *
 * The harness renders core Scene fixtures through the public engine API, reads
 * the linear accumulator through captureFrame(), writes human-inspectable PNGs,
 * and records machine-checkable signal/delta/agreement metrics.
 */

import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const exampleDir = resolve(repoRoot, 'examples/pt-webgl2-direct');
const outputDir = resolve(repoRoot, 'tools/reference-renders/pt-webgl2-local');
const statusPath = resolve(scriptDir, 'pt-webgl2-local-status.json');
const port = Number(process.env.VITRUM_PT_WEBGL2_PROOF_PORT ?? '5192');
const width = Number(process.env.VITRUM_WIDTH ?? '64');
const height = Number(process.env.VITRUM_HEIGHT ?? '64');
const spp = Number(process.env.VITRUM_SPP ?? '32');
const bounces = Number(process.env.VITRUM_BOUNCES ?? '6');
const bdptMaxLightBounces = process.env.VITRUM_BDPT_MAX_LIGHT_BOUNCES == null
  ? null
  : Number(process.env.VITRUM_BDPT_MAX_LIGHT_BOUNCES);
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? '120000');
const browserGlBackend = process.env.VITRUM_PLAYWRIGHT_GL ?? 'swiftshader';
const traceProbeStage = Number(process.env.VITRUM_PT_WEBGL2_PROBE_STAGE ?? '0');
const bsdfOracleSeedOffsets = (
  process.env.VITRUM_PT_WEBGL2_ORACLE_SEEDS ?? '0,2654435761'
)
  .split(',')
  .map((value) => Number(value.trim()));
const bsdfOracleRepeats = Number(
  process.env.VITRUM_PT_WEBGL2_ORACLE_REPEATS ?? '2',
);
const captureFrameSeedOffset = Number(
  process.env.VITRUM_FRAME_SEED_OFFSET ?? '0',
);

const ALL_CASES = [
  { id: 'smoke-sky', scenario: 'smoke-sky', spectral: false, bdpt: false, sampling: 'pcg' },
  { id: 'smoke-triangle', scenario: 'smoke-triangle', spectral: false, bdpt: false, sampling: 'pcg' },
  { id: 'cornell-pcg', scenario: 'cornell', spectral: false, bdpt: false, sampling: 'pcg' },
  { id: 'cornell-sobol', scenario: 'cornell', spectral: false, bdpt: false, sampling: 'sobol' },
  { id: 'cornell-spectral', scenario: 'cornell', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'beer-flat', scenario: 'beer-flat', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'beer-curve', scenario: 'beer-curve', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'thinfilm-off', scenario: 'thinfilm-off', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'thinfilm-on', scenario: 'thinfilm-on', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'dispersion-high', scenario: 'dispersion-high', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'dispersion-low', scenario: 'dispersion-low', spectral: true, bdpt: false, sampling: 'pcg' },
  { id: 'cornell-bdpt', scenario: 'cornell', spectral: false, bdpt: true, sampling: 'pcg' },
];
const BSDF_ORACLE_CASE = {
  id: 'bsdf-oracle',
  scenario: 'bsdf-oracle',
  spectral: true,
  bdpt: false,
  sampling: 'pcg',
};
const KNOWN_CASES = [...ALL_CASES, BSDF_ORACLE_CASE];
const requestedCases = new Set(
  (process.env.VITRUM_PT_WEBGL2_CASES ?? '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean),
);
const CASES = requestedCases.size === 0
  ? traceProbeStage === 32
    ? [BSDF_ORACLE_CASE]
    : ALL_CASES
  : KNOWN_CASES.filter(({ id }) => requestedCases.has(id));

const EFFECT_PAIRS = [
  {
    id: 'hero-wavelength',
    baseline: 'cornell-pcg',
    feature: 'cornell-spectral',
    minimumRelativeMae: 0.005,
  },
  {
    id: 'spectral-beer-lambert',
    baseline: 'beer-flat',
    feature: 'beer-curve',
    minimumRelativeMae: 0.005,
  },
  {
    id: 'multi-layer-thin-film',
    baseline: 'thinfilm-off',
    feature: 'thinfilm-on',
    minimumRelativeMae: 0.002,
  },
  {
    id: 'cauchy-dispersion',
    baseline: 'dispersion-high',
    feature: 'dispersion-low',
    minimumRelativeMae: 0.002,
  },
];

const AGREEMENT_PAIRS = [
  {
    id: 'sobol-sequence',
    baseline: 'cornell-pcg',
    feature: 'cornell-sobol',
    maximumMeanLuminanceRelativeError: 0.20,
  },
  {
    id: 'bounded-bdpt',
    baseline: 'cornell-pcg',
    feature: 'cornell-bdpt',
    maximumMeanLuminanceRelativeError: 0.20,
  },
];

validateConfiguration();
if (CASES.length === 0) {
  throw new Error(
    `VITRUM_PT_WEBGL2_CASES selected no known cases; expected one of ${KNOWN_CASES.map(({ id }) => id).join(', ')}`,
  );
}
await mkdir(outputDir, { recursive: true });

// Serve an immutable production bundle. A dev-server watcher can reload the
// proof page when another validation lane edits a transitive workspace source,
// tearing down an in-flight GL context and turning a valid render into a timeout.
const viteCli = resolve(repoRoot, 'node_modules/vite/bin/vite.js');
const build = spawnSync(process.execPath, [viteCli, 'build'], {
  cwd: exampleDir,
  env: process.env,
  encoding: 'utf8',
});
if (build.status !== 0) {
  throw new Error(
    `pt-webgl2 proof bundle failed (${String(build.status)}):\n${build.stdout ?? ''}\n${build.stderr ?? ''}`,
  );
}

const server = spawn(
  process.execPath,
  [
    viteCli,
    'preview',
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--strictPort',
  ],
  {
    cwd: exampleDir,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  },
);

let serverLog = '';
server.stdout.on('data', (chunk) => { serverLog += chunk.toString(); });
server.stderr.on('data', (chunk) => { serverLog += chunk.toString(); });

let browser;
let status;
try {
  await waitForServer();
  const { chromium } = await import('playwright');
  browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      ...browserGlArgs(browserGlBackend),
    ],
  });
  const browserVersion = browser.version();
  const captures = [];
  const linearById = new Map();
  let bsdfOracle = null;
  for (const captureCase of CASES) {
    if (traceProbeStage === 32 && captureCase.id === 'bsdf-oracle') {
      const runs = [];
      for (const seedOffset of bsdfOracleSeedOffsets) {
        for (let repeat = 0; repeat < bsdfOracleRepeats; repeat += 1) {
          const suffix = `seed-${seedOffset}-repeat-${repeat + 1}`;
          const genericCase = {
            ...captureCase,
            id: `bsdf-oracle-generic-${suffix}`,
          };
          const compactCase = {
            ...captureCase,
            id: `bsdf-oracle-compact-${suffix}`,
          };
          const generic = await captureCaseInBrowser(
            browser,
            genericCase,
            32,
            seedOffset,
          );
          const compact = await captureCaseInBrowser(
            browser,
            compactCase,
            33,
            seedOffset,
          );
          captures.push(generic.summary, compact.summary);
          linearById.set(genericCase.id, generic.rgba);
          linearById.set(compactCase.id, compact.rgba);
          runs.push({
            seedOffset,
            repeat: repeat + 1,
            genericHash: generic.summary.linearSha256,
            compactHash: compact.summary.linearSha256,
            ...compareBsdfOracle(generic.rgba, compact.rgba),
          });
        }
      }
      bsdfOracle = summarizeBsdfOracleRuns(runs);
      console.log(
        `[pt-webgl2-local] bsdf-oracle: logicalSamples=${bsdfOracle.totalLogicalSamples} ` +
        `runs=${runs.length} ` +
        `maxRelativeError=${bsdfOracle.maxRelativeError}`,
      );
      continue;
    }
    const result = await captureCaseInBrowser(
      browser,
      captureCase,
      traceProbeStage,
      captureFrameSeedOffset,
    );
    captures.push(result.summary);
    linearById.set(captureCase.id, result.rgba);
    console.log(
      `[pt-webgl2-local] ${captureCase.id}: meanLum=${result.summary.signal.meanLuminance.toFixed(6)} ` +
      `hash=${result.summary.linearSha256.slice(0, 12)}`,
    );
  }

  const capturedIds = new Set(captures.map(({ id }) => id));
  const effects = EFFECT_PAIRS
    .filter((pair) => capturedIds.has(pair.baseline) && capturedIds.has(pair.feature))
    .map((pair) => {
    const metrics = compareLinear(linearById.get(pair.baseline), linearById.get(pair.feature));
    return {
      ...pair,
      metrics,
      verdict: metrics.relativeMae >= pair.minimumRelativeMae ? 'PASS' : 'FAIL',
    };
  });
  const agreements = AGREEMENT_PAIRS
    .filter((pair) => capturedIds.has(pair.baseline) && capturedIds.has(pair.feature))
    .map((pair) => {
    const baseline = captureById(captures, pair.baseline);
    const feature = captureById(captures, pair.feature);
    const metrics = compareLinear(linearById.get(pair.baseline), linearById.get(pair.feature));
    const meanLuminanceRelativeError = relativeError(
      feature.signal.meanLuminance,
      baseline.signal.meanLuminance,
    );
    return {
      ...pair,
      metrics,
      meanLuminanceRelativeError,
      verdict:
        meanLuminanceRelativeError <= pair.maximumMeanLuminanceRelativeError
          ? 'PASS'
          : 'FAIL',
    };
  });

  const structurePass = captures.every((capture) => capture.signal.verdict === 'PASS');
  const verdict =
    structurePass &&
    effects.every((pair) => pair.verdict === 'PASS') &&
    agreements.every((pair) => pair.verdict === 'PASS') &&
    (bsdfOracle == null || bsdfOracle.verdict === 'PASS')
      ? 'PASS'
      : 'FAIL';
  status = {
    generatedAt: new Date().toISOString(),
    harness: 'renderer-fidelity-proof:pt-webgl2-local-browser',
    verdict,
    backend: 'pt-webgl2',
    browserGlBackend,
    browserVersion,
    resolution: { width, height },
    samplesPerPixel: spp,
    traceProbeStage,
    fixture: 'core-scene-cornell-plus-deterministic-material-panels',
    captures,
    effects,
    agreements,
    bsdfOracle,
    materialFixtureParity: capturedIds.has('cornell-pcg')
      ? {
          verdict: captureById(captures, 'cornell-pcg').signal.verdict,
          basis:
            'The public pt-webgl2 engine rendered the shared core Scene Cornell fixture with finite, non-black, spatially varying linear output.',
          captureId: 'cornell-pcg',
        }
      : null,
    provenance: await provenance(browserVersion),
  };
} catch (error) {
  status = {
    generatedAt: new Date().toISOString(),
    harness: 'renderer-fidelity-proof:pt-webgl2-local-browser',
    verdict: 'FAIL',
    backend: 'pt-webgl2',
    browserGlBackend,
    traceProbeStage,
    error: String(error?.stack ?? error),
    serverLog: serverLog.slice(-4000),
    provenance: await provenance(browser?.version?.() ?? null),
  };
} finally {
  await bestEffortWithin(browser?.close(), 5000);
  await stopServer();
}

await writeFile(statusPath, `${JSON.stringify(status, null, 2)}\n`);
console.log(`[pt-webgl2-local] ${status.verdict}: ${relative(repoRoot, statusPath)}`);
process.exit(status.verdict === 'PASS' ? 0 : 1);

async function captureCaseInBrowser(
  activeBrowser,
  captureCase,
  probeStage = traceProbeStage,
  frameSeedOffset = 0,
) {
  const context = await activeBrowser.newContext({ viewport: { width, height } });
  const page = await context.newPage();
  const consoleLines = [];
  page.on('console', (message) => consoleLines.push(`${message.type()}: ${message.text()}`));
  page.on('pageerror', (error) => consoleLines.push(`pageerror: ${String(error)}`));
  try {
    const url = new URL(`http://127.0.0.1:${port}/`);
    url.searchParams.set('vitrumSpp', String(spp));
    url.searchParams.set('vitrumBounces', String(bounces));
    url.searchParams.set('vitrumScenario', captureCase.scenario);
    url.searchParams.set('vitrumSpectral', captureCase.spectral ? '1' : '0');
    url.searchParams.set('vitrumBdpt', captureCase.bdpt ? '1' : '0');
    if (captureCase.bdpt && bdptMaxLightBounces != null) {
      url.searchParams.set(
        'vitrumBdptMaxLightBounces',
        String(bdptMaxLightBounces),
      );
    }
    url.searchParams.set('vitrumSampling', captureCase.sampling);
    url.searchParams.set('vitrumCaptureMode', '1');
    url.searchParams.set('vitrumProbeStage', String(probeStage));
    url.searchParams.set('vitrumFrameSeedOffset', String(frameSeedOffset));
    await page.goto(url.toString(), { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await page.waitForFunction(
      () => globalThis.VITRUM_CAPTURE_READY === true || globalThis.VITRUM_CAPTURE_ERROR != null,
      null,
      { timeout: timeoutMs },
    );
    const firstSignal = await page.evaluate(() => ({
      error: globalThis.VITRUM_CAPTURE_ERROR ?? null,
      ready: globalThis.VITRUM_CAPTURE_READY === true,
      diagnostics: globalThis.VITRUM_CAPTURE_DIAGNOSTICS ?? null,
    }));
    if (firstSignal.error != null) throw new Error(String(firstSignal.error));
    if (firstSignal.diagnostics?.firstNonzeroSpp == null) {
      throw new Error(
        `${captureCase.id} reached its ${spp}-SPP bound without a finite non-black accumulation frame`,
      );
    }
    const payload = await withTimeout(
      page.evaluate(async (probeStage) => {
        if (globalThis.VITRUM_CAPTURE_ERROR != null) {
          throw new Error(String(globalThis.VITRUM_CAPTURE_ERROR));
        }
        if (typeof globalThis.VITRUM_CAPTURE_FRAME !== 'function') {
          throw new Error('VITRUM_CAPTURE_FRAME is unavailable');
        }
        const canvas = document.querySelector('canvas');
        const gl = canvas instanceof HTMLCanvasElement ? canvas.getContext('webgl2') : null;
        const debugInfo = gl?.getExtension('WEBGL_debug_renderer_info') ?? null;
        const frame = await globalThis.VITRUM_CAPTURE_FRAME(
          probeStage === 8 ? 'output' : 'linear',
        );
        return {
          frame,
          telemetry: globalThis.VITRUM_CAPTURE_TELEMETRY ?? null,
          diagnostics: globalThis.VITRUM_CAPTURE_DIAGNOSTICS ?? null,
          adapter: gl == null
            ? null
            : {
                renderer:
                  debugInfo == null
                    ? String(gl.getParameter(gl.RENDERER))
                    : String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)),
                vendor:
                  debugInfo == null
                    ? String(gl.getParameter(gl.VENDOR))
                    : String(gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)),
                version: String(gl.getParameter(gl.VERSION)),
                shadingLanguageVersion: String(gl.getParameter(gl.SHADING_LANGUAGE_VERSION)),
                extensions: gl.getSupportedExtensions() ?? [],
              },
        };
      }, probeStage),
      timeoutMs,
      `${captureCase.id} linear capture timed out`,
    );
    const frame = payload.frame;
    if (
      frame == null ||
      frame.width !== width ||
      frame.height !== height ||
      !Array.isArray(frame.rgba) ||
      frame.rgba.length !== width * height * 4
    ) {
      throw new Error(`${captureCase.id} returned an invalid linear frame`);
    }
    const rgba = Float32Array.from(frame.rgba);
    const signal = analyzeSignal(rgba, probeStage > 0);
    const pngPath = resolve(outputDir, `${captureCase.id}.png`);
    await writeFile(pngPath, linearToPng(rgba));
    return {
      rgba,
      summary: {
        ...captureCase,
        telemetry: payload.telemetry,
        diagnostics: payload.diagnostics,
        adapter: payload.adapter,
        signal,
        linearSha256: hashFloat32(rgba),
        pngPath: relative(repoRoot, pngPath),
        pngSha256: createHash('sha256').update(await readFile(pngPath)).digest('hex'),
        console: consoleLines.slice(-40),
      },
    };
  } catch (error) {
    throw new Error(
      `${captureCase.id} browser capture failed: ${String(error?.stack ?? error)}\n` +
      `console:\n${consoleLines.slice(-80).join('\n')}`,
    );
  } finally {
    await bestEffortWithin(page.evaluate(() => {
      if (typeof globalThis.VITRUM_DISPOSE === 'function') globalThis.VITRUM_DISPOSE();
    }), 3000);
    await bestEffortWithin(page.close({ runBeforeUnload: false }), 3000);
    await bestEffortWithin(context.close(), 3000);
  }
}

function compareBsdfOracle(generic, compact) {
  if (
    !(generic instanceof Float32Array) ||
    !(compact instanceof Float32Array) ||
    generic.length !== compact.length ||
    generic.length % 8 !== 0
  ) {
    throw new Error(
      'compact BSDF oracle requires equal even-pixel Float32Array captures',
    );
  }
  const maximumRelativeError = 2e-5;
  const maximumAbsoluteError = 2e-6;
  let maxRelativeError = 0;
  let maxAbsoluteError = 0;
  let failingChannels = 0;
  let finiteChannels = 0;
  for (let i = 0; i < generic.length; i += 1) {
    const reference = generic[i];
    const candidate = compact[i];
    if (!Number.isFinite(reference) || !Number.isFinite(candidate)) continue;
    finiteChannels += 1;
    const absoluteError = Math.abs(candidate - reference);
    const relativeError = absoluteError / Math.max(Math.abs(reference), 1e-6);
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError);
    maxRelativeError = Math.max(maxRelativeError, relativeError);
    if (
      absoluteError > maximumAbsoluteError &&
      relativeError > maximumRelativeError
    ) {
      failingChannels += 1;
    }
  }
  const finiteFraction = finiteChannels / generic.length;
  return {
    logicalSamples: generic.length / 8,
    finiteFraction,
    maxRelativeError,
    maxAbsoluteError,
    maximumRelativeError,
    maximumAbsoluteError,
    failingChannels,
    comparedChannels: generic.length,
    verdict:
      finiteFraction === 1 && failingChannels === 0 ? 'PASS' : 'FAIL',
  };
}

function summarizeBsdfOracleRuns(runs) {
  if (runs.length === 0) throw new Error('compact BSDF oracle produced no runs');
  const coverage = analyzeBsdfOracleCoverage(
    (width * height) / 2,
    bsdfOracleSeedOffsets,
  );
  const repeatDeterminism = bsdfOracleSeedOffsets.map((seedOffset) => {
    const seedRuns = runs.filter((run) => run.seedOffset === seedOffset);
    const genericHashes = new Set(seedRuns.map((run) => run.genericHash));
    const compactHashes = new Set(seedRuns.map((run) => run.compactHash));
    return {
      seedOffset,
      genericUniqueHashes: genericHashes.size,
      compactUniqueHashes: compactHashes.size,
      verdict:
        seedRuns.length === bsdfOracleRepeats &&
        genericHashes.size === 1 &&
        compactHashes.size === 1
          ? 'PASS'
          : 'FAIL',
    };
  });
  return {
    logicalSamplesPerRun: runs[0].logicalSamples,
    totalLogicalSamples: runs.reduce(
      (sum, run) => sum + run.logicalSamples,
      0,
    ),
    deterministicSeedOffsets: bsdfOracleSeedOffsets,
    repeatsPerSeed: bsdfOracleRepeats,
    finiteFraction: Math.min(...runs.map((run) => run.finiteFraction)),
    maxRelativeError: Math.max(...runs.map((run) => run.maxRelativeError)),
    maxAbsoluteError: Math.max(...runs.map((run) => run.maxAbsoluteError)),
    maximumRelativeError: runs[0].maximumRelativeError,
    maximumAbsoluteError: runs[0].maximumAbsoluteError,
    failingChannels: runs.reduce(
      (sum, run) => sum + run.failingChannels,
      0,
    ),
    comparedChannels: runs.reduce(
      (sum, run) => sum + run.comparedChannels,
      0,
    ),
    coverage,
    repeatDeterminism,
    runs,
    verdict:
      runs.every((run) => run.verdict === 'PASS') &&
      coverage.verdict === 'PASS' &&
      repeatDeterminism.every((entry) => entry.verdict === 'PASS')
        ? 'PASS'
        : 'FAIL',
  };
}

function analyzeBsdfOracleCoverage(logicalSamples, seedOffsets) {
  const branches = {
    opaqueDielectric: false,
    opaqueMetal: false,
    transmissionAndIor: false,
    nearSmoothRoughness: false,
    fullyRough: false,
    clearcoat: false,
    clearcoatRoughnessExtremes: false,
    sheen: false,
    sheenRoughnessExtremes: false,
    isotropic: false,
    anisotropic: false,
    iridescence: false,
    thinWalled: false,
    multilayerThinFilm: false,
    angleDependentThinFilm: false,
    frontFace: false,
    backFace: false,
    reflectionHemisphere: false,
    transmissionHemisphere: false,
    outgoingGrazing: false,
    incomingGrazing: false,
    wavelengthEndpoints: false,
  };
  let sawClearcoatSmooth = false;
  let sawClearcoatRough = false;
  let sawSheenSmooth = false;
  let sawSheenRough = false;
  let sawWavelengthLow = false;
  let sawWavelengthHigh = false;
  for (const seedOffset of seedOffsets) {
    const shaderSeed = (1013904223 + seedOffset) >>> 0;
    const seedIndex = shaderSeed % 4093;
    for (let logicalIndex = 0; logicalIndex < logicalSamples; logicalIndex += 1) {
      const index = logicalIndex + seedIndex;
      const roughness = index % 9;
      const metalness = Math.floor(index / 9) % 5;
      const transmission = Math.floor(index / 45) % 5;
      const clearcoat = Math.floor(index / 225) % 5;
      const clearcoatRoughness = Math.floor(index / 17) % 9;
      const sheen = Math.floor(index / 7) % 5;
      const sheenRoughness = Math.floor(index / 19) % 9;
      const iridescence = Math.floor(index / 11) % 5;
      const ior = Math.floor(index / 41) % 9;
      const anisotropy = Math.floor(index / 71) % 5;
      const wavelength = Math.floor(index / 67) % 17;
      branches.opaqueDielectric ||= transmission === 0 && metalness === 0;
      branches.opaqueMetal ||= transmission === 0 && metalness === 4;
      branches.transmissionAndIor ||= transmission > 0 && ior > 0;
      branches.nearSmoothRoughness ||= roughness === 0;
      branches.fullyRough ||= roughness === 8;
      branches.clearcoat ||= clearcoat > 0;
      branches.sheen ||= sheen > 0;
      branches.isotropic ||= anisotropy === 0;
      branches.anisotropic ||= anisotropy > 0;
      branches.iridescence ||= iridescence > 0;
      branches.thinWalled ||= Math.floor(index / 13) % 2 === 1;
      branches.multilayerThinFilm ||= Math.floor(index / 31) % 2 === 1;
      branches.angleDependentThinFilm ||= Math.floor(index / 37) % 2 === 1;
      branches.frontFace ||= Math.floor(index / 79) % 2 === 0;
      branches.backFace ||= Math.floor(index / 79) % 2 === 1;
      branches.reflectionHemisphere ||= Math.floor(index / 61) % 2 === 0;
      branches.transmissionHemisphere ||= Math.floor(index / 61) % 2 === 1;
      branches.outgoingGrazing ||= Math.floor(index / 43) % 11 === 0;
      branches.incomingGrazing ||= Math.floor(index / 47) % 11 === 0;
      sawClearcoatSmooth ||= clearcoatRoughness === 0;
      sawClearcoatRough ||= clearcoatRoughness === 8;
      sawSheenSmooth ||= sheenRoughness === 0;
      sawSheenRough ||= sheenRoughness === 8;
      sawWavelengthLow ||= wavelength === 0;
      sawWavelengthHigh ||= wavelength === 16;
    }
  }
  branches.clearcoatRoughnessExtremes =
    sawClearcoatSmooth && sawClearcoatRough;
  branches.sheenRoughnessExtremes = sawSheenSmooth && sawSheenRough;
  branches.wavelengthEndpoints = sawWavelengthLow && sawWavelengthHigh;
  const missingBranches = Object.entries(branches)
    .filter(([, covered]) => !covered)
    .map(([branch]) => branch);
  return {
    branches,
    missingBranches,
    verdict: missingBranches.length === 0 ? 'PASS' : 'FAIL',
  };
}

function analyzeSignal(rgba, probeMode = false) {
  let finiteChannels = 0;
  let nonBlackPixels = 0;
  let meanLuminance = 0;
  let minLuminance = Infinity;
  let maxLuminance = -Infinity;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    if (Number.isFinite(r)) finiteChannels += 1;
    if (Number.isFinite(g)) finiteChannels += 1;
    if (Number.isFinite(b)) finiteChannels += 1;
    const luminance =
      Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)
        ? 0.2126 * r + 0.7152 * g + 0.0722 * b
        : 0;
    if (luminance > 1e-5) nonBlackPixels += 1;
    meanLuminance += luminance;
    minLuminance = Math.min(minLuminance, luminance);
    maxLuminance = Math.max(maxLuminance, luminance);
  }
  const pixelCount = rgba.length / 4;
  meanLuminance /= pixelCount;
  const finiteFraction = finiteChannels / (pixelCount * 3);
  const nonBlackFraction = nonBlackPixels / pixelCount;
  const luminanceRange = maxLuminance - minLuminance;
  const pass =
    finiteFraction === 1 &&
    meanLuminance > 1e-5 &&
    nonBlackFraction > 0.02 &&
    (probeMode || luminanceRange > 1e-4);
  return {
    finiteFraction,
    nonBlackFraction,
    meanLuminance,
    minLuminance,
    maxLuminance,
    luminanceRange,
    thresholds: {
      finiteFraction: 1,
      minimumMeanLuminance: 1e-5,
      minimumNonBlackFraction: 0.02,
      minimumLuminanceRange: 1e-4,
    },
    verdict: pass ? 'PASS' : 'FAIL',
  };
}

function compareLinear(a, b) {
  if (!(a instanceof Float32Array) || !(b instanceof Float32Array) || a.length !== b.length) {
    throw new Error('linear comparison requires equal Float32Array captures');
  }
  let absolute = 0;
  let squared = 0;
  let baselineMagnitude = 0;
  let maxAbs = 0;
  let channels = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const av = a[i + channel];
      const bv = b[i + channel];
      const delta = Math.abs(av - bv);
      absolute += delta;
      squared += delta * delta;
      baselineMagnitude += Math.abs(av);
      maxAbs = Math.max(maxAbs, delta);
      channels += 1;
    }
  }
  const mae = absolute / channels;
  return {
    mae,
    rmse: Math.sqrt(squared / channels),
    maxAbs,
    relativeMae: absolute / Math.max(baselineMagnitude, 1e-20),
  };
}

function linearToPng(rgba) {
  const png = new PNG({ width, height });
  for (let i = 0; i < rgba.length; i += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const linear = Math.max(0, Number.isFinite(rgba[i + channel]) ? rgba[i + channel] : 0);
      const mapped = linear / (1 + linear);
      const srgb = mapped <= 0.0031308
        ? 12.92 * mapped
        : 1.055 * Math.pow(mapped, 1 / 2.4) - 0.055;
      png.data[i + channel] = Math.round(Math.min(1, Math.max(0, srgb)) * 255);
    }
    png.data[i + 3] = 255;
  }
  return PNG.sync.write(png);
}

function hashFloat32(values) {
  return createHash('sha256')
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest('hex');
}

function relativeError(a, b) {
  return Math.abs(a - b) / Math.max(Math.abs(b), 1e-20);
}

function captureById(captures, id) {
  const capture = captures.find((candidate) => candidate.id === id);
  if (capture == null) throw new Error(`missing capture ${id}`);
  return capture;
}

async function provenance(browserVersion) {
  const files = [
    'tools/renderer-fidelity-proof/capture-pt-webgl2-local.mjs',
    'examples/pt-webgl2-direct/src/main.ts',
    'packages/pt-webgl2/src/index.ts',
  ];
  return {
    schema: 'vitrum.renderer-fidelity.pt-webgl2-local.v1',
    browserVersion,
    files: await Promise.all(files.map(async (path) => ({
      path,
      sha256: createHash('sha256').update(await readFile(resolve(repoRoot, path))).digest('hex'),
    }))),
  };
}

function validateConfiguration() {
  for (const [label, value] of [
    ['width', width],
    ['height', height],
    ['spp', spp],
    ['bounces', bounces],
    ['timeoutMs', timeoutMs],
    ['port', port],
    ['traceProbeStage', traceProbeStage],
  ]) {
    if (!Number.isInteger(value) || value < (label === 'traceProbeStage' ? 0 : 1)) {
      throw new Error(
        label === 'traceProbeStage'
          ? 'traceProbeStage must be an integer from 0 through 33'
          : `${label} must be a positive integer`,
      );
    }
  }
  if (traceProbeStage > 33) throw new Error('traceProbeStage must be an integer from 0 through 33');
  if (
    !Number.isInteger(captureFrameSeedOffset) ||
    captureFrameSeedOffset < 0 ||
    captureFrameSeedOffset > 0xffffffff
  ) {
    throw new Error('VITRUM_FRAME_SEED_OFFSET must be a uint32 integer');
  }
  if (
    bdptMaxLightBounces != null &&
    (!Number.isInteger(bdptMaxLightBounces) ||
      bdptMaxLightBounces < 1 ||
      bdptMaxLightBounces > 8)
  ) {
    throw new Error('VITRUM_BDPT_MAX_LIGHT_BOUNCES must be an integer from 1 through 8');
  }
  if (traceProbeStage === 32) {
    if (width * height % 2 !== 0) {
      throw new Error('BSDF oracle resolution must contain an even number of pixels');
    }
    if (!Number.isInteger(bsdfOracleRepeats) || bsdfOracleRepeats < 2) {
      throw new Error('VITRUM_PT_WEBGL2_ORACLE_REPEATS must be an integer of at least 2');
    }
    if (
      bsdfOracleSeedOffsets.length < 2 ||
      new Set(bsdfOracleSeedOffsets).size !== bsdfOracleSeedOffsets.length ||
      bsdfOracleSeedOffsets.some((seed) =>
        !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff
      )
    ) {
      throw new Error(
        'VITRUM_PT_WEBGL2_ORACLE_SEEDS must contain at least two distinct uint32 integers',
      );
    }
  }
  if (!['swiftshader', 'angle-gl', 'angle-vulkan', 'egl'].includes(browserGlBackend)) {
    throw new Error(
      `VITRUM_PLAYWRIGHT_GL must be swiftshader, angle-gl, angle-vulkan, or egl; got ${browserGlBackend}`,
    );
  }
}

function browserGlArgs(backend) {
  switch (backend) {
    case 'angle-gl':
      return ['--use-gl=angle', '--use-angle=gl'];
    case 'angle-vulkan':
      return ['--use-gl=angle', '--use-angle=vulkan'];
    case 'egl':
      return ['--use-gl=egl'];
    default:
      return ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'];
  }
}

async function waitForServer() {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (server.exitCode != null) {
      throw new Error(`vite exited early with code ${server.exitCode}: ${serverLog}`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/`);
      if (response.ok) return;
    } catch {
      // Retry until the deadline.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error(`vite did not become ready within ${timeoutMs}ms: ${serverLog}`);
}

async function stopServer() {
  if (server.exitCode != null || server.signalCode != null) return;
  server.kill('SIGTERM');
  await new Promise((resolvePromise) => {
    const timer = setTimeout(() => {
      if (server.exitCode == null && server.signalCode == null) server.kill('SIGKILL');
      resolvePromise(undefined);
    }, 1500);
    server.once('exit', () => {
      clearTimeout(timer);
      resolvePromise(undefined);
    });
  });
}

async function withTimeout(promise, milliseconds, message) {
  let timer;
  const guarded = Promise.resolve(promise);
  guarded.catch(() => undefined);
  try {
    return await Promise.race([
      guarded,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function bestEffortWithin(promise, milliseconds) {
  if (promise == null) return;
  await Promise.race([
    Promise.resolve(promise).catch(() => undefined),
    new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds)),
  ]);
}
