import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import { GAP_CLOSURE_SCENARIOS } from './scenario-presets.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const inputDir = resolve(
  repoRoot,
  process.env.VITRUM_PTWEBGL_FIDELITY_DIR ?? 'tools/reference-renders/pt-webgl-fidelity',
);
const outPath = resolve(
  repoRoot,
  process.env.VITRUM_PTWEBGL_FIDELITY_OUT ??
    `tools/benchmark-runner/results/pt-webgl-fidelity-${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
);
const strict = process.env.VITRUM_PTWEBGL_FIDELITY_STRICT === '1';
const defaultRequiredScenarios = GAP_CLOSURE_SCENARIOS
  .map((scenario) => scenario.scenarioId)
  .filter((id) => /^rfe\d+/i.test(id))
  .sort();
const requiredScenarios = (
  process.env.VITRUM_PTWEBGL_FIDELITY_REQUIRED ??
  defaultRequiredScenarios.join(',')
).split(',').map((s) => s.trim()).filter(Boolean);
const minPsnr = Number(process.env.VITRUM_PTWEBGL_FIDELITY_MIN_PSNR ?? '28');
const PERFECT_MATCH_PSNR = 999;
const minPsnrByScenario = (() => {
  const raw = process.env.VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO;
  if (!raw) return {};
  const parsed = JSON.parse(raw);
  if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO must be a JSON object.');
  }
  /** @type {Record<string, number>} */
  const out = {};
  for (const [key, value] of Object.entries(parsed)) {
    const v = Number(value);
    if (!Number.isFinite(v)) {
      throw new Error(`VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO["${key}"] must be numeric.`);
    }
    out[key] = v;
  }
  return out;
})();

function scenarioMatches(id, pattern) {
  return id === pattern || id.startsWith(`${pattern}.`) || id.startsWith(`${pattern}-`);
}

function thresholdForScenario(id) {
  if (Object.hasOwn(minPsnrByScenario, id)) return minPsnrByScenario[id];
  for (const [pattern, threshold] of Object.entries(minPsnrByScenario)) {
    if (scenarioMatches(id, pattern)) return threshold;
  }
  return minPsnr;
}

function assertSameSize(a, b, name) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(
      `${name} size mismatch: baseline=${a.width}x${a.height} candidate=${b.width}x${b.height}`,
    );
  }
}

function rgbMetrics(a, b) {
  let mse = 0;
  let mae = 0;
  let n = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const dr = (a.data[i] ?? 0) - (b.data[i] ?? 0);
    const dg = (a.data[i + 1] ?? 0) - (b.data[i + 1] ?? 0);
    const db = (a.data[i + 2] ?? 0) - (b.data[i + 2] ?? 0);
    mse += (dr * dr + dg * dg + db * db) / 3;
    mae += (Math.abs(dr) + Math.abs(dg) + Math.abs(db)) / 3;
    n += 1;
  }
  const meanSquared = n > 0 ? mse / n : 0;
  const meanAbs = n > 0 ? mae / (n * 255) : 0;
  const psnr = meanSquared <= 1e-12 ? PERFECT_MATCH_PSNR : 10 * Math.log10((255 * 255) / meanSquared);
  return { psnr, meanAbsDelta: meanAbs };
}

async function readPng(path) {
  return PNG.sync.read(await readFile(path));
}

async function main() {
  const files = await readdir(inputDir);
  const baselineFiles = files.filter((f) => f.endsWith('.baseline.png'));
  const results = [];

  for (const baselineFile of baselineFiles) {
    const id = baselineFile.slice(0, -'.baseline.png'.length);
    const candidateFile = `${id}.candidate.png`;
    if (!files.includes(candidateFile)) continue;

    const baselinePath = resolve(inputDir, baselineFile);
    const candidatePath = resolve(inputDir, candidateFile);
    const baseline = await readPng(baselinePath);
    const candidate = await readPng(candidatePath);
    assertSameSize(baseline, candidate, id);
    const metrics = rgbMetrics(baseline, candidate);
    const threshold = thresholdForScenario(id);
    results.push({
      scenarioId: id,
      baseline: basename(baselinePath),
      candidate: basename(candidatePath),
      width: baseline.width,
      height: baseline.height,
      psnr: metrics.psnr,
      meanAbsDelta: metrics.meanAbsDelta,
      minPsnr: threshold,
      pass: Number.isFinite(metrics.psnr) ? metrics.psnr >= threshold : true,
    });
  }

  const matchedRequired = Object.fromEntries(
    requiredScenarios.map((id) => [
      id,
      results.filter((r) => scenarioMatches(r.scenarioId, id)).map((r) => r.scenarioId),
    ]),
  );
  const missingRequired = requiredScenarios.filter((id) => (matchedRequired[id] ?? []).length === 0);
  const failingScenarios = results.filter((r) => !r.pass).map((r) => r.scenarioId);
  const report = {
    schemaVersion: 'pt-webgl-fidelity-2026-05-25',
    generatedAt: new Date().toISOString(),
    inputDir,
    requiredScenarios,
    minPsnr,
    minPsnrByScenario,
    matchedRequired,
    missingRequired,
    failingScenarios,
    allRequiredPresent: missingRequired.length === 0,
    allPassing: failingScenarios.length === 0 && missingRequired.length === 0,
    results,
  };

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`VITRUM_PTWEBGL_FIDELITY_METRICS=${outPath}`);
  if (strict && (!report.allRequiredPresent || !report.allPassing)) {
    throw new Error(
      `PT-WebGL fidelity strict mode failed: ` +
        `missingRequired=${report.missingRequired.join(',') || 'none'}, ` +
        `failingScenarios=${report.failingScenarios.join(',') || 'none'}.`,
    );
  }
}

main().catch((error) => {
  console.error(String(error));
  process.exit(1);
});

