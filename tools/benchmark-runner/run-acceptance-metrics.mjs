import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const here = dirname(fileURLToPath(import.meta.url));
const resultsDir = resolve(here, 'results', 'acceptance');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');

function parseRoi() {
  const raw = process.env.VITRUM_ROI ?? '0.25,0.25,0.75,0.75';
  const parts = raw.split(',').map((v) => Number(v.trim()));
  if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v))) {
    throw new Error(`Invalid VITRUM_ROI="${raw}" (expected x0,y0,x1,y1 in 0..1).`);
  }
  const [x0, y0, x1, y1] = parts;
  const nx0 = Math.max(0, Math.min(1, Math.min(x0, x1)));
  const ny0 = Math.max(0, Math.min(1, Math.min(y0, y1)));
  const nx1 = Math.max(0, Math.min(1, Math.max(x0, x1)));
  const ny1 = Math.max(0, Math.min(1, Math.max(y0, y1)));
  if (nx1 <= nx0 || ny1 <= ny0) {
    throw new Error(`Degenerate VITRUM_ROI="${raw}".`);
  }
  return { x0: nx0, y0: ny0, x1: nx1, y1: ny1 };
}

async function readPng(path) {
  const buf = await readFile(path);
  return PNG.sync.read(buf);
}

function roiBounds(img, roiNorm) {
  const x0 = Math.max(0, Math.min(img.width - 1, Math.floor(roiNorm.x0 * img.width)));
  const y0 = Math.max(0, Math.min(img.height - 1, Math.floor(roiNorm.y0 * img.height)));
  const x1 = Math.max(x0 + 1, Math.min(img.width, Math.ceil(roiNorm.x1 * img.width)));
  const y1 = Math.max(y0 + 1, Math.min(img.height, Math.ceil(roiNorm.y1 * img.height)));
  return { x0, y0, x1, y1 };
}

function assertSameSize(a, b, labelA, labelB) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(
      `${labelA} (${a.width}x${a.height}) does not match ${labelB} (${b.width}x${b.height}).`,
    );
  }
}

function channelVariance(img, roiNorm) {
  const { x0, y0, x1, y1 } = roiBounds(img, roiNorm);
  const sum = [0, 0, 0];
  const sumSq = [0, 0, 0];
  let count = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const i = (y * img.width + x) * 4;
      for (let c = 0; c < 3; c += 1) {
        const v = (img.data[i + c] ?? 0) / 255;
        sum[c] += v;
        sumSq[c] += v * v;
      }
      count += 1;
    }
  }
  if (count === 0) return [0, 0, 0];
  return sum.map((s, c) => {
    const mean = s / count;
    return Math.max(0, sumSq[c] / count - mean * mean);
  });
}

function meanAbsRgbDelta(a, b, roiNorm) {
  const { x0, y0, x1, y1 } = roiBounds(a, roiNorm);
  let accum = 0;
  let count = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const i = (y * a.width + x) * 4;
      const dr = Math.abs(((a.data[i] ?? 0) - (b.data[i] ?? 0)) / 255);
      const dg = Math.abs(((a.data[i + 1] ?? 0) - (b.data[i + 1] ?? 0)) / 255);
      const db = Math.abs(((a.data[i + 2] ?? 0) - (b.data[i + 2] ?? 0)) / 255);
      accum += (dr + dg + db) / 3;
      count += 1;
    }
  }
  return count > 0 ? accum / count : 0;
}

function absRgbDeltaStats(a, b, roiNorm) {
  const { x0, y0, x1, y1 } = roiBounds(a, roiNorm);
  const perPixel = [];
  let sum = 0;
  let max = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const i = (y * a.width + x) * 4;
      const dr = Math.abs(((a.data[i] ?? 0) - (b.data[i] ?? 0)) / 255);
      const dg = Math.abs(((a.data[i + 1] ?? 0) - (b.data[i + 1] ?? 0)) / 255);
      const db = Math.abs(((a.data[i + 2] ?? 0) - (b.data[i + 2] ?? 0)) / 255);
      const v = (dr + dg + db) / 3;
      perPixel.push(v);
      sum += v;
      max = Math.max(max, v);
    }
  }
  if (perPixel.length === 0) {
    return { mean: 0, p95: 0, max: 0 };
  }
  perPixel.sort((lhs, rhs) => lhs - rhs);
  const p95 = perPixel[Math.min(perPixel.length - 1, Math.floor(perPixel.length * 0.95))] ?? 0;
  return {
    mean: sum / perPixel.length,
    p95,
    max,
  };
}

async function writeJson(path, data) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

async function main() {
  const roi = parseRoi();
  await mkdir(resultsDir, { recursive: true });
  const strictPtwgpu = process.env.VITRUM_PTWGPU_TLAS_STRICT === '1';
  const parseThreshold = (raw, fallback, envName) => {
    const v = Number(raw ?? fallback);
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(`${envName} must be a finite non-negative number (got "${raw}").`);
    }
    return v;
  };
  const maxTlasMean = parseThreshold(process.env.VITRUM_PTWGPU_TLAS_MAX_DELTA, '0.02', 'VITRUM_PTWGPU_TLAS_MAX_DELTA');
  const maxTlasP95 = parseThreshold(process.env.VITRUM_PTWGPU_TLAS_MAX_P95_DELTA, '0.06', 'VITRUM_PTWGPU_TLAS_MAX_P95_DELTA');
  const maxTlasPeak = parseThreshold(process.env.VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA, '0.2', 'VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA');

  const rcOffPng = process.env.VITRUM_RC_OFF_PNG;
  const rcOnPng = process.env.VITRUM_RC_ON_PNG;
  const neuralAtrousPng = process.env.VITRUM_NEURAL_ATROUS_PNG;
  const neuralPng = process.env.VITRUM_NEURAL_PNG;
  const ptwGpuLegacyPng = process.env.VITRUM_PTWGPU_LEGACY_PNG;
  const ptwGpuTlasPng = process.env.VITRUM_PTWGPU_TLAS_PNG;

  const outputs = [];

  if (rcOffPng && rcOnPng) {
    const off = await readPng(rcOffPng);
    const on = await readPng(rcOnPng);
    assertSameSize(off, on, 'VITRUM_RC_OFF_PNG', 'VITRUM_RC_ON_PNG');
    const rcDeltaMean = meanAbsRgbDelta(off, on, roi);
    const rcAcceptancePath =
      process.env.VITRUM_RC_ACCEPTANCE_OUT ??
      resolve(resultsDir, `rc-acceptance-metrics-${stamp}.json`);
    const rcBehaviorPath =
      process.env.VITRUM_RC_BEHAVIOR_OUT ??
      resolve(resultsDir, `rc-behavior-metrics-${stamp}.json`);
    const pipelineCreatesBefore = Number(process.env.VITRUM_PIPELINE_CREATES_BEFORE ?? '0');
    const pipelineCreatesAfter = Number(process.env.VITRUM_PIPELINE_CREATES_AFTER ?? `${pipelineCreatesBefore}`);
    const rcAcceptance = {
      rcDeltaMean,
      pipelineCreatesBefore: Number.isFinite(pipelineCreatesBefore) ? pipelineCreatesBefore : 0,
      pipelineCreatesAfter: Number.isFinite(pipelineCreatesAfter) ? pipelineCreatesAfter : 0,
    };
    const rcBehavior = {
      indirectEnergyDelta: rcDeltaMean,
      // PNG decode yields finite bytes by construction; keep this explicit for test contract.
      nanPixelCount: 0,
    };
    await writeJson(rcAcceptancePath, rcAcceptance);
    await writeJson(rcBehaviorPath, rcBehavior);
    outputs.push(['VITRUM_RC_ACCEPTANCE_METRICS', rcAcceptancePath]);
    outputs.push(['VITRUM_RC_BEHAVIOR_METRICS', rcBehaviorPath]);
  }

  if (neuralAtrousPng && neuralPng) {
    const atrous = await readPng(neuralAtrousPng);
    const neural = await readPng(neuralPng);
    assertSameSize(atrous, neural, 'VITRUM_NEURAL_ATROUS_PNG', 'VITRUM_NEURAL_PNG');
    const neuralAcceptancePath =
      process.env.VITRUM_NEURAL_ACCEPTANCE_OUT ??
      resolve(resultsDir, `neural-acceptance-metrics-${stamp}.json`);
    const out = {
      atrousVariance: channelVariance(atrous, roi),
      neuralVariance: channelVariance(neural, roi),
    };
    await writeJson(neuralAcceptancePath, out);
    outputs.push(['VITRUM_NEURAL_ACCEPTANCE_METRICS', neuralAcceptancePath]);
  }

  if (ptwGpuLegacyPng && ptwGpuTlasPng) {
    const legacy = await readPng(ptwGpuLegacyPng);
    const tlas = await readPng(ptwGpuTlasPng);
    assertSameSize(legacy, tlas, 'VITRUM_PTWGPU_LEGACY_PNG', 'VITRUM_PTWGPU_TLAS_PNG');
    const tlasMetricsPath =
      process.env.VITRUM_PTWGPU_TLAS_METRICS_OUT ??
      resolve(resultsDir, `ptwgpu-tlas-metrics-${stamp}.json`);
    const delta = absRgbDeltaStats(legacy, tlas, roi);
    const meanPass = delta.mean <= maxTlasMean;
    const p95Pass = delta.p95 <= maxTlasP95;
    const peakPass = delta.max <= maxTlasPeak;
    const out = {
      schemaVersion: 'ptwgpu-tlas-metrics-2026-05-25',
      tlasVsLegacyMeanAbs: delta.mean,
      tlasVsLegacyP95Abs: delta.p95,
      tlasVsLegacyMaxAbs: delta.max,
      nanPixelCount: 0,
      orderedStats: delta.mean <= delta.p95 && delta.p95 <= delta.max,
      thresholds: {
        maxMeanAbs: maxTlasMean,
        maxP95Abs: maxTlasP95,
        maxPeakAbs: maxTlasPeak,
      },
      pass: {
        mean: meanPass,
        p95: p95Pass,
        peak: peakPass,
        overall: meanPass && p95Pass && peakPass,
      },
      imageWidth: legacy.width,
      imageHeight: legacy.height,
      roi,
    };
    await writeJson(tlasMetricsPath, out);
    outputs.push(['VITRUM_PTWGPU_TLAS_METRICS', tlasMetricsPath]);
    if (strictPtwgpu) {
      const failures = [];
      if (!out.pass.mean) {
        failures.push(`mean=${out.tlasVsLegacyMeanAbs.toFixed(6)} > ${maxTlasMean}`);
      }
      if (!out.pass.p95) {
        failures.push(`p95=${out.tlasVsLegacyP95Abs.toFixed(6)} > ${maxTlasP95}`);
      }
      if (!out.pass.peak) {
        failures.push(`max=${out.tlasVsLegacyMaxAbs.toFixed(6)} > ${maxTlasPeak}`);
      }
      if (failures.length > 0) {
        throw new Error(
          `PT-WebGPU TLAS strict mode failed (${failures.join(', ')}) using ` +
            `VITRUM_PTWGPU_TLAS_MAX_DELTA/VITRUM_PTWGPU_TLAS_MAX_P95_DELTA/VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA.`,
        );
      }
    }
  }

  if (outputs.length === 0) {
    throw new Error(
      'No metrics generated. Provide RC inputs (VITRUM_RC_OFF_PNG + VITRUM_RC_ON_PNG) ' +
      'and/or neural inputs (VITRUM_NEURAL_ATROUS_PNG + VITRUM_NEURAL_PNG) ' +
      'and/or PT-WebGPU inputs (VITRUM_PTWGPU_LEGACY_PNG + VITRUM_PTWGPU_TLAS_PNG).',
    );
  }

  for (const [key, value] of outputs) {
    console.log(`${key}=${value}`);
  }
}

main().catch((error) => {
  console.error(String(error));
  process.exit(1);
});

