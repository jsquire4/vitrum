import {
  acquirePtDevice,
  buildCausticScene,
  buildCornellScene,
  H,
  relativeError,
  renderScene,
  rmseROI,
  W,
} from './helpers.mjs';

const REF_FRAMES = 40;
const CANDIDATE_FRAMES = 12;
const RESULT_PATH = new URL('./results-sobol.json', import.meta.url);
const FULL_ROI = { x0: 0, y0: 0, x1: W - 1, y1: H - 1 };
const PROMOTION = {
  defaultReady: false,
  evidenceClass: 'wsl-lite-equal-frame-proxy',
  reason: 'WSL-lite evidence bounds correctness but does not show equal-time convergence superiority.',
  requiredEvidence: 'full-tier/real-adapter equal-time Sobol RMSE A/B',
};

const SCENES = [
  {
    id: 'cornell-indirect',
    build: buildCornellScene,
    roi: { x0: 18, y0: 18, x1: 62, y1: 62 },
    description: 'diffuse Cornell-box indirect transport',
  },
  {
    id: 'caustic-floor',
    build: buildCausticScene,
    roi: { x0: 26, y0: 48, x1: 56, y1: 74 },
    description: 'caustic-floor convergence stress scene',
  },
];

function finiteNumber(value) {
  return Number.isFinite(value) && value >= 0;
}

function ratioOrInfinity(numerator, denominator) {
  if (Math.abs(denominator) <= 1e-9) {
    return Math.abs(numerator) <= 1e-9 ? 1.0 : Number.POSITIVE_INFINITY;
  }
  return numerator / denominator;
}

function rmseOverRoi(rgbaA, rgbaB, roi) {
  return rmseROI(rgbaA, rgbaB, W, roi.x0, roi.y0, roi.x1, roi.y1);
}

function luminanceVarianceROI(rgba, roi) {
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = roi.y0; y <= roi.y1; y++) {
    for (let x = roi.x0; x <= roi.x1; x++) {
      const i = y * W + x;
      const lum = 0.2126 * rgba[i * 4 + 0] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2];
      sum += lum;
      sumSq += lum * lum;
      count++;
    }
  }
  if (count <= 1) return 0;
  const mean = sum / count;
  return Math.max(0, sumSq / count - mean * mean);
}

async function timedRender(label, engineOpts, scene, frames, device) {
  const startedAt = performance.now();
  const result = await renderScene(engineOpts, scene, frames, device);
  const elapsedMs = performance.now() - startedAt;
  result.engine.dispose();
  await device.queue.onSubmittedWorkDone();
  return {
    label,
    rgba: result.rgba,
    elapsedMs,
  };
}

function compareScene(sceneCase, reference, pcg, sobol) {
  const pcgGlobalRmse = rmseOverRoi(pcg.rgba, reference.rgba, FULL_ROI);
  const sobolGlobalRmse = rmseOverRoi(sobol.rgba, reference.rgba, FULL_ROI);
  const pcgRoiRmse = rmseOverRoi(pcg.rgba, reference.rgba, sceneCase.roi);
  const sobolRoiRmse = rmseOverRoi(sobol.rgba, reference.rgba, sceneCase.roi);
  const pcgRoiSpatialVariance = luminanceVarianceROI(pcg.rgba, sceneCase.roi);
  const sobolRoiSpatialVariance = luminanceVarianceROI(sobol.rgba, sceneCase.roi);
  const globalRmseRatio = ratioOrInfinity(sobolGlobalRmse, pcgGlobalRmse);
  const roiRmseRatio = ratioOrInfinity(sobolRoiRmse, pcgRoiRmse);
  const roiSpatialVarianceRatio = ratioOrInfinity(sobolRoiSpatialVariance, pcgRoiSpatialVariance);
  const timeRatio = ratioOrInfinity(sobol.elapsedMs, pcg.elapsedMs);

  return {
    id: sceneCase.id,
    description: sceneCase.description,
    roi: sceneCase.roi,
    referenceFrames: REF_FRAMES,
    candidateFrames: CANDIDATE_FRAMES,
    pcg: {
      elapsedMs: Number(pcg.elapsedMs.toFixed(3)),
      globalRmse: pcgGlobalRmse,
      roiRmse: pcgRoiRmse,
      roiSpatialVariance: pcgRoiSpatialVariance,
    },
    sobol: {
      elapsedMs: Number(sobol.elapsedMs.toFixed(3)),
      globalRmse: sobolGlobalRmse,
      roiRmse: sobolRoiRmse,
      roiSpatialVariance: sobolRoiSpatialVariance,
    },
    ratios: {
      globalRmse: globalRmseRatio,
      roiRmse: roiRmseRatio,
      roiSpatialVariance: roiSpatialVarianceRatio,
      elapsedMs: timeRatio,
    },
    relativeErrors: {
      globalRmse: relativeError(sobolGlobalRmse, pcgGlobalRmse),
      roiRmse: relativeError(sobolRoiRmse, pcgRoiRmse),
      roiSpatialVariance: relativeError(sobolRoiSpatialVariance, pcgRoiSpatialVariance),
    },
    pass: finiteNumber(globalRmseRatio)
      && finiteNumber(roiRmseRatio)
      && finiteNumber(roiSpatialVarianceRatio)
      && globalRmseRatio <= 1.5
      && roiRmseRatio <= 1.5
      && timeRatio <= 20.0,
  };
}

async function main() {
  const device = await acquirePtDevice(true);
  const scenes = [];

  try {
    for (const sceneCase of SCENES) {
      const scene = sceneCase.build();
      const commonOptions = {
        traceTier: 'lite',
        requireRadiometricSignal: true,
      };

      const reference = await timedRender(
        `${sceneCase.id}:reference-pcg`,
        { ...commonOptions, sampling: 'pcg' },
        scene,
        REF_FRAMES,
        device,
      );
      const pcg = await timedRender(
        `${sceneCase.id}:candidate-pcg`,
        { ...commonOptions, sampling: 'pcg' },
        scene,
        CANDIDATE_FRAMES,
        device,
      );
      const sobol = await timedRender(
        `${sceneCase.id}:candidate-sobol`,
        { ...commonOptions, sampling: 'sobol' },
        scene,
        CANDIDATE_FRAMES,
        device,
      );

      scenes.push(compareScene(sceneCase, reference, pcg, sobol));
    }
  } finally {
    device.destroy();
  }

  const verdict = scenes.every((scene) => scene.pass) ? 'PASS' : 'FAIL';
  const result = {
    ab: 'sobol-equal-frame-rmse',
    verdict,
    generatedAt: new Date().toISOString(),
    resolution: { width: W, height: H },
    reference: {
      sampling: 'pcg',
      frames: REF_FRAMES,
      note: 'Higher-frame PCG reference. Candidate arms use equal frame budgets and record wall time.',
    },
    candidateFrames: CANDIDATE_FRAMES,
    traceTier: 'lite',
    promotion: PROMOTION,
    thresholds: {
      maxGlobalRmseRatio: 1.5,
      maxRoiRmseRatio: 1.5,
      maxElapsedMsRatio: 20.0,
    },
    roi: SCENES[0].roi,
    scenes,
  };

  await Deno.writeTextFile(RESULT_PATH, `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
  if (verdict !== 'PASS') {
    Deno.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  Deno.exit(1);
});
