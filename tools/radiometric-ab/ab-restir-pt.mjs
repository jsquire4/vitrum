#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/ab-restir-pt.mjs
 *
 * A/B #3: ReSTIR-PT reuse on vs off — bias check + variance reduction
 *
 * One-edge GRIS reconnection (`oneEdgeReconnectionReuse:true`) composites the
 * reconnection-indirect
 * estimate into the beauty accumulator via the COMPOSITE megakernel path
 * (A1, kernel.wgsl.ts:308-312).  The estimator split is E0-direct-only in
 * the megakernel + rpt_result indirect from the resolve pass.  If the split
 * is double-count-free and unbiased, the converged mean should agree with
 * the plain path tracer (`oneEdgeReconnectionReuse:false`).
 *
 * This A/B tests:
 *   (a) Unbiasedness: equal-spp global mean luminance must agree within 10%,
 *       and a paired independent-seed 95% confidence interval must fit wholly
 *       inside that equivalence margin.
 *   (b) Variance: 8×8-frame runs estimate per-pixel variance in an
 *       indirect-lit ROI. Reconnection reuse should not INCREASE variance
 *       beyond 2×.
 *
 * Device requirement: oneEdgeReconnectionReuse=true requires full-tier limits
 * (maxStorageBuffersPerShaderStage ≥ 28).  The gate acquires these limits.
 * On lavapipe, the full-tier flag is auto-set by acquirePtDevice(true).
 *
 * Usage
 * ─────
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *     deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
 *     tools/radiometric-ab/ab-restir-pt.mjs
 */

import {
  buildCornellScene,
  renderScene,
  renderMultipleRuns,
  ptRadiometricSeedManifest,
  meanLuminanceROI,
  varianceROI,
  relativeError,
  W, H,
} from "./helpers.mjs";
import {
  PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
  radiometricResultProvenance,
} from "./resultProvenance.mjs";
import {
  buildRestirPtCaptureConfig,
  buildRestirPtResult,
} from "./restirPtResultContract.mjs";

console.log("=== A/B #3: ReSTIR-PT reuse on vs off (bias check + variance) ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}`);
console.log("");

const scene = buildCornellScene();
const FULL_TIER = { traceTier: "full", requireFullTier: true, requireRadiometricSignal: true };
const EQUIVALENCE_MARGIN = 0.10;
const VARIANCE_RATIO_MAX = 2.0;
const PAIRED_T_CRITICAL_95_DF7 = 2.364624251;

function arithmeticMean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStdDev(values, mean) {
  if (values.length < 2) return 0;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function pairedConfidence95(values) {
  const mean = arithmeticMean(values);
  const standardDeviation = sampleStdDev(values, mean);
  const halfWidth = PAIRED_T_CRITICAL_95_DF7 * standardDeviation / Math.sqrt(values.length);
  return { mean, standardDeviation, lower: mean - halfWidth, upper: mean + halfWidth };
}

function assertSameDeviceIdentity(reference, candidate, label) {
  if (JSON.stringify(reference) !== JSON.stringify(candidate)) {
    throw new Error(`${label} resolved a different device/adapter identity`);
  }
}

// Indirect ROI: same as BDPT test — back wall + indirect contributions
const ROI = { x0: 20, y0: 25, x1: 60, y1: 55 };
console.log(`Indirect ROI: cols ${ROI.x0}–${ROI.x1}, rows ${ROI.y0}–${ROI.y1} (${(ROI.x1-ROI.x0+1)*(ROI.y1-ROI.y0+1)} pixels)`);
console.log("");

// ── Part 1: Mean luminance comparison (60 frames each) ───────────────────────
const MEAN_FRAMES = 60;
console.log(`Part 1: Mean luminance comparison (${MEAN_FRAMES} frames each)...`);

console.log("  Rendering BASE (oneEdgeReconnectionReuse:false, default path)...");
const baseResult = await renderScene({ ...FULL_TIER, oneEdgeReconnectionReuse: false }, scene, MEAN_FRAMES);
const deviceIdentity = baseResult.deviceIdentity;
const baseGlobalLum = meanLuminanceROI(baseResult.rgba, W, 0, 0, W-1, H-1);
const baseROILum    = meanLuminanceROI(baseResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
baseResult.engine.dispose();
baseResult.device.destroy();
console.log(`  BASE: global lum = ${baseGlobalLum.toFixed(5)}, ROI lum = ${baseROILum.toFixed(5)}`);

console.log("  Rendering RPT (oneEdgeReconnectionReuse:true, composite path)...");
const rptResult = await renderScene({
  ...FULL_TIER,
  oneEdgeReconnectionReuse: true,
  captureRestirPtReservoirStats: true,
}, scene, MEAN_FRAMES);
const rptGlobalLum = meanLuminanceROI(rptResult.rgba, W, 0, 0, W-1, H-1);
const rptROILum    = meanLuminanceROI(rptResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
const reservoirWeightStats = rptResult.restirPtReservoirStats;
if (reservoirWeightStats == null) throw new Error("missing uncapped ReSTIR-PT reservoir weight statistics");
assertSameDeviceIdentity(deviceIdentity, rptResult.deviceIdentity, "RPT mean arm");
rptResult.engine.dispose();
rptResult.device.destroy();
console.log(`  RPT:  global lum = ${rptGlobalLum.toFixed(5)}, ROI lum = ${rptROILum.toFixed(5)}`);
console.log("");

const globalRelErr = relativeError(rptGlobalLum, baseGlobalLum);
const roiRelErr    = relativeError(rptROILum, baseROILum);
console.log(`  Global mean relative error (RPT vs BASE): ${(globalRelErr * 100).toFixed(2)}%`);
console.log(`  ROI mean relative error (RPT vs BASE):    ${(roiRelErr * 100).toFixed(2)}%`);
// If double-count-free + unbiased: < 10% at 60 spp.
// Known residual per road-to-100: RPT is composite A1 (wired but V28-queue'd).
// At 60 spp on lavapipe the mean should agree unless there is a systematic bias.
const highFrameMeanAgreement = globalRelErr < EQUIVALENCE_MARGIN;
console.log(`  High-frame mean agreement (< 10% threshold): ${highFrameMeanAgreement ? "PASS" : "FINDING"}`);
console.log(`  Resolved W > 10: ${reservoirWeightStats.aboveDiagnosticClampCount}/${reservoirWeightStats.nonEmptyCount}`);
console.log(`  Weight mass removed by W=10 clamp: ${(reservoirWeightStats.clippedWeightMassFraction * 100).toFixed(2)}%`);
console.log("");

// ── Part 2: Variance estimate (8 runs × 8 frames) ───────────────────────────
const VAR_RUNS   = 8;
const VAR_FRAMES = 8;
console.log(`Part 2: Variance estimate (${VAR_RUNS} runs × ${VAR_FRAMES} frames)...`);

console.log("  Rendering BASE variance runs...");
const baseRuns = await renderMultipleRuns({ ...FULL_TIER, oneEdgeReconnectionReuse: false }, scene, VAR_FRAMES, VAR_RUNS);
assertSameDeviceIdentity(deviceIdentity, baseRuns.deviceIdentity, "BASE variance arm");
const baseVar  = varianceROI(baseRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  BASE variance (ROI): ${baseVar.toFixed(6)}`);

console.log("  Rendering RPT variance runs...");
const rptRuns = await renderMultipleRuns({ ...FULL_TIER, oneEdgeReconnectionReuse: true }, scene, VAR_FRAMES, VAR_RUNS);
assertSameDeviceIdentity(deviceIdentity, rptRuns.deviceIdentity, "RPT variance arm");
const rptVar  = varianceROI(rptRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  RPT variance (ROI): ${rptVar.toFixed(6)}`);
console.log("");

const varRatio = baseVar > 0 ? rptVar / baseVar : (rptVar > 0 ? Infinity : 1.0);
console.log(`  RPT/BASE variance ratio in ROI: ${varRatio.toFixed(4)}`);
// At 8 frames per run the variance estimate is noisy, but a professional
// promotion proof still fails closed if reuse more than doubles it.
const varianceNotWorse = varRatio <= VARIANCE_RATIO_MAX;
console.log(`  Variance not worse than the 2× ceiling: ${varianceNotWorse ? "PASS" : "FINDING"}`);
console.log("");

const pairedSeedRuns = baseRuns.map((baseRgba, run) => {
  const rptRgba = rptRuns[run];
  const baseGlobal = meanLuminanceROI(baseRgba, W, 0, 0, W-1, H-1);
  const rptGlobal = meanLuminanceROI(rptRgba, W, 0, 0, W-1, H-1);
  const baseRoi = meanLuminanceROI(baseRgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
  const rptRoi = meanLuminanceROI(rptRgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
  return {
    run,
    baseGlobalLum: baseGlobal,
    rptGlobalLum: rptGlobal,
    baseRoiLum: baseRoi,
    rptRoiLum: rptRoi,
    signedGlobalRelativeDifference: (rptGlobal - baseGlobal) / baseGlobal,
    signedRoiRelativeDifference: (rptRoi - baseRoi) / baseRoi,
  };
});
const pairedGlobal95 = pairedConfidence95(pairedSeedRuns.map((run) => run.signedGlobalRelativeDifference));
const pairedRoi95 = pairedConfidence95(pairedSeedRuns.map((run) => run.signedRoiRelativeDifference));
const pairedGlobalEquivalent =
  pairedGlobal95.lower > -EQUIVALENCE_MARGIN && pairedGlobal95.upper < EQUIVALENCE_MARGIN;
const pairedRoiEquivalent =
  pairedRoi95.lower > -EQUIVALENCE_MARGIN && pairedRoi95.upper < EQUIVALENCE_MARGIN;
const pairedMeanEquivalent = pairedGlobalEquivalent && pairedRoiEquivalent;
const meanAgreement = highFrameMeanAgreement && pairedMeanEquivalent;
console.log("=== Independent-seed paired mean confidence ===");
console.log(`Global signed Δ 95% CI: [${(pairedGlobal95.lower * 100).toFixed(2)}%, ${(pairedGlobal95.upper * 100).toFixed(2)}%]`);
console.log(`ROI signed Δ 95% CI:    [${(pairedRoi95.lower * 100).toFixed(2)}%, ${(pairedRoi95.upper * 100).toFixed(2)}%]`);
console.log(`Global equivalence CI wholly inside ±10%: ${pairedGlobalEquivalent ? "PASS" : "FINDING"}`);
console.log(`ROI equivalence CI wholly inside ±10%:    ${pairedRoiEquivalent ? "PASS" : "FINDING"}`);
console.log("");

// ── Results table ─────────────────────────────────────────────────────────────
console.log("=== ReSTIR-PT vs BASE Summary Table ===");
console.log(`${"Metric".padEnd(40)} ${"BASE".padEnd(12)} ${"RPT".padEnd(12)} Notes`);
console.log(`${"Global mean lum ("+MEAN_FRAMES+" frames)".padEnd(40)} ${baseGlobalLum.toFixed(5).padEnd(12)} ${rptGlobalLum.toFixed(5).padEnd(12)} relErr=${(globalRelErr*100).toFixed(2)}%`);
console.log(`${"ROI mean lum ("+MEAN_FRAMES+" frames)".padEnd(40)} ${baseROILum.toFixed(5).padEnd(12)} ${rptROILum.toFixed(5).padEnd(12)} relErr=${(roiRelErr*100).toFixed(2)}%`);
console.log(`${"Variance in ROI ("+VAR_RUNS+"×"+VAR_FRAMES+" frames)".padEnd(40)} ${baseVar.toFixed(6).padEnd(12)} ${rptVar.toFixed(6).padEnd(12)} ratio=${varRatio.toFixed(4)}`);
console.log("");

const verdict = (meanAgreement && varianceNotWorse) ? "PASS" : "FINDING";
console.log(`=== Verdict: ${verdict} ===`);
if (verdict === "PASS") {
  console.log("ReSTIR-PT composite path agrees with the default megakernel.");
  console.log(`  Unbiasedness: global relErr = ${(globalRelErr*100).toFixed(2)}% < 10%.`);
  console.log(`  Paired-seed global 95% CI = [${(pairedGlobal95.lower*100).toFixed(2)}%, ${(pairedGlobal95.upper*100).toFixed(2)}%].`);
  console.log(`  Paired-seed ROI 95% CI = [${(pairedRoi95.lower*100).toFixed(2)}%, ${(pairedRoi95.upper*100).toFixed(2)}%].`);
  console.log(`  Variance ratio RPT/BASE = ${varRatio.toFixed(4)} (≤ ${VARIANCE_RATIO_MAX.toFixed(1)} threshold met).`);
} else {
  console.log("FINDING:");
  if (!meanAgreement) {
    if (!highFrameMeanAgreement) {
      console.log(`  Global relative error ${(globalRelErr*100).toFixed(2)}% exceeds 10%.`);
    }
    if (!pairedGlobalEquivalent) console.log(`  Paired-seed global 95% CI is not wholly inside ±10%.`);
    if (!pairedRoiEquivalent) console.log(`  Paired-seed ROI 95% CI is not wholly inside ±10%.`);
  }
  if (!varianceNotWorse) {
    console.log(`  RPT variance ratio ${varRatio.toFixed(4)} > ${VARIANCE_RATIO_MAX.toFixed(1)} — temporal reuse is injecting noise.`);
  }
}
console.log("");

// ── Write results JSON ────────────────────────────────────────────────────────
const results = buildRestirPtResult({
  provenance: await radiometricResultProvenance(
    import.meta.url,
    "tools/radiometric-ab/ab-restir-pt.mjs",
    "tools/radiometric-ab/results-restir-pt.json",
    {
      repoRootImportMetaUrl: new URL("../../", import.meta.url).href,
      sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    },
  ),
  date: new Date().toISOString(),
  resolution: { W, H },
  roi: ROI,
  meanFrames: MEAN_FRAMES,
  varianceRuns: VAR_RUNS,
  varianceFramesPerRun: VAR_FRAMES,
  captureConfig: buildRestirPtCaptureConfig({
    scene: "cornell-indirect-v1",
    traceTier: "full",
    colorSpace: "linear",
    requireFullTier: true,
    requireRadiometricSignal: true,
    maxBounces: 6,
    resolution: { W, H },
    roi: ROI,
    meanFrames: MEAN_FRAMES,
    varianceRuns: VAR_RUNS,
    varianceFramesPerRun: VAR_FRAMES,
    effectiveMClamp: 20,
    seeds: ptRadiometricSeedManifest(MEAN_FRAMES, VAR_RUNS, VAR_FRAMES),
  }),
  deviceIdentity,
  base: { globalLum: baseGlobalLum, roiLum: baseROILum, variance: baseVar },
  rpt:  { globalLum: rptGlobalLum,  roiLum: rptROILum,  variance: rptVar },
  globalRelErr,
  roiRelErr,
  varRatio,
  reservoirWeightStats,
  pairedSeedAnalysis: {
    confidenceLevel: 0.95,
    tCritical: PAIRED_T_CRITICAL_95_DF7,
    equivalenceMargin: EQUIVALENCE_MARGIN,
    runs: pairedSeedRuns,
    global: pairedGlobal95,
    roi: pairedRoi95,
    globalEquivalent: pairedGlobalEquivalent,
    roiEquivalent: pairedRoiEquivalent,
    equivalent: pairedMeanEquivalent,
  },
  highFrameMeanAgreement,
  meanAgreement,
  varianceNotWorse,
  verdict,
});

const outPath = new URL("./results-restir-pt.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2));
console.log(`Results written to: ${outPath}`);

Deno.exit(verdict === "PASS" ? 0 : 1);
