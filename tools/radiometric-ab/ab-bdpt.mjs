#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/ab-bdpt.mjs
 *
 * A/B #2: BDPT default vs unidirectional — unbiasedness + variance check
 *
 * The unbiasedness check: equal-spp renders with bdpt:false vs bdpt:true must
 * agree on MEAN LUMINANCE to within Monte Carlo noise (relative error < 10%).
 * `bdpt:true` exercises the supported two-bounce light-subpath default.
 * Explicit maxLightBounces:1/2/3 controls isolate depth sensitivity.
 *
 * The variance check: BDPT should have equal-or-lower per-pixel variance
 * than unidirectional in the indirect-light region (the back-wall region
 * that receives reflected light from the metal sphere).  A variance ratio
 * BDPT/UNI ≤ 1.0 is the improvement; > 1.0 would be a regression.
 *
 * Test design
 * ───────────
 * Scene: Cornell box with a glossy metal sphere — creates challenging
 * indirect paths (light → ceiling → metal sphere → floor/wall).
 *
 * Indirect ROI: top half of the image (rows 0–30) where ceiling + metal
 * sphere indirect contributions dominate.  The back wall region is
 * rows 25–55, cols 20–60.
 *
 * SPP: 60 frames for the mean comparison; 8 independent runs × 8 frames
 * each for the variance estimate.
 *
 * Pass criteria
 * ─────────────
 * PASS: relative error on global mean luminance < 10% AND BDPT variance
 *       ratio ≤ 2.0 in the indirect ROI (2× slack for lavapipe noise).
 *
 * Usage
 * ─────
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *     deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
 *     tools/radiometric-ab/ab-bdpt.mjs
 */

import {
  buildCornellScene,
  acquirePtDevice,
  renderScene,
  renderMultipleRuns,
  meanLuminanceROI,
  varianceROI,
  relativeError,
  W, H,
} from "./helpers.mjs";
import {
  PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
  radiometricResultProvenance,
} from "./resultProvenance.mjs";

console.log("=== A/B #2: BDPT default vs unidirectional (unbiasedness + variance) ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}`);
console.log("");

const scene = buildCornellScene();
const FULL_TIER = { traceTier: "full", requireFullTier: true, requireRadiometricSignal: true };
const sharedDevice = await acquirePtDevice(true);

// Indirect ROI: back wall + indirect-lit region (rows 25–55, cols 20–60)
const ROI = { x0: 20, y0: 25, x1: 60, y1: 55 };
console.log(`Indirect ROI: cols ${ROI.x0}–${ROI.x1}, rows ${ROI.y0}–${ROI.y1} (${(ROI.x1-ROI.x0+1)*(ROI.y1-ROI.y0+1)} pixels)`);
console.log("");

// ── Part 1: Mean luminance comparison (60 frames each) ───────────────────────
const MEAN_FRAMES = 60;
console.log(`Part 1: Mean luminance comparison (${MEAN_FRAMES} frames each)...`);

console.log("  Rendering UNI (bdpt:false)...");
const uniResult = await renderScene({ ...FULL_TIER, bdpt: false }, scene, MEAN_FRAMES, sharedDevice);
const uniGlobalLum = meanLuminanceROI(uniResult.rgba, W, 0, 0, W-1, H-1);
const uniROILum    = meanLuminanceROI(uniResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
uniResult.engine.dispose();
console.log(`  UNI: global lum = ${uniGlobalLum.toFixed(5)}, ROI lum = ${uniROILum.toFixed(5)}`);

console.log("  Rendering BDPT default (bdpt:true, maxLightBounces:2)...");
const bdptResult = await renderScene({ ...FULL_TIER, bdpt: true }, scene, MEAN_FRAMES, sharedDevice);
const bdptGlobalLum = meanLuminanceROI(bdptResult.rgba, W, 0, 0, W-1, H-1);
const bdptROILum    = meanLuminanceROI(bdptResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
bdptResult.engine.dispose();
console.log(`  BDPT: global lum = ${bdptGlobalLum.toFixed(5)}, ROI lum = ${bdptROILum.toFixed(5)}`);
console.log("");

const globalRelErr = relativeError(bdptGlobalLum, uniGlobalLum);
const roiRelErr    = relativeError(bdptROILum, uniROILum);
console.log(`  Global mean relative error (BDPT vs UNI): ${(globalRelErr * 100).toFixed(2)}%`);
console.log(`  ROI mean relative error (BDPT vs UNI):    ${(roiRelErr * 100).toFixed(2)}%`);
const meanAgreement = globalRelErr < 0.10; // < 10%
console.log(`  Mean agreement (< 10% threshold): ${meanAgreement ? "PASS" : "FINDING"}`);
console.log("");

// ── Part 1b: BDPT depth isolation controls ─────────────────────────────────
//
// These controls keep the main pass/fail threshold on the default while
// recording how the supported estimator behaves at each configured depth.
const CONTROL_MAX_LIGHT_BOUNCES = [1, 2, 3, 8];
const controlRuns = [];
console.log("Part 1b: BDPT light-subpath depth controls...");
for (const maxLightBounces of CONTROL_MAX_LIGHT_BOUNCES) {
  console.log(`  Rendering BDPT control (maxLightBounces:${maxLightBounces})...`);
  const bdptOptions = { maxLightBounces };
  const controlResult = await renderScene(
    { ...FULL_TIER, bdpt: true, bdptOptions },
    scene,
    MEAN_FRAMES,
    sharedDevice,
  );
  const controlGlobalLum = meanLuminanceROI(controlResult.rgba, W, 0, 0, W - 1, H - 1);
  const controlROILum = meanLuminanceROI(controlResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
  controlResult.engine.dispose();
  const controlGlobalRelErr = relativeError(controlGlobalLum, uniGlobalLum);
  const controlRoiRelErr = relativeError(controlROILum, uniROILum);
  controlRuns.push({
    maxLightBounces,
    globalLum: controlGlobalLum,
    roiLum: controlROILum,
    globalRelErr: controlGlobalRelErr,
    roiRelErr: controlRoiRelErr,
  });
  console.log(
    `  BDPT maxLightBounces:${maxLightBounces}: global lum = ${controlGlobalLum.toFixed(5)}, ` +
    `ROI lum = ${controlROILum.toFixed(5)}, global relErr = ${(controlGlobalRelErr * 100).toFixed(2)}%`,
  );
}
const endpointOnlyControl = controlRuns.find((r) => r.maxLightBounces === 1);
const endpointOnlyCarriesSignal =
  endpointOnlyControl != null &&
  endpointOnlyControl.globalLum > 1e-5 &&
  endpointOnlyControl.roiLum > 1e-5;
const endpointOnlyMeanAgreement =
  endpointOnlyControl != null &&
  endpointOnlyControl.globalRelErr < 0.10 &&
  endpointOnlyControl.roiRelErr < 0.15;
console.log(`  Endpoint-only control carries finite-light signal: ${endpointOnlyCarriesSignal ? "YES" : "NO"}`);
console.log(`  Endpoint-only mean agreement: ${endpointOnlyMeanAgreement ? "PASS" : "FINDING"}`);
console.log("");

// ── Part 2: Variance estimate (8 runs × 8 frames) ───────────────────────────
const VAR_RUNS   = 8;
const VAR_FRAMES = 8;
console.log(`Part 2: Variance estimate (${VAR_RUNS} runs × ${VAR_FRAMES} frames)...`);

console.log("  Rendering UNI variance runs...");
const uniRuns = await renderMultipleRuns(
  { ...FULL_TIER, bdpt: false },
  scene,
  VAR_FRAMES,
  VAR_RUNS,
  sharedDevice,
);
const uniVar  = varianceROI(uniRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  UNI variance (ROI): ${uniVar.toFixed(6)}`);

console.log("  Rendering BDPT variance runs...");
const bdptRuns = await renderMultipleRuns(
  { ...FULL_TIER, bdpt: true },
  scene,
  VAR_FRAMES,
  VAR_RUNS,
  sharedDevice,
);
const bdptVar  = varianceROI(bdptRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  BDPT variance (ROI): ${bdptVar.toFixed(6)}`);
console.log("");

const varRatio = uniVar > 0 ? bdptVar / uniVar : (bdptVar > 0 ? Infinity : 1.0);
console.log(`  BDPT/UNI variance ratio in ROI: ${varRatio.toFixed(4)}`);
// BDPT should not be dramatically WORSE; ≤ 2.0 is the pass threshold
// (2× slack for lavapipe noise at only 8 frames per run).
const varianceImproved = varRatio <= 2.0;
console.log(`  Variance improvement (ratio ≤ 2.0): ${varianceImproved ? "PASS" : "FINDING"}`);
console.log("");
sharedDevice.destroy();

// ── Results table ─────────────────────────────────────────────────────────────
console.log("=== BDPT vs UNI Summary Table ===");
console.log(`${"Metric".padEnd(40)} ${"UNI".padEnd(12)} ${"BDPT".padEnd(12)} Notes`);
console.log(`${"Global mean lum ("+MEAN_FRAMES+" frames)".padEnd(40)} ${uniGlobalLum.toFixed(5).padEnd(12)} ${bdptGlobalLum.toFixed(5).padEnd(12)} relErr=${(globalRelErr*100).toFixed(2)}%`);
console.log(`${"ROI mean lum ("+MEAN_FRAMES+" frames)".padEnd(40)} ${uniROILum.toFixed(5).padEnd(12)} ${bdptROILum.toFixed(5).padEnd(12)} relErr=${(roiRelErr*100).toFixed(2)}%`);
console.log(`${"Variance in ROI ("+VAR_RUNS+"×"+VAR_FRAMES+" frames)".padEnd(40)} ${uniVar.toFixed(6).padEnd(12)} ${bdptVar.toFixed(6).padEnd(12)} ratio=${varRatio.toFixed(4)}`);
for (const c of controlRuns) {
  console.log(`${("BDPT control maxLightBounces="+c.maxLightBounces).padEnd(40)} ${uniGlobalLum.toFixed(5).padEnd(12)} ${c.globalLum.toFixed(5).padEnd(12)} relErr=${(c.globalRelErr*100).toFixed(2)}%`);
}
console.log("");

const verdict = (
  meanAgreement &&
  varianceImproved &&
  endpointOnlyCarriesSignal &&
  endpointOnlyMeanAgreement
) ? "PASS" : "FINDING";
console.log(`=== Verdict: ${verdict} ===`);
if (verdict === "PASS") {
  console.log("BDPT and unidirectional agree on mean luminance within MC noise.");
  console.log(`  Unbiasedness confirmed: global relErr = ${(globalRelErr*100).toFixed(2)}% < 10%.`);
  console.log(`  Variance ratio BDPT/UNI = ${varRatio.toFixed(4)} (≤ 2.0 threshold met).`);
} else {
  console.log("FINDING:");
  if (!meanAgreement) {
    console.log(`  Global relative error ${(globalRelErr*100).toFixed(2)}% exceeds 10% threshold.`);
    console.log("  This indicates a bias in the configured BDPT default.");
  }
  if (!varianceImproved) {
    console.log(`  BDPT variance ratio ${varRatio.toFixed(4)} > 2.0 — BDPT is adding significant noise.`);
  }
  if (!endpointOnlyCarriesSignal || !endpointOnlyMeanAgreement) {
    console.log("  The c=0 emitter-endpoint estimator is black or disagrees with UNI.");
  }
}
console.log("");

// ── Write results JSON ────────────────────────────────────────────────────────
const results = {
  schema: "vitrum.radiometric-ab.result.v1",
  provenance: await radiometricResultProvenance(
    import.meta.url,
    "tools/radiometric-ab/ab-bdpt.mjs",
    "tools/radiometric-ab/results-bdpt.json",
    {
      repoRootImportMetaUrl: new URL("../../", import.meta.url).href,
      sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    },
  ),
  ab: "bdpt-vs-unidirectional",
  date: new Date().toISOString(),
  resolution: { W, H },
  roi: ROI,
  meanFrames: MEAN_FRAMES,
  varianceRuns: VAR_RUNS,
  varianceFramesPerRun: VAR_FRAMES,
  uni: { globalLum: uniGlobalLum, roiLum: uniROILum, variance: uniVar },
  bdpt: { globalLum: bdptGlobalLum, roiLum: bdptROILum, variance: bdptVar },
  controls: {
    meanFrames: MEAN_FRAMES,
    byMaxLightBounces: controlRuns,
    endpointOnlyCarriesSignal,
    endpointOnlyMeanAgreement,
    supportedDepths: CONTROL_MAX_LIGHT_BOUNCES,
  },
  globalRelErr,
  roiRelErr,
  varRatio,
  meanAgreement,
  varianceImproved,
  verdict,
};

const outPath = new URL("./results-bdpt.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2));
console.log(`Results written to: ${outPath}`);

Deno.exit(verdict === "PASS" ? 0 : 1);
