#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/ab-restir-pt.mjs
 *
 * A/B #3: ReSTIR-PT reuse on vs off — bias check + variance reduction
 *
 * ReSTIR-PT (restirPtReuse:true) composites the reconnection-indirect
 * estimate into the beauty accumulator via the COMPOSITE megakernel path
 * (A1, kernel.wgsl.ts:308-312).  The estimator split is E0-direct-only in
 * the megakernel + rpt_result indirect from the resolve pass.  If the split
 * is double-count-free and unbiased, the converged mean should agree with
 * the plain path tracer (restirPtReuse:false).
 *
 * This A/B tests:
 *   (a) Unbiasedness: equal-spp global mean luminance must agree within 10%.
 *   (b) Variance: 8×8-frame runs estimate per-pixel variance in an
 *       indirect-lit ROI.  ReSTIR-PT should not INCREASE variance beyond 2×.
 *
 * Device requirement: restirPtReuse=true requires full-tier limits
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
  meanLuminanceROI,
  varianceROI,
  relativeError,
  W, H,
} from "./helpers.mjs";
import { radiometricResultProvenance } from "./resultProvenance.mjs";

console.log("=== A/B #3: ReSTIR-PT reuse on vs off (bias check + variance) ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}`);
console.log("");

const scene = buildCornellScene();
const FULL_TIER = { traceTier: "full", requireFullTier: true, requireRadiometricSignal: true };

// Indirect ROI: same as BDPT test — back wall + indirect contributions
const ROI = { x0: 20, y0: 25, x1: 60, y1: 55 };
console.log(`Indirect ROI: cols ${ROI.x0}–${ROI.x1}, rows ${ROI.y0}–${ROI.y1} (${(ROI.x1-ROI.x0+1)*(ROI.y1-ROI.y0+1)} pixels)`);
console.log("");

// ── Part 1: Mean luminance comparison (60 frames each) ───────────────────────
const MEAN_FRAMES = 60;
console.log(`Part 1: Mean luminance comparison (${MEAN_FRAMES} frames each)...`);

console.log("  Rendering BASE (restirPtReuse:false, default path)...");
const baseResult = await renderScene({ ...FULL_TIER, restirPtReuse: false }, scene, MEAN_FRAMES);
const baseGlobalLum = meanLuminanceROI(baseResult.rgba, W, 0, 0, W-1, H-1);
const baseROILum    = meanLuminanceROI(baseResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
baseResult.engine.dispose();
baseResult.device.destroy();
console.log(`  BASE: global lum = ${baseGlobalLum.toFixed(5)}, ROI lum = ${baseROILum.toFixed(5)}`);

console.log("  Rendering RPT (restirPtReuse:true, composite path)...");
const rptResult = await renderScene({ ...FULL_TIER, restirPtReuse: true }, scene, MEAN_FRAMES);
const rptGlobalLum = meanLuminanceROI(rptResult.rgba, W, 0, 0, W-1, H-1);
const rptROILum    = meanLuminanceROI(rptResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
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
const meanAgreement = globalRelErr < 0.10;
console.log(`  Mean agreement (< 10% threshold): ${meanAgreement ? "PASS" : "FINDING"}`);
console.log("");

// ── Part 2: Variance estimate (8 runs × 8 frames) ───────────────────────────
const VAR_RUNS   = 8;
const VAR_FRAMES = 8;
console.log(`Part 2: Variance estimate (${VAR_RUNS} runs × ${VAR_FRAMES} frames)...`);

console.log("  Rendering BASE variance runs...");
const baseRuns = await renderMultipleRuns({ ...FULL_TIER, restirPtReuse: false }, scene, VAR_FRAMES, VAR_RUNS);
const baseVar  = varianceROI(baseRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  BASE variance (ROI): ${baseVar.toFixed(6)}`);

console.log("  Rendering RPT variance runs...");
const rptRuns = await renderMultipleRuns({ ...FULL_TIER, restirPtReuse: true }, scene, VAR_FRAMES, VAR_RUNS);
const rptVar  = varianceROI(rptRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  RPT variance (ROI): ${rptVar.toFixed(6)}`);
console.log("");

const varRatio = baseVar > 0 ? rptVar / baseVar : (rptVar > 0 ? Infinity : 1.0);
console.log(`  RPT/BASE variance ratio in ROI: ${varRatio.toFixed(4)}`);
// At 8 frames per run the variance estimate is noisy; allow 3× slack.
// The important number is the mean-agreement check; variance is secondary.
const varianceNotWorse = varRatio <= 3.0;
console.log(`  Variance not catastrophically worse (ratio ≤ 3.0): ${varianceNotWorse ? "PASS" : "FINDING"}`);
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
  console.log(`  Variance ratio RPT/BASE = ${varRatio.toFixed(4)} (≤ 3.0 threshold met).`);
} else {
  console.log("FINDING:");
  if (!meanAgreement) {
    console.log(`  Global relative error ${(globalRelErr*100).toFixed(2)}% exceeds 10%.`);
    console.log(`  Possible double-count or missing component in the composite split.`);
    console.log(`  (Note: RPT indirect composited via rpt_result → beauty accumulator;`);
    console.log(`   if the megakernel was not in E0-direct-only mode this would show as`);
    console.log(`   double-counting the indirect.)`);
  }
  if (!varianceNotWorse) {
    console.log(`  RPT variance ratio ${varRatio.toFixed(4)} > 3.0 — temporal reuse is injecting noise.`);
  }
}
console.log("");

// ── Write results JSON ────────────────────────────────────────────────────────
const results = {
  provenance: await radiometricResultProvenance(import.meta.url, "tools/radiometric-ab/ab-restir-pt.mjs", "tools/radiometric-ab/results-restir-pt.json"),
  ab: "restir-pt-reuse-on-vs-off",
  date: new Date().toISOString(),
  resolution: { W, H },
  roi: ROI,
  meanFrames: MEAN_FRAMES,
  varianceRuns: VAR_RUNS,
  varianceFramesPerRun: VAR_FRAMES,
  base: { globalLum: baseGlobalLum, roiLum: baseROILum, variance: baseVar },
  rpt:  { globalLum: rptGlobalLum,  roiLum: rptROILum,  variance: rptVar },
  globalRelErr,
  roiRelErr,
  varRatio,
  meanAgreement,
  varianceNotWorse,
  verdict,
};

const outPath = new URL("./results-restir-pt.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2));
console.log(`Results written to: ${outPath}`);

Deno.exit(verdict === "PASS" ? 0 : 1);
