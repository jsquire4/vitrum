#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/ab-restir-pt-glossy-research.mjs
 *
 * Bounded A/B for the opt-in ReSTIR-PT glossy/metallic visible-vertex reuse
 * branch. This is deliberately separate from `ab-restir-pt.mjs`, whose artifact
 * proves the default diffuse-safe mode. The branch tested here is research mode:
 *
 *   restirPtReuse: true
 *   restirPtReuseOptions: { experimentalGlossyReuse: true }
 *
 * The result is allowed to be a FINDING. A successful run means the branch is
 * measured and classified, not promoted.
 */

import {
  buildCornellScene,
  renderScene,
  renderMultipleRuns,
  meanLuminanceROI,
  varianceROI,
  relativeError,
  W,
  H,
} from "./helpers.mjs";

console.log("=== A/B #3b: ReSTIR-PT glossy research reuse vs base ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}x${H}`);
console.log("");

const scene = buildCornellScene();
const FULL_TIER = { traceTier: "full", requireFullTier: true, requireRadiometricSignal: true };
const BASE_OPTS = { ...FULL_TIER, restirPtReuse: false };
const GLOSSY_RESEARCH_OPTS = {
  ...FULL_TIER,
  restirPtReuse: true,
  restirPtReuseOptions: { experimentalGlossyReuse: true },
};
const ROI = { x0: 20, y0: 25, x1: 60, y1: 55 };
const MEAN_FRAMES = 60;
const VAR_RUNS = 8;
const VAR_FRAMES = 8;
const THRESHOLDS = {
  globalRelErrMax: 0.10,
  varRatioMax: 3.0,
};
const GLOSSY_RESEARCH_WARNING_CODE = "pt-webgpu.restir-pt-glossy-reuse-research-mode";
const GLOSSY_RESEARCH_BLOCKER = "glossy-visible-vertex-reuse-outside-diffuse-safe-validation-envelope";
const GLOSSY_RESEARCH_REQUIRED_EVIDENCE = "glossy-material-furnace-reference-ab-and-browser-real-adapter-recapture";

console.log(
  `Indirect ROI: cols ${ROI.x0}-${ROI.x1}, rows ${ROI.y0}-${ROI.y1} ` +
  `(${(ROI.x1 - ROI.x0 + 1) * (ROI.y1 - ROI.y0 + 1)} pixels)`,
);
console.log("");

console.log(`Part 1: Mean luminance comparison (${MEAN_FRAMES} frames each)...`);
console.log("  Rendering BASE (restirPtReuse:false)...");
const baseResult = await renderScene(BASE_OPTS, scene, MEAN_FRAMES);
const baseGlobalLum = meanLuminanceROI(baseResult.rgba, W, 0, 0, W - 1, H - 1);
const baseROILum = meanLuminanceROI(baseResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
baseResult.engine.dispose();
baseResult.device.destroy();
console.log(`  BASE: global lum = ${baseGlobalLum.toFixed(5)}, ROI lum = ${baseROILum.toFixed(5)}`);

console.log("  Rendering GLOSSY-RPT (experimentalGlossyReuse:true)...");
const glossyResult = await renderScene(GLOSSY_RESEARCH_OPTS, scene, MEAN_FRAMES);
const glossyGlobalLum = meanLuminanceROI(glossyResult.rgba, W, 0, 0, W - 1, H - 1);
const glossyROILum = meanLuminanceROI(glossyResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
glossyResult.engine.dispose();
glossyResult.device.destroy();
console.log(`  GLOSSY-RPT: global lum = ${glossyGlobalLum.toFixed(5)}, ROI lum = ${glossyROILum.toFixed(5)}`);
console.log("");

const globalRelErr = relativeError(glossyGlobalLum, baseGlobalLum);
const roiRelErr = relativeError(glossyROILum, baseROILum);
const meanAgreement = globalRelErr < THRESHOLDS.globalRelErrMax;
console.log(`  Global mean relative error: ${(globalRelErr * 100).toFixed(2)}%`);
console.log(`  ROI mean relative error:    ${(roiRelErr * 100).toFixed(2)}%`);
console.log(`  Mean agreement (< 10% threshold): ${meanAgreement ? "PASS" : "FINDING"}`);
console.log("");

console.log(`Part 2: Variance estimate (${VAR_RUNS} runs x ${VAR_FRAMES} frames)...`);
console.log("  Rendering BASE variance runs...");
const baseRuns = await renderMultipleRuns(BASE_OPTS, scene, VAR_FRAMES, VAR_RUNS);
const baseVar = varianceROI(baseRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  BASE variance (ROI): ${baseVar.toFixed(6)}`);

console.log("  Rendering GLOSSY-RPT variance runs...");
const glossyRuns = await renderMultipleRuns(GLOSSY_RESEARCH_OPTS, scene, VAR_FRAMES, VAR_RUNS);
const glossyVar = varianceROI(glossyRuns, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
console.log(`  GLOSSY-RPT variance (ROI): ${glossyVar.toFixed(6)}`);
console.log("");

const varRatio = baseVar > 0 ? glossyVar / baseVar : (glossyVar > 0 ? Infinity : 1.0);
const varianceNotWorse = varRatio <= THRESHOLDS.varRatioMax;
const verdict = meanAgreement && varianceNotWorse ? "PASS" : "FINDING";
console.log(`  GLOSSY-RPT/BASE variance ratio in ROI: ${varRatio.toFixed(4)}`);
console.log(`  Variance not catastrophically worse (ratio <= 3.0): ${varianceNotWorse ? "PASS" : "FINDING"}`);
console.log("");
console.log(`=== Verdict: ${verdict} ===`);
if (verdict === "PASS") {
  console.log("The research branch stayed within the committed coarse radiometric envelope.");
  console.log("This is not default-promotion evidence; broader glossy/material-furnace proof is still required.");
} else {
  console.log("The research branch remains outside the committed safe-default envelope.");
  if (!meanAgreement) {
    console.log(`  Global relative error ${(globalRelErr * 100).toFixed(2)}% exceeds 10%.`);
  }
  if (!varianceNotWorse) {
    console.log(`  Variance ratio ${varRatio.toFixed(4)} exceeds 3.0.`);
  }
}
console.log("");

const results = {
  ab: "restir-pt-glossy-research-vs-base",
  date: new Date().toISOString(),
  mode: "research",
  resolution: { W, H },
  roi: ROI,
  meanFrames: MEAN_FRAMES,
  varianceRuns: VAR_RUNS,
  varianceFramesPerRun: VAR_FRAMES,
  thresholds: THRESHOLDS,
  reference: { restirPtReuse: false },
  candidate: {
    restirPtReuse: true,
    restirPtReuseOptions: { experimentalGlossyReuse: true },
  },
  base: { globalLum: baseGlobalLum, roiLum: baseROILum, variance: baseVar },
  glossyResearch: { globalLum: glossyGlobalLum, roiLum: glossyROILum, variance: glossyVar },
  globalRelErr,
  roiRelErr,
  varRatio,
  meanAgreement,
  varianceNotWorse,
  promotion: {
    defaultReady: false,
    reason: verdict === "PASS"
      ? "Single repaired-Cornell research-mode capture is bounded but insufficient for default promotion."
      : "Experimental glossy/metallic visible-vertex reuse is outside the committed safe-default radiometric envelope.",
  },
  researchFindings: {
    restirPtGlossyResearch: {
      verdict,
      defaultReady: false,
      warningCode: GLOSSY_RESEARCH_WARNING_CODE,
      blocker: GLOSSY_RESEARCH_BLOCKER,
      requiredEvidence: GLOSSY_RESEARCH_REQUIRED_EVIDENCE,
      globalRelErr,
      varRatio,
      evidencePath: "tools/radiometric-ab/results-restir-pt-glossy-research.json",
    },
  },
  verdict,
};

const outPath = new URL("./results-restir-pt-glossy-research.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2) + "\n");
console.log(`Results written to: ${outPath}`);

// FINDING is an expected research classification, not a harness failure.
Deno.exit(0);
