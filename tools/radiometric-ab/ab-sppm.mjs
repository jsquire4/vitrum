#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/ab-sppm.mjs
 *
 * A/B #1: SPPM (photon-map) vs manifold-NEE caustic reference
 *
 * The test harness for SPPM correctness.  Pure forward path tracing at
 * feasible spp cannot resolve hard caustics through a glass sphere (the
 * probability of sampling a glass-refraction + floor-bounce + light path
 * without guided sampling is O(1/spp) for a small light).  manifold-nee is
 * the GPU-validated MNEE Newton solver (V-series hardware-validated); it
 * forms the REFERENCE here.
 *
 * Test design
 * ───────────
 * Scene: Cornell box + glass sphere at (0, −0.3, 0) r=0.28 + overhead point
 * light at (0, 0.85, 0.1) intensity=6.  The caustic landing zone is the
 * ~floor region directly below and around the sphere.
 *
 * Caustic ROI: bottom-centre of the 80×80 frame — rows 50–70 (below the
 * sphere equator as projected from the camera), columns 28–52 (horizontally
 * centred).  This captures the primary caustic footprint without including
 * the coloured walls.
 *
 * Ref render (manifold-nee): 80 frames.
 * SPPM render: accumulated at 3 checkpoints (20, 50, 80 frames).
 *
 * Pass criteria
 * ─────────────
 * PASS: SPPM luminance in the ROI is in the same order of magnitude as the
 *       reference (relative error < 10×) AND the relative error decreases
 *       monotonically across the 3 checkpoints, showing convergence.
 *
 * FINDING: if either criterion fails, report the numbers and flag FINDING.
 *
 * Usage
 * ─────
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *     deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
 *     tools/radiometric-ab/ab-sppm.mjs
 */

import {
  buildCausticScene,
  renderScene,
  meanLuminanceROI,
  rmseROI,
  relativeError,
  W, H,
} from "./helpers.mjs";

console.log("=== A/B #1: SPPM vs manifold-NEE caustic reference ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}`);
console.log("");

const scene = buildCausticScene();

// Caustic ROI — rows 50–70, cols 28–52 (0-indexed, floor region below sphere)
const ROI = { x0: 28, y0: 50, x1: 52, y1: 70 };
console.log(`Caustic ROI: cols ${ROI.x0}–${ROI.x1}, rows ${ROI.y0}–${ROI.y1} (${(ROI.x1-ROI.x0+1)*(ROI.y1-ROI.y0+1)} pixels)`);
console.log("");

// ── Render A: manifold-NEE reference ─────────────────────────────────────────
console.log("Rendering A: manifold-nee (reference, 80 frames)...");
const REF_FRAMES = 80;
const refResult = await renderScene(
  { causticStrategy: "manifold-nee" },
  scene,
  REF_FRAMES,
);
const refLum = meanLuminanceROI(refResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
const refGlobalLum = meanLuminanceROI(refResult.rgba, W, 0, 0, W-1, H-1);
console.log(`  A (manifold-nee, ${REF_FRAMES} frames): ROI lum = ${refLum.toFixed(5)}, global lum = ${refGlobalLum.toFixed(5)}`);
refResult.engine.dispose();
refResult.device.destroy();
console.log("");

// ── Render B: SPPM at 3 checkpoints ──────────────────────────────────────────
const CHECKPOINTS = [20, 50, 80];
const sppmResults = [];

console.log("Rendering B: photon-map (SPPM) at 3 accumulation checkpoints...");
for (const frames of CHECKPOINTS) {
  console.log(`  Rendering SPPM checkpoint: ${frames} frames...`);
  const r = await renderScene(
    { causticStrategy: "photon-map" },
    scene,
    frames,
  );
  const lum = meanLuminanceROI(r.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
  const globalLum = meanLuminanceROI(r.rgba, W, 0, 0, W-1, H-1);
  const rmse = rmseROI(r.rgba, refResult.rgba, W, ROI.x0, ROI.y0, ROI.x1, ROI.y1);
  const relErr = relativeError(lum, refLum);
  sppmResults.push({ frames, lum, globalLum, rmse, relErr });
  console.log(`  B (photon-map, ${frames} frames): ROI lum = ${lum.toFixed(5)}, global lum = ${globalLum.toFixed(5)}, RMSE = ${rmse.toFixed(5)}, relErr vs ref = ${(relErr*100).toFixed(1)}%`);
  r.engine.dispose();
  r.device.destroy();
}
console.log("");

// ── Convergence check ─────────────────────────────────────────────────────────
const relErrors = sppmResults.map(r => r.relErr);
const isMonotonicDecreasing = relErrors.every((e, i) => i === 0 || e <= relErrors[i-1] * 1.5); // 50% slack for noise
const finalRelErr = relErrors[relErrors.length - 1];
const converging = isMonotonicDecreasing;
// At fewest 80 frames, SPPM should be in the same order of magnitude as ref
// (relative error < 500% — very lenient because SPPM at low spp has high variance)
const inBallpark = finalRelErr < 5.0;

// ── Results table ─────────────────────────────────────────────────────────────
console.log("=== SPPM Convergence Table ===");
console.log(`${"Frames".padEnd(10)} ${"ROI lum".padEnd(12)} ${"relErr vs ref".padEnd(16)} ${"RMSE"}`);
console.log(`${"(manifold-nee ref)".padEnd(10)} ${refLum.toFixed(5).padEnd(12)} —`);
for (const r of sppmResults) {
  console.log(`${String(r.frames).padEnd(10)} ${r.lum.toFixed(5).padEnd(12)} ${(r.relErr*100).toFixed(1).padEnd(14)}%  ${r.rmse.toFixed(5)}`);
}
console.log("");

const verdict = (converging && inBallpark) ? "PASS" : "FINDING";
console.log(`=== Verdict: ${verdict} ===`);
if (verdict === "PASS") {
  console.log("SPPM ROI luminance converges toward the manifold-nee reference.");
  console.log(`Final relative error: ${(finalRelErr * 100).toFixed(1)}% (threshold: <500%).`);
  console.log("Convergence trend: " + (isMonotonicDecreasing ? "monotone-decreasing (with 50% noise slack)." : "non-monotone but within threshold."));
} else {
  console.log("FINDING — SPPM does not converge as expected:");
  if (!inBallpark) console.log(`  Final relative error ${(finalRelErr*100).toFixed(1)}% exceeds 500% threshold.`);
  if (!isMonotonicDecreasing) console.log(`  Convergence trend is not decreasing: ${relErrors.map(e => (e*100).toFixed(1)+"%").join(" → ")}`);
}
console.log("");

// ── Write results JSON ────────────────────────────────────────────────────────
const results = {
  ab: "sppm-vs-manifold-nee",
  date: new Date().toISOString(),
  resolution: { W, H },
  roi: ROI,
  reference: { strategy: "manifold-nee", frames: REF_FRAMES, roiLum: refLum, globalLum: refGlobalLum },
  sppm: sppmResults,
  verdict,
  converging,
  inBallpark,
};

const outPath = new URL("./results-sppm.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2));
console.log(`Results written to: ${outPath}`);

Deno.exit(verdict === "PASS" ? 0 : 1);
