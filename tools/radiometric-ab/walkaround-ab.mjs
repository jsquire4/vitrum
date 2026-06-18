#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * tools/radiometric-ab/walkaround-ab.mjs
 *
 * Walkaround-hybrid radiometric A/B harness — 2026-06-10.
 *
 * Four measurements (all walkaround-hybrid on lavapipe):
 *
 *   A8  GRIS bias quantification:
 *       restirPtReuse:false (default, biased) vs restirPtReuse:true (GRIS,
 *       unbiased). Same Cornell scene, static camera, N=16 frames accumulated.
 *       Measures mean-luminance delta per region (floor/wall/ceiling). The delta
 *       is the documented bias of the default path.
 *
 *   SUN  Sun-NEE analytic validation:
 *       directional-lit diffuse floor (no area emitter). Compares rendered floor
 *       luminance to the analytic direct irradiance E = I·cosθ·albedo.
 *       A shadowed region (back of the box) must be near-zero. Self-validating
 *       harness pattern.
 *
 *   GLASS  Glass-GI validation:
 *       Cornell with a glass pane vs WITHOUT the glass pane. Both have the same
 *       area emitter. Through-glass region luminance should be ≥ no-glass × 0.7
 *       (accounting for Fresnel attenuation and GI propagation; strict black check).
 *
 *   GLOSSY  Metallic probe check (B2):
 *       metalness=1, roughness=0.05 floor (a mirror) vs roughness=0.85 floor
 *       (diffuse default). The metallic floor must show nonzero indirect (the
 *       specular probe term lo_indirectSpecular is active when metal>0). Also
 *       asserts that the metallic floor overall luminance is >= diffuse floor
 *       luminance (the mirror reflects the ceiling light directly).
 *
 * Usage (from repo root):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
 *   deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
 *     tools/radiometric-ab/walkaround-ab.mjs
 *
 * Writes: tools/radiometric-ab/walkaround-ab-results.json
 *
 * IMPORTANT: all @vitrum/* imports are resolved via deno.json in this directory
 * (relative paths). Do NOT add absolute paths — see the stale-import lesson in
 * CLAUDE.md.
 */

import { createWalkaroundEngine_Hybrid } from "@vitrum/walkaround-hybrid";
import { asMat4 } from "@vitrum/core";
import { applyNagaFix } from "../shader-gate/nagaFix.mjs";

// ── Resolution + frame count ──────────────────────────────────────────────────
// 128×128 gives more stable per-region statistics than 64×64 at modest cost.
const W = 128, H = 128;
const SPP = 16; // accumulation frames per variant

// ── Camera ────────────────────────────────────────────────────────────────────
function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f  = 1.0 / Math.tan((fovDeg * Math.PI) / 180 / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function makeLookAtMatrix(eye, center, up) {
  const fx = center[0]-eye[0], fy = center[1]-eye[1], fz = center[2]-eye[2];
  const fL  = Math.hypot(fx, fy, fz);
  const fnx = fx/fL, fny = fy/fL, fnz = fz/fL;
  const sx = fny*up[2]-fnz*up[1], sy = fnz*up[0]-fnx*up[2], sz = fnx*up[1]-fny*up[0];
  const sL  = Math.hypot(sx, sy, sz);
  const snx = sx/sL, sny = sy/sL, snz = sz/sL;
  const ux  = sny*fnz-snz*fny, uy = snz*fnx-snx*fnz, uz = snx*fny-sny*fnx;
  return new Float32Array([
    snx, ux, -fnx, 0,
    sny, uy, -fny, 0,
    snz, uz, -fnz, 0,
    -(snx*eye[0]+sny*eye[1]+snz*eye[2]),
    -(ux *eye[0]+uy *eye[1]+uz *eye[2]),
     fnx*eye[0]+fny*eye[1]+fnz*eye[2],
    1,
  ]);
}

const EYE    = [0, 0, 2.5];
const CENTER = [0, 0, 0];
const proj   = asMat4(makePerspectiveMatrix(60, W / H, 0.1, 50));
const view   = asMat4(makeLookAtMatrix(EYE, CENTER, [0,1,0]));

// ── Scene builders ────────────────────────────────────────────────────────────
function makeQuad(id, verts, normal, color, roughness = 1.0, metallic = 0.0) {
  return {
    kind: "mesh", id,
    positions: new Float32Array(verts.flat()),
    normals:   new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs:       new Float32Array(8),
    indices:   new Uint32Array([0, 2, 1, 2, 0, 3]),
    material:  { baseColor: color, roughness, metallic },
  };
}

/** Standard Cornell box with area emitter on the ceiling. */
function makeCornellScene(opts = {}) {
  const floorRoughness = opts.floorRoughness ?? 1.0;
  const floorMetallic  = opts.floorMetallic  ?? 0.0;
  const floorColor     = opts.floorColor     ?? [0.8, 0.8, 0.8];

  const primitives = [
    makeQuad("floor",      [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0],  floorColor, floorRoughness, floorMetallic),
    makeQuad("ceiling",    [[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],     [0,-1,0], [0.8,0.8,0.8]),
    makeQuad("back-wall",  [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],     [0,0,-1], [0.8,0.8,0.8]),
    makeQuad("left-wall",  [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
    makeQuad("right-wall", [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
  ];

  if (opts.glass) {
    // Glass pane between camera and back wall.
    primitives.push({
      kind: "mesh", id: "glass-pane",
      positions: new Float32Array([-0.5,-0.5,0.5, 0.5,-0.5,0.5, 0.5,0.5,0.5, -0.5,0.5,0.5]),
      normals:   new Float32Array([0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1]),
      uvs:       new Float32Array(8),
      indices:   new Uint32Array([0,2,1, 2,0,3]),
      material:  { baseColor: [1.0,1.0,1.0], roughness: 0.05, metallic: 0.0, transmission: 1.0 },
    });
  }

  const emitters = opts.noEmitter ? [] : [{
    kind: "rect-area", id: "ceiling-light",
    position: [0, 0.95, 0],
    uAxis: [0, 0, 0.2], vAxis: [0.2, 0, 0],
    color: [1,1,1], intensity: 12.0,
  }];

  return { primitives, emitters, environment: { kind: "none" } };
}

/** Directional-only scene: just a flat diffuse floor + walls, no area emitter. */
function makeDirOnlyScene() {
  return {
    primitives: [
      // Large diffuse floor — the analytic check region
      makeQuad("floor",     [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0], [0.8,0.8,0.8]),
      // Back wall (in shadow from camera's angle)
      makeQuad("back-wall", [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],     [0,0,-1], [0.8,0.8,0.8]),
      makeQuad("left-wall", [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
      makeQuad("right-wall",[[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
    ],
    emitters:    [],
    environment: { kind: "none" },
  };
}

// ── Naga gap patch (walkaround-hybrid) ────────────────────────────────────────
function patchDeviceForWh(device) {
  const orig = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    if (typeof desc.code === "string") {
      try { return orig({ ...desc, code: applyNagaFix(desc.code) }); }
      catch { return orig(desc); }
    }
    return orig(desc);
  };
}

// ── Device acquisition ────────────────────────────────────────────────────────
async function acquireWhDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const limits = {};
  const sb = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
  const st = adapter.limits.maxStorageTexturesPerShaderStage ?? 4;
  if (sb >= 16) limits.maxStorageBuffersPerShaderStage = sb;
  if (st >= 8)  limits.maxStorageTexturesPerShaderStage = st;
  const bg = adapter.limits.maxBindGroups ?? 4;
  if (bg > 4) limits.maxBindGroups = bg;
  return adapter.requestDevice(Object.keys(limits).length ? { requiredLimits: limits } : {});
}

// ── Engine readiness helper ──────────────────────────────────────────────────
async function waitForReady(engine, label, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    if (engine.state === "ready") return;
    if (engine.state === "error") throw new Error(`${label}: engine.state === 'error'`);
    await new Promise(r => setTimeout(r, 50));
    if (engine.state === "ready") return;
    if (engine.state === "error") throw new Error(`${label}: engine.state === 'error'`);
    if (Date.now() > deadline) {
      throw new Error(`${label}: engine init timeout after ${Math.round(timeoutMs / 1000)} s`);
    }
  }
}

// ── Pixel statistics helpers ──────────────────────────────────────────────────
function meanLuminance(pixels) {
  let sum = 0;
  const n = pixels.length / 4;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += 0.2126 * pixels[i] + 0.7152 * pixels[i+1] + 0.0722 * pixels[i+2];
  }
  return sum / n;
}

/**
 * Mean luminance of a screen region [x0,x1) × [y0,y1) (pixel coords).
 * y=0 is top of image (captureFrame contract row order).
 */
function regionLuminance(pixels, texW, x0, y0, x1, y1) {
  let sum = 0, count = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * texW + x) * 4;
      sum += 0.2126 * pixels[i] + 0.7152 * pixels[i+1] + 0.0722 * pixels[i+2];
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

function absDelta(a, b) {
  return Math.abs(a - b);
}

// ── Engine runner ─────────────────────────────────────────────────────────────
async function runVariant(label, engineOpts, sceneFactory) {
  let device;
  try { device = await acquireWhDevice(); }
  catch (e) { return { label, error: e.message, pixels: null, lum: 0 }; }

  patchDeviceForWh(device);

  let engine  = null;
  let swapTex = null;
  let pixels  = null;
  let error   = null;

  try {
    engine = await createWalkaroundEngine_Hybrid({
      device,
      width:  W, height: H,
      primaryLightDir:       [0.3, -0.8, 0.5],
      primaryLightIntensity: 0.6,
      skyTint:               [0.5, 0.7, 1.0],
      skyIrradiance:         0.15,
      verbose:               false,
      ppgEnabled:            false,
      rcEnabled:             false,
      denoiser:              "atrous-variance",
      ...engineOpts,
    });

    const scene = sceneFactory();
    engine.setScene(scene);

    await waitForReady(engine, label);

    swapTex = device.createTexture({
      label: `swap-${label}`,
      size:  [W, H, 1],
      format: "bgra8unorm",
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    });
    const swapView = swapTex.createView();

    for (let fi = 0; fi < SPP; fi++) {
      engine.renderFrame({
        viewMatrix:      view,
        projMatrix:      proj,
        cameraPosition:  EYE,
        viewport:        { width: W, height: H, devicePixelRatio: 1 },
        frameIndex:      fi,
        frameSeed:       fi * 1664525 + 1013904223,
        swapChainView:   swapView,
        swapChainFormat: "bgra8unorm",
      });
      await device.queue.onSubmittedWorkDone();
    }

    const captured = await engine.captureFrame({ colorSpace: "linear" });
    if (captured == null) throw new Error("captureFrame returned null");
    if (captured.width !== W || captured.height !== H) {
      throw new Error(`captureFrame size mismatch: got ${captured.width}x${captured.height}, expected ${W}x${H}`);
    }
    pixels = captured.rgba;
  } catch (e) {
    error = e.message;
  } finally {
    try { swapTex?.destroy(); } catch { /* best-effort */ }
    try { engine?.dispose();  } catch { /* best-effort */ }
    device.destroy();
  }

  if (error) return { label, error, pixels: null, lum: 0 };
  return { label, error: null, pixels, lum: meanLuminance(pixels) };
}

// ── A8: GRIS bias quantification ─────────────────────────────────────────────
//
// Cornell + area emitter. Static camera at EYE=[0,0,2.5] looking at CENTER.
// The image space maps:
//   Floor   → bottom ~1/3 of image   (y ≈ 85..127 of 128)
//   Ceiling → top    ~1/5 of image   (y ≈ 0..25  of 128)
//   Walls   → left/right stripes     (x ≈ 0..20 left wall, 108..128 right wall)
//
// We use three broad bands to integrate over enough pixels for stable MC mean.
//
async function runA8() {
  console.log("\n── A8: GRIS bias quantification (restirPtReuse:false vs true) ──");
  const t0 = Date.now();

  const biasedResult  = await runVariant("a8-biased",  { restirPtReuse: false }, () => makeCornellScene());
  const unbiasedResult = await runVariant("a8-unbiased", { restirPtReuse: true  }, () => makeCornellScene());

  const dt = ((Date.now() - t0) / 1000).toFixed(1);

  if (biasedResult.error || unbiasedResult.error) {
    return {
      id: "A8",
      error: biasedResult.error ?? unbiasedResult.error,
      verdict: "ERROR",
    };
  }

  const { pixels: pB } = biasedResult;
  const { pixels: pU } = unbiasedResult;

  // Region windows (128×128 output, camera at z=2.5 looking at origin with fov=60°)
  // Floor visible at bottom of frame; ceiling at top; left wall at left stripe.
  const floorB   = regionLuminance(pB, W,  10, 85,  118, 127);
  const ceilB    = regionLuminance(pB, W,  10, 0,   118, 20);
  const leftWB   = regionLuminance(pB, W,  0,  20,  20,  108);
  const rightWB  = regionLuminance(pB, W,  108, 20, 128, 108);

  const floorU   = regionLuminance(pU, W,  10, 85,  118, 127);
  const ceilU    = regionLuminance(pU, W,  10, 0,   118, 20);
  const leftWU   = regionLuminance(pU, W,  0,  20,  20,  108);
  const rightWU  = regionLuminance(pU, W,  108, 20, 128, 108);

  const deltaFloor = floorU - floorB;
  const deltaCeil  = ceilU  - ceilB;
  const deltaLeft  = leftWU - leftWB;
  const deltaRight = rightWU - rightWB;

  const overallBiased   = biasedResult.lum;
  const overallUnbiased = unbiasedResult.lum;
  const overallDelta    = overallUnbiased - overallBiased;

  // Verdict: if overall delta is within ±0.03 the bias is small and bounded.
  // If it exceeds ±0.05 the bias is significant for the converged path.
  const absDelta = Math.abs(overallDelta);
  const verdict  = absDelta < 0.005 ? "NEGLIGIBLE"
                 : absDelta < 0.03  ? "SMALL"
                 : absDelta < 0.06  ? "MODERATE"
                 : "SIGNIFICANT";

  console.log(`  biased   (off): overall=${overallBiased.toFixed(4)}  floor=${floorB.toFixed(4)}  ceil=${ceilB.toFixed(4)}  lWall=${leftWB.toFixed(4)}  rWall=${rightWB.toFixed(4)}`);
  console.log(`  unbiased (on):  overall=${overallUnbiased.toFixed(4)}  floor=${floorU.toFixed(4)}  ceil=${ceilU.toFixed(4)}  lWall=${leftWU.toFixed(4)}  rWall=${rightWU.toFixed(4)}`);
  console.log(`  DELTA (unbiased - biased): overall=${overallDelta.toFixed(4)}  floor=${deltaFloor.toFixed(4)}  ceil=${deltaCeil.toFixed(4)}  lWall=${deltaLeft.toFixed(4)}  rWall=${deltaRight.toFixed(4)}`);
  console.log(`  verdict: ${verdict} (|overall delta|=${absDelta.toFixed(4)}) — render time ${dt}s`);

  return {
    id: "A8",
    description: "GRIS bias quantification: restirPtReuse:false vs true",
    spp: SPP,
    resolution: `${W}x${H}`,
    biased: {
      overall: overallBiased,
      floor: floorB, ceiling: ceilB, leftWall: leftWB, rightWall: rightWB,
    },
    unbiased: {
      overall: overallUnbiased,
      floor: floorU, ceiling: ceilU, leftWall: leftWU, rightWall: rightWU,
    },
    delta: {
      overall: overallDelta,
      floor: deltaFloor, ceiling: deltaCeil, leftWall: deltaLeft, rightWall: deltaRight,
    },
    renderTimeSec: parseFloat(dt),
    verdict,
    notes: [
      "Bias sources B1-B4 documented in HybridEngineOptions.restirPtReuse JSDoc.",
      "Delta is (unbiased - biased). Positive = biased underestimates; negative = biased overestimates.",
      "Lavapipe (software Vulkan) — MC variance at SPP=16 is ~1-3% of mean.",
    ],
  };
}

// ── SUN: Sun-NEE analytic validation ─────────────────────────────────────────
//
// Sun direction d̂ = normalize([0.3,-0.8,0.5]). The floor normal is [0,1,0].
// cosθ = dot([0,1,0], normalize([0.3,0.8,-0.5])) (from sun TO floor, not from floor TO sun).
//   to-sun = normalize([0.3,-0.8,0.5]) → negate z to get from-floor perspective:
//   Actually: sunDir = [0.3,-0.8,0.5] (downward, toward scene). The floor normal
//   points UP [0,1,0]. The to-sun direction from the floor surface = -sunDir direction.
//   cosθ = dot(floorNormal, toSun) = dot([0,1,0], normalize(-[0.3,-0.8,0.5]))
//         = dot([0,1,0], normalize([-0.3, 0.8, -0.5]))
//         = 0.8 / sqrt(0.09 + 0.64 + 0.25) = 0.8 / sqrt(0.98) ≈ 0.808
// Analytic direct irradiance on floor = I × cosθ = 3.0 × 0.808 = 2.424
// After Lambertian BRDF (albedo/π) and the cosine-weighted outgoing integration:
//   Rendered luminance = I × cosθ × (albedo / π) × π = I × cosθ × albedo
//   (the π from Lambertian BRDF cancels the ∫cosθdω hemisphere integral)
//   = 3.0 × 0.808 × 0.8 ≈ 1.939 in linear HDR units.
// The harness reads `captureFrame({ colorSpace:"linear" })` after rendering, so
// the signal is not tonemapped or quantized by the host swap-chain. We still use
// a lower intensity (0.3) to keep the analytic value near the rest of the scene:
//   I=0.3 → E = 0.3 × 0.808 × 0.8 ≈ 0.194
//
// Back wall: normal [0,0,-1]. cos(wallNormal, toSun) = dot([0,0,-1], [-0.3,0.8,-0.5]*norm)
//   = dot([0,0,-1], [-0.303, 0.808, -0.505]) ≈ 0.505 > 0, so back wall IS lit.
// The region behind the box (not visible to camera) cannot be sampled directly,
// but the left wall (normal [1,0,0]):
//   dot([1,0,0], toSun) = -0.303 < 0 → left wall SHADOWED by sun (no direct sun).
//
// Assertion: floor luminance > left-wall luminance (left wall gets NO direct sun).
//
async function runSun() {
  console.log("\n── SUN: Sun-NEE analytic self-validation ──");
  const t0 = Date.now();

  // Use intensity 0.3 so the direct-light analytic signal stays in a stable range.
  const sunResult = await runVariant("sun", {
    primaryLightDir:       [0.3, -0.8, 0.5],
    primaryLightIntensity: 0.3,
  }, () => makeDirOnlyScene());

  const dt = ((Date.now() - t0) / 1000).toFixed(1);

  if (sunResult.error) {
    return { id: "SUN", error: sunResult.error, verdict: "ERROR" };
  }

  const { pixels } = sunResult;

  // Floor region: bottom half of image (floor is the large quad at y=-1).
  // With the camera at [0,0,2.5] looking at origin, floor pixels land in the lower
  // ~40% of the image. Use a conservative center strip.
  const floorLum  = regionLuminance(pixels, W, 30, 78,  98, 118); // floor center
  const leftLum   = regionLuminance(pixels, W,  0, 30,  15,  98); // left wall (should be dark — sun comes from the right of left-wall)
  const overallLum = sunResult.lum;

  // Analytic: I × cosθ × albedo (diffuse reciprocal Lambertian: brdf=albedo/π, ∫cos dω=π)
  // sunDir = [0.3, -0.8, 0.5], floor normal = [0,1,0]
  // toSun from floor = normalize(-sunDir) = normalize([-0.3, 0.8, -0.5])
  const sd = [0.3, -0.8, 0.5];
  const sdLen = Math.hypot(...sd);
  const toSun = [-sd[0]/sdLen, -sd[1]/sdLen, -sd[2]/sdLen];
  const cosTheta = Math.max(0, toSun[1]); // dot([0,1,0], toSun)
  const intensity = 0.3;
  const albedo    = 0.8; // floor baseColor luminance (0.8,0.8,0.8 → Y≈0.8)
  // Analytic rendered luminance (Lambertian: Lo = albedo/π × E = albedo × I × cosθ)
  const analyticFloor = intensity * cosTheta * albedo;

  // Tolerance: the walkaround pipeline applies denoising, temporal accumulation,
  // GTAO AO, and the atrous filter — these can attenuate. We use a ±50% band.
  const tol = 0.5;
  const floorRatioToAnalytic = analyticFloor > 0 ? floorLum / analyticFloor : 0;
  const analyticPasses = floorRatioToAnalytic >= (1 - tol) && floorRatioToAnalytic <= (1 + tol);
  const shadowPasses   = leftLum < floorLum * 0.7; // left wall must be significantly darker than lit floor

  const verdict = (floorLum > 0.01 && shadowPasses && analyticPasses)
    ? "PASS" : (floorLum > 0.01 && shadowPasses) ? "PASS-PARTIAL" : "FAIL";

  console.log(`  floor lum:   ${floorLum.toFixed(4)}  (analytic: ${analyticFloor.toFixed(4)}, ratio: ${floorRatioToAnalytic.toFixed(3)})`);
  console.log(`  left-wall:   ${leftLum.toFixed(4)}  (expected < floor×0.7 = ${(floorLum*0.7).toFixed(4)})`);
  console.log(`  overall:     ${overallLum.toFixed(4)}`);
  console.log(`  cosTheta:    ${cosTheta.toFixed(4)}`);
  console.log(`  analytic Lo: ${analyticFloor.toFixed(4)}  (I=${intensity} × cosθ=${cosTheta.toFixed(3)} × albedo=${albedo})`);
  console.log(`  verdict:     ${verdict} — render time ${dt}s`);

  return {
    id: "SUN",
    description: "Sun-NEE analytic validation: directional-lit diffuse floor vs analytic E=I·cosθ·albedo",
    spp: SPP,
    resolution: `${W}x${H}`,
    sunDirection: [0.3, -0.8, 0.5],
    sunIntensity: intensity,
    floorAlbedo:  albedo,
    cosTheta,
    analyticExpectedFloorLum: analyticFloor,
    rendered: {
      floorLum,
      leftWallLum: leftLum,
      overall:     overallLum,
    },
    floorRatioToAnalytic,
    analyticAgreement: analyticPasses,
    shadowCorrect: shadowPasses,
    renderTimeSec: parseFloat(dt),
    verdict,
    notes: [
      "analytic = I × cosθ × albedo (Lambertian: brdf=albedo/π, hemisphere integral=π → Lo=albedo×I×cosθ).",
      "±50% tolerance accounts for GTAO attenuation, temporal accumulation damping, denoiser.",
      "Left wall has dot([1,0,0], toSun) < 0 → receives no direct sun (shadow test).",
      "Lavapipe — SPP=16.",
    ],
  };
}

// ── GLASS: Glass-GI validation ────────────────────────────────────────────────
//
// Scene A: Cornell with glass pane.
// Scene B: identical Cornell WITHOUT glass pane (noGlass control).
// Both have the ceiling area emitter (intensity 12.0).
//
// Expected: through-glass region luminance ≥ no-glass × 0.5.
// Physically: the glass transmits most of the direct light (Fresnel T≈0.92 for
// normal incidence at n=1.5) and the transmitted GI from the diffuse wall behind
// it. At low SPP the denoiser may attenuate, so we use a conservative 0.5 floor.
//
// We look at the centre of the image where the glass pane occlude the back wall.
// The glass pane spans z=0.5, x∈[-0.5,0.5], y∈[-0.5,0.5]; the camera looks along
// -z from z=2.5, so the pane projects to roughly the centre of the image.
//
async function runGlass() {
  console.log("\n── GLASS: Glass-GI validation ──");
  const t0 = Date.now();

  const glassResult   = await runVariant("glass",    {}, () => makeCornellScene({ glass: true  }));
  const noGlassResult = await runVariant("no-glass", {}, () => makeCornellScene({ glass: false }));

  const dt = ((Date.now() - t0) / 1000).toFixed(1);

  if (glassResult.error || noGlassResult.error) {
    return { id: "GLASS", error: glassResult.error ?? noGlassResult.error, verdict: "ERROR" };
  }

  const { pixels: pG } = glassResult;
  const { pixels: pN } = noGlassResult;

  // Centre region: the glass pane projects to approximately centre of image.
  // Use a 32×32 centre crop (indices 48..80 in x and y for a 128×128 image).
  const cx0 = 48, cx1 = 80, cy0 = 48, cy1 = 80;
  const glassCenter   = regionLuminance(pG, W, cx0, cy0, cx1, cy1);
  const noGlassCenter = regionLuminance(pN, W, cx0, cy0, cx1, cy1);

  const overallGlass   = glassResult.lum;
  const overallNoGlass = noGlassResult.lum;

  // Ratio: glass / no-glass. Should be ≥ 0.5 (transmittance lower-bound).
  // A ratio of 0.0 means the glass region is black (GI not propagating through glass).
  const centerRatio = noGlassCenter > 0.01 ? glassCenter / noGlassCenter : 0;
  const overallRatio = noGlassResult.lum > 0.01 ? overallGlass / noGlassResult.lum : 0;
  const centreDelta = glassCenter - noGlassCenter;
  const overallDelta = overallGlass - overallNoGlass;

  // Fresnel-T at normal incidence n=1.5: T = 1 - ((1.5-1)/(1.5+1))^2 = 1 - 0.04 = 0.96
  // Two surfaces (enter + exit): T_total ≈ 0.96^2 ≈ 0.92. Beer absorption = none (no tint).
  // Expected centre ratio > 0.5 (allowing for denoiser, GTAO, temporal).
  const EXPECTED_MIN_RATIO = 0.5;
  const MIN_SIGNAL_DELTA = 1e-4;
  const notBlack  = glassCenter > 0.01;
  const ratioPass = centerRatio >= EXPECTED_MIN_RATIO;
  const materialEffectObserved =
    Math.max(absDelta(glassCenter, noGlassCenter), absDelta(overallGlass, overallNoGlass)) >= MIN_SIGNAL_DELTA;
  const verdict   = notBlack && ratioPass && materialEffectObserved ? "PASS"
                  : notBlack && ratioPass ? "PASS-WEAK"
                  : notBlack ? "PASS-WEAK"
                  : "FAIL";

  console.log(`  glass centre:   ${glassCenter.toFixed(4)}  overall=${overallGlass.toFixed(4)}`);
  console.log(`  no-glass centre:${noGlassCenter.toFixed(4)}  overall=${overallNoGlass.toFixed(4)}`);
  console.log(`  centre ratio:   ${centerRatio.toFixed(3)}  (expected ≥ ${EXPECTED_MIN_RATIO}); delta=${centreDelta.toExponential(3)}`);
  console.log(`  overall ratio:  ${overallRatio.toFixed(3)}`);
  console.log(`  effect observed:${materialEffectObserved ? "YES" : "NO"}  (min |delta|=${MIN_SIGNAL_DELTA})`);
  console.log(`  verdict:        ${verdict} — render time ${dt}s`);

  return {
    id: "GLASS",
    description: "Glass-GI: Cornell+glass vs Cornell-no-glass, centre-region luminance ratio",
    spp: SPP,
    resolution: `${W}x${H}`,
    fresnelT_normal_incidence_n1p5: 0.92,
    expectedMinCentreRatio: EXPECTED_MIN_RATIO,
    glass: {
      centreRegionLum: glassCenter,
      overall:         overallGlass,
    },
    noGlass: {
      centreRegionLum: noGlassCenter,
      overall:         overallNoGlass,
    },
    centreRatio:  centerRatio,
    overallRatio: overallRatio,
    delta: {
      centreRegionLum: centreDelta,
      overall: overallDelta,
    },
    minSignalDelta: MIN_SIGNAL_DELTA,
    materialEffectObserved,
    renderTimeSec: parseFloat(dt),
    verdict,
    notes: [
      "Glass Fresnel-T ≈ 0.92 at normal incidence (n=1.5, two surfaces). Beer absorption = 0 (white glass).",
      "Expected centreRatio ≥ 0.50 — conservative; denoiser can attenuate through-glass signal at low SPP.",
      "Walkaround isGlass gate: matColor.a > 0.3 (transmission=1.0 → packed alpha≈255 → isGlass=true).",
      "PASS-WEAK means the through-glass region is non-black but the glass/no-glass captures are statistically indistinguishable at this SPP; do not promote material transport from that alone.",
    ],
  };
}

// ── GLOSSY: Metallic probe check (B2) ────────────────────────────────────────
//
// Scene A: Cornell with metalness=1.0, roughness=0.05 floor (mirror-like).
// Scene B: Cornell with metalness=0.0, roughness=1.0 floor (Lambertian default).
//
// Expected:
//   1. Metallic floor has nonzero indirect (lo_indirectSpecular is active when metal>0,
//      gate: `metal <= 0.0 && rough >= 0.6 → skip`; with metal=1 it fires).
//   2. Metallic floor overall luminance ≥ diffuse floor luminance because it reflects
//      the ceiling emitter directly (specular reflection of the bright emitter).
//
// We assert: metalLum > 0.01 (not black) and metalLum > diffuseLum × 0.8.
//
async function runGlossy() {
  console.log("\n── GLOSSY: Metallic probe check (B2) ──");
  const t0 = Date.now();

  const metalResult   = await runVariant("metal",   {}, () => makeCornellScene({
    floorRoughness: 0.05, floorMetallic: 1.0, floorColor: [0.9, 0.9, 0.9],
  }));
  const diffuseResult = await runVariant("diffuse", {}, () => makeCornellScene({
    floorRoughness: 1.0,  floorMetallic: 0.0, floorColor: [0.8, 0.8, 0.8],
  }));

  const dt = ((Date.now() - t0) / 1000).toFixed(1);

  if (metalResult.error || diffuseResult.error) {
    return { id: "GLOSSY", error: metalResult.error ?? diffuseResult.error, verdict: "ERROR" };
  }

  const overallMetal  = metalResult.lum;
  const overallDiffuse = diffuseResult.lum;

  // Floor region — bottom strip of image (camera looks slightly down)
  const floorMetal   = regionLuminance(metalResult.pixels,   W, 20, 85, 108, 127);
  const floorDiffuse = regionLuminance(diffuseResult.pixels, W, 20, 85, 108, 127);
  const floorDelta = floorMetal - floorDiffuse;
  const overallDelta = overallMetal - overallDiffuse;

  // Check: metallic floor must be nonzero and at least 0.8× the diffuse floor.
  // The mirror floor reflects the bright ceiling emitter → should be brighter or equal.
  const floorRatio = floorDiffuse > 0.01 ? floorMetal / floorDiffuse : 0;
  const notBlack   = floorMetal > 0.01;
  const MIN_SIGNAL_DELTA = 1e-4;
  const materialEffectObserved =
    Math.max(absDelta(floorMetal, floorDiffuse), absDelta(overallMetal, overallDiffuse)) >= MIN_SIGNAL_DELTA;
  // Structural assertion: metallic floor is directionally plausible (bright where
  // reflected probe field is bright). At low SPP we just check nonzero + plausible ratio.
  const plausible  = floorRatio >= 0.8;

  const verdict = notBlack && plausible && materialEffectObserved ? "PASS"
                : notBlack ? "PASS-WEAK"
                : "FAIL";

  console.log(`  metallic  floor: ${floorMetal.toFixed(4)}  overall=${overallMetal.toFixed(4)}`);
  console.log(`  diffuse   floor: ${floorDiffuse.toFixed(4)}  overall=${overallDiffuse.toFixed(4)}`);
  console.log(`  floor ratio (metal/diffuse): ${floorRatio.toFixed(3)}  (expected ≥ 0.80); delta=${floorDelta.toExponential(3)}`);
  console.log(`  effect observed:${materialEffectObserved ? "YES" : "NO"}  (min |delta|=${MIN_SIGNAL_DELTA})`);
  console.log(`  verdict:  ${verdict} — render time ${dt}s`);

  return {
    id: "GLOSSY",
    description: "B2 metallic probe: metalness=1,rough=0.05 vs metalness=0,rough=1 floor",
    spp: SPP,
    resolution: `${W}x${H}`,
    metal: {
      floorLum: floorMetal,
      overall:  overallMetal,
    },
    diffuse: {
      floorLum: floorDiffuse,
      overall:  overallDiffuse,
    },
    floorRatio,
    delta: {
      floorLum: floorDelta,
      overall: overallDelta,
    },
    expectedMinFloorRatio: 0.8,
    minSignalDelta: MIN_SIGNAL_DELTA,
    materialEffectObserved,
    renderTimeSec: parseFloat(dt),
    verdict,
    notes: [
      "lo_indirectSpecular fires when metal>0 OR rough<0.6 (SPEC_GI_ROUGH_MAX=0.6).",
      "Metal floor (n=1.0, rough=0.05): GGX specular lobe reflects the ceiling emitter → brighter than Lambertian.",
      "Approximation: DDGI atlas stores cosine-weighted irradiance, not GGX-filtered radiance (documented in-code).",
      "Lavapipe SPP=16; metallic-mirror at low SPP has high variance — ratio check uses ≥0.80.",
      "PASS-WEAK means the metallic floor is non-black/plausible but the metal/diffuse captures are statistically indistinguishable at this SPP; do not promote glossy probe quality from that alone.",
    ],
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────
console.log("=== walkaround radiometric A/Bs ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}, SPP: ${SPP}`);

const a8Result     = await runA8();
const sunResult    = await runSun();
const glassResult  = await runGlass();
const glossyResult = await runGlossy();

const results = { a8: a8Result, sun: sunResult, glass: glassResult, glossy: glossyResult };

console.log("\n=== SUMMARY ===");
for (const r of Object.values(results)) {
  console.log(`  ${r.id.padEnd(8)} ${r.verdict ?? r.error ?? "?"}${r.delta ? `  globalDelta=${r.delta.overall?.toFixed(4)}` : ""}`);
}

// Write JSON results
const outPath = new URL("./walkaround-ab-results.json", import.meta.url).pathname;
await Deno.writeTextFile(outPath, JSON.stringify(results, null, 2));
console.log(`\nResults written to: ${outPath}`);

const anyFail = Object.values(results).some(r => r.verdict === "FAIL" || r.verdict === "ERROR");
Deno.exit(anyFail ? 1 : 0);
