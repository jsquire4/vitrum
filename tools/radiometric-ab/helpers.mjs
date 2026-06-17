// @ts-nocheck
/**
 * tools/radiometric-ab/helpers.mjs
 *
 * Shared engine-boot, scene-build, and readback helpers for the radiometric
 * A/B harness.  Imports and reuses the naga-gap patches from the behavioral
 * gate verbatim so we exercise the same engine path the gate runs.
 *
 * Key contract: captureFrame() returns the raw RGBA float32 linear-HDR
 * accumulator (not the tonemapped presentTexture) — that is what makes these
 * comparisons radiometric rather than display-encoded.
 */

import { createPTEngine_WebGPU } from "@vitrum/pt-webgpu";
import { asMat4 } from "@vitrum/core";

// ── Naga gap patches (copied from tools/behavioral-gate/gate.mjs) ─────────────
// These are the same patches the behavioral gate uses; without them lavapipe
// rejects the shader at createShaderModule.

function stripBdptMipArg(wgsl) {
  const NEEDLE = "textureLoad(bdptLightPath,";
  let result = "", i = 0;
  while (i < wgsl.length) {
    const start = wgsl.indexOf(NEEDLE, i);
    if (start < 0) { result += wgsl.slice(i); break; }
    const openParen = start + "textureLoad".length;
    result += wgsl.slice(i, openParen + 1);
    let depth = 1, j = openParen + 1;
    const commas = [];
    let closeParen = -1;
    while (j < wgsl.length) {
      const ch = wgsl[j];
      if (ch === "(") depth++;
      else if (ch === ")") { depth--; if (depth === 0) { closeParen = j; break; } }
      else if (ch === "," && depth === 1) commas.push(j);
      j++;
    }
    if (closeParen < 0) { result += wgsl.slice(openParen + 1); break; }
    if (commas.length >= 2) {
      result += wgsl.slice(openParen + 1, commas[commas.length - 1]) + ")";
    } else {
      result += wgsl.slice(openParen + 1, closeParen) + ")";
    }
    i = closeParen + 1;
  }
  return result;
}

function addBdptMipArg(wgsl) {
  const NEEDLE = "textureLoad(bdptLightPath,";
  let result = "", i = 0;
  while (i < wgsl.length) {
    const start = wgsl.indexOf(NEEDLE, i);
    if (start < 0) { result += wgsl.slice(i); break; }
    const openParen = start + "textureLoad".length;
    result += wgsl.slice(i, openParen + 1);
    let depth = 1, j = openParen + 1;
    const commas = [];
    let closeParen = -1;
    while (j < wgsl.length) {
      const ch = wgsl[j];
      if (ch === "(") depth++;
      else if (ch === ")") { depth--; if (depth === 0) { closeParen = j; break; } }
      else if (ch === "," && depth === 1) commas.push(j);
      j++;
    }
    if (closeParen < 0) { result += wgsl.slice(openParen + 1); break; }
    if (commas.length === 1) {
      result += wgsl.slice(openParen + 1, closeParen) + ", 0)";
    } else {
      result += wgsl.slice(openParen + 1, closeParen) + ")";
    }
    i = closeParen + 1;
  }
  return result;
}

export function applyPtNagaGapFix(wgsl, bdptOn) {
  let fixed = stripBdptMipArg(wgsl);
  if (fixed.includes("isNan(") || fixed.includes("isInf(")) {
    const helpers = `\nfn isNan(v: vec3f) -> vec3<bool> { return v != v; }\nfn isInf(v: vec3f) -> vec3<bool> { return abs(v) >= vec3f(1e38); }\n`;
    const idx = fixed.indexOf('\nfn ');
    if (idx > 0) fixed = fixed.slice(0, idx) + helpers + fixed.slice(idx);
  }
  if (!bdptOn && fixed.includes('texture_storage_2d<rgba32float, read_write>')) {
    fixed = fixed.replace('texture_storage_2d<rgba32float, read_write>', 'texture_2d<f32>');
    fixed = addBdptMipArg(fixed);
    fixed = fixed.replace(/textureStore\s*\(\s*bdptLightPath\s*,[^;]+;/g,
      '// naga-gap-fix: textureStore(bdptLightPath) removed');
  }
  return fixed;
}

export function patchDeviceForPt(device, bdptOn) {
  const orig = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    if (typeof desc.code === "string" && desc.code.includes("bdptLightPath")) {
      return orig({ ...desc, code: applyPtNagaGapFix(desc.code, bdptOn) });
    }
    return orig(desc);
  };
  return () => { device.createShaderModule = orig; };
}

// ── Device acquisition ─────────────────────────────────────────────────────────

export async function acquirePtDevice(wantsFullTier) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter — is VK_ICD_FILENAMES set?");
  const limits = {};
  if (wantsFullTier) {
    const sb = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
    const st = adapter.limits.maxStorageTexturesPerShaderStage ?? 4;
    if (sb >= 28) limits.maxStorageBuffersPerShaderStage = sb;
    if (st >= 5)  limits.maxStorageTexturesPerShaderStage = st;
  }
  const bg = adapter.limits.maxBindGroups ?? 4;
  if (bg > 4) limits.maxBindGroups = bg;
  return adapter.requestDevice(Object.keys(limits).length ? { requiredLimits: limits } : {});
}

// ── Camera ─────────────────────────────────────────────────────────────────────

export function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f  = 1.0 / Math.tan((fovDeg * Math.PI) / 180 / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function makeLookAtMatrix(eye, center, up) {
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

// ── Scene builders ─────────────────────────────────────────────────────────────

/** Build a quad mesh for Cornell box walls. */
export function makeQuad(id, verts, normal, color) {
  return {
    kind: "mesh", id,
    positions: new Float32Array(verts.flat()),
    normals:   new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs:       new Float32Array(8),
    indices:   new Uint32Array([0, 2, 1, 2, 0, 3]),
    material:  { baseColor: color, roughness: 1.0, metallic: 0.0 },
  };
}

/**
 * Caustic scene for SPPM and BDPT A/Bs:
 * - Cornell box walls
 * - Glass sphere in the box (creates hard-to-sample caustic below it)
 * - Point light above the sphere (overhead, forces light through the glass)
 *
 * The glass sphere is at y=−0.3, radius=0.25; the floor is at y=−1.
 * The caustic region on the floor is approximately the circle directly below
 * the sphere (x in [−0.3,+0.3], z in [−0.3,+0.3] → 4 floor pixels at 64×64
 * resolution covering roughly [i_x ∈ 16..47, i_y ∈ 16..47] in screen-space).
 * We define the caustic ROI as ALL floor pixels at y=−0.95 as seen from the
 * camera — in practice we sum the BOTTOM HALF of the image (below the sphere
 * equator) for a robust but still caustic-dominated region.
 */
export function buildCausticScene() {
  const primitives = [
    makeQuad("floor",      [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0],  [0.8,0.8,0.8]),
    makeQuad("ceiling",    [[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],     [0,-1,0], [0.8,0.8,0.8]),
    // The camera sits at +Z looking toward the origin, so the Cornell box must
    // be open toward +Z and place its back wall at -Z. A previous version put
    // the wall at +Z, causing local-light A/Bs to measure the unlit outside face.
    makeQuad("back-wall",  [[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1]],   [0,0,1],  [0.8,0.8,0.8]),
    makeQuad("left-wall",  [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
    makeQuad("right-wall", [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
    // Glass sphere: transmission=1.0 → material.transmission decoded as glass
    // in the WGSL BSDF selector.  Placed at centre-low so the caustic lands on
    // the floor where the camera sees it.
    {
      kind: "mesh", id: "glass-sphere",
      positions: glassSpherePositions(0, -0.3, 0, 0.28, 12, 8),
      normals:   glassSphereNormals(0, -0.3, 0, 12, 8),
      uvs:       new Float32Array(glassSphereVertexCount(12, 8) * 2),
      indices:   glassSphereIndices(12, 8),
      material:  { baseColor: [1.0, 1.0, 1.0], roughness: 0.0, metallic: 0.0, transmission: 1.0, ior: 1.5 },
    },
  ];

  const emitters = [
    // Point light overhead and slightly forward so paths through the sphere
    // are plentiful but the DIRECT path is blocked by the sphere for floor points.
    { kind: "point", id: "top-light", position: [0, 0.85, 0.1], color: [1,1,1], intensity: 6.0 },
  ];

  return { primitives, emitters, environment: { kind: "none" } };
}

/**
 * Cornell box with a perfectly diffuse back wall and left+right coloured
 * walls.  The ceiling emitter is the only light.  A metal sphere is placed
 * in the box to drive indirect-light paths that benefit from BDPT and
 * ReSTIR-PT: the sphere reflects light from the ceiling onto the back wall
 * and from the left/right walls onto the floor.
 */
export function buildCornellScene() {
  const primitives = [
    makeQuad("floor",      [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0],  [0.8,0.8,0.8]),
    makeQuad("ceiling",    [[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],     [0,-1,0], [0.8,0.8,0.8]),
    // Open-front Cornell box for the fixed +Z camera; see buildCausticScene.
    makeQuad("back-wall",  [[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1]],   [0,0,1],  [0.8,0.8,0.8]),
    makeQuad("left-wall",  [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
    makeQuad("right-wall", [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
    // Metal sphere in centre — glossy specular, low roughness.
    // Creates indirect-light paths (light→metal→floor, light→metal→wall)
    // that a unidirectional tracer samples poorly at low bounce counts but
    // BDPT/ReSTIR-PT handle well.
    {
      kind: "mesh", id: "metal-sphere",
      positions: glassSpherePositions(0.2, -0.5, 0.1, 0.35, 12, 8),
      normals:   glassSphereNormals(0.2, -0.5, 0.1, 12, 8),
      uvs:       new Float32Array(glassSphereVertexCount(12, 8) * 2),
      indices:   glassSphereIndices(12, 8),
      material:  { baseColor: [0.95, 0.8, 0.2], roughness: 0.08, metallic: 1.0 },
    },
  ];

  const emitters = [
    { kind: "rect-area", id: "ceiling-light",
      position: [0, 0.95, 0],
      uAxis: [0, 0, 0.2], vAxis: [0.2, 0, 0],
      color: [1,1,1], intensity: 12.0 },
  ];

  return { primitives, emitters, environment: { kind: "none" } };
}

// ── Sphere geometry helpers ────────────────────────────────────────────────────

export function glassSphereVertexCount(stacks, slices) {
  return (stacks + 1) * (slices + 1);
}

/** Generate sphere vertex positions (cx,cy,cz = centre, r = radius). */
export function glassSpherePositions(cx, cy, cz, r, stacks, slices) {
  const verts = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = Math.PI * i / stacks;
    for (let j = 0; j <= slices; j++) {
      const theta = 2 * Math.PI * j / slices;
      verts.push(
        cx + r * Math.sin(phi) * Math.cos(theta),
        cy + r * Math.cos(phi),
        cz + r * Math.sin(phi) * Math.sin(theta),
      );
    }
  }
  return new Float32Array(verts);
}

/** Generate sphere vertex normals (outward unit normals, same layout as positions). */
export function glassSphereNormals(cx, cy, cz, stacks, slices) {
  const norms = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = Math.PI * i / stacks;
    for (let j = 0; j <= slices; j++) {
      const theta = 2 * Math.PI * j / slices;
      norms.push(
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      );
    }
  }
  return new Float32Array(norms);
}

/** Generate sphere triangle indices (GL_TRIANGLES). */
export function glassSphereIndices(stacks, slices) {
  const idx = [];
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j;
      const b = a + (slices + 1);
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return new Uint32Array(idx);
}

// ── Radiometric statistics ─────────────────────────────────────────────────────

/**
 * Compute mean linear luminance of an RGBA Float32Array (W×H×4 interleaved).
 * Uses Rec.709 luma coefficients.
 */
export function meanLuminance(rgba, W, H) {
  const n = W * H;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const R = rgba[i * 4 + 0];
    const G = rgba[i * 4 + 1];
    const B = rgba[i * 4 + 2];
    sum += 0.2126 * R + 0.7152 * G + 0.0722 * B;
  }
  return sum / n;
}

/**
 * Compute mean linear luminance of a rectangular ROI (pixel-coordinate box,
 * inclusive on both ends).  Used to measure the caustic region on the floor.
 */
export function meanLuminanceROI(rgba, W, x0, y0, x1, y1) {
  let sum = 0, count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x;
      const R = rgba[i * 4 + 0];
      const G = rgba[i * 4 + 1];
      const B = rgba[i * 4 + 2];
      sum += 0.2126 * R + 0.7152 * G + 0.0722 * B;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Per-pixel RMSE of luminance over a rectangular ROI between two RGBA images.
 * Both images must be W×H×4 Float32 interleaved.
 */
export function rmseROI(rgbaA, rgbaB, W, x0, y0, x1, y1) {
  let sumSq = 0, count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x;
      const lumA = 0.2126 * rgbaA[i*4+0] + 0.7152 * rgbaA[i*4+1] + 0.0722 * rgbaA[i*4+2];
      const lumB = 0.2126 * rgbaB[i*4+0] + 0.7152 * rgbaB[i*4+1] + 0.0722 * rgbaB[i*4+2];
      sumSq += (lumA - lumB) ** 2;
      count++;
    }
  }
  return count > 0 ? Math.sqrt(sumSq / count) : 0;
}

/**
 * Per-pixel variance estimate of luminance over a ROI from an array of N
 * independent Float32 RGBA images (N short runs).
 * Returns the mean pixel-variance (averaged over the ROI).
 */
export function varianceROI(rgbaImages, W, x0, y0, x1, y1) {
  const N = rgbaImages.length;
  if (N < 2) return 0;
  let sumVar = 0, count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x;
      let mean = 0;
      for (let k = 0; k < N; k++) {
        mean += 0.2126 * rgbaImages[k][i*4+0]
              + 0.7152 * rgbaImages[k][i*4+1]
              + 0.0722 * rgbaImages[k][i*4+2];
      }
      mean /= N;
      let variance = 0;
      for (let k = 0; k < N; k++) {
        const lum = 0.2126 * rgbaImages[k][i*4+0]
                  + 0.7152 * rgbaImages[k][i*4+1]
                  + 0.0722 * rgbaImages[k][i*4+2];
        variance += (lum - mean) ** 2;
      }
      sumVar += variance / (N - 1); // Bessel-corrected
      count++;
    }
  }
  return count > 0 ? sumVar / count : 0;
}

// ── Engine runner ──────────────────────────────────────────────────────────────

const W = 80, H = 80;
const EYE    = [-0.05, 0, 2.75];
const CENTER = [-0.05, -0.15, 0];

export { W, H, EYE, CENTER };

export function makePtCamera(W, H) {
  const proj = asMat4(makePerspectiveMatrix(40, W / H, 0.1, 50));
  const view = asMat4(makeLookAtMatrix(EYE, CENTER, [0,1,0]));
  return { proj, view };
}

function splitHarnessOptions(engineOpts) {
  const {
    requireFullTier = false,
    requireRadiometricSignal = false,
    ...ptOptions
  } = engineOpts;
  return { requireFullTier, requireRadiometricSignal, ptOptions };
}

function assertRequiredTier(engine, requireFullTier, label) {
  const resolvedLite = engine.capabilities.experimentalFeatures?.has("pt-webgpu-lite-tier") === true;
  if (requireFullTier && resolvedLite) {
    throw new Error(
      `${label} requires pt-webgpu full tier, but the adapter resolved to lite. ` +
      `Do not use this radiometric A/B as evidence on lite-tier adapters; run it through ` +
      `the wsl-gpu/dzn full-tier lane or another adapter with full pt-webgpu limits.`,
    );
  }
}

function assertRadiometricSignal(rgba, requireRadiometricSignal, label) {
  if (!requireRadiometricSignal) return;
  const lum = meanLuminance(rgba, W, H);
  if (!(lum > 1e-5)) {
    throw new Error(
      `${label} produced no radiometric signal (mean luminance ${lum}). ` +
      `This A/B would otherwise be a false PASS if both arms are black; fix the ` +
      `capture scene/backend path before using the result as evidence.`,
    );
  }
}

/**
 * Boot a pt-webgpu engine, render `totalFrames` frames, and return:
 *   { rgba: Float32Array, W, H, samplesAccumulated }
 * via captureFrame() (linear HDR, not tonemapped).
 *
 * @param {object} engineOpts  — options forwarded to createPTEngine_WebGPU
 * @param {object} scene       — scene returned by a build*Scene helper
 * @param {number} totalFrames — number of renderFrame calls
 * @param {GPUDevice?} device  — supply to reuse; if null a fresh one is acquired
 * @returns {Promise<{rgba: Float32Array, W: number, H: number, samples: number, device: GPUDevice, engine: object}>}
 */
export async function renderScene(engineOpts, scene, totalFrames, device = null) {
  const { requireFullTier, requireRadiometricSignal, ptOptions } = splitHarnessOptions(engineOpts);
  const bdptOn = ptOptions.bdpt === true;
  const isLite = ptOptions.traceTier === "lite";
  let ownDevice = false;
  if (!device) {
    device = await acquirePtDevice(!isLite);
    ownDevice = true;
  }

  const unpatch = patchDeviceForPt(device, bdptOn);
  let engine = null;
  let rgba = null;
  let samples = 0;
  let errorMsg = null;

  const { proj, view } = makePtCamera(W, H);

  try {
    engine = await createPTEngine_WebGPU({
      device,
      maxBounces: 6,
      maxSamplesPerPixel: totalFrames,
      ...ptOptions,
    });
    assertRequiredTier(engine, requireFullTier, "renderScene");

    engine.setScene(scene);

    for (let frame = 0; frame < totalFrames; frame++) {
      const seed = Number(BigInt.asUintN(32, BigInt(frame) * 6364136223846793005n + 1442695040888963407n));
      engine.renderFrame({
        viewMatrix: view,
        projMatrix: proj,
        cameraPosition: EYE,
        viewport: { width: W, height: H, devicePixelRatio: 1 },
        frameIndex: frame, frameSeed: seed,
        quality: { samplesTarget: totalFrames, bounces: 6, resolutionFactor: 1 },
      });
      await device.queue.onSubmittedWorkDone();
    }
    samples = totalFrames;

    // captureFrame with colorSpace:'linear' → raw accumTexture (pre-tonemap float32)
    const captured = await engine.captureFrame({ colorSpace: "linear" });
    if (captured != null) {
      rgba = captured.rgba;
      assertRadiometricSignal(rgba, requireRadiometricSignal, "renderScene");
    } else {
      errorMsg = "captureFrame returned null";
    }
  } catch (e) {
    errorMsg = e.message ?? String(e);
  } finally {
    unpatch();
  }

  if (errorMsg) throw new Error(errorMsg);
  if (!rgba) throw new Error("No frame captured");

  return { rgba, W, H, samples, device, engine, ownDevice };
}

/**
 * Render multiple independent short runs for variance estimation.
 * Returns an array of N Float32 RGBA images (each from a fresh engine instance
 * seeded differently via frameIndex offsets).
 */
export async function renderMultipleRuns(engineOpts, scene, framesPerRun, numRuns) {
  const { requireFullTier, requireRadiometricSignal, ptOptions } = splitHarnessOptions(engineOpts);
  const device = await acquirePtDevice(ptOptions.traceTier !== "lite");
  const results = [];
  const bdptOn = ptOptions.bdpt === true;
  const unpatch = patchDeviceForPt(device, bdptOn);

  try {
    const { proj, view } = makePtCamera(W, H);

    for (let run = 0; run < numRuns; run++) {
      let engine = null;
      try {
        engine = await createPTEngine_WebGPU({
          device,
          maxBounces: 6,
          maxSamplesPerPixel: framesPerRun,
          ...ptOptions,
        });
        assertRequiredTier(engine, requireFullTier, `renderMultipleRuns run ${run}`);

        engine.setScene(scene);

        // Offset frame seeds per-run so runs are independent
        const seedOffset = run * 97;
        for (let frame = 0; frame < framesPerRun; frame++) {
          const globalFrame = run * framesPerRun + frame;
          const seed = Number(BigInt.asUintN(32, BigInt(globalFrame) * 6364136223846793005n + 1442695040888963407n));
          engine.renderFrame({
            viewMatrix: view,
            projMatrix: proj,
            cameraPosition: EYE,
            viewport: { width: W, height: H, devicePixelRatio: 1 },
            frameIndex: frame, frameSeed: seed + seedOffset,
            quality: { samplesTarget: framesPerRun, bounces: 6, resolutionFactor: 1 },
          });
          await device.queue.onSubmittedWorkDone();
        }

        const captured = await engine.captureFrame({ colorSpace: "linear" });
        if (captured?.rgba) {
          assertRadiometricSignal(captured.rgba, requireRadiometricSignal, `renderMultipleRuns run ${run}`);
          results.push(captured.rgba);
        }
      } finally {
        try { engine?.dispose(); } catch { /* ignore */ }
        // Wait between runs so the device settles
        await device.queue.onSubmittedWorkDone();
      }
    }
  } finally {
    unpatch();
    try { device.destroy(); } catch { /* ignore */ }
  }

  return results;
}

export function relativeError(a, b) {
  if (b === 0) return a === 0 ? 0 : Infinity;
  return Math.abs(a - b) / b;
}
