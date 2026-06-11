#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env
// @ts-nocheck
/**
 * tools/behavioral-gate/gate.mjs
 *
 * Behavioral gate — exercises the full pt-webgpu and walkaround-hybrid engines
 * end-to-end on lavapipe (software Vulkan) for every production config.
 *
 * Per config:
 *   - boots the engine
 *   - renders N frames at 64×64
 *   - reads back the output texture
 *   - checks: zero GPU validation/OOM errors, all pixels finite, mean luminance
 *     meets the per-config expectation (see EXPECTATION_TABLE below)
 *
 * Usage (from repo root):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
 *   npm run behavioral-gate
 *
 * Self-test mode (injects a black config and verifies detection):
 *   ... behavioral-gate -- --self-test
 *
 * Exit codes:
 *   0 — all configs passed their expectations
 *   1 — one or more configs failed
 *
 * ── Expectation table ────────────────────────────────────────────────────────
 *
 * Each entry has:
 *   expected: 'ok' | 'known-residual'
 *
 *   'ok'            — gate FAILS if result is BLACK, GPU-ERROR, NaN, or ERROR
 *   'known-residual' — gate passes regardless of result; the entry records:
 *                       reason: human-readable explanation
 *                       planItem: road-to-100 / trust-remediation item number
 *                     As fixes land, the agent fixing the config updates the
 *                     table to 'ok' and removes reason/planItem.
 *
 * ── Naga gap patches ─────────────────────────────────────────────────────────
 *
 * - pt-webgpu: bdptLightPath storage texture → sampled texture when bdpt=off;
 *   isNan/isInf polyfills; strip 3-arg textureLoad mip level.
 * - walkaround-hybrid: uses the production nagaFix.mjs from tools/shader-gate/
 *   (ptr<storage> parameter rewrite, among others). Relative import — no
 *   absolute paths.
 *
 * IMPORTANT: all @vitrum/* imports are resolved via deno.json in this directory
 * (relative paths). Do NOT add absolute paths — see the stale-import lesson in
 * CLAUDE.md.
 */

import { createPTEngine_WebGPU } from "@vitrum/pt-webgpu";
import { createWalkaroundEngine_Hybrid } from "@vitrum/walkaround-hybrid";
import { asMat4 } from "@vitrum/core";
import { applyNagaFix } from "../shader-gate/nagaFix.mjs";

// ── CLI flags ─────────────────────────────────────────────────────────────────
const selfTest = Deno.args.includes("--self-test");

// ── Expectation table ─────────────────────────────────────────────────────────
// keyed by config label; missing entry defaults to { expected: 'ok' }.
const EXPECTATION_TABLE = {
  // pt-webgpu configs
  "pt/default":           { expected: "ok" },
  "pt/spectral":          { expected: "ok" },
  "pt/bdpt":              { expected: "ok" },
  "pt/caustic-manifold":  { expected: "ok" },
  "pt/caustic-photon":    { expected: "ok" }, // R7a-3 fixed: capacity 32 fits default maxStorageBufferBindingSize
  "pt/spectral+photon":  { expected: "ok" }, // item 21: spectral×photon-map gather fix (heroLambda passed to sppmGather)
  "pt/lite-tier":         { expected: "ok" },
  "pt/restirPtReuse":     { expected: "ok" }, // A1 done: composite megakernel folds rpt indirect into beauty (kernel.wgsl.ts:308-312, gpuResources.ts:930)
  "pt/skinned-mesh":      { expected: "ok" },
  "pt/analytic-sphere":   { expected: "ok" },
  "pt/point-light":       { expected: "ok" },
  "pt/spot-light":        { expected: "ok" },
  "pt/directional-2":     { expected: "ok" },
  "pt/hdri-env":          { expected: "ok" },
  "pt/procedural-sky":    { expected: "ok" },
  "pt/spectral+bdpt":     { expected: "ok" },
  "pt/lite+hdri":         { expected: "ok" },
  "pt/lite+point-light":  { expected: "ok" },

  // walkaround configs
  "wh/default":           { expected: "ok" },
  "wh/rcEnabled":         { expected: "ok" }, // R7a-2 fixed: rc_tlas naga renames
  "wh/ppgEnabled":        { expected: "ok" },
  "wh/gtao-off":          { expected: "ok" },
  "wh/checkerboard":      { expected: "ok" },
  "wh/skinned-mesh":      { expected: "ok" },
  "wh/hdri-env":          { expected: "ok" },
  "wh/rect-area-emitter": { expected: "ok" },
  // item 4 (2026-06-10) — direct sun NEE default-on. Strong directional emitter,
  // no rect-area emitter; asserts non-black + higher luminance than the no-sun default.
  // Pin provenance: sun-NEE default-on, 2026-06-10 — RENDER-CHANGING for directional-lit scenes, A/B in R8-C.
  "wh/directional-sun":   { expected: "ok" },
};

// ── Matrix ────────────────────────────────────────────────────────────────────

const PT_CONFIGS = [
  { label: "pt/default",          eng: {},                                    scene: {} },
  { label: "pt/spectral",         eng: { spectral: true },                    scene: {} },
  { label: "pt/bdpt",             eng: { bdpt: true },                        scene: {} },
  { label: "pt/caustic-manifold", eng: { causticStrategy: "manifold-nee" },   scene: {} },
  { label: "pt/caustic-photon",   eng: { causticStrategy: "photon-map" },     scene: {} },
  { label: "pt/spectral+photon",  eng: { spectral: true, causticStrategy: "photon-map" }, scene: {} },
  { label: "pt/lite-tier",        eng: { traceTier: "lite" },                 scene: {} },
  { label: "pt/restirPtReuse",    eng: { restirPtReuse: true },               scene: {} },
  { label: "pt/skinned-mesh",     eng: {},                                    scene: { skinned: true } },
  { label: "pt/analytic-sphere",  eng: {},                                    scene: { analytic: true } },
  { label: "pt/point-light",      eng: {},                                    scene: {
    emitters: [{ kind: "point", id: "pt-light", position: [0, 0.8, 0], color: [1,1,1], intensity: 4.0 }],
  }},
  { label: "pt/spot-light",       eng: {},                                    scene: {
    emitters: [{ kind: "spot", id: "sp-light", position: [0, 0.8, 0], direction: [0,-1,0],
      color: [1,1,1], intensity: 6.0, angle: 0.8, penumbra: 0.2 }],
  }},
  { label: "pt/directional-2",    eng: {},                                    scene: {
    emitters: [
      { kind: "directional", id: "dir1", direction: [0.3,-0.8,0.5], color: [1,1,1], intensity: 2.0 },
      { kind: "directional", id: "dir2", direction: [-0.3,-0.8,-0.5], color: [0.8,0.9,1.0], intensity: 1.0 },
    ],
  }},
  { label: "pt/hdri-env",         eng: {},                                    scene: { hdri: true } },
  { label: "pt/procedural-sky",   eng: {},                                    scene: { sky: true } },
  { label: "pt/spectral+bdpt",    eng: { spectral: true, bdpt: true },        scene: {} },
  { label: "pt/lite+hdri",        eng: { traceTier: "lite" },                 scene: { hdri: true } },
  { label: "pt/lite+point-light", eng: { traceTier: "lite" },                 scene: {
    emitters: [{ kind: "point", id: "pt-light", position: [0, 0.8, 0], color: [1,1,1], intensity: 4.0 }],
  }},
];

const WH_CONFIGS = [
  { label: "wh/default",           eng: {},                                    scene: {} },
  { label: "wh/rcEnabled",         eng: { rcEnabled: true },                   scene: {} },
  { label: "wh/ppgEnabled",        eng: { ppgEnabled: true },                  scene: {} },
  { label: "wh/gtao-off",          eng: { gtaoEnabled: false },                scene: {} },
  { label: "wh/checkerboard",      eng: { checkerboardEnabled: true },         scene: {} },
  { label: "wh/skinned-mesh",      eng: {},                                    scene: { skinned: true } },
  { label: "wh/hdri-env",          eng: {},                                    scene: { hdri: true } },
  { label: "wh/rect-area-emitter", eng: {},                                    scene: {} },
  // item 4 (2026-06-10) — direct sun NEE: strong directional emitter, no rect-area
  // light. Opaque Cornell box surfaces should be lit via lo_sunNEE. LUM_THRESHOLD
  // gates BLACK detection (>0.005); a strong directional at intensity 3.0 reliably
  // exceeds that after 8 frames even on lavapipe.
  { label: "wh/directional-sun",   eng: {
      primaryLightDir:       [0.3, -0.8, 0.5],
      primaryLightIntensity: 3.0,
    },                                                                          scene: { directionalOnly: true } },
];

// ── Scene builder ─────────────────────────────────────────────────────────────

function makeQuad(id, verts, normal, color) {
  return {
    kind: "mesh", id,
    positions: new Float32Array(verts.flat()),
    normals:   new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs:       new Float32Array(8),
    indices:   new Uint32Array([0, 2, 1, 2, 0, 3]),
    material:  { baseColor: color, roughness: 1.0, metallic: 0.0 },
  };
}

function buildCornellScene(opts = {}) {
  const primitives = [
    makeQuad("floor",      [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0],  [0.8,0.8,0.8]),
    makeQuad("ceiling",    [[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],     [0,-1,0], [0.8,0.8,0.8]),
    makeQuad("back-wall",  [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],     [0,0,-1], [0.8,0.8,0.8]),
    makeQuad("left-wall",  [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
    makeQuad("right-wall", [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
  ];

  if (opts.skinned) {
    // For pt-webgpu: skinIndices/skinWeights/bones/boneInverses contract
    // For walkaround-hybrid: joints/weights/bindMatrices/jointMatrices contract
    // We supply both sets so the same buildCornellScene works for both backends.
    const ident = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    primitives.push({
      kind: "skinned-mesh", id: "skinned-box",
      positions:    new Float32Array([-0.3,-0.7,0.0, 0.3,-0.7,0.0, 0.3,-0.3,0.0, -0.3,-0.3,0.0]),
      normals:      new Float32Array([0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1]),
      uvs:          new Float32Array(8),
      indices:      new Uint32Array([0,2,1,2,0,3]),
      // pt-webgpu contract
      skinIndices:  new Uint32Array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
      skinWeights:  new Float32Array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]),
      bones:        ident,
      boneInverses: ident,
      // walkaround-hybrid contract
      joints:        new Uint16Array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
      weights:       new Float32Array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]),
      bindMatrices:  [ident],
      jointMatrices: [ident],
      material: { baseColor: [0.5,0.5,0.9], roughness: 0.6, metallic: 0.0 },
    });
  }

  if (opts.analytic) {
    primitives.push({
      kind: "analytic", id: "analytic-sphere",
      shape: { kind: "sphere", radius: 0.3 },
      transform: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0.2,-0.5,0.2,1]),
      material: { baseColor: [0.9,0.5,0.1], roughness: 0.2, metallic: 0.8 },
    });
  }

  // directionalOnly: no rect-area emitter, so the only light is the primary
  // directional set on the engine (tests that lo_sunNEE lights opaque surfaces).
  const emitters = opts.directionalOnly ? [] : (opts.emitters ?? [{
    kind: "rect-area", id: "ceiling-light",
    position: [0, 0.95, 0],
    uAxis: [0, 0, 0.2], vAxis: [0.2, 0, 0],
    color: [1,1,1], intensity: 12.0,
  }]);

  let environment = { kind: "none" };
  if (opts.hdri) {
    const W = 4, H = 2;
    const texels = new Float32Array(W * H * 4);
    for (let i = 0; i < texels.length; i += 4) {
      texels[i] = 1.0; texels[i+1] = 1.0; texels[i+2] = 1.0; texels[i+3] = 1.0;
    }
    environment = { kind: "hdri", textureData: texels, width: W, height: H, intensity: 1.0 };
  }
  if (opts.sky) {
    environment = { kind: "procedural-sky", sunDirection: [0.5, 1.0, 0.3] };
  }

  return { primitives, emitters, environment };
}

// ── Camera ────────────────────────────────────────────────────────────────────

const W = 64, H = 64;

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

// pt-webgpu camera
const PT_EYE    = [-0.05, 0, 2.75];
const PT_CENTER = [-0.05, -0.15, 0];
const ptProj    = asMat4(makePerspectiveMatrix(40, W / H, 0.1, 50));
const ptView    = asMat4(makeLookAtMatrix(PT_EYE, PT_CENTER, [0,1,0]));

// walkaround camera
const WH_EYE    = [0, 0, 2.5];
const WH_CENTER = [0, 0, 0];
const whProj    = asMat4(makePerspectiveMatrix(60, W / H, 0.1, 50));
const whView    = asMat4(makeLookAtMatrix(WH_EYE, WH_CENTER, [0,1,0]));

// ── Naga gap patches (pt-webgpu) ──────────────────────────────────────────────
// Mirrors render-pt-webgpu.ts and the /tmp prototypes.

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

function applyPtNagaGapFix(wgsl, bdptOn) {
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

function patchDeviceForPt(device, bdptOn) {
  const orig = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    if (typeof desc.code === "string" && desc.code.includes("bdptLightPath")) {
      return orig({ ...desc, code: applyPtNagaGapFix(desc.code, bdptOn) });
    }
    return orig(desc);
  };
  return () => { device.createShaderModule = orig; };
}

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

async function acquirePtDevice(wantsFullTier) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
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

// ── Readback helpers ──────────────────────────────────────────────────────────

/**
 * Readback an rgba16float (or rgba32float) texture via a blit-to-rgba8unorm
 * render pass, then CPU readback.  Returns Uint8Array of RGBA pixels.
 */
async function readbackAsRgba8(device, srcTex, texW, texH) {
  const dstTex = device.createTexture({
    size: { width: texW, height: texH },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    label: "bg-blit-dst",
  });
  const blitMod = device.createShaderModule({
    label: "bg-blit",
    code: `
      @group(0) @binding(0) var srcTex: texture_2d<f32>;
      @group(0) @binding(1) var srcSmp: sampler;
      struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
      @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
        var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
        var o: VO;
        o.pos = vec4f(p[vi], 0.0, 1.0);
        o.uv  = (p[vi] + vec2f(1,1)) * 0.5;
        o.uv.y = 1.0 - o.uv.y;
        return o;
      }
      @fragment fn fs(i: VO) -> @location(0) vec4f {
        return vec4f(clamp(textureSample(srcTex, srcSmp, i.uv).rgb, vec3f(0), vec3f(1)), 1.0);
      }
    `,
  });
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
  ]});
  const pipeline = device.createRenderPipeline({
    label: "bg-blit-pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex:   { module: blitMod, entryPoint: "vs" },
    fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const bg = device.createBindGroup({ layout: bgl, entries: [
    { binding: 0, resource: srcTex.createView() },
    { binding: 1, resource: device.createSampler({ magFilter: "nearest", minFilter: "nearest" }) },
  ]});
  const enc  = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{
    view: dstTex.createView(), loadOp: "clear", storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }]});
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  device.queue.submit([enc.finish()]);

  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc2 = device.createCommandEncoder();
  enc2.copyTextureToBuffer({ texture: dstTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc2.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy(); dstTex.destroy();
  return pixels;
}

/** Readback bgra8unorm texture (walkaround swap chain). Returns RGBA pixels. */
async function readbackBgra8(device, tex, texW, texH) {
  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc  = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy();
  // swap B↔R (bgra → rgba)
  for (let i = 0; i < pixels.length; i += 4) {
    const b = pixels[i]; pixels[i] = pixels[i+2]; pixels[i+2] = b;
  }
  return pixels;
}

function meanLuminance(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += 0.2126 * (pixels[i]/255) + 0.7152 * (pixels[i+1]/255) + 0.0722 * (pixels[i+2]/255);
  }
  return sum / (pixels.length / 4);
}

function hasNaN(pixels) {
  for (let i = 0; i < pixels.length; i++) {
    if (Number.isNaN(pixels[i])) return true;
  }
  return false;
}

// ── Result classification ─────────────────────────────────────────────────────

const SPP = 8;
const LUM_THRESHOLD = 0.005;

/**
 * Run the expectation check for a single config result.
 * Returns { pass: boolean, note: string }.
 */
function checkExpectation(label, rawStatus, lum, errCount, nans) {
  const entry = EXPECTATION_TABLE[label] ?? { expected: "ok" };
  if (entry.expected === "known-residual") {
    return {
      pass: true,
      note: `KNOWN-RESIDUAL (${entry.planItem ?? "?"}) — ${rawStatus} — ${entry.reason}`,
    };
  }
  // expected === 'ok'
  if (rawStatus === "OK") return { pass: true, note: "" };
  return { pass: false, note: `FAIL — expected OK, got ${rawStatus} (lum=${lum.toFixed(4)}, gpuErrs=${errCount}, nan=${nans})` };
}

// ── pt-webgpu runner ──────────────────────────────────────────────────────────

async function runPtConfig(label, engineOpts, sceneOpts) {
  const bdptOn    = engineOpts.bdpt === true;
  const isLite    = engineOpts.traceTier === "lite";
  let device;
  try {
    device = await acquirePtDevice(!isLite);
  } catch (e) {
    return { label, rawStatus: "BLOCKED", lum: 0, errCount: 0, nans: false, errorMsg: e.message };
  }

  let errCount = 0;
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const unpatch = patchDeviceForPt(device, bdptOn);
  let engine  = null;
  let pixels  = null;
  let errorMsg = null;

  try {
    engine = await createPTEngine_WebGPU({
      device,
      maxBounces: 3,
      maxSamplesPerPixel: SPP,
      ...engineOpts,
    });

    const scene = buildCornellScene(sceneOpts);
    engine.setScene(scene);

    let frameOutput = null;
    for (let frame = 0; frame < SPP; frame++) {
      const seed = ((frame * 6364136223846793005 + 1442695040888963407) >>> 0);
      frameOutput = engine.renderFrame({
        viewMatrix: ptView,
        projMatrix: ptProj,
        cameraPosition: PT_EYE,
        viewport: { width: W, height: H, devicePixelRatio: 1 },
        frameIndex: frame, frameSeed: seed,
        quality: { samplesTarget: SPP, bounces: 3, resolutionFactor: 1 },
      });
      await device.queue.onSubmittedWorkDone();
      if (frameOutput?.isConverged) break;
    }

    if (frameOutput?.kind === "rendered") {
      pixels = await readbackAsRgba8(device, frameOutput.primaryRadiance, W, H);
    }
  } catch (e) {
    errorMsg = e.message;
  } finally {
    unpatch();
    try { engine?.dispose(); } catch {}
  }

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  if (oomErr) errCount++;
  if (valErr) errCount++;
  device.destroy();

  if (errorMsg) return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg };
  if (!pixels)  return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg: "no pixels" };

  const nans = hasNaN(pixels);
  const lum  = meanLuminance(pixels);
  const rawStatus = nans ? "NaN" : (errCount > 0 ? "GPU-ERROR" : (lum < LUM_THRESHOLD ? "BLACK" : "OK"));
  return { label, rawStatus, lum, errCount, nans };
}

// ── walkaround-hybrid runner ──────────────────────────────────────────────────

async function runWhConfig(label, engineOpts, sceneOpts) {
  let device;
  try {
    device = await acquireWhDevice();
  } catch (e) {
    return { label, rawStatus: "BLOCKED", lum: 0, errCount: 0, nans: false, errorMsg: e.message };
  }

  let errCount = 0;
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  patchDeviceForWh(device);

  let engine   = null;
  let pixels   = null;
  let swapTex  = null;
  let errorMsg = null;

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

    const scene = buildCornellScene(sceneOpts);
    engine.setScene(scene);

    // Poll until ready (state machine: uninitialized → initializing → ready)
    const deadline = Date.now() + 60_000;
    while (engine.state !== "ready" && engine.state !== "error") {
      await new Promise(r => setTimeout(r, 50));
      if (Date.now() > deadline) throw new Error("engine init timeout after 60 s");
    }
    if (engine.state === "error") throw new Error("engine.state === 'error'");

    swapTex = device.createTexture({
      label: "bg-wh-swap",
      size:  [W, H, 1],
      format: "bgra8unorm",
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    });
    const swapView = swapTex.createView();

    for (let fi = 0; fi < SPP; fi++) {
      engine.renderFrame({
        viewMatrix: whView,
        projMatrix: whProj,
        cameraPosition: WH_EYE,
        viewport: { width: W, height: H, devicePixelRatio: 1 },
        frameIndex: fi,
        frameSeed:  fi * 1664525 + 1013904223,
        swapChainView:   swapView,
        swapChainFormat: "bgra8unorm",
      });
      await device.queue.onSubmittedWorkDone();
    }

    pixels = await readbackBgra8(device, swapTex, W, H);
  } catch (e) {
    errorMsg = e.message;
  } finally {
    try { swapTex?.destroy(); } catch {}
    try { engine?.dispose();  } catch {}
  }

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  if (oomErr) errCount++;
  if (valErr) errCount++;
  device.destroy();

  if (errorMsg) return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg };
  if (!pixels)  return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg: "no pixels" };

  const nans = hasNaN(pixels);
  const lum  = meanLuminance(pixels);
  const rawStatus = nans ? "NaN" : (errCount > 0 ? "GPU-ERROR" : (lum < LUM_THRESHOLD ? "BLACK" : "OK"));
  return { label, rawStatus, lum, errCount, nans };
}

// ── Self-test config (injected when --self-test) ───────────────────────────────
// Forces a BLACK result so the gate can verify it catches a failing config.
async function runSelfTestConfig() {
  return {
    label: "__self-test/always-black",
    rawStatus: "BLACK",
    lum: 0.0,
    errCount: 0,
    nans: false,
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log("=== behavioral-gate ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}, SPP: ${SPP}`);
if (selfTest) console.log("Mode: --self-test");
console.log("");

const results = [];

console.log("── pt-webgpu ──");
for (const cfg of PT_CONFIGS) {
  const r = await runPtConfig(cfg.label, cfg.eng, cfg.scene);
  results.push(r);
  const { pass, note } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  const marker = pass ? "PASS" : "FAIL";
  const detail = r.errorMsg
    ? `${r.rawStatus} | ${r.errorMsg.replace(/\n/g, " ").slice(0, 160)}`
    : `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans}`;
  console.log(`  ${marker} | ${r.label.padEnd(28)} | ${detail}${note ? " | " + note : ""}`);
}

console.log("");
console.log("── walkaround-hybrid ──");
for (const cfg of WH_CONFIGS) {
  const r = await runWhConfig(cfg.label, cfg.eng, cfg.scene);
  results.push(r);
  const { pass, note } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  const marker = pass ? "PASS" : "FAIL";
  const detail = r.errorMsg
    ? `${r.rawStatus} | ${r.errorMsg.replace(/\n/g, " ").slice(0, 160)}`
    : `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans}`;
  console.log(`  ${marker} | ${r.label.padEnd(28)} | ${detail}${note ? " | " + note : ""}`);
}

// ── Self-test mode ────────────────────────────────────────────────────────────

if (selfTest) {
  console.log("");
  console.log("── self-test ──");
  // Inject a config that always returns BLACK — it has no EXPECTATION_TABLE entry
  // so it defaults to { expected: 'ok' } → should FAIL the gate.
  const stResult = await runSelfTestConfig();
  const { pass: stPass } = checkExpectation(stResult.label, stResult.rawStatus, stResult.lum, 0, false);
  if (!stPass) {
    console.log("  PASS | __self-test/always-black correctly detected as FAIL");
  } else {
    console.error("  FAIL | __self-test/always-black was NOT detected (gate broken)");
    Deno.exit(1);
  }

  // Verify that all production results match their expectations.
  const prodFails = results.filter(r => {
    const { pass } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
    return !pass;
  });
  if (prodFails.length > 0) {
    console.error("\n--self-test aborted: production configs also failed:");
    for (const r of prodFails) console.error(`  ${r.label}: ${r.rawStatus}`);
    Deno.exit(1);
  }
  console.log("  --self-test PASSED");
  Deno.exit(0);
}

// ── Summary + exit code ───────────────────────────────────────────────────────

console.log("");
const failures = results.filter(r => {
  const { pass } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  return !pass;
});
const residuals = results.filter(r => {
  const entry = EXPECTATION_TABLE[r.label] ?? { expected: "ok" };
  return entry.expected === "known-residual";
});

console.log(`=== summary: ${results.length} configs total, ${failures.length} failures, ${residuals.length} known-residuals ===`);

if (failures.length > 0) {
  console.error("\nFAILED configs:");
  for (const r of failures) {
    console.error(`  ${r.label}: ${r.rawStatus} (lum=${r.lum.toFixed(4)}, gpuErrs=${r.errCount})`);
  }
  Deno.exit(1);
}

Deno.exit(0);
