// @ts-nocheck
/**
 * tools/lib/ptNagaGapFix.mjs
 *
 * Shared naga-gap WGSL patches for the native (lavapipe/wgpu-native) GPU
 * harnesses. Extracted verbatim from tools/behavioral-gate/gate.mjs — the
 * behavioral gate and the radiometric-ab harness (tools/radiometric-ab/
 * helpers.mjs) previously carried BYTE-IDENTICAL copies of these functions.
 * They MUST stay a single source of truth so every native harness exercises the
 * exact same shader-rewrite path (D17-4 / T8).
 *
 * Naga (used by Deno's built-in wgpu-native WebGPU + Firefox) rejects a few
 * WGSL constructs that Tint/Chrome accept. These patches rewrite the shipped
 * pt-webgpu kernel source at createShaderModule time so lavapipe accepts it:
 *   - strip the extra mip arg from textureLoad(bdptLightPath, …)
 *   - inject isNan/isInf vec3 helpers
 *   - when BDPT is off, demote the read_write rgba32float storage texture to a
 *     sampled texture_2d<f32> and drop the (dead) textureStore calls.
 *
 * Pure string transforms — no GPU or vitrum imports — so both deno import maps
 * can consume it by relative path.
 */

export function stripBdptMipArg(wgsl) {
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

export function addBdptMipArg(wgsl) {
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
