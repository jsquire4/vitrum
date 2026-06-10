// tonemap_functions.glsl.js — GLSL port of the @vitrum/shared-samplers tonemap
// operators (packages/shared-samplers/src/tonemap.ts + wgsl/tonemap.wgsl.ts).
//
// CANONICAL SOURCE: @vitrum/shared-samplers is the single source of truth for
// the operator math.  This GLSL port MUST be kept numerically identical to the
// WGSL twin (tonemapWgsl() in shared-samplers/src/wgsl/tonemap.wgsl.ts) and
// the TS reference (applyTonemap() in shared-samplers/src/tonemap.ts).  If the
// operators change there, update here in lockstep.
//
// Wired 2026-06-10: FrameQualitySettings.tonemap / .exposure / .outputColorSpace
// live via the pt-webgl2 present pass (glResources.ts #runPresentPass).

/** GLSL 300 es fragment body providing `vitrumTonemap` + `vt_linearToSrgb`.
 *  Included verbatim into the present-pass fragment shader body. */
export const tonemap_functions = /* glsl */ `
// --- vitrum tonemap operators (port of @vitrum/shared-samplers) ---

vec3 vt_aces(vec3 x) {
  float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

float vt_agx_curve(float x) {
  float x2 = x * x; float x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 vt_agx(vec3 c) {
  vec3 v = max(c, vec3(1e-6));
  vec3 lx = clamp((log2(v) + vec3(12.47393)) / (12.47393 + 4.026069), vec3(0.0), vec3(1.0));
  return clamp(vec3(vt_agx_curve(lx.x), vt_agx_curve(lx.y), vt_agx_curve(lx.z)), vec3(0.0), vec3(1.0));
}

// Linear -> sRGB OETF (IEC 61966-2-1). Skipped when outputColorSpace=1 (linear).
vec3 vt_linearToSrgb(vec3 c) {
  vec3 v = max(c, vec3(0.0));
  vec3 lo = v * 12.92;
  vec3 hi = 1.055 * pow(v, vec3(1.0 / 2.4)) - vec3(0.055);
  return mix(hi, lo, vec3(lessThanEqual(v, vec3(0.0031308))));
}

// mode: 0=aces(default) 1=agx 2=reinhard 3=linear(clamped) 4=none.
// Exposure is applied first (same as WGSL vitrumTonemap + TS applyTonemap).
vec3 vitrumTonemap(vec3 color, int mode, float exposure) {
  vec3 x = color * exposure;
  if (mode == 1) { return vt_agx(x); }
  if (mode == 2) { return x / (1.0 + max(x, vec3(0.0))); }
  if (mode == 3) { return clamp(x, vec3(0.0), vec3(1.0)); }
  if (mode == 4) { return x; }
  return vt_aces(x); // default: mode==0 (aces)
}
`;
