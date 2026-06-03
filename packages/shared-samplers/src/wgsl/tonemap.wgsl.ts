// tonemap.wgsl.ts — GPU twin of `../tonemap.ts` (P4). Emits `vitrumTonemap`,
// the operator the backends apply at composite/output. mode indices match
// `TONEMAP_MODE_INDEX`; kept in lockstep by `__tests__/tonemap.test.ts`.

export function tonemapWgsl(): string {
  return /* wgsl */ `
fn vt_aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
fn vt_agx_curve(x: f32) -> f32 {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn vt_agx(c: vec3f) -> vec3f {
  let v = max(c, vec3f(1e-6));
  let lx = clamp((log2(v) + vec3f(12.47393)) / (12.47393 + 4.026069), vec3f(0.0), vec3f(1.0));
  return clamp(vec3f(vt_agx_curve(lx.x), vt_agx_curve(lx.y), vt_agx_curve(lx.z)), vec3f(0.0), vec3f(1.0));
}
// Linear → sRGB OETF (default output encode; skipped for 'linear' colorspace).
fn vt_linearToSrgb(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  let lo = v * 12.92;
  let hi = 1.055 * pow(v, vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(hi, lo, v <= vec3f(0.0031308));
}
// mode: 0=aces 1=agx 2=reinhard 3=linear(clamped) 4=none. Exposure applied first.
fn vitrumTonemap(color: vec3f, mode: u32, exposure: f32) -> vec3f {
  let x = color * exposure;
  if (mode == 1u) { return vt_agx(x); }
  if (mode == 2u) { return x / (1.0 + max(x, vec3f(0.0))); }
  if (mode == 3u) { return clamp(x, vec3f(0.0), vec3f(1.0)); }
  if (mode == 4u) { return x; }
  return vt_aces(x);
}
`;
}
