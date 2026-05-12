/**
 * Octahedral encode/decode only (Cigolle et al. JCGT 2014).
 * Atlas UV helpers live in `@vitrum/shared-bvh` (DDGI probes).
 */

export const OCTAHEDRAL_CORE_WGSL = /* wgsl */ `
fn octEncode(dir: vec3f) -> vec2f {
  let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
  if (n.z >= 0.0) {
    return n.xy;
  }
  // Lower-hemisphere fold — Item #39 / Cigolle 2014 §A.1:
  // sign(0)=0 in WGSL collapses the fold when n.x or n.y is exactly 0.
  // Use select() so that 0 maps to +1, matching the paper's convention.
  let sx = select(-1.0, 1.0, n.x >= 0.0);
  let sy = select(-1.0, 1.0, n.y >= 0.0);
  return vec2f((1.0 - abs(n.y)) * sx, (1.0 - abs(n.x)) * sy);
}

fn octDecode(oct: vec2f) -> vec3f {
  let n = vec3f(oct, 1.0 - abs(oct.x) - abs(oct.y));
  if (n.z < 0.0) {
    let xy = (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
    return normalize(vec3f(xy, n.z));
  }
  return normalize(n);
}
`;
