/**
 * Ported from stainedGlass walkaround WGSL helpers.
 * Octahedral direction encoding is useful for compact normal/ray direction
 * storage in future temporal caches and aux buffers.
 */
export const OCTAHEDRAL_WGSL = /* wgsl */ `
fn octEncode(dir: vec3f) -> vec2f {
  let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
  if (n.z >= 0.0) {
    return n.xy;
  }
  return (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
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
