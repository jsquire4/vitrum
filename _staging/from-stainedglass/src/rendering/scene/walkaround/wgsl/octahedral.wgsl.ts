/**
 * Octahedral encoding / decoding and atlas UV helpers for DDGI probe maps.
 * Per Cigolle et al. "A Survey of Efficient Representations for Independent
 * Unit Vectors." JCGT 2014.
 */

export const OCTAHEDRAL_WGSL = /* wgsl */`

// Encode a unit direction to octahedral coordinates in [-1,1]^2.
fn octEncode(dir: vec3f) -> vec2f {
  let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
  if (n.z >= 0.0) {
    return n.xy;
  }
  // Fold the lower hemisphere.
  return (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
}

// Decode octahedral coordinates [-1,1]^2 to a unit direction.
fn octDecode(oct: vec2f) -> vec3f {
  let n = vec3f(oct, 1.0 - abs(oct.x) - abs(oct.y));
  if (n.z < 0.0) {
    let xy = (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
    return normalize(vec3f(xy, n.z));
  }
  return normalize(n);
}

// Compute the atlas UV for a probe's irradiance texel.
// probeIdx  = flat probe index (x + y*dimsX + z*dimsX*dimsY)
// octUv     = octahedral UV in [0,1]^2
// atlasW, atlasH = atlas dimensions in texels
// cellW, cellH   = per-probe cell size (irradiance: 8, visibility: 16)
// with +2 border padding each side, so stride = cellW + 2
fn irradianceAtlasUv(
  probeIdx: u32,
  octUv: vec2f,
  atlasW: f32, atlasH: f32,
  gridDims: vec3u,
) -> vec2f {
  let CELL = 8u;
  let STRIDE = CELL + 2u;
  let px = probeIdx % gridDims.x;
  let tmp = probeIdx / gridDims.x;
  let py = tmp % gridDims.y;
  let pz = tmp / gridDims.y;
  let cellX = f32(px * STRIDE) + 1.0;    // +1 for border
  let cellY = f32((py + pz * gridDims.y) * STRIDE) + 1.0;
  let u = (cellX + octUv.x * f32(CELL)) / atlasW;
  let v = (cellY + octUv.y * f32(CELL)) / atlasH;
  return vec2f(u, v);
}

// Same for visibility atlas (16×16 per probe).
fn visibilityAtlasUv(
  probeIdx: u32,
  octUv: vec2f,
  atlasW: f32, atlasH: f32,
  gridDims: vec3u,
) -> vec2f {
  let CELL = 16u;
  let STRIDE = CELL + 2u;
  let px = probeIdx % gridDims.x;
  let tmp = probeIdx / gridDims.x;
  let py = tmp % gridDims.y;
  let pz = tmp / gridDims.y;
  let cellX = f32(px * STRIDE) + 1.0;
  let cellY = f32((py + pz * gridDims.y) * STRIDE) + 1.0;
  let u = (cellX + octUv.x * f32(CELL)) / atlasW;
  let v = (cellY + octUv.y * f32(CELL)) / atlasH;
  return vec2f(u, v);
}

`;
