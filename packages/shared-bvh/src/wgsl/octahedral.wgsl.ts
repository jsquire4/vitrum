/**
 * Octahedral encoding / decoding and atlas UV helpers for DDGI probe maps.
 * Encode/decode core is shared with `@vitrum/shared-samplers` / pt-webgpu.
 * Per Cigolle et al. JCGT 2014.
 */

import { OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

export const OCTAHEDRAL_WGSL =
  OCTAHEDRAL_CORE_WGSL.trimEnd() +
  /* wgsl */ `

// Compute the atlas UV for a probe's texel. Parameterized by cell size:
//   irradiance atlas uses cell=8, visibility atlas uses cell=16.
// Each cell carries a +1 px border on every side, so the per-probe stride
// is cell + 2.
//   probeIdx  = flat probe index (x + y*dimsX + z*dimsX*dimsY)
//   octUv     = octahedral UV in [0,1]^2
//   atlasW/H  = atlas dimensions in texels
//   gridDims  = probe grid dims (x,y,z)
//   cell      = per-probe texel side (8 for irradiance, 16 for visibility)
fn probeAtlasUv(
  probeIdx: u32,
  octUv: vec2f,
  atlasW: f32, atlasH: f32,
  gridDims: vec3u,
  cell: u32,
) -> vec2f {
  let stride = cell + 2u;
  let px = probeIdx % gridDims.x;
  let tmp = probeIdx / gridDims.x;
  let py = tmp % gridDims.y;
  let pz = tmp / gridDims.y;
  let cellX = f32(px * stride) + 1.0;    // +1 for border
  let cellY = f32((py + pz * gridDims.y) * stride) + 1.0;
  let u = (cellX + octUv.x * f32(cell)) / atlasW;
  let v = (cellY + octUv.y * f32(cell)) / atlasH;
  return vec2f(u, v);
}

fn irradianceAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: f32, atlasH: f32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 8u);
}

fn visibilityAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: f32, atlasH: f32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 16u);
}

`;
