/**
 * Octahedral encoding / decoding and atlas UV helpers for DDGI probe maps.
 * Encode/decode core is shared with `@vitrum/shared-samplers` / pt-webgpu.
 * Per Cigolle et al. JCGT 2014.
 */

import { OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

export const OCTAHEDRAL_WGSL =
  OCTAHEDRAL_CORE_WGSL.trimEnd() +
  /* wgsl */ `

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
