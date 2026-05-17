/**
 * Octahedral encoding / decoding and atlas UV helpers for DDGI probe maps.
 * Encode/decode core is shared with `@vitrum/shared-samplers` / pt-webgpu.
 * Per Cigolle et al. JCGT 2014.
 */

import { OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

export const OCTAHEDRAL_WGSL =
  OCTAHEDRAL_CORE_WGSL.trimEnd() +
  /* wgsl */ `

// Integer texel origin (upper-left corner inside the +1 px border) of a
// probe's per-probe cell within an octahedral atlas.  Returned as a vec2u
// so callers writing via textureStore() can add a pixel offset directly:
//
//   let origin = probeAtlasCellOrigin(probeIdx, gridDims, IRR_CELL);
//   let texel  = origin + pixel;   // vec2u
//   textureStore(irrOut, texel, ...);
//
// Sampling-side callers compose this with the float octahedral UV via
// probeAtlasUv() below.  Single source of truth for atlas-coordinate
// arithmetic (W2-C4 dedup) — was previously re-derived in three places:
// probeUpdateBlend.wgsl.ts (irradiance + visibility), ddgiSampleWgsl.ts
// (irradiance + visibility), and ddgi/probeUpdateRays.wgsl.ts (via
// irradianceAtlasUv).
fn probeAtlasCellOrigin(probeIdx: u32, gridDims: vec3u, cell: u32) -> vec2u {
  let stride = cell + 2u;
  let px = probeIdx % gridDims.x;
  let tmp = probeIdx / gridDims.x;
  let py = tmp % gridDims.y;
  let pz = tmp / gridDims.y;
  return vec2u(
    px * stride + 1u,
    (py + pz * gridDims.y) * stride + 1u,
  );
}

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
  let origin = probeAtlasCellOrigin(probeIdx, gridDims, cell);
  let u = (f32(origin.x) + octUv.x * f32(cell)) / atlasW;
  let v = (f32(origin.y) + octUv.y * f32(cell)) / atlasH;
  return vec2f(u, v);
}

fn irradianceAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: f32, atlasH: f32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 8u);
}

fn visibilityAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: f32, atlasH: f32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 16u);
}

`;
