/**
 * TSL/WGSL helper for sampling DDGI irradiance in a fragment shader.
 *
 * Exports a single `ddgiSample` function as one self-contained wgslFn
 * source. Used by `applyDDGIShading.ts` to inject DDGI diffuse-indirect
 * into PBR materials.
 *
 * Why a single inlined function (no helpers): Three.js's wgslFn parser
 * (three.webgpu.js:38176) matches only the FIRST `fn` declaration in
 * the source via `/^[fn]*\s*([a-z_0-9]+)?\s*\(([\s\S]*?)\)/`. If the
 * source contains helper functions before the entry, the parser
 * registers the wrong function (with the wrong input count). Calling
 * the wgslFn with N arguments where the parser found a function with
 * fewer than N inputs throws `TypeError: Cannot read properties of
 * undefined (reading 'type')` from `generateInput` every frame, freezing
 * the JS thread within seconds.
 *
 * Mitigation: inline every helper (octEncodeDDGI, irrAtlasUvDDGI,
 * visAtlasUvDDGI) directly inside the single `ddgiSample` body. The
 * parser sees one entry function with exactly 16 inputs, matching the
 * call site's 16 arguments.
 */

import { IRR_CELL, VIS_CELL, IRR_STRIDE, VIS_STRIDE } from './ddgiAtlasLayout';

// Atlas-layout constants are template-substituted at module-load time so
// the producer (probeGrid.allocateAtlases) and the two consumers
// (this file + engines/restir/shaders/shade.wgsl.ts) read the same values
// from one source of truth (ddgiAtlasLayout.ts).
export const DDGI_SAMPLE_WGSL = /* wgsl */`
fn ddgiSample(
  worldPos: vec3f,
  surfaceNormal: vec3f,
  irradianceAtlas: texture_2d<f32>,
  visibilityAtlas: texture_2d<f32>,
  samp: sampler,
  gridOriginX: f32, gridOriginY: f32, gridOriginZ: f32,
  gridSpacing: f32,
  gridDimsX: u32, gridDimsY: u32, gridDimsZ: u32,
  irrW: f32, irrH: f32, visW: f32, visH: f32,
) -> vec3f {
  let gridOrigin = vec3f(gridOriginX, gridOriginY, gridOriginZ);
  let gridDims   = vec3u(gridDimsX, gridDimsY, gridDimsZ);

  let gridPos  = (worldPos - gridOrigin) / gridSpacing;
  let baseIdx3 = vec3i(floor(gridPos));
  let frac     = fract(gridPos);

  var sum         = vec3f(0.0);
  var totalWeight = 0.0;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let co  = vec3u((i & 1u), (i >> 1u) & 1u, (i >> 2u) & 1u);
    let pi3 = baseIdx3 + vec3i(co);
    if (any(pi3 < vec3i(0)) || any(pi3 >= vec3i(gridDims))) { continue; }

    let probeFlatIdx = u32(pi3.x) +
                       u32(pi3.y) * gridDims.x +
                       u32(pi3.z) * gridDims.x * gridDims.y;
    let probeWorld   = gridOrigin + vec3f(pi3) * gridSpacing;

    // Trilinear weight.
    let tw = mix(vec3f(1.0) - frac, frac, vec3f(co));
    var w  = tw.x * tw.y * tw.z;

    // Smooth backface modulation (DDGI paper Eq. 9).
    let toProbe   = probeWorld - worldPos;
    let probeDist = length(toProbe);
    if (probeDist > 1e-3) {
      let probeDir = toProbe / probeDist;
      let nDotP    = dot(surfaceNormal, probeDir);
      let bw       = pow((nDotP + 1.0) * 0.5, 2.0) + 0.2;
      w = w * bw;
    }

    // Octahedral-encode the surface→probe direction (visibility lookup).
    let probeDirToSurf = normalize(worldPos - probeWorld);
    let dirV       = -probeDirToSurf;
    let absV       = abs(dirV);
    let nv         = dirV / (absV.x + absV.y + absV.z);
    var octV: vec2f;
    if (nv.z >= 0.0) { octV = nv.xy; }
    else { octV = (1.0 - abs(nv.yx)) * vec2f(sign(nv.x), sign(nv.y)); }
    octV = octV * 0.5 + 0.5;

    // Visibility atlas UV (cell + 2px border, 1px each side). Strides
    // come from ddgiAtlasLayout.ts via template substitution.
    let visStride = ${VIS_STRIDE}u;
    let visCell   = ${VIS_CELL}u;
    let visPx     = probeFlatIdx % gridDims.x;
    let visTmpY   = probeFlatIdx / gridDims.x;
    let visPy     = visTmpY % gridDims.y;
    let visPz     = visTmpY / gridDims.y;
    let visCx     = f32(visPx * visStride) + 1.0 + octV.x * f32(visCell);
    let visCy     = f32((visPy + visPz * gridDims.y) * visStride) + 1.0 + octV.y * f32(visCell);
    let visUv     = vec2f(visCx / visW, visCy / visH);
    let vis       = textureSampleLevel(visibilityAtlas, samp, visUv, 0.0).rg;
    let mean      = vis.x;
    let variance  = abs(vis.y - mean * mean);
    let chebyshev = select(
      variance / (variance + max(0.0, probeDist - mean) * max(0.0, probeDist - mean)),
      1.0,
      probeDist <= mean,
    );
    w = w * max(chebyshev, 0.0);

    // Octahedral-encode the surface normal (irradiance lookup).
    let absN = abs(surfaceNormal);
    let nN   = surfaceNormal / (absN.x + absN.y + absN.z);
    var octN: vec2f;
    if (nN.z >= 0.0) { octN = nN.xy; }
    else { octN = (1.0 - abs(nN.yx)) * vec2f(sign(nN.x), sign(nN.y)); }
    octN = octN * 0.5 + 0.5;

    // Irradiance atlas UV (cell + 2px border, 1px each side). Strides
    // come from ddgiAtlasLayout.ts via template substitution.
    let irrStride = ${IRR_STRIDE}u;
    let irrCell   = ${IRR_CELL}u;
    let irrPx     = probeFlatIdx % gridDims.x;
    let irrTmpY   = probeFlatIdx / gridDims.x;
    let irrPy     = irrTmpY % gridDims.y;
    let irrPz     = irrTmpY / gridDims.y;
    let irrCx     = f32(irrPx * irrStride) + 1.0 + octN.x * f32(irrCell);
    let irrCy     = f32((irrPy + irrPz * gridDims.y) * irrStride) + 1.0 + octN.y * f32(irrCell);
    let irrUv     = vec2f(irrCx / irrW, irrCy / irrH);
    let irr       = textureSampleLevel(irradianceAtlas, samp, irrUv, 0.0).rgb;

    sum         = sum + irr * w;
    totalWeight = totalWeight + w;
  }

  if (totalWeight < 1e-4) {
    // No probes contributed — return neutral grey indirect.
    return vec3f(0.05);
  }
  return sum / totalWeight;
}
`;
