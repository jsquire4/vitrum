/**
 * PPG guided indirect — extra @group(3) bindings + helpers for shade.wgsl.
 *
 * Reads the same atomic fixed-point leaf layout as ppgUpdate.wgsl (not vec2f bins).
 * Injected only when `ppgEnabled` at pipeline compile (see pipelineCompiler).
 */

/** Declarations + functions appended after PPG train bindings (group 3, slots 4–5). */
export const SHADE_PPG_GUIDE_WGSL = /* wgsl */`
const PPG_GUIDE_RADIANCE_SCALE: f32 = 65536.0;
const PPG_GUIDE_INDIRECT_BLEND: f32 = 0.35;

struct PPGGuideSpatialCell {
  position:  vec3f,
  _pad:      f32,
  leafIndex: u32,
  _pad2x:    u32,
  _pad2y:    u32,
  _pad2z:    u32,
};

struct PPGGuideKdNode {
  child0: u32,
  child1: u32,
  meta:   u32,
  split:  f32,
};

struct PPGShadeMeta {
  cellCount: u32,
  _pad0:     u32,
  _pad1:     u32,
  _pad2:     u32,
};

@group(3) @binding(6) var<storage, read> ppgGuideCells: array<PPGGuideSpatialCell>;
@group(3) @binding(7) var<storage, read_write> ppgGuideLeafData: array<atomic<u32>>;
@group(3) @binding(8) var<storage, read> ppgGuideKdNodes: array<PPGGuideKdNode>;
@group(3) @binding(9) var<uniform> ppgShadeMeta: PPGShadeMeta;

fn ppgGuideLeafSlot(leafIdx: u32, binIdx: u32, field: u32) -> u32 {
  return leafIdx * 64u + binIdx * 2u + field;
}

fn ppgGuideBinToOctahedral(binIdx: u32) -> vec2f {
  let row = f32(binIdx / 4u);
  let col = f32(binIdx % 4u);
  let u = (col + 0.5) / 4.0;
  let v = (row + 0.5) / 4.0;
  return vec2f(u, v);
}

fn ppgGuideOctahedralToDir(oct: vec2f) -> vec3f {
  let f = oct * 2.0 - vec2f(1.0);
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

fn ppgGuideAxisComp(v: vec3f, axis: u32) -> f32 {
  if (axis == 0u) { return v.x; }
  if (axis == 1u) { return v.y; }
  return v.z;
}

fn ppgGuideFindCellBrute(worldPos: vec3f) -> u32 {
  let cellCount = ppgShadeMeta.cellCount;
  if (cellCount == 0u) { return 0u; }
  var bestIdx  = 0u;
  var bestDist = 1e20;
  for (var i = 0u; i < cellCount; i++) {
    let d = ppgGuideCells[i].position - worldPos;
    let dist2 = dot(d, d);
    if (dist2 < bestDist) {
      bestDist = dist2;
      bestIdx  = i;
    }
  }
  return bestIdx;
}

fn ppgGuideKdFindCell(worldPos: vec3f) -> u32 {
  let nk = arrayLength(&ppgGuideKdNodes);
  let cellCount = ppgShadeMeta.cellCount;
  if (nk == 0u || cellCount == 0u) { return 0u; }
  let root = ppgGuideKdNodes[0];
  if (root.child0 == 0xFFFFFFFFu && root.child1 == 0xFFFFFFFFu) {
    return ppgGuideFindCellBrute(worldPos);
  }
  var bestIdx  = 0u;
  var bestDist2 = 1e38;
  var stN: array<u32, 48>;
  var stK: array<u32, 48>;
  var stFar: array<u32, 48>;
  var stD2: array<f32, 48>;
  var sp = 0u;
  stN[sp] = 0u;
  stK[sp] = 0u;
  stFar[sp] = 0u;
  stD2[sp] = 0.0;
  sp = sp + 1u;
  while (sp > 0u) {
    sp = sp - 1u;
    if (stK[sp] == 1u) {
      if (stD2[sp] < bestDist2 && sp < 48u) {
        stN[sp] = stFar[sp];
        stK[sp] = 0u;
        sp = sp + 1u;
      }
      continue;
    }
    let nid = stN[sp];
    if (nid >= nk) { continue; }
    let node = ppgGuideKdNodes[nid];
    let meta = node.meta;
    if ((meta & 0x80000000u) != 0u) {
      let cellIdx = meta & 0x7FFFFFFFu;
      if (cellIdx < cellCount) {
        let d = ppgGuideCells[cellIdx].position - worldPos;
        let dist2 = dot(d, d);
        if (dist2 < bestDist2) {
          bestDist2 = dist2;
          bestIdx = cellIdx;
        }
      }
      continue;
    }
    let axis = meta & 3u;
    let split = node.split;
    let c0 = node.child0;
    let c1 = node.child1;
    let d0 = ppgGuideAxisComp(worldPos, axis) - split;
    let d2plane = d0 * d0;
    let nearI = select(c1, c0, d0 < 0.0);
    let farI = select(c0, c1, d0 < 0.0);
    if (sp + 2u > 48u) {
      return ppgGuideFindCellBrute(worldPos);
    }
    stFar[sp] = farI;
    stD2[sp] = d2plane;
    stK[sp] = 1u;
    sp = sp + 1u;
    stN[sp] = nearI;
    stK[sp] = 0u;
    stFar[sp] = 0u;
    stD2[sp] = 0.0;
    sp = sp + 1u;
  }
  return bestIdx;
}

fn ppgGuideFindCellIndex(worldPos: vec3f) -> u32 {
  return ppgGuideKdFindCell(worldPos);
}

fn ppgGuideBuildCDF(leafIdx: u32, cdf: ptr<function, array<f32, 16>>) -> f32 {
  var total = 0.0;
  for (var b = 0u; b < 16u; b++) {
    let slot = ppgGuideLeafSlot(leafIdx, b, 0u);
    let raw = atomicLoad(&ppgGuideLeafData[slot]);
    let rad = f32(raw) / PPG_GUIDE_RADIANCE_SCALE;
    total += rad;
    (*cdf)[b] = total;
  }
  return total;
}

fn ppgGuideLeafIsEmpty(leafIdx: u32) -> bool {
  for (var b = 0u; b < 16u; b++) {
    let slot = ppgGuideLeafSlot(leafIdx, b, 1u);
    let cnt = atomicLoad(&ppgGuideLeafData[slot]);
    if (cnt > 0u) { return false; }
  }
  return true;
}

fn ppgGuideSampleDirection(worldPos: vec3f, n: vec3f, u1: f32, u2: f32,
                          rng: ptr<function, u32>) -> vec3f {
  let cellIdx = ppgGuideFindCellIndex(worldPos);
  let leafIdx = ppgGuideCells[cellIdx].leafIndex;
  if (ppgGuideLeafIsEmpty(leafIdx)) {
    return sampleCosineHemisphere(n, rng);
  }
  var cdf: array<f32, 16>;
  let total = ppgGuideBuildCDF(leafIdx, &cdf);
  if (total <= 0.0) {
    return sampleCosineHemisphere(n, rng);
  }
  let target = u2 * total;
  var selectedBin = 15u;
  for (var b = 0u; b < 16u; b++) {
    if (cdf[b] >= target) {
      selectedBin = b;
      break;
    }
  }
  let row   = f32(selectedBin / 4u);
  let col   = f32(selectedBin % 4u);
  let jitterU = (col + fract(u1 * 4.0)) / 4.0;
  let jitterV = (row + fract(u1 * 16.0)) / 4.0;
  let octUV   = vec2f(clamp(jitterU, 0.0, 1.0), clamp(jitterV, 0.0, 1.0));
  let localDir = ppgGuideOctahedralToDir(octUV);
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  return normalize(localDir.x * T + localDir.y * B + localDir.z * n);
}

fn ppgGuidePDF(worldPos: vec3f, n: vec3f, dir: vec3f) -> f32 {
  let cellIdx = ppgGuideFindCellIndex(worldPos);
  let leafIdx = ppgGuideCells[cellIdx].leafIndex;
  if (ppgGuideLeafIsEmpty(leafIdx)) {
    return cosineHemispherePdf(n, dir);
  }
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  let localDir = vec3f(dot(dir, T), dot(dir, B), dot(dir, n));
  let lenL1 = abs(localDir.x) + abs(localDir.y) + abs(localDir.z);
  var oct: vec2f;
  if (lenL1 > 0.0) {
    oct = localDir.xy / lenL1;
  } else {
    oct = vec2f(0.0);
  }
  if (localDir.z < 0.0) {
    let tmp = oct;
    oct.x = (1.0 - abs(tmp.y)) * select(-1.0, 1.0, tmp.x >= 0.0);
    oct.y = (1.0 - abs(tmp.x)) * select(-1.0, 1.0, tmp.y >= 0.0);
  }
  oct = oct * 0.5 + vec2f(0.5);
  oct = clamp(oct, vec2f(0.0), vec2f(1.0));
  let col = u32(oct.x * 4.0);
  let row = u32(oct.y * 4.0);
  let binIdx = clamp(row * 4u + col, 0u, 15u);
  var cdf: array<f32, 16>;
  let total = ppgGuideBuildCDF(leafIdx, &cdf);
  if (total <= 0.0) {
    return cosineHemispherePdf(n, dir);
  }
  let radSlot = ppgGuideLeafSlot(leafIdx, binIdx, 0u);
  let binRadiance = f32(atomicLoad(&ppgGuideLeafData[radSlot])) / PPG_GUIDE_RADIANCE_SCALE;
  let binProb     = binRadiance / total;
  let binSolidAngle = 2.0 * 3.14159265358979 / 16.0;
  return binProb / binSolidAngle;
}
`;

const TRAIN_HEAD_ANCHOR =
  '@group(3) @binding(5) var<storage, read_write> ppgTrainHead: array<atomic<u32>>;';

const COMBINED_ANCHOR = `  let combined = Lo_emit + Lo_direct + Lo_sunCaustic
               + Lo_skyAperture * 0.08
               + Lo_ddgi * DDGI_DIFFUSE_BLEND;`;

const COMBINED_REPLACEMENT = `  var Lo_ppgBounce = vec3f(0.0);
  if (!isGlass && !isMetal && ppgShadeMeta.cellCount > 0u) {
    let uPpgA = rand_f32(&rng);
    let uPpgB = rand_f32(&rng);
    let wiPpg = ppgGuideSampleDirection(pos, normal, uPpgA, uPpgB, &rng);
    let pdfPpg = ppgGuidePDF(pos, normal, wiPpg);
    let rayPpg = Ray(pos + normal * 1e-3, wiPpg);
    let hitPpg = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, rayPpg);
    if (!hitPpg.didHit) {
      let nDotWi = max(0.0, dot(normal, wiPpg));
      let LiSky = ubo.skyTint * ubo.skyIrradiance;
      Lo_ppgBounce = LiSky * albedo * INV_PI * nDotWi / max(pdfPpg, 1e-6);
    }
  }
  let combined = Lo_emit + Lo_direct + Lo_sunCaustic
               + Lo_skyAperture * 0.08
               + Lo_ddgi * DDGI_DIFFUSE_BLEND
               + Lo_ppgBounce * PPG_GUIDE_INDIRECT_BLEND;`;

export function injectPpgGuideDeclsIntoShadeWgsl(shadeWgsl: string): string {
  if (!shadeWgsl.includes(TRAIN_HEAD_ANCHOR)) {
    throw new Error('[shade PPG guide] train head anchor not found — inject train bindings first');
  }
  return shadeWgsl.replace(TRAIN_HEAD_ANCHOR, `${TRAIN_HEAD_ANCHOR}\n${SHADE_PPG_GUIDE_WGSL}`);
}

export function injectPpgGuideBounceIntoShadeWgsl(shadeWgsl: string): string {
  if (!shadeWgsl.includes(COMBINED_ANCHOR)) {
    throw new Error('[shade PPG guide] combined radiance anchor not found');
  }
  return shadeWgsl.replace(COMBINED_ANCHOR, COMBINED_REPLACEMENT);
}
