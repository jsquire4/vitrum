/**
 * PPG update shader — Sprint 11.
 *
 * Compute shader for updating PPG directional bin statistics from completed-path samples.
 * Reads from a per-frame sample buffer (written by the shade pass) and
 * accumulates radiance into the appropriate cell's directional bin.
 *
 * Ping-pong convention:
 *   - Even frames (frameParity == 0): read ppgSampleBuffer → write ppgLeaves.
 *   - Odd  frames (frameParity == 1): skip update (read-only frame).
 * This halves the update bandwidth and mirrors the checkerboard pattern
 * used by the resolve pass (Sprint 9).
 *
 * Per-frame flow:
 *   1. Each thread handles one path-completion sample from ppgSampleBuffer.
 *   2. Find the nearest PPG spatial cell to the sample's world position
 *      via kd-tree on `ppgKdNodes` (same traversal as `ppgSample.wgsl.ts`);
 *      falls back to linear scan when the kd buffer holds the disabled sentinel.
 *   3. Encode the sample's incident direction to an octahedral bin index.
 *   4. Atomic-add the sample's radiance into the cell's bin.
 *   5. Atomic-increment the bin's sampleCount.
 *
 * Atomic update strategy:
 *   WebGPU does NOT support atomic f32 add on storage buffers (only
 *   atomicAdd on u32/i32). We use a fixed-point representation:
 *     stored_value = round(radianceSum * PPG_RADIANCE_SCALE)
 *   where PPG_RADIANCE_SCALE = 65536.0 (16-bit fixed point headroom).
 *   The leaf buffer uses array<atomic<u32>> for radianceSum and sampleCount
 *   columns. The host-side type (PPGDirectionalBin) stores f32 for JS use
 *   only; the GPU layout is all-u32 for atomic compatibility.
 *
 * References:
 *   - Müller et al. 2017, "Practical Path Guiding for Efficient Light-Transport
 *     Simulation", Computer Graphics Forum 36(4).
 *   - WebGPU atomic spec: https://gpuweb.github.io/gpuweb/wgsl/#atomic-types
 *
 * @since Sprint 11, 2026-05-09
 */

export const PPG_UPDATE_WGSL = /* wgsl */`

// ============================================================
// Constants
// ============================================================

// Fixed-point scale for atomic radiance accumulation.
// radianceSum stored as round(value * PPG_RADIANCE_SCALE) in a u32.
// Max representable radiance: 2^32 / 65536 = 65536 — sufficient for
// HDR scene values in the 0–1000 nit range.
const PPG_RADIANCE_SCALE: f32 = 65536.0;

// ============================================================
// PPG structs (GPU-side, atomic layout)
// ============================================================

// PPGSpatialCell — 32 bytes, matches ppgSample.wgsl layout exactly.
struct PPGSpatialCell {
  position:  vec3f,
  _pad:      f32,
  leafIndex: u32,
  _pad2x:    u32,
  _pad2y:    u32,
  _pad2z:    u32,
};

// 16-byte kd-tree node (matches buildPpgKdTree.ts / ppgSample.wgsl.ts).
struct PPGKdNode {
  child0: u32,
  child1: u32,
  meta:   u32,
  split:  f32,
};

// PPGPathSample — one completed-path record written by the shade pass.
// Layout (48 bytes):
//   bytes  0-11: worldPos xyz (vec3f)
//   bytes 12-15: _pad (f32)
//   bytes 16-27: incidentDir xyz (vec3f) — direction of incoming radiance
//   bytes 28-31: _pad2 (f32)
//   bytes 32-43: radiance rgb (vec3f) — luminance only used; stored as vec3f
//                for future spectral extension
//   bytes 44-47: _pad3 (f32)
struct PPGPathSample {
  worldPos:    vec3f,
  _pad:        f32,
  incidentDir: vec3f,
  _pad2:       f32,
  radiance:    vec3f,
  _pad3:       f32,
};

// ============================================================
// PPG uniforms
// ============================================================

struct PPGUpdateUniforms {
  sampleCount:  u32,  // number of samples in ppgSampleBuffer this frame
  frameParity:  u32,  // 0 = even frame (update); 1 = odd frame (skip)
  cellCount:    u32,  // active cells in ppgCells
  _pad:         u32,
};

// ============================================================
// Bind group (deferred — group number assigned by pipelineCompiler)
// Integration note: the bind group index below is a placeholder.
// When Sprint 11 integration wires ppgUpdate into the frame dispatch,
// pipelineCompiler.ts assigns the actual group index.
// ============================================================

@group(0) @binding(0) var<uniform>         u_ppg:          PPGUpdateUniforms;
@group(0) @binding(1) var<storage, read>   ppgSamples:     array<PPGPathSample>;
@group(0) @binding(2) var<storage, read>   ppgCells:       array<PPGSpatialCell>;
// ppgLeafRadiance: atomic u32 per (leafIdx, binIdx) — radianceSum in fixed point.
// Layout: leafIdx * PPG_DIRECTIONS * 2 + binIdx * 2 + 0 = radianceSum slot
//          leafIdx * PPG_DIRECTIONS * 2 + binIdx * 2 + 1 = sampleCount slot
// Each leaf occupies PPG_DIRECTIONS * 2 u32 slots = 16 * 2 * 4 bytes = 128 bytes.
// This matches the PPG_LEAF_BYTE_STRIDE (256 byte allocation; 128 bytes used).
@group(0) @binding(3) var<storage, read_write> ppgLeafData: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read>   ppgKdNodes:     array<PPGKdNode>;

// ============================================================
// Internal helpers
// ============================================================

fn ppgUpdateAxisComp(v: vec3f, axis: u32) -> f32 {
  if (axis == 0u) { return v.x; }
  if (axis == 1u) { return v.y; }
  return v.z;
}

// Brute nearest cell over active indices [0 .. cellCount-1] (strict less-than on distance²).
fn ppgUpdateFindCellBrute(worldPos: vec3f, cellCount: u32) -> u32 {
  if (cellCount == 0u) { return 0u; }

  var bestIdx  = 0u;
  var bestDist2 = 1e38;

  for (var i = 0u; i < cellCount; i++) {
    let d = ppgCells[i].position - worldPos;
    let dist2 = dot(d, d);
    if (dist2 < bestDist2) {
      bestDist2 = dist2;
      bestIdx  = i;
    }
  }
  return bestIdx;
}

// kd-tree NN (iterative); matches ppgSample.wgsl.ts ppgKdFindCell but uses
// uniform cellCount instead of arrayLength(&ppgCells) for valid leaf indices.
fn ppgUpdateKdFindCell(worldPos: vec3f, cellCount: u32) -> u32 {
  let nk = arrayLength(&ppgKdNodes);
  if (nk == 0u || cellCount == 0u) { return 0u; }
  let root = ppgKdNodes[0];
  if (root.child0 == 0xFFFFFFFFu && root.child1 == 0xFFFFFFFFu) {
    return ppgUpdateFindCellBrute(worldPos, cellCount);
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
    let node = ppgKdNodes[nid];
    let meta = node.meta;
    if ((meta & 0x80000000u) != 0u) {
      let cellIdx = meta & 0x7FFFFFFFu;
      if (cellIdx < cellCount) {
        let d = ppgCells[cellIdx].position - worldPos;
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
    let d0 = ppgUpdateAxisComp(worldPos, axis) - split;
    let d2plane = d0 * d0;
    let nearI = select(c1, c0, d0 < 0.0);
    let farI = select(c0, c1, d0 < 0.0);

    if (sp + 2u > 48u) {
      return ppgUpdateFindCellBrute(worldPos, cellCount);
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

fn ppgUpdateFindCell(worldPos: vec3f, cellCount: u32) -> u32 {
  return ppgUpdateKdFindCell(worldPos, cellCount);
}

// Encode a world-space incident direction to a bin index [0..15].
// Directions must be in the upper hemisphere (z >= 0 in local frame).
// We encode globally (no per-cell normal frame): bins cover the full
// unit sphere mapped to octahedral [0,1]², then quantised to 4×4.
fn ppgDirToBinIdx(dir: vec3f) -> u32 {
  // Encode to octahedral [−1,1]²
  let lenL1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
  var oct: vec2f;
  if (lenL1 > 1e-8) {
    oct = dir.xy / lenL1;
    // Fold negative hemisphere
    if (dir.z < 0.0) {
      let tmp = oct;
      oct.x = (1.0 - abs(tmp.y)) * select(-1.0, 1.0, tmp.x >= 0.0);
      oct.y = (1.0 - abs(tmp.x)) * select(-1.0, 1.0, tmp.y >= 0.0);
    }
  }
  // Map [−1,1]² → [0,1]²
  oct = oct * 0.5 + vec2f(0.5);
  oct = clamp(oct, vec2f(0.0), vec2f(0.9999));

  let col = u32(oct.x * 4.0);
  let row = u32(oct.y * 4.0);
  return clamp(row * 4u + col, 0u, 15u);
}

// Compute flat index into ppgLeafData for (leafIdx, binIdx, field).
// field: 0 = radianceSum, 1 = sampleCount.
//
// Stride derivation (matches ppgSample.wgsl PPGDirectionalLeaf layout):
//   PPGDirectionalLeaf = 256 bytes = 64 u32 slots per leaf.
//   PPG_LEAF_BYTE_STRIDE (types.ts:129) = 256 bytes = 64 × sizeof(u32).
//   Each bin occupies 2 u32 slots (radianceSum + sampleCount).
//   → leafIdx * 64u + binIdx * 2u + field
//
// AUDIT FIX H-1 (2026-05-09): was "leafIdx * 32u", which addressed 128-byte
// offsets while ppgSample.wgsl expects 256-byte (64-slot) offsets. For any
// scene with more than one occupied PPG cell the old formula corrupted
// directional bin data in all cells beyond cell 0.
fn ppgLeafSlot(leafIdx: u32, binIdx: u32, field: u32) -> u32 {
  return leafIdx * 64u + binIdx * 2u + field;
}

// ============================================================
// Compute kernel
// ============================================================

@compute @workgroup_size(64, 1, 1)
fn ppgUpdateKernel(@builtin(global_invocation_id) id: vec3<u32>) {
  let sampleIdx = id.x;

  // Ping-pong: skip update on odd frames.
  if (u_ppg.frameParity != 0u) { return; }

  // Bounds check.
  if (sampleIdx >= u_ppg.sampleCount) { return; }

  let s = ppgSamples[sampleIdx];

  // Compute luminance of this sample's radiance (scalar contribution).
  let lum = dot(s.radiance, vec3f(0.2126, 0.7152, 0.0722));
  if (lum <= 0.0) { return; }

  // Find nearest cell.
  let cellIdx = ppgUpdateFindCell(s.worldPos, u_ppg.cellCount);
  let leafIdx = ppgCells[cellIdx].leafIndex;

  // Encode direction to bin.
  let binIdx = ppgDirToBinIdx(normalize(s.incidentDir));

  // Atomic accumulate in fixed-point.
  // Max representable radiance = 2^32 / PPG_RADIANCE_SCALE = 4294967295 / 65536 ≈ 65536 nits.
  // AUDIT FIX H-2 (2026-05-09): was f32(0xFFFFFFu) = 16,777,215 → max ≈ 256 nits, saturating
  // HDR emitters (sun-through-glass can exceed 1000 nits). Corrected to full u32 range.
  let lumFixed = u32(clamp(lum * PPG_RADIANCE_SCALE, 0.0, f32(0xFFFFFFFFu)));
  let radianceSlot = ppgLeafSlot(leafIdx, binIdx, 0u);
  let countSlot    = ppgLeafSlot(leafIdx, binIdx, 1u);

  atomicAdd(&ppgLeafData[radianceSlot], lumFixed);
  atomicAdd(&ppgLeafData[countSlot],    1u);
}

`;
