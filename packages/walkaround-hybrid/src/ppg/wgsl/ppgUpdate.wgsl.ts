/**
 * PPG update shader — Sprint 11.
 *
 * Compute shader for updating the PPG kd-tree with completed-path samples.
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
 *      (linear scan — consistent with ppgSample.wgsl.ts's brute-force
 *      approach; optimisation deferred to post-Sprint-11).
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

// ============================================================
// Internal helpers
// ============================================================

// Find the nearest PPG spatial cell to worldPos (brute-force linear scan).
// Returns the cell index, or 0 if cellCount == 0.
fn ppgUpdateFindCell(worldPos: vec3f, cellCount: u32) -> u32 {
  if (cellCount == 0u) { return 0u; }

  var bestIdx  = 0u;
  var bestDist = 1e20;

  for (var i = 0u; i < cellCount; i++) {
    let d = ppgCells[i].position - worldPos;
    let dist2 = dot(d, d);
    if (dist2 < bestDist) {
      bestDist = dist2;
      bestIdx  = i;
    }
  }
  return bestIdx;
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
fn ppgLeafSlot(leafIdx: u32, binIdx: u32, field: u32) -> u32 {
  return leafIdx * 32u + binIdx * 2u + field;
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
  let lumFixed = u32(clamp(lum * PPG_RADIANCE_SCALE, 0.0, f32(0xFFFFFFu)));
  let radianceSlot = ppgLeafSlot(leafIdx, binIdx, 0u);
  let countSlot    = ppgLeafSlot(leafIdx, binIdx, 1u);

  atomicAdd(&ppgLeafData[radianceSlot], lumFixed);
  atomicAdd(&ppgLeafData[countSlot],    1u);
}

`;
