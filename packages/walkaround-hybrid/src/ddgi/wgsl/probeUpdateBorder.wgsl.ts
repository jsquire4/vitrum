/**
 * DDGI Probe Update — Pass 3: Octahedral Atlas Border Fill.
 *
 * Two WGSL modules (one for irradiance, one for visibility) that populate
 * the 1-pixel border ring around each probe's octahedral cell.
 *
 * Background: the atlas allocates (CELL + BORDER) × (CELL + BORDER) pixels
 * per probe, where BORDER=2 (1 pixel on each side). The blend pass writes
 * only the interior CELL × CELL pixels. The border pixels are zero-initialized
 * and, without this pass, bilinear sampling at octahedral seam coordinates
 * reads a zero border → systematic darkening at every probe-cell edge.
 *
 * Fix (Majercik 2019 §3.2, Cigolle 2014 §A.1): for each border texel,
 * copy the octahedral-mirrored interior texel that represents the same
 * direction on the octahedral map. The "fold" convention is:
 *
 *   Cell occupies (CELL+2) × (CELL+2) pixels; interior at (1..CELL, 1..CELL).
 *   Local coordinates (lx, ly) in [0, CELL+1].
 *   N = CELL (interior dimension).
 *
 *   Top edge    (ly=0,   1 ≤ lx ≤ N): copy from interior (N+1-lx, 2)
 *   Bottom edge (ly=N+1, 1 ≤ lx ≤ N): copy from interior (N+1-lx, N-1)
 *   Left edge   (lx=0,   1 ≤ ly ≤ N): copy from interior (2,       N+1-ly)
 *   Right edge  (lx=N+1, 1 ≤ ly ≤ N): copy from interior (N-1,     N+1-ly)
 *   Corner (0,0):       copy from interior (N,   N)
 *   Corner (N+1,0):     copy from interior (1,   N)
 *   Corner (0,N+1):     copy from interior (N,   1)
 *   Corner (N+1,N+1):   copy from interior (1,   1)
 *
 * WebGPU restricts simultaneous texture_2d (read) + texture_storage_2d
 * (write) bindings to the SAME texture within a single pipeline pass.
 * This pass uses a ping-pong copy: the host does
 *   copyTextureToTexture(writeAtlas → scratchAtlas)
 * so the border pass reads from `scratchAtlas` and writes to `writeAtlas`.
 * The scratch texture holds the blend output just before border fill.
 *
 * References:
 *   Majercik et al. 2019, "Dynamic Diffuse Global Illumination with Ray-Traced
 *   Irradiance Fields", JCGT §3.2 (atlas border update convention).
 *   Cigolle et al. 2014, "A Survey of Efficient Representations for Independent
 *   Unit Vectors", JCGT §A.1 (octahedral seam border mirror derivation).
 */

// Irradiance border pass:
//   @group(0) @binding(0) — atlasRead  (scratch copy of blend output, rgba16float)
//   @group(0) @binding(1) — atlasWrite (blend output to be border-filled, rgba16float)
//   @group(0) @binding(2) — ubo (BorderUBO)
//
// Dispatch: (probeCount, 1, 1) workgroups of (BORDER_WG_SIZE, 1, 1).
// Each workgroup handles one probe cell.
// BORDER_WG_SIZE = 4*(CELL+2) covers every border texel in one thread stripe.
// Interior threads are discarded cheaply by the border-check below.

export const PROBE_UPDATE_BORDER_IRR_WGSL = /* wgsl */`

const IRR_CELL:   u32 = 8u;
const IRR_STRIDE: u32 = 10u;   // CELL + BORDER (BORDER=2)

// N = IRR_CELL (interior dimension, used in mirror formulas).
// Border workgroup size: 4*(CELL+2) = 4*10 = 40 threads per probe.
// We only need to cover (CELL+2) border positions on each of 4 edges,
// minus 4 corners handled separately → dispatch at most (CELL+2)*4 per probe.
// We over-allocate to a round number and discard out-of-range threads.
const BORDER_THREADS: u32 = 48u;  // ≥ 4*(IRR_STRIDE) = 40; next ^2 bound

struct BorderUBO {
  numProbes:   u32,
  atlasWidth:  u32,
  atlasHeight: u32,
  _pad0:       u32,
  // probe grid dims (x, y, z) packed as 3 u32.
  gridDimX:    u32,
  gridDimY:    u32,
  gridDimZ:    u32,
  _pad1:       u32,
};

@group(0) @binding(0) var atlasRead:  texture_2d<f32>;
@group(0) @binding(1) var atlasWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform>   ubo: BorderUBO;

// Compute the atlas origin of probe probeIdx's cell (top-left corner
// including the 1-pixel border, i.e. the cell is at atlasOrigin..(+STRIDE)).
fn irrCellOrigin(probeIdx: u32) -> vec2u {
  let px  = probeIdx % ubo.gridDimX;
  let tmp = probeIdx / ubo.gridDimX;
  let py  = tmp % ubo.gridDimY;
  let pz  = tmp / ubo.gridDimY;
  return vec2u(
    px * IRR_STRIDE,
    (py + pz * ubo.gridDimY) * IRR_STRIDE,
  );
}

// Mirror a border local coordinate (lx, ly) in [0, IRR_STRIDE) to the
// interior source coordinate (mx, my) in [0, IRR_STRIDE) that holds the
// same octahedral direction reflected across the seam.
// Interior pixels occupy local coords (1 ≤ x ≤ N, 1 ≤ y ≤ N) where N=IRR_CELL.
// Returns interior sentinel (same coord) if (lx, ly) is not a border pixel.
fn irrMirror(lx: u32, ly: u32) -> vec2u {
  let N = IRR_CELL;  // = 8
  // Corners
  if (lx == 0u && ly == 0u)            { return vec2u(N,   N);   }
  if (lx == N + 1u && ly == 0u)        { return vec2u(1u,  N);   }
  if (lx == 0u && ly == N + 1u)        { return vec2u(N,   1u);  }
  if (lx == N + 1u && ly == N + 1u)    { return vec2u(1u,  1u);  }
  // Edges
  if (ly == 0u)                         { return vec2u(N + 1u - lx, 2u);         }
  if (ly == N + 1u)                     { return vec2u(N + 1u - lx, N - 1u);     }
  if (lx == 0u)                         { return vec2u(2u,           N + 1u - ly); }
  if (lx == N + 1u)                     { return vec2u(N - 1u,       N + 1u - ly); }
  // Interior — caller must discard this thread.
  return vec2u(lx, ly);  // sentinel: same coord → interior (unused)
}

fn irrIsBorder(lx: u32, ly: u32) -> bool {
  let N = IRR_CELL;
  return (lx == 0u || lx == N + 1u || ly == 0u || ly == N + 1u);
}

@compute @workgroup_size(48, 1, 1)
fn probeUpdateBorderIrradiance(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(workgroup_id)         wid: vec3u,
  @builtin(local_invocation_id)  lid: vec3u,
) {
  let probeIdx = wid.x;
  if (probeIdx >= ubo.numProbes) { return; }

  // Map thread linear index → (lx, ly) in [0, IRR_STRIDE) × [0, IRR_STRIDE).
  // IRR_STRIDE = 10 = CELL+2. We enumerate 10*10 = 100 positions using
  // BORDER_THREADS=48 threads by processing two strips: thread t covers
  // positions t and t+48. Interior threads early-exit.
  let stride = IRR_STRIDE;  // 10
  let t0 = lid.x;
  let t1 = lid.x + 48u;

  let origin = irrCellOrigin(probeIdx);

  for (var pass = 0u; pass < 2u; pass = pass + 1u) {
    let t = select(t0, t1, pass == 1u);
    // t ranges [0,48) for pass=0 and [48,96) for pass=1. Positions above
    // stride*stride = 100 are skipped.
    if (t >= stride * stride) { continue; }
    let lx = t % stride;
    let ly = t / stride;
    if (!irrIsBorder(lx, ly)) { continue; }
    let src = irrMirror(lx, ly);
    let srcAtlas = vec2u(origin.x + src.x, origin.y + src.y);
    let dstAtlas = vec2u(origin.x + lx,    origin.y + ly);
    let val = textureLoad(atlasRead, srcAtlas, 0);
    textureStore(atlasWrite, dstAtlas, val);
  }
}
`;

// -----------------------------------------------------------------
// Visibility border pass (VIS_CELL = 16)
// -----------------------------------------------------------------
export const PROBE_UPDATE_BORDER_VIS_WGSL = /* wgsl */`

const VIS_CELL:   u32 = 16u;
const VIS_STRIDE: u32 = 18u;   // CELL + BORDER

// (VIS_STRIDE)² = 18*18 = 324 positions. Dispatch 256 threads per probe;
// each covers 2 positions to reach all 324 (256+68 → 256*2=512 ≥ 324).
const BORDER_THREADS_V: u32 = 256u;

struct BorderUBOV {
  numProbes:   u32,
  atlasWidth:  u32,
  atlasHeight: u32,
  _pad0:       u32,
  gridDimX:    u32,
  gridDimY:    u32,
  gridDimZ:    u32,
  _pad1:       u32,
};

@group(0) @binding(0) var atlasReadV:  texture_2d<f32>;
@group(0) @binding(1) var atlasWriteV: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform>    uboV: BorderUBOV;

fn visCellOrigin(probeIdx: u32) -> vec2u {
  let px  = probeIdx % uboV.gridDimX;
  let tmp = probeIdx / uboV.gridDimX;
  let py  = tmp % uboV.gridDimY;
  let pz  = tmp / uboV.gridDimY;
  return vec2u(
    px * VIS_STRIDE,
    (py + pz * uboV.gridDimY) * VIS_STRIDE,
  );
}

fn visIsBorder(lx: u32, ly: u32) -> bool {
  let N = VIS_CELL;
  return (lx == 0u || lx == N + 1u || ly == 0u || ly == N + 1u);
}

fn visMirror(lx: u32, ly: u32) -> vec2u {
  let N = VIS_CELL;  // = 16
  if (lx == 0u && ly == 0u)            { return vec2u(N,   N);   }
  if (lx == N + 1u && ly == 0u)        { return vec2u(1u,  N);   }
  if (lx == 0u && ly == N + 1u)        { return vec2u(N,   1u);  }
  if (lx == N + 1u && ly == N + 1u)    { return vec2u(1u,  1u);  }
  if (ly == 0u)                         { return vec2u(N + 1u - lx, 2u);         }
  if (ly == N + 1u)                     { return vec2u(N + 1u - lx, N - 1u);     }
  if (lx == 0u)                         { return vec2u(2u,           N + 1u - ly); }
  if (lx == N + 1u)                     { return vec2u(N - 1u,       N + 1u - ly); }
  return vec2u(lx, ly);
}

@compute @workgroup_size(256, 1, 1)
fn probeUpdateBorderVisibility(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let probeIdx = wid.x;
  if (probeIdx >= uboV.numProbes) { return; }

  let stride = VIS_STRIDE;  // 18
  let total = stride * stride;  // 324
  let origin = visCellOrigin(probeIdx);

  for (var pass = 0u; pass < 2u; pass = pass + 1u) {
    let t = lid.x + pass * 256u;
    if (t >= total) { continue; }
    let lx = t % stride;
    let ly = t / stride;
    if (!visIsBorder(lx, ly)) { continue; }
    let src = visMirror(lx, ly);
    let srcAtlas = vec2u(origin.x + src.x, origin.y + src.y);
    let dstAtlas = vec2u(origin.x + lx,    origin.y + ly);
    let val = textureLoad(atlasReadV, srcAtlas, 0);
    textureStore(atlasWriteV, dstAtlas, val);
  }
}
`;
