/**
 * DDGI Probe Update — Pass 3: Octahedral Atlas Border Fill.
 *
 * Two WGSL modules (one for irradiance, one for visibility) that populate
 * the 1-pixel border ring around each probe's octahedral cell. Both are
 * emitted from a single parameterized factory `makeBorderFillWGSL` — the
 * passes differ ONLY by cell size (8 vs 16), stride (10 vs 18), workgroup
 * size (48 vs 256), and entry-point name. The cell + stride are sourced
 * from `ddgiAtlasLayout.ts` (the single source of truth shared with the
 * producer and samplers), so the border pass cannot silently drift from
 * the atlas layout.
 *
 * For backward compatibility the previous string exports
 * (`PROBE_UPDATE_BORDER_IRR_WGSL` / `PROBE_UPDATE_BORDER_VIS_WGSL`) remain
 * available as consts computed from the factory at module-load time.
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
 * Dispatch: (probeCount, 1, 1) workgroups of (workgroupSize, 1, 1).
 * Each workgroup handles one probe cell. A workgroup of `workgroupSize`
 * threads covers ALL (CELL+2)² = stride² local positions in
 * ⌈stride²/workgroupSize⌉ strips — thread `t` handles positions
 * `lid.x + passIdx*workgroupSize` for `passIdx` in `[0, ⌈stride²/workgroupSize⌉)`.
 * The strip count is derived from stride²/workgroupSize at emit time so the
 * pass ALWAYS covers every border texel regardless of workgroupSize (the
 * previous fixed 2-strip loop left the last `stride² - 2*workgroupSize`
 * positions unwritten — e.g. for irradiance, stride=10, workgroupSize=48:
 * 2*48=96 < 100, leaving the four bottom-edge texels lx∈{6,7,8,9}, ly=9
 * with a zero border → seam darkening at the bottom octahedral edge).
 * Interior threads early-exit cheaply via the border check.
 *
 *   @group(0) @binding(0) — atlasRead  (scratch copy of blend output, rgba16float)
 *   @group(0) @binding(1) — atlasWrite (blend output to be border-filled, rgba16float)
 *   @group(0) @binding(2) — ubo (BorderUBO)
 *
 * References:
 *   Majercik et al. 2019, "Dynamic Diffuse Global Illumination with Ray-Traced
 *   Irradiance Fields", JCGT §3.2 (atlas border update convention).
 *   Cigolle et al. 2014, "A Survey of Efficient Representations for Independent
 *   Unit Vectors", JCGT §A.1 (octahedral seam border mirror derivation).
 */

import { IRR_CELL, VIS_CELL, IRR_STRIDE, VIS_STRIDE } from '../ddgiAtlasLayout.js';

/** Parameters that distinguish the irradiance vs visibility border pass. */
export interface BorderFillParams {
  /** Interior octahedral cell dimension (N). IRR_CELL=8 / VIS_CELL=16. */
  cell: number;
  /** In-atlas stride = cell + BORDER. IRR_STRIDE=10 / VIS_STRIDE=18. */
  stride: number;
  /**
   * Compute workgroup size (threads per probe cell). Any positive value is
   * correct: the emitted loop runs `⌈stride²/workgroupSize⌉` strips so every
   * local position is visited regardless of how this divides stride². IRR=48
   * (⌈100/48⌉=3 strips), VIS=256 (⌈324/256⌉=2 strips).
   */
  workgroupSize: number;
  /** WGSL entry-point name for the compute pass. */
  entryPoint: string;
}

/**
 * Build a DDGI octahedral atlas border-fill WGSL module. Emitted twice —
 * once per atlas (irradiance / visibility) — with cell + stride sourced
 * from {@link ddgiAtlasLayout}.
 */
export function makeBorderFillWGSL(params: BorderFillParams): string {
  const { cell, stride, workgroupSize, entryPoint } = params;
  // Number of workgroup-sized strips needed to visit all stride² local
  // positions. Derived (not hardcoded) so the loop covers every border texel
  // for any workgroupSize — see the four-texel coverage bug in the module
  // docblock. ⌈100/48⌉=3 (irradiance), ⌈324/256⌉=2 (visibility).
  const stripCount = Math.ceil((stride * stride) / workgroupSize);
  return /* wgsl */`

const CELL:   u32 = ${cell}u;
const STRIDE: u32 = ${stride}u;   // CELL + BORDER (BORDER=2)

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
fn cellOrigin(probeIdx: u32) -> vec2u {
  let px  = probeIdx % ubo.gridDimX;
  let tmp = probeIdx / ubo.gridDimX;
  let py  = tmp % ubo.gridDimY;
  let pz  = tmp / ubo.gridDimY;
  return vec2u(
    px * STRIDE,
    (py + pz * ubo.gridDimY) * STRIDE,
  );
}

// Mirror a border local coordinate (lx, ly) in [0, STRIDE) to the interior
// source coordinate (mx, my) in [0, STRIDE) that holds the same octahedral
// direction reflected across the seam.
// Interior pixels occupy local coords (1 ≤ x ≤ N, 1 ≤ y ≤ N) where N=CELL.
// Returns interior sentinel (same coord) if (lx, ly) is not a border pixel.
fn mirror(lx: u32, ly: u32) -> vec2u {
  let N = CELL;
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

fn isBorder(lx: u32, ly: u32) -> bool {
  let N = CELL;
  return (lx == 0u || lx == N + 1u || ly == 0u || ly == N + 1u);
}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn ${entryPoint}(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let probeIdx = wid.x;
  if (probeIdx >= ubo.numProbes) { return; }

  // Enumerate ALL STRIDE*STRIDE local positions using ${workgroupSize}
  // threads in ${stripCount} strips: thread t covers position
  // lid.x + passIdx*${workgroupSize}. The strip count ⌈STRIDE²/${workgroupSize}⌉
  // guarantees full coverage; positions past stride*stride are skipped and
  // interior threads early-exit.
  let total = STRIDE * STRIDE;
  let origin = cellOrigin(probeIdx);

  // 'pass' is a WGSL reserved word; the loop counter is named 'passIdx'.
  for (var passIdx = 0u; passIdx < ${stripCount}u; passIdx = passIdx + 1u) {
    let t = lid.x + passIdx * ${workgroupSize}u;
    if (t >= total) { continue; }
    let lx = t % STRIDE;
    let ly = t / STRIDE;
    if (!isBorder(lx, ly)) { continue; }
    let src = mirror(lx, ly);
    let srcAtlas = vec2u(origin.x + src.x, origin.y + src.y);
    let dstAtlas = vec2u(origin.x + lx,    origin.y + ly);
    let val = textureLoad(atlasRead, srcAtlas, 0);
    textureStore(atlasWrite, dstAtlas, val);
  }
}
`;
}

// -----------------------------------------------------------------
// Irradiance border pass (IRR_CELL = 8, stride 10). Workgroup 48: the
// derived ⌈100/48⌉ = 3-strip loop covers local positions
// [0,48) ∪ [48,96) ∪ [96,144) ⊇ [0,100), so ALL 100 cell positions (every
// border texel) are written. The historical hand-written shader used a fixed
// 2-strip loop ([0,96)) which left the four bottom-edge texels lx∈{6,7,8,9},
// ly=9 unfilled → a zero border and seam darkening at the bottom octahedral
// edge; the strip count is now derived from stride²/workgroupSize so the
// pass cannot under-cover. The visibility pass (stride 18, 324 positions)
// uses 256 threads: ⌈324/256⌉ = 2 strips, covering [0,256) ∪ [256,512) ⊇
// [0,324) (behavior unchanged).
// -----------------------------------------------------------------
/** Build the irradiance-atlas border-fill WGSL (IRR_CELL/IRR_STRIDE). */
export function makeProbeUpdateBorderIrrWGSL(): string {
  return makeBorderFillWGSL({
    cell: IRR_CELL,
    stride: IRR_STRIDE,
    workgroupSize: 48,
    entryPoint: 'probeUpdateBorderIrradiance',
  });
}

/** Build the visibility-atlas border-fill WGSL (VIS_CELL/VIS_STRIDE). */
export function makeProbeUpdateBorderVisWGSL(): string {
  return makeBorderFillWGSL({
    cell: VIS_CELL,
    stride: VIS_STRIDE,
    workgroupSize: 256,
    entryPoint: 'probeUpdateBorderVisibility',
  });
}

/**
 * @deprecated Prefer {@link makeProbeUpdateBorderIrrWGSL}. Retained as a
 * module-load const so any external consumer importing the old name keeps
 * working; computed from the factory so it still reflects ddgiAtlasLayout.
 */
export const PROBE_UPDATE_BORDER_IRR_WGSL = makeProbeUpdateBorderIrrWGSL();

/**
 * @deprecated Prefer {@link makeProbeUpdateBorderVisWGSL}. Retained as a
 * module-load const so any external consumer importing the old name keeps
 * working; computed from the factory so it still reflects ddgiAtlasLayout.
 */
export const PROBE_UPDATE_BORDER_VIS_WGSL = makeProbeUpdateBorderVisWGSL();
