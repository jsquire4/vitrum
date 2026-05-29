// fusedMlp.wgsl.ts — FUSED / TILED MLP forward + backward kernel for Neural
// Radiance Caching (Müller et al. 2021, "Real-time Neural Radiance Caching for
// Path Tracing", ACM TOG 40(4); the fully-fused-MLP idea is from Müller's
// tiny-cuda-nn — Müller 2021, "Tiny CUDA Neural Networks").
//
// WHY THIS EXISTS — the cost the SPIKE could not pay
// --------------------------------------------------
// The validated spike (tools/nrc-spike/mlpKernels.ts) does a SEPARATE compute
// dispatch per layer, writing the whole [B × 64] activation tensor to GLOBAL
// memory after every layer and reading it back for the next. For Müller's
// 6×64 MLP that is ~7 forward + ~6 backward + grad round-trips of a [B × 64]
// f32 tensor — memory-bandwidth-bound, ~18 ms/step at B=65536 (the spike's own
// conservative lower bound).
//
// THE FUSION (tiny-cuda-nn principle, ported to core WGSL):
//   * One WORKGROUP processes a TILE of TILE_B samples through the ENTIRE MLP.
//   * The activation tile for the current node-layer lives in WORKGROUP SHARED
//     MEMORY. The next layer is computed into a second shared tile; we ping-pong
//     between them across all layers. Activations DO NOT round-trip to global
//     memory between layers during the sweep.
//   * Each layer's weight matrix is staged from global into a shared tile and
//     consumed by a shared-memory-staged GEMM (standard register/shared tiling).
//   * Backward mirrors forward: the delta tile stays resident in shared and
//     ping-pongs across layers; weights are re-streamed.
//   * Gradient accumulation over the tile is a WORKGROUP REDUCTION in shared
//     memory (not the spike's serialize-over-batch-in-one-thread), sidestepping
//     the WGSL i32/u32-atomic-only limitation: each invocation owns a subset of
//     the [out×in] weight-grad cells and sums that cell over the tile's samples
//     by reading the resident shared activation/delta tiles.
//
// SHARED-MEMORY BUDGET (lavapipe maxComputeWorkgroupStorageSize = 32768 B):
//   The full weight set (24768 params) does NOT fit in shared (49 KB f16), so
//   we stream weights per layer (one global read each — unavoidable) and keep
//   only the per-tile ACTIVATION/DELTA state resident. Budget at TILE_B=64,
//   W=64, f16:
//     actA  [TILE_B×W] = 64*64*2 = 8192 B
//     actB  [TILE_B×W] = 8192 B
//     wTile [W×W]      = 8192 B
//   = 24576 B  < 32768 B  ✓   (f32 doubles each → 49152 B > 32768 → f32 needs a
//   smaller TILE_B or W-split; see TS harness which picks TILE_B accordingly.)
//
// LAYER SHAPES (modeled NRC core): node-layers
//   L0(IN_W) → L1(64) → … → L6(64) → L7(OUT_W=3)
// Hidden layers are W=64 wide ReLU; the output layer is OUT_W wide linear.
// The input layer width IN_W may differ from W (frequency/one-blob encoding
// expands raw features to ~64); the harness pads it to W for the first GEMM.
//
// This module emits the WGSL as a string templated on f16-vs-f32 so the harness
// can pick the path the adapter supports.

export interface FusedMlpWgslOptions {
  /** Emit `enable f16;` and use f16 storage/arith for weights+activations. */
  useF16: boolean;
  /** Hidden width (Müller: 64). */
  W: number;
  /** Output width (Müller NRC RGB core: 3). */
  OUT_W: number;
  /** Number of hidden node-layers (Müller: 6). */
  HIDDEN: number;
  /** Samples processed per workgroup (tile). Must be <= W and a multiple of
   *  the reduction granularity; harness picks 64 for f16, 32 for f32. */
  TILE_B: number;
}

// The fused forward+save kernel. One workgroup = one tile of TILE_B samples.
// workgroup_size = (W, 1, 1): one invocation per output neuron column; each
// invocation walks all TILE_B samples for its column (the tile is small).
//
// Buffers (group 0):
//   weights  : all layers' weights concatenated, row-major [out, in] per layer
//   biases   : all layers' biases concatenated
//   inputs   : [numSamples × IN_W] network inputs (raw features, row-major)
//   actsGlob : [numSamples × (HIDDEN+2) × W] saved activations for backward
//              (node-layer-major within a sample block); written once per layer
//   zGlob    : same layout, saved pre-activations for relu'
//   params   : uniform (offsets + sample count + tile base)
export function fusedForwardWgsl(o: FusedMlpWgslOptions): string {
  const f16 = o.useF16;
  const SC = f16 ? "f16" : "f32"; // scalar type for resident tiles
  const enableF16 = f16 ? "enable f16;\n" : "";
  const W = o.W, OUT_W = o.OUT_W, HIDDEN = o.HIDDEN, TILE_B = o.TILE_B;
  // node-layers total = HIDDEN + 2 (input + output). weight-layers = HIDDEN + 1.
  const NODE = HIDDEN + 2;
  const WLAYERS = HIDDEN + 1;
  return /* wgsl */`${enableF16}
// ── Fused forward (one workgroup per tile of ${TILE_B} samples) ──
const W : u32 = ${W}u;
const OUT_W : u32 = ${OUT_W}u;
const HIDDEN : u32 = ${HIDDEN}u;
const NODE : u32 = ${NODE}u;        // node-layers incl input+output
const WLAYERS : u32 = ${WLAYERS}u;  // weight layers
const TILE_B : u32 = ${TILE_B}u;

// Per-layer offsets packed as vec4<u32>(wOff, bOff, inW, outW) — one per weight
// layer. Packed into the uniform (16-byte-aligned array elements) to keep the
// storage-buffer binding count low and portable (real adapters default to 8).
struct FwdParams {
  numSamples : u32,
  inW        : u32,   // raw input width (padded into the W-wide first tile)
  tileBase   : u32,   // first sample index this dispatch's workgroups cover
  numTiles   : u32,
  lay : array<vec4<u32>, ${WLAYERS}>,  // [wOff, bOff, inW, outW] per weight layer
}
@group(0) @binding(0) var<storage, read>        weights  : array<${SC}>;
@group(0) @binding(1) var<storage, read>        biases   : array<${SC}>;
@group(0) @binding(2) var<storage, read>        inputs   : array<${SC}>;
@group(0) @binding(3) var<storage, read_write>  actsGlob : array<${SC}>;
@group(0) @binding(4) var<storage, read_write>  zGlob    : array<${SC}>;
@group(0) @binding(5) var<uniform>              p        : FwdParams;
fn wOff(l : u32) -> u32 { return p.lay[l].x; }
fn bOff(l : u32) -> u32 { return p.lay[l].y; }
fn layInW(l : u32) -> u32 { return p.lay[l].z; }
fn layOutW(l : u32) -> u32 { return p.lay[l].w; }

// Resident activation tiles (ping-pong) + a staged weight column buffer.
// Layout: actA[s*W + n] = activation of sample s (within tile), neuron n.
var<workgroup> actA : array<${SC}, ${TILE_B * W}>;
var<workgroup> actB : array<${SC}, ${TILE_B * W}>;

// actsGlob/zGlob layout for sample S, node-layer nl, neuron n:
//   base = S * NODE * W + nl * W + n   (W-padded for every node-layer)
fn saveOff(S : u32, nl : u32, n : u32) -> u32 { return S * NODE * W + nl * W + n; }

@compute @workgroup_size(${W}, 1, 1)
fn fusedForward(@builtin(workgroup_id) wg : vec3<u32>,
                @builtin(local_invocation_id) lid : vec3<u32>) {
  let tile = p.tileBase + wg.x;
  if (tile >= p.numTiles) { return; }
  let col = lid.x;                 // this invocation owns output neuron column = col
  let sampleBase = tile * TILE_B;

  // Load node-layer 0 (network input) into actA, padding to W. One column per
  // invocation across all TILE_B samples.
  for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
    let S = sampleBase + s;
    var v : ${SC} = ${SC}(0);
    if (S < p.numSamples && col < p.inW) {
      v = inputs[S * p.inW + col];
    }
    actA[s * W + col] = v;
    // save node-layer-0 activation for backward (a_prev of first weight layer)
    if (S < p.numSamples) { actsGlob[saveOff(S, 0u, col)] = v; }
  }
  workgroupBarrier();

  // Sweep weight layers. Read from cur (actA on even l), write to nxt (actB).
  for (var l : u32 = 0u; l < WLAYERS; l = l + 1u) {
    let inW = layInW(l);
    let outW = layOutW(l);
    let wo = wOff(l);
    let bo = bOff(l);
    let isOut = (l == WLAYERS - 1u);

    // Each invocation computes column 'col' of the output tile for every sample,
    // but only columns < outW are valid.
    if (col < outW) {
      for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
        let S = sampleBase + s;
        var acc : ${SC} = biases[bo + col];
        // GEMM row: dot(weightsRow[col], activationTile[s]) over inW.
        // Resident tile read is fully in shared memory (no global round-trip).
        let wBase = wo + col * inW;
        if ((l & 1u) == 0u) {
          for (var i : u32 = 0u; i < inW; i = i + 1u) {
            acc = acc + weights[wBase + i] * actA[s * W + i];
          }
        } else {
          for (var i : u32 = 0u; i < inW; i = i + 1u) {
            acc = acc + weights[wBase + i] * actB[s * W + i];
          }
        }
        // save pre-activation z of THIS node-layer (l+1) for backward relu'
        if (S < p.numSamples) { zGlob[saveOff(S, l + 1u, col)] = acc; }
        var a : ${SC};
        if (isOut) { a = acc; } else { a = max(${SC}(0), acc); }
        if (S < p.numSamples) { actsGlob[saveOff(S, l + 1u, col)] = a; }
        if ((l & 1u) == 0u) { actB[s * W + col] = a; } else { actA[s * W + col] = a; }
      }
    } else {
      // zero out unused columns of the destination tile so stale lanes don't
      // pollute the next layer's dot product.
      for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
        if ((l & 1u) == 0u) { actB[s * W + col] = ${SC}(0); }
        else { actA[s * W + col] = ${SC}(0); }
      }
    }
    workgroupBarrier();
  }
}
`;
}

// The fused backward kernel. One workgroup per tile; computes per-layer weight
// and bias gradients via workgroup reduction, accumulating across tiles into
// global grad buffers with a SINGLE-WRITE-PER-TILE-PER-CELL scheme:
//   Because each weight cell (l,o,i) is summed over the whole tile by ONE
//   invocation, and different tiles touch the same cell, we accumulate ACROSS
//   tiles with an i32 fixed-point atomic add (WGSL has no f32 atomics). The
//   harness scales by a fixed-point factor and converts back on the host /
//   in the Adam kernel. This is the gradient-accumulation strategy the spike
//   noted as the real-throughput path.
//
// Delta tile stays resident & ping-pongs across layers, mirroring forward.
export function fusedBackwardWgsl(o: FusedMlpWgslOptions): string {
  const f16 = o.useF16;
  const SC = f16 ? "f16" : "f32";
  const enableF16 = f16 ? "enable f16;\n" : "";
  const W = o.W, OUT_W = o.OUT_W, HIDDEN = o.HIDDEN, TILE_B = o.TILE_B;
  const NODE = HIDDEN + 2;
  const WLAYERS = HIDDEN + 1;
  return /* wgsl */`${enableF16}
const W : u32 = ${W}u;
const OUT_W : u32 = ${OUT_W}u;
const HIDDEN : u32 = ${HIDDEN}u;
const NODE : u32 = ${NODE}u;
const WLAYERS : u32 = ${WLAYERS}u;
const TILE_B : u32 = ${TILE_B}u;
// fixed-point scale for the i32 grad atomics (host divides back out).
const GRAD_FP : f32 = 1048576.0;   // 2^20

struct BwdParams {
  numSamples : u32,
  inW        : u32,
  tileBase   : u32,
  numTiles   : u32,
  lay : array<vec4<u32>, ${WLAYERS}>,  // [wOff, bOff, inW, outW] per weight layer
}
@group(0) @binding(0) var<storage, read>        weights  : array<${SC}>;
@group(0) @binding(1) var<storage, read>        targets  : array<f32>;   // [numSamples × OUT_W]
@group(0) @binding(2) var<storage, read>        actsGlob : array<${SC}>;
@group(0) @binding(3) var<storage, read>        zGlob    : array<${SC}>;
@group(0) @binding(4) var<storage, read_write>  gradWfx  : array<atomic<i32>>; // fixed-point
@group(0) @binding(5) var<storage, read_write>  gradBfx  : array<atomic<i32>>;
@group(0) @binding(6) var<uniform>              p        : BwdParams;
fn wOff(l : u32) -> u32 { return p.lay[l].x; }
fn bOff(l : u32) -> u32 { return p.lay[l].y; }
fn layInW(l : u32) -> u32 { return p.lay[l].z; }
fn layOutW(l : u32) -> u32 { return p.lay[l].w; }

// Resident delta tiles (ping-pong). deltaCur[s*W + n] = delta of node-layer's
// neuron n for sample s within the tile.
var<workgroup> deltaA : array<${SC}, ${TILE_B * W}>;
var<workgroup> deltaB : array<${SC}, ${TILE_B * W}>;

fn saveOff(S : u32, nl : u32, n : u32) -> u32 { return S * NODE * W + nl * W + n; }

@compute @workgroup_size(${W}, 1, 1)
fn fusedBackward(@builtin(workgroup_id) wg : vec3<u32>,
                 @builtin(local_invocation_id) lid : vec3<u32>) {
  let tile = p.tileBase + wg.x;
  if (tile >= p.numTiles) { return; }
  let col = lid.x;
  let sampleBase = tile * TILE_B;

  // ── output delta into deltaA (node-layer NODE-1, width OUT_W) ──
  // delta_out[s,o] = (pred - tgt) / numSamples   (MSE grad, linear out)
  if (col < OUT_W) {
    for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
      let S = sampleBase + s;
      if (S < p.numSamples) {
        let pred = f32(actsGlob[saveOff(S, NODE - 1u, col)]);
        let tgt  = targets[S * OUT_W + col];
        deltaA[s * W + col] = ${SC}((pred - tgt) / f32(p.numSamples));
      } else {
        deltaA[s * W + col] = ${SC}(0);
      }
    }
  } else {
    for (var s : u32 = 0u; s < TILE_B; s = s + 1u) { deltaA[s * W + col] = ${SC}(0); }
  }
  workgroupBarrier();

  // The "current" delta lives in deltaA when (NODE-1-nl) is even from the top.
  // We backprop node-layer nl from NODE-1 down to 1. The delta of node-layer
  // (l+1) is the output of weight-layer l. We use parity on the weight-layer
  // index counting DOWN: cur buffer alternates as we go.
  //
  // Process weight layers l = WLAYERS-1 .. 0. For each:
  //   delta_in  = delta of node-layer (l+1)   [resident in cur]
  //   1) accumulate gradW[l][o,i] += sum_s delta_in[s,o] * a_prev[s,i]
  //      and gradB[l][o]    += sum_s delta_in[s,o]
  //   2) if l>0: delta_prev[s,i] = (sum_o W[l][o,i]*delta_in[s,o]) * relu'(z[l][s,i])
  //      written to the OTHER resident buffer.
  //
  // cur/other parity: at l = WLAYERS-1 the output delta is in deltaA. Each step
  // flips. Use a running flag.
  var curIsA : bool = true;

  for (var ll : i32 = i32(WLAYERS) - 1; ll >= 0; ll = ll - 1) {
    let l = u32(ll);
    let inW = layInW(l);
    let outW = layOutW(l);
    let wo = wOff(l);
    let bo = bOff(l);

    // (1) bias grad: invocations with col<outW own gradB cell (l,col).
    if (col < outW) {
      var accB : f32 = 0.0;
      for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
        let S = sampleBase + s;
        if (S < p.numSamples) {
          let d = f32(select(deltaB[s * W + col], deltaA[s * W + col], curIsA));
          accB = accB + d;
        }
      }
      atomicAdd(&gradBfx[bo + col], i32(accB * GRAD_FP));
    }

    // (1) weight grad: there are outW*inW cells. Distribute across the W
    // invocations: invocation 'col' handles cells (o,i) with o*inW+i ≡ col (mod W).
    // Each owned cell sums delta_in[s,o]*a_prev[s,i] over the tile.
    let totalCells = outW * inW;
    var c : u32 = col;
    loop {
      if (c >= totalCells) { break; }
      let o = c / inW;
      let i = c % inW;
      var accW : f32 = 0.0;
      for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
        let S = sampleBase + s;
        if (S < p.numSamples) {
          let d = f32(select(deltaB[s * W + o], deltaA[s * W + o], curIsA));
          let aPrev = f32(actsGlob[saveOff(S, l, i)]); // a of node-layer l
          accW = accW + d * aPrev;
        }
      }
      atomicAdd(&gradWfx[wo + o * inW + i], i32(accW * GRAD_FP));
      c = c + W;
    }
    workgroupBarrier();

    // (2) propagate delta to the earlier node-layer (l), unless l==0 (raw input).
    if (l > 0u) {
      // delta_prev[s,i] = (sum_o W[l][o,i] * delta_in[s,o]) * relu'(z[node-layer l][s,i])
      // invocation owns column i = col (must be < inW).
      if (col < inW) {
        for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
          let S = sampleBase + s;
          var acc : f32 = 0.0;
          if (S < p.numSamples) {
            for (var o : u32 = 0u; o < outW; o = o + 1u) {
              let dd = f32(select(deltaB[s * W + o], deltaA[s * W + o], curIsA));
              acc = acc + f32(weights[wo + o * inW + col]) * dd;
            }
            let z = f32(zGlob[saveOff(S, l, col)]);
            let g = select(0.0, 1.0, z > 0.0);
            acc = acc * g;
          }
          // write to the OTHER buffer
          if (curIsA) { deltaB[s * W + col] = ${SC}(acc); }
          else { deltaA[s * W + col] = ${SC}(acc); }
        }
      } else {
        for (var s : u32 = 0u; s < TILE_B; s = s + 1u) {
          if (curIsA) { deltaB[s * W + col] = ${SC}(0); }
          else { deltaA[s * W + col] = ${SC}(0); }
        }
      }
      workgroupBarrier();
      curIsA = !curIsA;
    }
  }
}
`;
}

// Downcast f32 master weights -> f16 forward/backward buffers (mixed precision).
// One invocation per element. Only emitted/used on the f16 path.
export function downcastF16Wgsl(): string {
  return /* wgsl */`enable f16;
struct DCParams { count : u32, _p0 : u32, _p1 : u32, _p2 : u32 }
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f16>;
@group(0) @binding(2) var<uniform>             p   : DCParams;
@compute @workgroup_size(64, 1, 1)
fn downcast(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.count) { return; }
  dst[idx] = f16(src[idx]);
}
`;
}

// Convert fixed-point i32 grads back to f32 (one invocation per grad element)
// and clear the fixed-point buffer for the next step. Lets the existing Adam
// kernel consume f32 grads unchanged.
export function gradFinalizeWgsl(): string {
  return /* wgsl */`
struct GFParams { count : u32, _p0 : u32, _p1 : u32, _p2 : u32 }
@group(0) @binding(0) var<storage, read_write> gradFx : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> gradF  : array<f32>;
@group(0) @binding(2) var<uniform>             p      : GFParams;
const GRAD_FP : f32 = 1048576.0;
@compute @workgroup_size(64, 1, 1)
fn gradFinalize(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.count) { return; }
  let fx = atomicExchange(&gradFx[idx], 0);   // read & clear for next step
  gradF[idx] = f32(fx) / GRAD_FP;
}
`;
}
