/**
 * bmfr.wgsl.ts — BMFR per-block feature-regression compute kernel.
 *
 * BMFR = Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala,
 * Takala. "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 * Reconstruction." ACM Transactions on Graphics 38(5), 2019.
 *
 * One workgroup per 32×32 pixel block (256 threads = 16×16, each thread owns a
 * 2×2 patch so a 16×16 workgroup covers the 32×32 block). The block:
 *   1. Builds each pixel's 10-feature row  [1, p.xyz, n.xyz, p².xyz]  where p is
 *      the block-local world position (block-mean-subtracted, /positionScale).
 *   2. Accumulates the symmetric normal matrix  M = AᵀA + λI  (10×10) and the
 *      three RHS vectors  r_ch = Aᵀ c_ch  in workgroup shared memory.
 *      The accumulation is the per-block analogue of solving the least-squares
 *      fit by QR on the feature matrix (Koskela 2019 §4.2) — forming the small
 *      dense normal system and QR-solving THAT is equivalent and is what the
 *      reference impl reduces to once the block size is fixed.
 *   3. A single thread Householder-QR-solves  M α_ch = r_ch  for each channel.
 *   4. Every thread reconstructs its pixel color  α_ch · feature_row  and
 *      temporally accumulates against the reprojected history texture.
 *
 * Color is filtered in DEMODULATED space (caller divides by albedo first); this
 * kernel does not see albedo. The block-mean position subtraction + division by
 * positionScale keeps the p² columns well-conditioned for the QR solve.
 *
 * Bind group 0 (entry point: bmfrMain):
 *   binding 0 — texture_2d<f32>                        bmfr_color    (noisy demodulated RGB, rgba16f)
 *   binding 1 — texture_2d<f32>                        bmfr_normal   (.xyz = world normal × 0.5 + 0.5)
 *   binding 2 — texture_2d<f32>                        bmfr_worldPos (.xyz = world position, .w unused)
 *   binding 3 — texture_2d<f32>                        bmfr_history  (previous reconstructed frame, rgba16f)
 *   binding 4 — texture_storage_2d<rgba16float, write> bmfr_out      (reconstructed + accumulated)
 *   binding 5 — var<uniform>                           bmfr_ubo      BmfrUBO
 */

/** Workgroup is 16×16 threads; each thread owns a 2×2 patch → covers 32×32. */
export const BMFR_WORKGROUP_SIZE = 16 as const;

/** Feature columns — must equal BMFR_FEATURE_COUNT in bmfrRegression.ts. */
export const BMFR_WGSL_FEATURE_COUNT = 10 as const;

export const BMFR_WGSL = /* wgsl */ `
const F: u32 = ${BMFR_WGSL_FEATURE_COUNT}u;   // feature count (10)
const PATCH: u32 = 2u;                          // 2×2 pixels per thread → 32×32 block

struct BmfrUBO {
  blockSize:      u32,
  blockStride:    u32,
  positionScale:  f32,
  temporalAlpha:  f32,
  regularisation: f32,
  hasHistory:     f32,
  // positionMode: 0 = sample bmfr_worldPos.xyz as world position (.w = validity);
  //               1 = screen-space proxy — position = (pixelX, pixelY, depth),
  //                   depth + validity read from bmfr_worldPos.w (the host binds
  //                   the gNormalDepth texture there). Lets the walkaround
  //                   pipeline avoid allocating a dedicated world-position
  //                   G-buffer; per-screen-block regression is well-posed with
  //                   the screen-space position proxy (positions are normalised
  //                   block-locally before the squared features anyway).
  positionMode:   u32,
};

@group(0) @binding(0) var bmfr_color:    texture_2d<f32>;
@group(0) @binding(1) var bmfr_normal:   texture_2d<f32>;
@group(0) @binding(2) var bmfr_worldPos: texture_2d<f32>;
@group(0) @binding(3) var bmfr_history:  texture_2d<f32>;
@group(0) @binding(4) var bmfr_out:      texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform>       bmfr_ubo: BmfrUBO;

// Resolve per-pixel (position.xyz, validity) honoring positionMode.
fn loadPosition(coord: vec2u) -> vec4f {
  let raw = textureLoad(bmfr_worldPos, coord, 0);
  if (bmfr_ubo.positionMode == 1u) {
    // Screen-space proxy: position = (pixel.x, pixel.y, depth), depth = .w.
    let depth = raw.w;
    return vec4f(f32(coord.x), f32(coord.y), depth, depth);
  }
  return raw; // mode 0: .xyz world position, .w validity
}

// ── Workgroup shared state ──────────────────────────────────────────────────
// f32 atomics are not core WGSL, so the normal-matrix accumulation + the dense
// 10×10 QR solve are done on a SINGLE thread (thread 0): 32×32 = 1024 pixels ×
// 10 features is cheap relative to the ray-tracing passes, and avoids any
// f32-atomic dependency or large shared-memory reduction buffer. Threads
// 1..255 wait on a barrier, then each reconstructs its own 2×2 patch from the
// alpha weights thread 0 publishes via alphaR/alphaG/alphaB. validCount (atomic
// u32) only counts valid surface pixels for the block-mean divisor; meanPos is
// the block-mean world position published by thread 0 for block-local
// centering of the squared position features.
var<workgroup> validCount: atomic<u32>;
var<workgroup> meanPos: vec3f;

var<workgroup> alphaR: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
var<workgroup> alphaG: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
var<workgroup> alphaB: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;

// Build the 10-feature row for a block-local position + normal.
fn featureRow(pLocal: vec3f, n: vec3f, row: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>) {
  (*row)[0] = 1.0;
  (*row)[1] = pLocal.x;
  (*row)[2] = pLocal.y;
  (*row)[3] = pLocal.z;
  (*row)[4] = n.x;
  (*row)[5] = n.y;
  (*row)[6] = n.z;
  (*row)[7] = pLocal.x * pLocal.x;
  (*row)[8] = pLocal.y * pLocal.y;
  (*row)[9] = pLocal.z * pLocal.z;
}

// MUST-MATCH MIRROR: householderSolve (GPU ↔ CPU)
//
// This WGSL kernel mirrors the CPU reference in
// bmfrRegression.ts::householderSolve. The two implementations MUST stay
// bit-for-bit equivalent on every convergence guard and back-substitution step:
//
//   • norm < 1e-20         — skip near-zero pivot columns       ← BOTH sides
//   • vNormSq < 1e-30      — skip near-degenerate reflectors    ← BOTH sides
//   • abs(diag) > 1e-20    — back-substitution singularity gate ← BOTH sides
//   • back-substitution traversal order: i = F-1 .. 0 (descending via ii)
//
// If you change any of these guards or the back-substitution order here,
// apply the IDENTICAL change in bmfrRegression.ts::householderSolve and vice-versa.
//
// Householder-QR solve of the dense f×f system M x = b. M is row-major in a
// flat 100-entry array; b is length 10. Returns x in out.
fn householderSolve(
  Min:  ptr<function, array<f32, 100>>,
  bin:  ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
  out:  ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
) {
  var R: array<f32, 100>;
  var y: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  for (var i = 0u; i < F * F; i = i + 1u) { R[i] = (*Min)[i]; }
  for (var i = 0u; i < F; i = i + 1u) { y[i] = (*bin)[i]; }

  for (var col = 0u; col < F; col = col + 1u) {
    var normSq = 0.0;
    for (var i = col; i < F; i = i + 1u) {
      let v = R[i * F + col];
      normSq = normSq + v * v;
    }
    var norm = sqrt(normSq);
    if (norm < 1e-20) { continue; } // MUST-MATCH bmfrRegression.ts
    let x0 = R[col * F + col];
    let sgn = select(-1.0, 1.0, x0 >= 0.0);
    norm = norm * sgn;
    var v: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    for (var i = 0u; i < F; i = i + 1u) { v[i] = 0.0; }
    v[col] = x0 + norm;
    for (var i = col + 1u; i < F; i = i + 1u) { v[i] = R[i * F + col]; }
    var vNormSq = 0.0;
    for (var i = col; i < F; i = i + 1u) { vNormSq = vNormSq + v[i] * v[i]; }
    if (vNormSq < 1e-30) { continue; } // MUST-MATCH bmfrRegression.ts

    for (var j = col; j < F; j = j + 1u) {
      var dot = 0.0;
      for (var i = col; i < F; i = i + 1u) { dot = dot + v[i] * R[i * F + j]; }
      let factor = (2.0 * dot) / vNormSq;
      for (var i = col; i < F; i = i + 1u) { R[i * F + j] = R[i * F + j] - factor * v[i]; }
    }
    var dotY = 0.0;
    for (var i = col; i < F; i = i + 1u) { dotY = dotY + v[i] * y[i]; }
    let fy = (2.0 * dotY) / vNormSq;
    for (var i = col; i < F; i = i + 1u) { y[i] = y[i] - fy * v[i]; }
  }

  // Back-substitution.
  for (var ii = 0u; ii < F; ii = ii + 1u) {
    let i = F - 1u - ii;
    var acc = y[i];
    for (var j = i + 1u; j < F; j = j + 1u) { acc = acc - R[i * F + j] * (*out)[j]; }
    let diag = R[i * F + i];
    (*out)[i] = select(0.0, acc / diag, abs(diag) > 1e-20); // MUST-MATCH bmfrRegression.ts
  }
}

@compute @workgroup_size(${BMFR_WORKGROUP_SIZE}, ${BMFR_WORKGROUP_SIZE}, 1)
fn bmfrMain(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) lindex: u32,
) {
  let dims = textureDimensions(bmfr_color);
  let blockSize = bmfr_ubo.blockSize;
  // Block origin in pixels. The host clamps blockStride >= blockSize so the
  // tiling partitions the image disjointly (no overlapping textureStore →
  // deterministic output); one workgroup per block origin = wid * blockStride.
  let blockOrigin = wid.xy * bmfr_ubo.blockStride;

  // Each thread covers a PATCH×PATCH sub-tile of the block.
  // Thread (lx,ly) owns block-local pixels [lx*PATCH .. lx*PATCH+PATCH).
  let baseLocal = lid.xy * PATCH;

  // ── Phase 1: count valid surface pixels in the block (mean divisor). ──────
  // Thread 0 re-reads the whole block in Phase 2 to compute the position mean
  // and form M/r, so threads here only tally the valid-pixel count via the
  // u32 atomic (the only reduction that needs cross-thread accumulation).
  if (lindex == 0u) {
    atomicStore(&validCount, 0u);
  }
  workgroupBarrier();

  var localValid = 0u;
  for (var py = 0u; py < PATCH; py = py + 1u) {
    for (var px = 0u; px < PATCH; px = px + 1u) {
      let bl = baseLocal + vec2u(px, py);
      if (bl.x >= blockSize || bl.y >= blockSize) { continue; }
      let coord = blockOrigin + bl;
      if (any(coord >= dims)) { continue; }
      let wp = loadPosition(coord);
      // Sky / miss pixels (depth-less) carry w<=0; skip from the position mean.
      if (wp.w <= 0.0) { continue; }
      localValid = localValid + 1u;
    }
  }
  atomicAdd(&validCount, localValid);
  workgroupBarrier();

  // ── Phase 2 + 3: thread 0 forms M, r and solves; publishes α. ─────────────
  if (lindex == 0u) {
    // Recompute the block-mean position over the whole block (thread 0 only).
    var msum = vec3f(0.0);
    let cnt = atomicLoad(&validCount);
    for (var by = 0u; by < blockSize; by = by + 1u) {
      for (var bx = 0u; bx < blockSize; bx = bx + 1u) {
        let coord = blockOrigin + vec2u(bx, by);
        if (any(coord >= dims)) { continue; }
        let wp = loadPosition(coord);
        if (wp.w <= 0.0) { continue; }
        msum = msum + wp.xyz;
      }
    }
    let mean = select(vec3f(0.0), msum / f32(cnt), cnt > 0u);
    meanPos = mean;

    let invScale = 1.0 / max(bmfr_ubo.positionScale, 1e-4);

    // Accumulate normal matrix M = AᵀA + λI and rhs r_ch = Aᵀ c_ch.
    var M: array<f32, 100>;
    for (var i = 0u; i < F * F; i = i + 1u) { M[i] = 0.0; }
    var rR: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    var rG: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    var rB: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    for (var i = 0u; i < F; i = i + 1u) { rR[i] = 0.0; rG[i] = 0.0; rB[i] = 0.0; }

    for (var by = 0u; by < blockSize; by = by + 1u) {
      for (var bx = 0u; bx < blockSize; bx = bx + 1u) {
        let coord = blockOrigin + vec2u(bx, by);
        if (any(coord >= dims)) { continue; }
        let wp = loadPosition(coord);
        if (wp.w <= 0.0) { continue; }
        let pLocal = (wp.xyz - mean) * invScale;
        let nrm = textureLoad(bmfr_normal, coord, 0).xyz * 2.0 - 1.0;
        let col = textureLoad(bmfr_color, coord, 0).rgb;
        var row: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
        featureRow(pLocal, nrm, &row);
        for (var i = 0u; i < F; i = i + 1u) {
          let ri = row[i];
          rR[i] = rR[i] + ri * col.r;
          rG[i] = rG[i] + ri * col.g;
          rB[i] = rB[i] + ri * col.b;
          for (var j = 0u; j < F; j = j + 1u) {
            M[i * F + j] = M[i * F + j] + ri * row[j];
          }
        }
      }
    }
    let lambda = bmfr_ubo.regularisation;
    for (var i = 0u; i < F; i = i + 1u) { M[i * F + i] = M[i * F + i] + lambda; }

    var aR: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    var aG: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    var aB: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
    for (var i = 0u; i < F; i = i + 1u) { aR[i] = 0.0; aG[i] = 0.0; aB[i] = 0.0; }
    householderSolve(&M, &rR, &aR);
    householderSolve(&M, &rG, &aG);
    householderSolve(&M, &rB, &aB);
    for (var i = 0u; i < F; i = i + 1u) {
      alphaR[i] = aR[i];
      alphaG[i] = aG[i];
      alphaB[i] = aB[i];
    }
  }
  workgroupBarrier();

  // ── Phase 4: every thread reconstructs its own 2×2 patch + accumulates. ───
  let mean = meanPos;
  let invScale = 1.0 / max(bmfr_ubo.positionScale, 1e-4);
  let useHistory = bmfr_ubo.hasHistory > 0.5;
  let alpha = bmfr_ubo.temporalAlpha;

  for (var py = 0u; py < PATCH; py = py + 1u) {
    for (var px = 0u; px < PATCH; px = px + 1u) {
      let bl = baseLocal + vec2u(px, py);
      if (bl.x >= blockSize || bl.y >= blockSize) { continue; }
      let coord = blockOrigin + bl;
      if (any(coord >= dims)) { continue; }

      let wp = loadPosition(coord);
      // Sky / miss: pass the raw color straight through (no surface to fit).
      if (wp.w <= 0.0) {
        let raw = textureLoad(bmfr_color, coord, 0).rgb;
        textureStore(bmfr_out, coord, vec4f(raw, 1.0));
        continue;
      }

      let pLocal = (wp.xyz - mean) * invScale;
      let nrm = textureLoad(bmfr_normal, coord, 0).xyz * 2.0 - 1.0;
      var row: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
      featureRow(pLocal, nrm, &row);

      var recon = vec3f(0.0);
      for (var i = 0u; i < F; i = i + 1u) {
        recon = recon + row[i] * vec3f(alphaR[i], alphaG[i], alphaB[i]);
      }
      // The fit can mildly undershoot to negatives on dark blocks; clamp ≥ 0.
      recon = max(recon, vec3f(0.0));

      var outColor = recon;
      if (useHistory) {
        let hist = textureLoad(bmfr_history, coord, 0).rgb;
        outColor = mix(hist, recon, alpha);
      }
      textureStore(bmfr_out, coord, vec4f(outColor, 1.0));
    }
  }
}
`;
