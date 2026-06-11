/**
 * nrcEncoding.ts — Neural Radiance Caching INPUT ENCODING CPU oracle.
 *
 * This is the load-bearing, THREE-free, GPU-free reference implementation of the
 * "full"-version NRC input encoding (Müller, Rousselle, Novák, Keller 2021,
 * "Real-time Neural Radiance Caching for Path Tracing", ACM TOG 40(4) §4),
 * assembled from:
 *
 *   1. Multiresolution **hash-grid positional encoding** of the cache-query
 *      vertex position (Müller, Evans, Schied, Keller 2022, "Instant Neural
 *      Graphics Primitives with a Multiresolution Hash Encoding", ACM TOG 41(4),
 *      §3 / Eq. 3–5). L resolution levels; each level has a hashed feature table
 *      of T entries × F features. A query position maps into each level's voxel
 *      grid, the 2^D corner entries are looked up by the spatial hash, and their
 *      features are TRILINEARLY interpolated (D=3 → 8 corners). The L·F level
 *      features are concatenated. The feature tables are TRAINABLE — this module
 *      provides the forward AND the backward GRADIENT SCATTER into the hashed
 *      table entries (Instant-NGP §4: the interpolation weight of each corner is
 *      the local gradient w.r.t. that corner's feature; multiple (level, corner)
 *      pairs may collide onto the same table row, and their gradients ACCUMULATE
 *      — that accumulation is the whole point of the learned hash encoding).
 *
 *   2. **One-blob encoding** of direction(s) (Müller, McWilliams, Rousselle,
 *      Gross, Novák 2019, "Neural Importance Sampling", ACM TOG 38(5), §4.3 /
 *      Eq. 12; used for the NRC scattered/view direction in Müller 2021 §4). A
 *      scalar u ∈ [0,1] is encoded as k Gaussian-kernel activations placed on a
 *      uniform 1-D grid — a soft one-hot that gives the MLP a smooth, localized
 *      representation of the angle. We encode a direction as two scalars
 *      (octahedral-mapped u,v ∈ [0,1]) × k bins each. One-blob has NO trainable
 *      parameters (it is a fixed kernel), so it has only a forward.
 *
 *   3. **Raw surface features** appended verbatim: surface normal (3, octahedral
 *      or raw xyz — we keep raw xyz here, the MLP learns the rest), roughness
 *      (1), diffuse albedo (3). These pass through with an identity Jacobian.
 *
 * The concatenated vector is the MLP input of width
 *   inW = L·F  (hash grid)  +  2·k  (one-blob dir)  +  3+1+3 (raw features).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * VERIFICATION ROLE
 * ════════════════════════════════════════════════════════════════════════════
 * This oracle plays the SAME role `reconnectionShift.ts` plays for GRIS and
 * `bdptConnectionMisFull.ts` plays for BDPT: a deterministic first-principles
 * reference the WGSL forward (`nrcEncoding.wgsl.ts`) is pinned against to ~1e-6
 * (f32). The hash-grid BACKWARD is checked two ways in the tests:
 *   - against an EXACT-ANALYTIC gradient (the interpolation weights ARE the
 *     analytic ∂feature/∂corner, so the scatter is exact — no FD needed), and
 *   - by a finite-difference probe of a downstream scalar loss for cross-check.
 * The trilinear interpolation is smooth (no ReLU kink), so the FD probe here is
 * clean — UNLIKE the MLP-internal FD which the kernel agent documented carries a
 * ReLU-kink artifact. We therefore gate the hash-grid backward on the
 * exact-analytic match and use FD only as a secondary sanity check.
 *
 * NOTE ON BIAS (honest acceptance criterion): NRC is a BIASED estimator — the
 * cache is a learned approximation of the path suffix, NOT an unbiased Monte
 * Carlo estimate. This module only concerns the ENCODING + its gradients, which
 * are exact; the bias enters at cache QUERY time (the MLP's prediction replaces
 * the true suffix integral). See the V-item in HARDWARE-VALIDATION-NEEDS.md.
 */

/** A single hash-grid resolution level's hyperparameters + feature table. */
export interface HashGridLevel {
  /** Voxel resolution N_l along each axis at this level (Eq. 3: N_l = floor(N_min·b^l)). */
  readonly resolution: number;
  /** Number of feature rows T in this level's hashed table. */
  readonly tableSize: number;
  /** Feature table, row-major [tableSize × featuresPerEntry]. TRAINABLE. */
  readonly table: Float32Array;
}

export interface HashGridConfig {
  /** Spatial dimension. NRC cache query is a 3-D world position → D = 3. */
  readonly dim: 3;
  /** Features per table entry F (Instant-NGP default 2). */
  readonly featuresPerEntry: number;
  /** Per-level config + tables (length L). */
  readonly levels: readonly HashGridLevel[];
  /**
   * AABB the query position is normalised into [0,1]^D against before the grid
   * lookup. NRC queries the scene-space vertex; we map it through the scene AABB.
   */
  readonly aabbMin: readonly [number, number, number];
  readonly aabbMax: readonly [number, number, number];
}

/** One-blob encoding hyperparameters (no trainable params). */
export interface OneBlobConfig {
  /** Number of Gaussian bins k per encoded scalar (Müller 2019 §4.3). */
  readonly bins: number;
  /**
   * Gaussian kernel bandwidth σ in grid-cell units. Müller 2019 uses
   * σ = 1 / bins (one cell). Larger σ = smoother / more overlap.
   */
  readonly sigma: number;
}

/**
 * Instant-NGP spatial hash (Müller 2022 §3, Eq. 4). π-constants are the
 * canonical large primes from the reference implementation. Returns a row index
 * in [0, tableSize). For coarse levels whose dense grid is smaller than the
 * table, Instant-NGP indexes densely (no hashing); we replicate by hashing
 * unconditionally — collisions on coarse levels are harmless because the hash is
 * a bijection on the small dense range when tableSize ≥ (resolution+1)^D, and
 * the tests cover both regimes. The XOR-of-prime-products form is exactly the
 * reference `grid.cu` hash.
 */
export function spatialHash3D(ix: number, iy: number, iz: number, tableSize: number): number {
  // The reference uses unsigned 32-bit wraparound; emulate with >>> 0 / Math.imul.
  const P1 = 1;            // x prime is implicitly 1 in the reference (no multiply)
  const P2 = 2654435761;   // 0x9E3779B1
  const P3 = 805459861;    // 0x30034BB7
  const h = (Math.imul(ix >>> 0, P1) ^ Math.imul(iy >>> 0, P2) ^ Math.imul(iz >>> 0, P3)) >>> 0;
  return h % tableSize;
}

/** Geometric per-level resolution growth (Instant-NGP Eq. 2/3). */
export function levelResolution(nMin: number, growth: number, level: number): number {
  return Math.floor(nMin * Math.pow(growth, level));
}

/**
 * The 8 trilinear corner contributions for a normalised query point at a given
 * level. Returns, for each of the 8 corners, its hashed table ROW and its
 * trilinear interpolation WEIGHT (the weights sum to 1). This is the shared core
 * of forward (weighted feature sum) and backward (weighted gradient scatter).
 */
export interface CornerContribution {
  readonly row: number;
  readonly weight: number;
}

export function trilinearCorners(
  level: HashGridLevel,
  // query already normalised to [0,1]^3
  nx: number, ny: number, nz: number,
): CornerContribution[] {
  const N = level.resolution;
  // Scale into the level's voxel grid. Position p ∈ [0, N].
  const px = nx * N, py = ny * N, pz = nz * N;
  const x0 = Math.floor(px), y0 = Math.floor(py), z0 = Math.floor(pz);
  const fx = px - x0, fy = py - y0, fz = pz - z0; // local frac ∈ [0,1)
  const out: CornerContribution[] = [];
  for (let c = 0; c < 8; c++) {
    const cx = (c & 1) ? 1 : 0;
    const cy = (c & 2) ? 1 : 0;
    const cz = (c & 4) ? 1 : 0;
    // Trilinear weight = product of per-axis (frac or 1-frac).
    const wx = cx ? fx : (1 - fx);
    const wy = cy ? fy : (1 - fy);
    const wz = cz ? fz : (1 - fz);
    const weight = wx * wy * wz;
    const row = spatialHash3D(x0 + cx, y0 + cy, z0 + cz, level.tableSize);
    out.push({ row, weight });
  }
  return out;
}

/** Normalise a world position into the grid's [0,1]^3 (clamped). */
export function normalizeToAabb(
  pos: readonly [number, number, number],
  aabbMin: readonly [number, number, number],
  aabbMax: readonly [number, number, number],
): [number, number, number] {
  const n: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const ext = aabbMax[a]! - aabbMin[a]!;
    const t = ext > 1e-20 ? (pos[a]! - aabbMin[a]!) / ext : 0;
    n[a] = Math.min(1, Math.max(0, t));
  }
  return n;
}

/**
 * Forward hash-grid encode: for each level, trilinearly interpolate the F
 * features of the 8 hashed corners and concatenate across levels. Output width
 * = L · F. (Instant-NGP §3.)
 */
export function hashGridForward(
  cfg: HashGridConfig,
  pos: readonly [number, number, number],
): Float32Array {
  const [nx, ny, nz] = normalizeToAabb(pos, cfg.aabbMin, cfg.aabbMax);
  const F = cfg.featuresPerEntry;
  const out = new Float32Array(cfg.levels.length * F);
  for (let l = 0; l < cfg.levels.length; l++) {
    const level = cfg.levels[l]!;
    const corners = trilinearCorners(level, nx, ny, nz);
    const base = l * F;
    for (const { row, weight } of corners) {
      const rb = row * F;
      for (let f = 0; f < F; f++) {
        out[base + f] = out[base + f]! + weight * level.table[rb + f]!;
      }
    }
  }
  return out;
}

/**
 * Backward hash-grid: scatter the upstream gradient dL/dFeature (length L·F)
 * into per-level gradient tables, accumulating each corner's weighted
 * contribution. Instant-NGP §4: dL/dtable[row][f] += weight · dL/dfeature[f],
 * summed over every (level, corner) pair that maps to that row. COLLISIONS
 * ACCUMULATE — two distinct corners/levels hashing to the same row add their
 * gradients, which is the defining behaviour of the learned hash encoding.
 *
 * Returns one Float32Array per level (same shape as the level's table), so the
 * caller can feed each into that level's Adam optimizer state.
 *
 * NOTE: this is EXACT-ANALYTIC. The trilinear interpolation weight IS the
 * analytic ∂feature/∂corner, so no finite difference is involved on the encode
 * Jacobian — the FD probe in the tests is only a downstream cross-check.
 */
export function hashGridBackward(
  cfg: HashGridConfig,
  pos: readonly [number, number, number],
  dOut: Float32Array, // length L·F
): Float32Array[] {
  const [nx, ny, nz] = normalizeToAabb(pos, cfg.aabbMin, cfg.aabbMax);
  const F = cfg.featuresPerEntry;
  const grads = cfg.levels.map((lvl) => new Float32Array(lvl.table.length));
  for (let l = 0; l < cfg.levels.length; l++) {
    const level = cfg.levels[l]!;
    const corners = trilinearCorners(level, nx, ny, nz);
    const base = l * F;
    const g = grads[l]!;
    for (const { row, weight } of corners) {
      const rb = row * F;
      for (let f = 0; f < F; f++) {
        // Accumulate (+=) — collisions onto the same row sum, per §4.
        g[rb + f] = g[rb + f]! + weight * dOut[base + f]!;
      }
    }
  }
  return grads;
}

/**
 * One-blob encode of a single scalar u ∈ [0,1] into `bins` Gaussian activations
 * on a uniform 1-D grid (Müller 2019 §4.3, Eq. 12). Bin i is centred at
 * (i + 0.5)/bins; activation = exp(-0.5·((u - center)/σ)²). We L1-normalise the
 * bin vector so it integrates to 1 (a soft one-hot), matching the reference's
 * partition-of-unity intent and keeping the MLP input magnitude stable.
 */
export function oneBlobEncodeScalar(u: number, cfg: OneBlobConfig): Float32Array {
  const k = cfg.bins;
  const out = new Float32Array(k);
  const uc = Math.min(1, Math.max(0, u));
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const center = (i + 0.5) / k;
    const d = (uc - center) / cfg.sigma;
    const a = Math.exp(-0.5 * d * d);
    out[i] = a;
    sum += a;
  }
  // Normalise to unit L1 (guard against all-tiny in degenerate σ).
  if (sum > 1e-20) {
    for (let i = 0; i < k; i++) out[i] = out[i]! / sum;
  }
  return out;
}

/**
 * Octahedral-map a unit direction to (u, v) ∈ [0,1]² (Cigolle et al. 2014;
 * the canonical octEncode used elsewhere in the engine — W2-C3). Used so a
 * 3-D direction is one-blob-encoded as two scalars.
 */
export function octEncodeDir(d: readonly [number, number, number]): [number, number] {
  const ax = Math.abs(d[0]), ay = Math.abs(d[1]), az = Math.abs(d[2]);
  const s = ax + ay + az;
  const inv = s > 1e-20 ? 1 / s : 0;
  let px = d[0] * inv;
  let py = d[1] * inv;
  if (d[2] < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox; py = oy;
  }
  // map [-1,1] → [0,1]
  return [px * 0.5 + 0.5, py * 0.5 + 0.5];
}

export interface NrcSurfaceFeatures {
  readonly position: readonly [number, number, number];
  readonly normal: readonly [number, number, number];
  readonly direction: readonly [number, number, number]; // e.g. view / scattered dir
  readonly roughness: number;
  readonly albedo: readonly [number, number, number];
}

export interface NrcEncodingConfig {
  readonly hashGrid: HashGridConfig;
  readonly oneBlob: OneBlobConfig;
}

/** Total encoded input width for a given config (= the MLP `inW`). */
export function nrcInputWidth(cfg: NrcEncodingConfig): number {
  const hg = cfg.hashGrid.levels.length * cfg.hashGrid.featuresPerEntry;
  const blob = 2 * cfg.oneBlob.bins; // octahedral u,v
  const raw = 3 + 1 + 3;             // normal + roughness + albedo
  return hg + blob + raw;
}

/**
 * Assemble the FULL NRC MLP input vector for one cache-query vertex:
 *   [ hash-grid(pos) | one-blob(octU(dir)) | one-blob(octV(dir)) | normal | rough | albedo ]
 * The layout is fixed and shared with the WGSL forward (nrcEncoding.wgsl.ts).
 */
export function assembleNrcInput(
  cfg: NrcEncodingConfig,
  s: NrcSurfaceFeatures,
): Float32Array {
  const hg = hashGridForward(cfg.hashGrid, s.position);
  const [ou, ov] = octEncodeDir(s.direction);
  const bu = oneBlobEncodeScalar(ou, cfg.oneBlob);
  const bv = oneBlobEncodeScalar(ov, cfg.oneBlob);
  const out = new Float32Array(nrcInputWidth(cfg));
  let o = 0;
  out.set(hg, o); o += hg.length;
  out.set(bu, o); o += bu.length;
  out.set(bv, o); o += bv.length;
  out[o++] = s.normal[0]!; out[o++] = s.normal[1]!; out[o++] = s.normal[2]!;
  out[o++] = s.roughness;
  out[o++] = s.albedo[0]!; out[o++] = s.albedo[1]!; out[o++] = s.albedo[2]!;
  return out;
}
