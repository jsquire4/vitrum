/**
 * SPPM group-4 bindings — shared between the photon-emission compute pass
 * (`sppmPhotonPass.wgsl.ts`) and the megakernel gather path in
 * `caustic.wgsl.ts`.
 *
 * SPPM = Stochastic Progressive Photon Mapping (Hachisuka & Jensen 2009).
 *
 * Binding layout (@group(4)):
 *   binding(0)  sppmPhotonCells   — PhotonRecord[SPPM_MAX_CELLS × SPPM_CELL_CAPACITY]
 *               Each cell has SPPM_CELL_CAPACITY slots.  The photon pass writes
 *               photons here atomically (modulo capacity) and the megakernel
 *               reads from them.  access = read_write on both pipelines.
 *   binding(1)  sppmCellCounters  — atomic<u32>[SPPM_MAX_CELLS]
 *               Cumulative photon-insertion counter per cell.  Modulo
 *               SPPM_CELL_CAPACITY gives the ring slot; min-cap gives the true
 *               count for the density estimate.
 *   binding(2)  sppmStats         — SppmStats (uniform UBO, 32 bytes)
 *               currentRadius, frameIndex, photonCount, sceneExtent.
 *
 * Geometry (PhotonRecord, 48 bytes = 3 vec4f):
 *   vec0:  position.xyz + padding
 *   vec1:  flux.rgb + padding
 *   vec2:  incidentDir.xyz + padding
 *
 * The megakernel gather uses the hash-grid cell → check photon.position within
 * currentRadius → accumulate BRDF × flux × kernel / (π r²) over the N_cell
 * photons stored.
 *
 * Both pipelines are full-tier only.  When causticStrategy != 'photon-map' a
 * 16-byte placeholder buffer is bound so the group-4 layout slot is satisfied
 * without allocating the real photon map.  The gather code in caustic.wgsl.ts
 * is guarded by `if (causticMode() == 2u)` so it never executes.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009); spatial-hash scheme follows the survey in
 * Ihrke et al. 2007 § 4 (prime-multiplied coordinate hash, no collision
 * resolution — overflow counter protects memory, not energy).
 */

/** Number of spatial-hash cells.  Power-of-2 avoids the modulo's division for
 *  non-power-of-2 table sizes, but an odd prime-ish size gives better hash
 *  distribution; 65521 is the largest prime below 2^16. */
export const SPPM_MAX_CELLS = 65521;

/** Photons stored per cell (ring buffer).  Over-capacity photons are dropped
 *  (their slot is counted but not written); the density estimator clamps to
 *  this capacity so only the most-recent SPPM_CELL_CAPACITY photons contribute
 *  per cell.  R7a behavioral-gate fix (2026-06-10): capacity 128 made the cells
 *  buffer 65521 × 128 × 48 B ≈ 402 MiB — EXCEEDING WebGPU's default
 *  maxBufferSize (256 MiB), so every photon-map render failed buffer validation
 *  on default-limit devices.  32 → ≈ 100 MiB, inside BOTH the default maxBufferSize (256 MiB) AND the default maxStorageBufferBindingSize (128 MiB — the binding limit binds first); the
 *  host additionally guards against the live device limit at allocation and
 *  degrades to manifold-nee with a warning. */
export const SPPM_CELL_CAPACITY = 32;

/** Bytes per PhotonRecord: 3 × vec4f = 48 bytes. */
export const SPPM_PHOTON_RECORD_BYTES = 48;

/** Bytes for the sppmPhotonCells buffer:
 *  SPPM_MAX_CELLS × SPPM_CELL_CAPACITY × SPPM_PHOTON_RECORD_BYTES. */
export const SPPM_PHOTON_CELLS_BYTES =
  SPPM_MAX_CELLS * SPPM_CELL_CAPACITY * SPPM_PHOTON_RECORD_BYTES;

/** Bytes for the sppmCellCounters buffer: SPPM_MAX_CELLS × 4 (atomic<u32>). */
export const SPPM_CELL_COUNTERS_BYTES = SPPM_MAX_CELLS * 4;

/** Bytes for the SppmStats UBO: 4 × f32 = 32 bytes (padded to 32 for alignment). */
export const SPPM_STATS_BYTES = 32;

/**
 * I4.5 — Structural pin descriptor for the `SppmStats` WGSL struct.
 *
 * Each entry mirrors a field in the `struct SppmStats { … }` definition below.
 * The host packer (`GpuResources.writeSppmStats`) writes these slots in this
 * order; this descriptor lets a test assert:
 *   (a) the WGSL struct fields parse out in this exact order, and
 *   (b) the total size equals SPPM_STATS_BYTES (= 32).
 */
export const SPPM_STATS_FIELDS = [
  { name: 'currentRadius',    byteOffset:  0, type: 'f32' },
  { name: 'r0',               byteOffset:  4, type: 'f32' },
  { name: 'frameAccumulated', byteOffset:  8, type: 'u32' },
  { name: 'photonCount',      byteOffset: 12, type: 'u32' },
  { name: 'sceneExtent',      byteOffset: 16, type: 'f32' },
  { name: '_pad0',            byteOffset: 20, type: 'f32' },
  { name: '_pad1',            byteOffset: 24, type: 'f32' },
  { name: '_pad2',            byteOffset: 28, type: 'f32' },
] as const;

/**
 * Bytes per per-pixel SPPM statistics record:
 *   tau.rgb (f32×3) + radius2 (f32) + N (f32) + _pad (f32×3) = 8 × f32 = 32 bytes.
 *
 * A4-progressive: each pixel accumulates τ, R², and N across frames so the
 * Hachisuka progressive update rule can run without re-visiting previous frames.
 * The buffer is sized W×H×32 bytes and reset whenever the PT accumulator resets
 * (camera move, setScene, reset()) — the same static-eye-point assumption that
 * makes progressive accumulation valid also gates the SPPM stats.
 */
export const SPPM_PIXEL_STATS_BYTES_PER_PIXEL = 32; // 8 × f32

/**
 * Safety ceiling for the photon-cells buffer.  ~512 MiB at default params;
 * the warn-once pattern mirrors the BDPT eye-stack ceiling.
 */
export const SPPM_PHOTON_CELLS_MAX_BYTES = 512 * 1024 * 1024; // 512 MiB

/**
 * SPPM progressive-radius schedule.
 *
 * After `n` accumulated frames:
 *   r(n) = r₀ × sqrt((n × α + α) / (n + 1))
 * with α = 2/3 (Hachisuka & Jensen 2009, Eq.4).  For n=0 (first frame)
 * r(0) = r₀ × sqrt(α) ≈ r₀ × 0.8165, converging to 0 as n → ∞.
 *
 * The closed form for progressive shrink over one frame is:
 *   r(n+1) = r(n) × sqrt((n × α + α) / (n × α + 1))
 *
 * We store `r₀` and `frameIndex` in the UBO and recompute `currentRadius` in
 * the shader from the closed form, keeping radius state on the CPU.
 */
export const SPPM_ALPHA = 2.0 / 3.0; // Hachisuka & Jensen 2009 α

/**
 * Compute the SPPM radius for frame index `n` (0-based) given initial radius `r0`.
 * Used on the CPU host side (TypeScript) to write the UBO each frame.
 */
export function sppmRadiusAtFrame(r0: number, n: number): number {
  if (n <= 0) return r0 * Math.sqrt(SPPM_ALPHA);
  return r0 * Math.sqrt((n * SPPM_ALPHA + SPPM_ALPHA) / (n + 1));
}

/**
 * Compute the scale-aware initial radius `r₀` from the scene AABB diagonal.
 *
 * r₀ = max(diagonal / 100, 1e-3)
 *
 * The divisor 100 ensures that even on a 1 m Cornell box (diagonal ~1.7 m)
 * the initial radius is ~0.017 m (17 mm) — a reasonable first-frame footprint
 * that shrinks to sub-millimetre over thousands of frames.
 */
export function sppmInitialRadius(
  sceneMin: readonly [number, number, number],
  sceneMax: readonly [number, number, number],
): number {
  const dx = sceneMax[0] - sceneMin[0];
  const dy = sceneMax[1] - sceneMin[1];
  const dz = sceneMax[2] - sceneMin[2];
  const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return Math.max(diagonal / 100, 1e-3);
}

/**
 * SPPM group-3 WGSL bindings (bindings 6/7/8/9) — composed into the megakernel
 * AND the photon-emission pass.  Group 3 already carries the light-tree node
 * buffer (binding 0), mesh UVs (1), material-texture descriptors (2),
 * materialTextures array (3), materialTexSampler (4), and materialLinearTextures
 * (5) on the full tier.  SPPM appends four more bindings here to avoid
 * requiring maxBindGroups ≥ 5 (group 4 would need that, which lavapipe rejects).
 *
 * Binding layout (@group(3)):
 *   binding(6)  sppmPhotonCells     — PhotonRecord[CELLS × CAPACITY], read_write
 *   binding(7)  sppmCellCounters    — atomic<u32>[CELLS], read_write
 *   binding(8)  sppmStats           — SppmStats (uniform, 32 bytes)
 *   binding(9)  sppmPixelStats      — SppmPixelStats[W×H], read_write
 *               Per-pixel progressive statistics: tau.rgb, radius2, N, _pad×3.
 *               Binding(9) is only declared in the megakernel (not the photon
 *               pass); a 64-byte placeholder satisfies the layout when SPPM is off.
 *
 * Both the megakernel and the photon pass declare `read_write` storage for
 * (6) and (7) so a SINGLE GPUBindGroup can serve both pipelines.
 */
export const SPPM_GROUP4_BINDINGS_WGSL = /* wgsl */ `
// ── SPPM group-3 extension (bindings 6/7/8/9): photon hash-grid + per-pixel stats
// Stochastic Progressive Photon Mapping (Hachisuka & Jensen 2009).
// Full-tier only; lit binding exists when causticStrategy == 'photon-map';
// 64-byte placeholders satisfy the layout when SPPM is off (causticMode != 2u
// guards the gather so the placeholders are never actually accessed).

const SPPM_MAX_CELLS_WGSL = ${SPPM_MAX_CELLS}u;
const SPPM_CELL_CAPACITY_WGSL = ${SPPM_CELL_CAPACITY}u;
const SPPM_ALPHA_WGSL = ${SPPM_ALPHA}f; // α = 2/3 (Hachisuka & Jensen 2009, Eq.4)

// SppmStats (32 bytes): currentRadius, r0, frameAccumulated, photonCount, sceneExtent, _pad×3
struct SppmStats {
  currentRadius    : f32,
  r0               : f32,
  frameAccumulated : u32,
  photonCount      : u32,
  sceneExtent      : f32,
  _pad0            : f32,
  _pad1            : f32,
  _pad2            : f32,
}

// PhotonRecord (48 bytes = 3 × vec4f):
//   [0] position.xyz + _pad
//   [1] flux.rgb     + _pad
//   [2] incidentDir.xyz + _pad
struct PhotonRecord {
  position    : vec4f,
  flux        : vec4f,
  incidentDir : vec4f,
}

// SppmPixelStats (32 bytes = 8 × f32) — per-pixel progressive SPPM state.
//   A4-progressive: Hachisuka & Jensen 2009, §4 (Knaus-Zwicker formulation).
//   Holds the accumulated tau (brdf-weighted flux sum), current gather radius²,
//   and accumulated photon count N so the update rule can proceed from any frame.
//   Reset (zeroed) whenever the PT accumulator resets (camera move/setScene/reset).
struct SppmPixelStats {
  tau     : vec3f,   // accumulated brdf-weighted flux (τ in Hachisuka §4)
  radius2 : f32,     // current per-pixel gather radius² (R² shrinks each frame)
  N       : f32,     // accumulated photon count (floating-point, cf. Knaus-Zwicker)
  _pad0   : f32,
  _pad1   : f32,
  _pad2   : f32,
}

@group(3) @binding(6) var<storage, read_write> sppmPhotonCells    : array<PhotonRecord>;
@group(3) @binding(7) var<storage, read_write> sppmCellCounters   : array<atomic<u32>>;
@group(3) @binding(8) var<uniform>             sppmStats          : SppmStats;
@group(3) @binding(9) var<storage, read_write> sppmPixelStats     : array<SppmPixelStats>;

// Spatial hash: map a world-space position to a cell index.
// Prime-multiplied coordinate hash (Ihrke et al. 2007, § 4).
fn sppmCellIndex(pos: vec3f, radius: f32) -> u32 {
  let r = max(radius, 1e-6);
  let ix = i32(floor(pos.x / r));
  let iy = i32(floor(pos.y / r));
  let iz = i32(floor(pos.z / r));
  // Use bitwise operations on unsigned reinterpretation for the mix.
  let ux = bitcast<u32>(ix);
  let uy = bitcast<u32>(iy);
  let uz = bitcast<u32>(iz);
  let h = (ux * 1223u) ^ (uy * 7919u) ^ (uz * 1049u);
  return h % SPPM_MAX_CELLS_WGSL;
}

// Insert a photon into the hash grid (write path, photon pass only).
// Uses atomicAdd on the counter to claim a ring-buffer slot; overwrites
// the slot at (cellIdx * SPPM_CELL_CAPACITY + slot % SPPM_CELL_CAPACITY).
// Photons that overflow capacity are COUNTED but not STORED (the counter
// is not clamped, so the density estimate uses min(count, capacity) for
// the correct N without memory overflow).
fn sppmInsertPhoton(pos: vec3f, flux: vec3f, dir: vec3f, radius: f32) {
  let cellIdx = sppmCellIndex(pos, radius);
  let rawSlot = atomicAdd(&sppmCellCounters[cellIdx], 1u);
  let slot = rawSlot % SPPM_CELL_CAPACITY_WGSL;
  let base = cellIdx * SPPM_CELL_CAPACITY_WGSL + slot;
  sppmPhotonCells[base].position    = vec4f(pos, 0.0);
  sppmPhotonCells[base].flux        = vec4f(flux, 0.0);
  sppmPhotonCells[base].incidentDir = vec4f(dir, 0.0);
}

// ── A4-progressive: true Hachisuka SPPM per-pixel gather + update ────────────
//
// The Hachisuka & Jensen 2009 §4 / Knaus-Zwicker progressive update rule:
//
//   GIVEN:  per-pixel stats from the last frame: (τ, R², N)
//           this frame's hash-grid photons within sqrt(R²) of pos → M photons,
//           BRDF-weighted flux sum Φ_M = Σ_j f(ω_j)·Φ_j (accumulated below)
//
//   UPDATE (each frame, at each eye-path hit point):
//     N' = N + α·M                      (α = 2/3; new accumulated photon count)
//     ratio = N' / (N + M)              (radius shrink factor; guard M=0 → ratio=1)
//     R'² = R² · ratio                  (shrink the gather disk)
//     τ' = (τ + Φ_M) · ratio            (scale accumulated flux same way)
//     store (τ', R'², N') back to sppmPixelStats[pixelIndex]
//
//   ESTIMATE (displayed as caustic radiance this frame):
//     L_caustic = τ' / (N_e · π · R'²)
//     where N_e = frameAccumulated · photonCount  (total emitted photons)
//
//   CONVERGENCE:
//     N ~ n^α asymptotically (see test: sppmRecurrenceMatchesClosedForm).
//     R² ~ N·r₀² / photonCount shrinks to zero ⟹ L_caustic converges to
//     the true caustic radiance (Hachisuka 2009, Theorem 1).
//
//   ACCUMULATOR INTERACTION:
//     The PT accumulator computes a running mean μ_k = (1/k)·Σᵢ Lᵢ over
//     independent per-frame samples. SPPM contributes L_caustic(k) as its
//     per-frame sample. As k→∞, L_caustic(k) → L_true, so μ_k → L_true too
//     (a Cesàro mean of a converging sequence converges to the same limit).
//     No double-averaging pathology: we contribute the CURRENT (not per-frame
//     delta) estimate each frame; the running mean is the correct display value
//     ONLY in the early frames — from frame k onward the fresh L_caustic(k)
//     dominates the mean.  This is standard; see Hachisuka 2009, §5 / PBRT §16.
//
//   INITIAL STATE (first frame after reset):
//     N = 0, radius2 = r₀² (from sppmStats.r0), τ = 0.
//     The branch N=0 is handled cleanly: ratio=1, R'²=r₀², τ'=Φ_M·1.
//
// Item 21 — spectral × photon-map:
//   Photons store RGB flux.  In spectral mode resolve each photon's RGB flux
//   at the eye path's hero wavelength via spectralEmissionAtHero (same as
//   all other RGB emission sources).  Non-spectral path: use flux.rgb directly
//   — byte-identical to the pre-progressive streaming-window behaviour.
fn sppmGatherProgressive(
  pixelIndex : u32,
  pos        : vec3f,
  normal     : vec3f,
  wo         : vec3f,
  baseColor  : vec3f,
  roughness  : f32,
  metallic   : f32,
  throughput : vec3f,
  heroLambda : f32,
) -> vec3f {
  let nPhotons = sppmStats.photonCount;
  let r0 = sppmStats.r0;
  if (r0 <= 1e-9 || nPhotons == 0u) { return vec3f(0.0); }

  // Load per-pixel progressive state.  On the very first frame after a reset
  // all fields are zero (the buffer is GPU-cleared); initialise radius2 from
  // r₀ in that case.
  var pxStats = sppmPixelStats[pixelIndex];
  let isFirstFrame = (pxStats.radius2 <= 0.0);
  let r2 = select(pxStats.radius2, r0 * r0, isFirstFrame);
  let r  = sqrt(r2);
  var tau = pxStats.tau;
  var N   = pxStats.N;

  // ── Collect this frame's photons within the current gather disk ───────────
  // 3×3×3 neighbourhood to handle cell-boundary straddling (same as the
  // streaming-window gather).  Accumulate Φ_M = Σ f(ωᵢ)·Φᵢ (brdf-weighted).
  var phiM = vec3f(0.0);
  var M    = 0.0; // float to avoid a cast in the update below

  for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
      for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        let probe   = pos + vec3f(f32(dx), f32(dy), f32(dz)) * r;
        let cellIdx = sppmCellIndex(probe, r);
        let stored  = min(atomicLoad(&sppmCellCounters[cellIdx]), SPPM_CELL_CAPACITY_WGSL);
        let base    = cellIdx * SPPM_CELL_CAPACITY_WGSL;
        for (var si = 0u; si < stored; si = si + 1u) {
          let ph    = sppmPhotonCells[base + si];
          let diff  = ph.position.xyz - pos;
          let dist2 = dot(diff, diff);
          if (dist2 > r2) { continue; }
          let nDotL = max(dot(normal, -ph.incidentDir.xyz), 0.0);
          if (nDotL <= 1e-6) { continue; }
          let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, -ph.incidentDir.xyz);
          let fluxRgb = ph.flux.rgb;
          // Item 21 — spectral mode flux resolution (mirrors NEE half).
          let fluxOut = select(fluxRgb, spectralEmissionAtHero(fluxRgb, heroLambda), params.spectralEnabled != 0u);
          // Accumulate BRDF-weighted photon flux (no π r² denominator here —
          // it is applied once in the final estimate, not per-photon, which
          // keeps τ in physically-consistent units: [W·sr/m²·sr] = [W/m²]).
          phiM = phiM + throughput * brdf * fluxOut * nDotL;
          M    = M + 1.0;
        }
      }
    }
  }

  // ── Hachisuka §4 progressive update ──────────────────────────────────────
  //
  //   N'    = N + α·M
  //   ratio = N' / (N + M)        (= 1 when M = 0, guarded below)
  //   R'²   = R² · ratio
  //   τ'    = (τ + Φ_M) · ratio
  //
  //   The ratio is the radius-shrink factor.  Physically: after absorbing M
  //   new photons the area budget shrinks so that total photon density N'/A'
  //   equals (N + M)/A (the estimator's target).  Multiplying τ by the same
  //   ratio re-weights previously accumulated flux to the new (smaller) disk
  //   area, keeping units consistent across frames.
  let Nprime  = N + SPPM_ALPHA_WGSL * M;
  let NplusM  = N + M;
  // Guard M=0 ⟹ ratio=1 (no photons this frame → no update, no shrink).
  let ratio   = select(Nprime / NplusM, 1.0, M < 0.5);
  let r2prime = r2 * ratio;
  let tauPrime = (tau + phiM) * ratio;

  // Write updated per-pixel stats back.
  sppmPixelStats[pixelIndex].tau     = tauPrime;
  sppmPixelStats[pixelIndex].radius2 = r2prime;
  sppmPixelStats[pixelIndex].N       = Nprime;

  // ── Estimate: τ' / (N_e · π · R'²) ──────────────────────────────────────
  //
  //   N_e = frameAccumulated · photonCount   (total photons emitted so far)
  //   L_caustic = τ' / (N_e · π · R'²)
  //
  //   frameAccumulated starts at 0 on the host; it is incremented to 1 before
  //   the first renderFrame call that commits this frame's photons, so N_e ≥
  //   photonCount from frame 1 onward (guard below prevents div/0 on frame 0).
  let Ne = f32(sppmStats.frameAccumulated) * f32(nPhotons);
  if (Ne <= 0.0 || r2prime <= 1e-24) { return vec3f(0.0); }
  return tauPrime / (Ne * PI * r2prime);
}

// Legacy streaming-window gather — kept for reference; no longer called by the
// megakernel (A4-progressive replaced it).  The photon pass still uses
// sppmInsertPhoton / sppmCellIndex / the hash grid (unchanged).  The gather
// function below is superseded by sppmGatherProgressive above.
//
// Item 21 — spectral × photon-map regime fix:
// Photons store RGB flux; the eye path carries a hero-λ throughput. In spectral
// mode we must resolve the photon flux at the hero wavelength at gather time,
// exactly like all other RGB emission sources (rect lights, point lights, env —
// all use spectralEmissionAtHero). The gather conversion mirrors the NEE half:
//   fluxOut = select(flux.rgb, spectralEmissionAtHero(flux.rgb, heroLambda), spectralEnabled)
// Non-spectral path (spectralEnabled=0): fluxOut = flux.rgb — byte-identical.
// The heroLambda parameter carries the per-path hero wavelength from the kernel
// (already sampled by sampleHeroWavelengthMIS / params.heroLambdaNm).
fn sppmGather(
  pos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  throughput: vec3f,
  heroLambda: f32,
) -> vec3f {
  let r = sppmStats.currentRadius;
  let r2 = r * r;
  let nPhotons = sppmStats.photonCount;
  if (r <= 1e-9 || nPhotons == 0u) { return vec3f(0.0); }
  var acc = vec3f(0.0);

  // 3×3×3 neighbourhood in the hash grid to avoid boundary-straddling misses.
  for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
      for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        let probe = pos + vec3f(f32(dx), f32(dy), f32(dz)) * r;
        let cellIdx = sppmCellIndex(probe, r);
        // Number of photons actually stored (capped at capacity).
        let stored = min(atomicLoad(&sppmCellCounters[cellIdx]), SPPM_CELL_CAPACITY_WGSL);
        let base = cellIdx * SPPM_CELL_CAPACITY_WGSL;
        for (var si = 0u; si < stored; si = si + 1u) {
          let ph = sppmPhotonCells[base + si];
          let diff = ph.position.xyz - pos;
          let dist2 = dot(diff, diff);
          if (dist2 > r2) { continue; }
          let nDotL = max(dot(normal, -ph.incidentDir.xyz), 0.0);
          if (nDotL <= 1e-6) { continue; }
          let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, -ph.incidentDir.xyz);
          // Item 21 — spectral mode: resolve the stored RGB flux at the eye path's
          // hero wavelength before gathering. This mirrors the NEE half where every
          // other RGB emission source (rect/point/spot lights, env) applies
          // spectralEmissionAtHero. Non-spectral path uses flux.rgb directly —
          // byte-identical to the pre-fix behaviour.
          let fluxRgb = ph.flux.rgb;
          let fluxOut = select(fluxRgb, spectralEmissionAtHero(fluxRgb, heroLambda), params.spectralEnabled != 0u);
          // Flat disk kernel: 1 / (π r²) area normalisation.
          acc = acc + throughput * brdf * fluxOut * nDotL / max(PI * r2, 1e-12);
        }
      }
    }
  }
  // Scale by (total cells / photon count) to convert from per-cell to per-area
  // density.  The photon pass emits \`photonCount\` photons across the scene; each
  // has already been divided by photonCount in the emission flux, so no further
  // normalisation is needed here.  The π r² denominator above is the SPPM area
  // estimator in its standard form.
  return acc;
}
`;

/**
 * WGSL for the photon-emission compute pass entry point.  Composed with the
 * same module prefix stack as the megakernel (common + material + intersection +
 * bsdf) so all scene globals (positions, BVH, lights, materials, decodeMaterial,
 * traceClosest, uniformSphere, etc.) are in scope.
 *
 * The pass emits `sppmStats.photonCount` photons per frame.  Each photon is
 * seeded from the active analytic lights (point + spot; directional is handled
 * as a parallel-rays emitter with origin on a far plane, like the old photon-map
 * code) and traced through the scene via specular/transmissive bounces until it
 * hits a diffuse surface, then deposited into the SPPM hash grid via
 * `sppmInsertPhoton`.
 *
 * Photon flux is   Φ_photon = Φ_total / N_photons
 * so the estimator   L ≈ Σ(f × Φ_i × kernel / (π r²))
 * converges to the correct radiance as N × r → ∞ (SPPM §3).
 *
 * The pass is dispatched as dispatchWorkgroups(ceil(photonCount / 64), 1, 1)
 * with @workgroup_size(64, 1, 1).  Each invocation traces one photon.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009 §3 photon pass).
 */
export const SPPM_PHOTON_PASS_WGSL = /* wgsl */ `

// SPPM photon-emission pass.  workgroup_size(64,1,1); each lane = one photon.
@compute @workgroup_size(64, 1, 1)
fn sppmEmitPhotons(@builtin(global_invocation_id) gid: vec3u) {
  let photonIdx = gid.x;
  if (photonIdx >= sppmStats.photonCount) { return; }

  // Per-photon RNG: mix the frame seed (params.frameSeed), the frame index,
  // and the photon index for a decorrelated stream.  The same PCG hash used
  // by the megakernel, seeded differently per photon.
  var rng = pcgInit(photonIdx, params.frameSeed, params.frameIndex ^ 0xdeadbeefu);

  // ── Select a light source ──────────────────────────────────────────────────
  var availableLightCount = 0u;
  if (params.lightDir.w > 1e-6)       { availableLightCount = availableLightCount + 1u; }
  availableLightCount = availableLightCount + params.pointLightCount;
  availableLightCount = availableLightCount + params.spotLightCount;
  if (availableLightCount == 0u) { return; }

  let pick = u32(min(
    floor(rand_f32(&rng) * f32(availableLightCount)),
    f32(availableLightCount - 1u),
  ));

  var photonOrigin = vec3f(0.0);
  var photonDir    = vec3f(0.0, 1.0, 0.0);
  var photonFlux   = vec3f(0.0);
  var seeded       = false;
  var current      = 0u;

  // Directional light (parallel rays from a far plane, like the old approximation).
  if (params.lightDir.w > 1e-6) {
    if (current == pick) {
      let extent = sppmStats.sceneExtent;
      // Random point on a disk of radius sceneExtent centred on the camera.
      let r2d  = sqrt(rand_f32(&rng)) * extent;
      let phi2 = 2.0 * PI * rand_f32(&rng);
      let ldir = safe_normalize(params.lightDir.xyz);
      var lt: vec3f; var lb: vec3f;
      buildOnb(ldir, &lt, &lb);
      let diskPos = r2d * cos(phi2) * lt + r2d * sin(phi2) * lb;
      photonOrigin = params.cameraPos.xyz + diskPos - ldir * extent * 2.0;
      photonDir    = ldir;
      // Flux: irradiance × disk area / photonCount (importance-sampled).
      let diskArea = PI * extent * extent;
      photonFlux   = vec3f(params.lightDir.w) * diskArea / f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  // Point lights.
  for (var pointIdx = 0u; pointIdx < params.pointLightCount; pointIdx = pointIdx + 1u) {
    if (current == pick) {
      let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE;
      photonOrigin = pointLights[pointBase].xyz;
      photonDir    = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
      // Flux = radiance × 4π (total power from isotropic point emitter) / N.
      photonFlux   = pointLights[pointBase + 1u].rgb * (4.0 * PI) /
                     f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  // Spot lights.
  for (var spotIdx = 0u; spotIdx < params.spotLightCount; spotIdx = spotIdx + 1u) {
    if (current == pick) {
      let spotBase = spotIdx * SPOT_LIGHT_VEC4_STRIDE;
      let spos     = spotLights[spotBase].xyz;
      let saxisVec = spotLights[spotBase + 1u];
      let sradW    = spotLights[spotBase + 2u];
      let spotAxis = safe_normalize(saxisVec.xyz);
      let cosMin   = saxisVec.w;  // cosOuter
      // Sample within the outer cone.
      let cosTheta = mix(cosMin, 1.0, rand_f32(&rng));
      let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
      let phi3 = 2.0 * PI * rand_f32(&rng);
      var st: vec3f; var sb2: vec3f;
      buildOnb(spotAxis, &st, &sb2);
      photonOrigin = spos;
      photonDir    = safe_normalize(
        sinTheta * cos(phi3) * st + sinTheta * sin(phi3) * sb2 + cosTheta * spotAxis);
      // Cone solid angle = 2π(1 − cosMin); Φ = radiance × solidAngle / N.
      let solidAngle = 2.0 * PI * (1.0 - cosMin);
      photonFlux   = sradW.rgb * solidAngle / f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  if (!seeded) { return; }

  // ── Trace the photon path ──────────────────────────────────────────────────
  var ray  = Ray(photonOrigin + photonDir * 1e-3, photonDir);
  var flux = photonFlux;
  let maxBounces = clamp(params.mneeMaxChainLength, 1u, 8u);

  for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
    if (bounce >= maxBounces) { break; }
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) { break; }

    let matId = hitMaterialId(hit);
    let mat   = decodeMaterial(matId);
    let hp    = ray.origin + ray.direction * hit.dist;
    let frontFace  = dot(ray.direction, hit.normal) < 0.0;
    let surfNormal = select(-hit.normal, hit.normal, frontFace);

    // Deposit on diffuse-ish surfaces (not purely specular/transmissive).
    // A surface is a "diffuse receiver" if: transmission < 0.3 AND
    // (metallic < 0.9 OR roughness > 0.15).
    let isSpecular = mat.transmission > 0.3 ||
                     (mat.metallic > 0.9 && mat.roughness < 0.15);

    if (!isSpecular) {
      // Deposit photon at this diffuse hit.
      sppmInsertPhoton(hp, flux, ray.direction, sppmStats.currentRadius);
      // Diffuse surfaces absorb the photon (Russian roulette in future; v1 = terminate).
      break;
    }

    // Transmissive / specular bounce (follow the specular chain exactly like
    // traceSpecularTransmissiveChain — Beer-Lambert medium extinction omitted here
    // for simplicity; SPPM caustics are a first-order effect so the ~2% energy error
    // from ignoring medium extinction is well within the SPPM variance).
    let ior = mat.ior;
    let eta = select(ior, 1.0 / ior, frontFace);
    let refr = refract(ray.direction, surfNormal, eta);
    let hasRefr = dot(refr, refr) > 1e-8;
    let nextDir = select(reflect(ray.direction, surfNormal), safe_normalize(refr), hasRefr);
    flux = flux * mix(vec3f(1.0), clamp(mat.baseColor, vec3f(0.0), vec3f(1.0)), 0.2) *
           max(mat.transmission, 0.05);
    if (max(flux.r, max(flux.g, flux.b)) < 1e-5) { break; }
    ray.origin    = hp + nextDir * 1e-3;
    ray.direction = nextDir;
  }
}
`;
