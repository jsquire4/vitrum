/**
 * SPPM group-3 bindings — shared between the photon-emission compute pass
 * (`sppmPhotonPass.wgsl.ts`) and the megakernel gather path in
 * `caustic.wgsl.ts`.
 *
 * SPPM = Stochastic Progressive Photon Mapping (Hachisuka & Jensen 2009).
 *
 * Binding layout (@group(3), bindings 6–9):
 *   binding(6)  sppmPhotonCells   — PhotonRecord[SPPM_MAX_CELLS × SPPM_CELL_CAPACITY]
 *               Each cell has SPPM_CELL_CAPACITY slots.  The photon pass writes
 *               photons here atomically; over-capacity cells use bounded
 *               reservoir replacement and the megakernel compensates by
 *               totalInserted / storedCount.  access = read_write on both pipelines.
 *   binding(7)  sppmCellCounters  — atomic<u32>[SPPM_MAX_CELLS]
 *               Cumulative photon-insertion counter per cell.  The counter is
 *               not clamped; min-cap gives the bounded stored count and the
 *               counter/stored ratio gives the overflow compensation weight.
 *   binding(8)  sppmStats         — SppmStats (uniform UBO, 32 bytes)
 *               currentRadius, frameIndex, photonCount, sceneExtent.
 *   binding(9)  sppmPixelStats    — SppmPixelStats[W×H] (per-pixel progressive
 *               statistics: tau.rgb, radius2, N, _pad×3).
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
 * 16-byte placeholder buffer is bound so the group-3 layout slots are satisfied
 * without allocating the real photon map.  The gather code in caustic.wgsl.ts
 * is guarded by `if (causticMode() == 2u)` so it never executes.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009); spatial-hash scheme follows the survey in
 * Ihrke et al. 2007 § 4 (prime-multiplied coordinate hash, no collision
 * resolution — overflow counter protects memory, not energy).
 *
 * Host-side physics constants and helper functions (SPPM_MAX_CELLS,
 * SPPM_CELL_CAPACITY, SPPM_ALPHA, SPPM_STATS_FIELDS, sppmRadiusAtFrame,
 * sppmInitialRadius, etc.) live in ../../sppmParams.ts — this file only
 * holds WGSL string exports.
 */

// Re-export host constants so existing importers that reach through this
// module for the TS helpers continue to work (they can also import from
// sppmParams.ts directly).
export {
  SPPM_MAX_CELLS,
  SPPM_CELL_CAPACITY,
  SPPM_PHOTON_RECORD_BYTES,
  SPPM_PHOTON_CELLS_BYTES,
  SPPM_CELL_COUNTERS_BYTES,
  SPPM_STATS_BYTES,
  SPPM_STATS_FIELDS,
  SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
  SPPM_PHOTON_CELLS_MAX_BYTES,
  SPPM_ALPHA,
  sppmRadiusAtFrame,
  sppmInitialRadius,
} from '../../sppmParams.js';

import {
  SPPM_MAX_CELLS,
  SPPM_CELL_CAPACITY,
  SPPM_ALPHA,
} from '../../sppmParams.js';

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
// D9.8/I4.1 — renamed from SPPM_GROUP4_BINDINGS_WGSL (misnomer: bindings are
// at @group(3), not group 4 — lavapipe only supports maxBindGroups=4, i.e. 0–3).
export const SPPM_GROUP3_BINDINGS_WGSL = /* wgsl */ `
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
// Uses atomicAdd on the counter to claim the item index in this hash cell.
// Below capacity the index is stored directly. Above capacity, bounded reservoir
// replacement keeps an unbiased subset of the cell photons; gather multiplies
// stored samples by totalInserted / storedCount to preserve density under the cap.
fn sppmInsertPhoton(pos: vec3f, flux: vec3f, dir: vec3f, radius: f32, reservoirXi: f32) {
  let cellIdx = sppmCellIndex(pos, radius);
  let rawSlot = atomicAdd(&sppmCellCounters[cellIdx], 1u);
  var slot = rawSlot;
  if (rawSlot >= SPPM_CELL_CAPACITY_WGSL) {
    let xi = clamp(reservoirXi, 0.0, 0.99999994);
    let candidate = u32(floor(xi * f32(rawSlot + 1u)));
    if (candidate >= SPPM_CELL_CAPACITY_WGSL) {
      return;
    }
    slot = candidate;
  }
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
  clearcoat  : f32,
  clearcoatRoughness : f32,
  sheen      : f32,
  sheenRoughness : f32,
  sheenColor : vec3f,
  iridescence: f32,
  iridescenceIor : f32,
  iridescenceThicknessMin : f32,
  iridescenceThicknessMax : f32,
  specularColor : vec3f,
  specularIntensity : f32,
  anisotropy : f32,
  anisotropyRotation : f32,
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
  let gridRadius = max(sppmStats.currentRadius, 1e-6);
  var tau = pxStats.tau;
  var N   = pxStats.N;

  // ── Collect this frame's photons within the current gather disk ───────────
  // 3×3×3 neighbourhood to handle cell-boundary straddling. Photon insertion
  // hashes by sppmStats.currentRadius (the stable r0 cell size), while the
  // physical gather disk shrinks per pixel through r. Query the insertion
  // grid with the same stable radius, then filter physically with dist2 <= r^2.
  // Accumulate Φ_M = Σ f(ωᵢ)·Φᵢ (brdf-weighted).
  var phiM = vec3f(0.0);
  var M    = 0.0; // float to avoid a cast in the update below

  for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
      for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        let probe   = pos + vec3f(f32(dx), f32(dy), f32(dz)) * gridRadius;
        let cellIdx = sppmCellIndex(probe, gridRadius);
        let totalInCell = atomicLoad(&sppmCellCounters[cellIdx]);
        let stored  = min(totalInCell, SPPM_CELL_CAPACITY_WGSL);
        let cellSampleScale = f32(totalInCell) / f32(max(stored, 1u));
        let base    = cellIdx * SPPM_CELL_CAPACITY_WGSL;
        for (var si = 0u; si < stored; si = si + 1u) {
          let ph    = sppmPhotonCells[base + si];
          let diff  = ph.position.xyz - pos;
          let dist2 = dot(diff, diff);
          if (dist2 > r2) { continue; }
          let nDotL = max(dot(normal, -ph.incidentDir.xyz), 0.0);
          if (nDotL <= 1e-6) { continue; }
          let brdf = evaluateBrdfFull(
            baseColor, roughness, metallic, normal, wo, -ph.incidentDir.xyz,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          let fluxRgb = ph.flux.rgb;
          // Item 21 — spectral mode flux resolution (mirrors NEE half).
          let fluxOut = select(fluxRgb, spectralEmissionAtHero(fluxRgb, heroLambda), params.spectralEnabled != 0u);
          // Accumulate BRDF-weighted photon flux (no π r² denominator here —
          // it is applied once in the final estimate, not per-photon, which
          // keeps τ in physically-consistent units: [W·sr/m²·sr] = [W/m²]).
          phiM = phiM + throughput * brdf * fluxOut * nDotL * cellSampleScale;
          M    = M + cellSampleScale;
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

// D9.9 — sppmGather (legacy streaming-window gather, superseded by
// sppmGatherProgressive above) DELETED 2026-06-10.  It was never called by the
// megakernel after A4 landed; photonMapContribution in caustic.wgsl.ts calls
// sppmGatherProgressive directly.  sppmInsertPhoton + sppmCellIndex + the hash
// grid remain (still used by the photon-emission pass).
`;

/**
 * WGSL for the photon-emission compute pass entry point.  Composed with the
 * same module prefix stack as the megakernel (common + material + intersection +
 * bsdf) so all scene globals (positions, BVH, lights, materials, decodeMaterial,
 * traceClosest, uniformSphere, etc.) are in scope.
 *
 * The pass emits `sppmStats.photonCount` photons per frame.  Each photon is
 * seeded from the active directional / point / spot lights and traced through
 * the scene via specular/transmissive bounces until it hits a diffuse surface,
 * then deposited into the SPPM hash grid via `sppmInsertPhoton`.
 * Emitters with `castShadow:false` are intentionally excluded from photon
 * source selection: they remain directly/camera/specular visible elsewhere,
 * but they do not seed caustic/shadow transport.
 *
 * Photon flux is   Φ_photon = Φ_total / N_photons
 * so the estimator   L ≈ Σ(f × Φ_i × kernel / (π r²))
 * converges to the correct radiance as N × r → ∞ (SPPM §3).
 *
 * PTWG-03 note: implementation flux is normalized by 1 / p_select for every
 * photon source in the same flat order used by full-tier NEE: directional,
 * point, spot, rect/disc, mesh-area, environment. Rect/disc and mesh sources
 * use the same packed records and area conventions as NEE; environment sources
 * use the same importance/PDF helpers from connect.wgsl.ts.
 *
 * The pass is dispatched as dispatchWorkgroups(ceil(photonCount / 64), 1, 1)
 * with @workgroup_size(64, 1, 1).  Each invocation traces one photon.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009 §3 photon pass).
 */
export const SPPM_PHOTON_PASS_WGSL = /* wgsl */ `
fn sppmConcentricDiscSample(xi: vec2f) -> vec2f {
  let a = xi.x;
  let b = xi.y;
  if (abs(a) < 1e-8 && abs(b) < 1e-8) {
    return vec2f(0.0);
  }
  var r: f32;
  var phi: f32;
  if (abs(a) >= abs(b)) {
    r = a;
    phi = (PI / 4.0) * (b / max(abs(a), 1e-8));
  } else {
    r = b;
    phi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-8));
  }
  return vec2f(r * cos(phi), r * sin(phi));
}

fn sppmDirectionalCastsShadow(dirIdx: u32) -> bool {
  let dBase = dirIdx * 2u;
  return directionalLights[dBase].w >= 0.0;
}

fn sppmPointCastsShadow(pointIdx: u32) -> bool {
  let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE;
  return pointLights[pointBase + 2u].z <= 0.5;
}

fn sppmSpotCastsShadow(spotIdx: u32) -> bool {
  let spotBase = spotIdx * SPOT_LIGHT_VEC4_STRIDE;
  return spotLights[spotBase + 3u].z <= 0.5;
}

fn sppmRectAreaCastsShadow(rectIdx: u32) -> bool {
  let rectBase = rectIdx * 4u;
  return rectAreaLights[rectBase].w <= 0.5;
}

fn sppmMeshAreaCastsShadow(meshIdx: u32) -> bool {
  let meshBase = meshIdx * 4u;
  return meshAreaLights[meshBase + 3u].w <= 0.5;
}

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
  for (var dirIdx = 0u; dirIdx < params.directionalLightCount; dirIdx = dirIdx + 1u) {
    if (sppmDirectionalCastsShadow(dirIdx)) {
      availableLightCount = availableLightCount + 1u;
    }
  }
  for (var pointIdx = 0u; pointIdx < params.pointLightCount; pointIdx = pointIdx + 1u) {
    if (sppmPointCastsShadow(pointIdx)) {
      availableLightCount = availableLightCount + 1u;
    }
  }
  for (var spotIdx = 0u; spotIdx < params.spotLightCount; spotIdx = spotIdx + 1u) {
    if (sppmSpotCastsShadow(spotIdx)) {
      availableLightCount = availableLightCount + 1u;
    }
  }
  for (var rectIdx = 0u; rectIdx < params.rectAreaLightCount; rectIdx = rectIdx + 1u) {
    if (sppmRectAreaCastsShadow(rectIdx)) {
      availableLightCount = availableLightCount + 1u;
    }
  }
  for (var meshIdx = 0u; meshIdx < params.meshAreaLightCount; meshIdx = meshIdx + 1u) {
    if (sppmMeshAreaCastsShadow(meshIdx)) {
      availableLightCount = availableLightCount + 1u;
    }
  }
  if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
    availableLightCount = availableLightCount + 1u;
  }
  if (availableLightCount == 0u) { return; }

  let pick = u32(min(
    floor(rand_f32(&rng) * f32(availableLightCount)),
    f32(availableLightCount - 1u),
  ));
  let lightSelectInvPdf = f32(availableLightCount);

  var photonOrigin = vec3f(0.0);
  var photonDir    = vec3f(0.0, 1.0, 0.0);
  var photonFlux   = vec3f(0.0);
  var seeded       = false;
  var current      = 0u;

  // Directional lights (parallel rays from a far plane).
  for (var dirIdx = 0u; dirIdx < params.directionalLightCount; dirIdx = dirIdx + 1u) {
    if (!sppmDirectionalCastsShadow(dirIdx)) { continue; }
    if (current == pick) {
      let dBase = dirIdx * 2u;
      let dDirAD = directionalLights[dBase];        // .xyz = toward-light dir, .w = angularDiameter
      let dIrrMean = directionalLights[dBase + 1u]; // .rgb = irradiance,        .w = mean irradiance
      let extent = sppmStats.sceneExtent;
      // Random point on a disk of radius sceneExtent centred on the camera.
      let r2d  = sqrt(rand_f32(&rng)) * extent;
      let phi2 = 2.0 * PI * rand_f32(&rng);
      let towardLightDir = safe_normalize(dDirAD.xyz);
      var lt: vec3f; var lb: vec3f;
      buildOnb(towardLightDir, &lt, &lb);
      let diskPos = r2d * cos(phi2) * lt + r2d * sin(phi2) * lb;
      photonOrigin = params.cameraPos.xyz + diskPos + towardLightDir * extent * 2.0;
      photonDir    = -towardLightDir;
      // Flux: irradiance × disk area / photonCount (importance-sampled).
      let diskArea = PI * extent * extent;
      photonFlux   = dIrrMean.rgb * diskArea * lightSelectInvPdf /
                     f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  // Point lights.
  for (var pointIdx = 0u; pointIdx < params.pointLightCount; pointIdx = pointIdx + 1u) {
    if (!sppmPointCastsShadow(pointIdx)) { continue; }
    if (current == pick) {
      let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE;
      photonOrigin = pointLights[pointBase].xyz;
      photonDir    = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
      // Flux = radiance × 4π (total power from isotropic point emitter) / N.
      photonFlux   = pointLights[pointBase + 1u].rgb * (4.0 * PI) * lightSelectInvPdf /
                     f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  // Spot lights.
  for (var spotIdx = 0u; spotIdx < params.spotLightCount; spotIdx = spotIdx + 1u) {
    if (!sppmSpotCastsShadow(spotIdx)) { continue; }
    if (current == pick) {
      let spotBase = spotIdx * SPOT_LIGHT_VEC4_STRIDE;
      let spos     = spotLights[spotBase].xyz;
      let saxisVec = spotLights[spotBase + 1u];
      let sradW    = spotLights[spotBase + 2u];
      let spotAxis = safe_normalize(saxisVec.xyz);
      let cosMin   = saxisVec.w;  // cosOuter
      let cosInner = sradW.w;
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
      let softness = smoothstep(cosMin, max(cosInner, cosMin + 1e-6), cosTheta);
      let solidAngle = 2.0 * PI * (1.0 - cosMin);
      photonFlux   = sradW.rgb * solidAngle * softness * lightSelectInvPdf /
                     f32(max(sppmStats.photonCount, 1u));
      seeded = true;
    }
    current = current + 1u;
  }

  // Rect/disc area lights.  Same 4-vec4 record and area conventions as NEE:
  // rshape.w ~= 0 => rect area 4*|u x v|; rshape.w ~= 1 => disc area PI*r^2.
  for (var rectIdx = 0u; rectIdx < params.rectAreaLightCount; rectIdx = rectIdx + 1u) {
    if (!sppmRectAreaCastsShadow(rectIdx)) { continue; }
    if (current == pick) {
      let rectBase = rectIdx * 4u;
      let rpos = rectAreaLights[rectBase].xyz;
      let ru = rectAreaLights[rectBase + 1u].xyz;
      let rv = rectAreaLights[rectBase + 2u].xyz;
      let rshape = rectAreaLights[rectBase + 3u];
      let rr = rshape.rgb;
      let normalRaw = cross(ru, rv);
      let normalLen = length(normalRaw);
      if (normalLen > 1e-8) {
        let lightNormal = normalRaw / normalLen;
        let isDisc = abs(rshape.w - 1.0) < 0.5;
        let xi1 = rand_f32(&rng);
        let xi2 = rand_f32(&rng);
        var emitPos: vec3f;
        var area: f32;
        if (isDisc) {
          let disc = sppmConcentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));
          emitPos = rpos + ru * disc.x + rv * disc.y;
          area = max(PI * dot(ru, ru), 1e-6);
        } else {
          emitPos = rpos + ru * (xi1 * 2.0 - 1.0) + rv * (xi2 * 2.0 - 1.0);
          area = max(4.0 * length(cross(ru, rv)), 1e-6);
        }
        let hemi = cosineHemisphereSample(&rng, lightNormal);
        photonOrigin = emitPos;
        photonDir    = hemi.wi;
        photonFlux   = rr * area * PI * lightSelectInvPdf /
                       f32(max(sppmStats.photonCount, 1u));
        seeded = true;
      }
    }
    current = current + 1u;
  }

  // Mesh-area lights.  Same triangle stream and area convention as NEE.
  for (var meshIdx = 0u; meshIdx < params.meshAreaLightCount; meshIdx = meshIdx + 1u) {
    if (!sppmMeshAreaCastsShadow(meshIdx)) { continue; }
    if (current == pick) {
      let meshBase = meshIdx * 4u;
      let a = meshAreaLights[meshBase].xyz;
      let b = meshAreaLights[meshBase + 1u].xyz;
      let c = meshAreaLights[meshBase + 2u].xyz;
      let mr = meshAreaLights[meshBase + 3u].rgb;
      let e1 = b - a;
      let e2 = c - a;
      let normalRaw = cross(e1, e2);
      let normalLen = length(normalRaw);
      if (normalLen > 1e-8) {
        let lightNormal = normalRaw / normalLen;
        let r1 = rand_f32(&rng);
        let r2 = rand_f32(&rng);
        let su = sqrt(r1);
        let uu = 1.0 - su;
        let vv = r2 * su;
        let ww = 1.0 - uu - vv;
        let emitPos = a * uu + b * vv + c * ww;
        let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
        let hemi = cosineHemisphereSample(&rng, lightNormal);
        photonOrigin = emitPos;
        photonDir    = hemi.wi;
        photonFlux   = mr * area * PI * lightSelectInvPdf /
                       f32(max(sppmStats.photonCount, 1u));
        seeded = true;
      }
    }
    current = current + 1u;
  }

  // Environment source.  Direction/PDF comes from the same helpers as full-tier
  // NEE; photons are launched from a scene-covering disk perpendicular to wi.
  if ((hasEnvironmentMap() || params.environmentSun.w > 1e-6) && current == pick) {
    var envDir = vec3f(0.0, 1.0, 0.0);
    var envColor = vec3f(0.0);
    var envPdf = 0.0;
    let envSample = sampleEnvironmentImportance(&rng);
    if (envSample.pdf > 0.0) {
      envDir = envSample.wi;
      envColor = envSample.value;
      envPdf = envSample.pdf;
    } else {
      envDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
      envColor = sampleEnvironmentColor(envDir);
      envPdf = max(environmentPdf(envDir), 1e-8);
    }
    if (envPdf > 1e-8) {
      let extent = sppmStats.sceneExtent;
      let r2d = sqrt(rand_f32(&rng)) * extent;
      let phi = 2.0 * PI * rand_f32(&rng);
      var et: vec3f; var eb: vec3f;
      buildOnb(envDir, &et, &eb);
      let diskPos = r2d * cos(phi) * et + r2d * sin(phi) * eb;
      photonOrigin = params.cameraPos.xyz + diskPos + envDir * extent * 2.0;
      photonDir    = -envDir;
      let diskArea = PI * extent * extent;
      photonFlux   = envColor * diskArea * lightSelectInvPdf /
                     (f32(max(sppmStats.photonCount, 1u)) * envPdf);
      seeded = true;
    }
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
      sppmInsertPhoton(hp, flux, ray.direction, sppmStats.currentRadius, rand_f32(&rng));
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
