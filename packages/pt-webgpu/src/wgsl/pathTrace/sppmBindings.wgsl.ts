/**
 * SPPM group-3 bindings shared by the photon-emission pass and megakernel gather.
 *
 * Each emitted lane owns one PhotonRecord. Binding 7 is a per-frame-cleared
 * atomic bucket-head table: insertion stores the previous encoded head in
 * PhotonRecord.nextEncoded and publishes recordIndex + 1 with atomicExchange.
 * Zero is the empty/end sentinel, so record zero remains reachable. The gather
 * deduplicates its 27 hashed neighbourhood buckets, follows each linked chain
 * with index, cycle, and photon-count guards, and rejects candidates outside the
 * physical per-pixel radius.
 *
 * Binding layout (@group(3), bindings 6–9):
 *   binding(6) PhotonRecord[photonCount] — unique records and encoded next links
 *   binding(7) atomic<u32>[SPPM_MAX_CELLS] — per-frame bucket heads
 *   binding(8) SppmStats — radii, emitted count, scene extent and center
 *   binding(9) SppmPixelStats[W×H×2] — separate surface/volume state
 *
 * Both pipelines are full-tier only. Placeholders satisfy group-3 when SPPM is
 * inactive; causticMode() == 2u guards every real access.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009); the prime-multiplied spatial hash follows Ihrke et
 * al. 2007 §4.
 */

// Re-export host constants so existing importers that reach through this
// module for the TS helpers continue to work (they can also import from
// sppmParams.ts directly).
export {
  SPPM_MAX_CELLS,
  SPPM_PHOTON_COUNT,
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
  SPPM_PHOTON_COUNT,
  SPPM_ALPHA,
} from '../../sppmParams.js';

/**
 * SPPM group-3 WGSL bindings (bindings 6/7/8/9) — composed into the megakernel
 * AND the photon-emission pass.  Group 3 already carries the light-tree node
 * buffer (binding 0), mesh UVs (1), material-texture descriptors (2),
 * materialTextures array (3), materialTexSampler (4), materialLinearTextures
 * (5), and materialTexturesEmissive (17, T1-6 rgba16float HDR emissive) on the
 * full tier.  SPPM appends four more bindings here to avoid
 * requiring maxBindGroups ≥ 5 (group 4 would need that, which lavapipe rejects).
 *
 * Binding layout (@group(3)):
 *   binding(6)  sppmPhotonCells     — PhotonRecord[photonCount], read_write
 *   binding(7)  sppmCellCounters    — atomic<u32>[CELLS], read_write
 *   binding(8)  sppmStats           — SppmStats (uniform, 32 bytes)
 *   binding(9)  sppmPixelStats      — SppmPixelStats[W×H×2], read_write
 *               Two per-pixel records keep surface and volume measures apart.
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
const SPPM_PHOTON_COUNT_WGSL = ${SPPM_PHOTON_COUNT}u;
const SPPM_ALPHA_WGSL = ${SPPM_ALPHA}f; // α = 2/3 (Hachisuka & Jensen 2009, Eq.4)
// Photon-walk depth is an SPPM integrator constant. It must not inherit the
// unrelated MNEE manifold-chain length knob.
const SPPM_PHOTON_MAX_BOUNCES = 8u;

// A single wavelength sample is shared by the photon pass and all SPPM eye
// gathers in a frame. This is required for a photon flux value to be evaluated
// by the receiver BSDF at the same wavelength. The sample changes every frame,
// so RGB reconstruction of newly absorbed flux remains an unbiased spectral MC
// estimate while avoiding an impossible per-photon wavelength payload.
fn sppmFrameHeroSample() -> vec3f {
  var heroRng = pcgInit(
    0x51f15e1du,
    params.frameSeed ^ 0xa511e9b3u,
    ptRngFrameKey(params.frameSeed, params.frameIndex),
  );
  return sampleHeroWavelengthMIS(rand_f32(&heroRng), rand_f32(&heroRng));
}

// SppmStats (32 bytes): radius/count fields plus scene AABB extent and center.
struct SppmStats {
  currentRadius    : f32,
  r0               : f32,
  frameAccumulated : u32,
  photonCount      : u32,
  sceneExtent      : f32,
  sceneCenterX     : f32,
  sceneCenterY     : f32,
  sceneCenterZ     : f32,
}

// PhotonRecord (48 bytes = 3 × 16-byte lanes):
//   [0] position.xyz    + encoded linked-list head
//   [1] flux.rgb        + (kind:1 | mediumMatId:31)
//   [2] incidentDir.xyz + phase anisotropy g
// Surface and volume photons intentionally share the same hash grid. The two
// typed u32 lanes preserve every metadata/link bit exactly (including values
// whose f32 bit patterns would be NaN or subnormal) while keeping the original
// 48-byte storage footprint.
struct PhotonRecord {
  position    : vec3f,
  nextEncoded : u32,
  flux        : vec3f,
  metadata    : u32,
  incidentDir : vec3f,
  phaseG      : f32,
}

const SPPM_PHOTON_KIND_SURFACE = 0u;
const SPPM_PHOTON_KIND_VOLUME = 1u;
const SPPM_PHOTON_MEDIUM_MASK = 0x7fffffffu;

fn sppmPhotonMetadata(kind: u32, mediumMatId: u32) -> u32 {
  return ((kind & 1u) << 31u) | (mediumMatId & SPPM_PHOTON_MEDIUM_MASK);
}

fn sppmPhotonKind(ph: PhotonRecord) -> u32 {
  return ph.metadata >> 31u;
}

fn sppmPhotonMediumMatId(ph: PhotonRecord) -> u32 {
  return ph.metadata & SPPM_PHOTON_MEDIUM_MASK;
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

// Publish one photon into the per-frame hash grid. Each invocation owns the
// record at photonIndex, writes its payload exactly once, then atomically pushes
// photonIndex+1 onto the bucket head. Zero is the empty-list sentinel; the
// previous encoded head is stored in nextEncoded. The gather runs in a later
// compute pass, so all payload writes are visible before it follows the links.
fn sppmInsertPhoton(
  photonIndex: u32,
  pos: vec3f,
  flux: vec3f,
  dir: vec3f,
  radius: f32,
  kind: u32,
  mediumMatId: u32,
  phaseG: f32,
) {
  if (photonIndex >= sppmStats.photonCount ||
      photonIndex >= arrayLength(&sppmPhotonCells)) {
    return;
  }
  let cellIdx = sppmCellIndex(pos, radius);
  sppmPhotonCells[photonIndex].position = pos;
  sppmPhotonCells[photonIndex].flux = flux;
  sppmPhotonCells[photonIndex].metadata =
    sppmPhotonMetadata(kind, mediumMatId);
  sppmPhotonCells[photonIndex].incidentDir = dir;
  sppmPhotonCells[photonIndex].phaseG = phaseG;
  let previousHead = atomicExchange(&sppmCellCounters[cellIdx], photonIndex + 1u);
  sppmPhotonCells[photonIndex].nextEncoded = previousHead;
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
//     For stationary expected M, N grows linearly with iteration count and
//     R² follows the exact product recurrence, asymptotically n^(α-1).
//     The density estimate converges under the conditions of Hachisuka §4.
//
//   ACCUMULATOR INTERACTION:
//     The PT accumulator computes a running mean μ_k = (1/k)·Σᵢ Lᵢ over
//     independent per-frame samples. SPPM contributes L_caustic(k) as its
//     per-frame sample. As k→∞, L_caustic(k) → L_true, so μ_k → L_true too
//     (a Cesàro mean of a converging sequence converges to the same limit).
//     The renderer contributes the current cumulative estimate each frame; its
//     Cesàro mean has the same asymptotic limit.
//
//   INITIAL STATE (first frame after reset):
//     N = 0, radius2 = r₀² (from sppmStats.r0), τ = 0. For M>0 the
//     first update ratio is α; for M=0 the guarded ratio is one.
//
// Encoded heads use recordIndex + 1; zero is the empty/end sentinel.
fn sppmNextEncodedHead(encodedHead: u32, nPhotons: u32) -> u32 {
  if (encodedHead == 0u) { return 0u; }
  let recordIndex = encodedHead - 1u;
  if (recordIndex >= nPhotons ||
      recordIndex >= arrayLength(&sppmPhotonCells)) {
    return 0u;
  }
  return sppmPhotonCells[recordIndex].nextEncoded;
}
fn sppmUpdateProgressiveKind(
  pixelIndex : u32,
  pos        : vec3f,
  normal     : vec3f,
  clearcoatNormal : vec3f,
  wo         : vec3f,
  baseColor  : vec3f,
  roughness  : f32,
  metallic   : f32,
  transmission : f32,
  etaTOverI : f32,
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
  thinFilm : ThinFilmInterface,
  throughput : vec3f,
  heroLambda : f32,
  heroPdf : f32,
  absorbedFluxInvPdf : f32,
  gatherKind : u32,
  gatherMediumMatId : u32,
) {
  let nPhotons = sppmStats.photonCount;
  let r0 = sppmStats.r0;
  if (r0 <= 1e-9 || nPhotons == 0u) { return; }

  // Load per-pixel progressive state.  On the very first frame after a reset
  // all fields are zero (the buffer is GPU-cleared); initialise radius2 from
  // r₀ in that case.
  // Surface tau has area-density units; volume tau has volume-density units.
  // They must remain distinct across frames even though one eye path is allowed
  // to update only one of them in a frame.
  let statsIndex = pixelIndex * 2u + gatherKind;
  var pxStats = sppmPixelStats[statsIndex];
  let isFirstFrame = (pxStats.radius2 <= 0.0);
  let r2 = select(pxStats.radius2, r0 * r0, isFirstFrame);
  let r  = sqrt(r2);
  let gridRadius = max(sppmStats.currentRadius, 1e-6);
  var tau = pxStats.tau;
  var N   = pxStats.N;

  // ── Collect this frame's photons within the current gather support ────────
  // 3×3×3 neighbourhood to handle cell-boundary straddling. Photon insertion
  // hashes by sppmStats.currentRadius (the stable r0 cell size), while the
  // physical gather disk shrinks per pixel through r. Query the insertion
  // grid with the same stable radius, then filter physically with dist2 <= r^2.
  // Accumulate Φ_M = Σ f(ωᵢ)·Φᵢ (brdf-weighted).
  var phiM = vec3f(0.0);
  var M    = 0.0; // float to avoid a cast in the update below

  var visitedCells: array<u32, 27>;
  var visitedCellCount = 0u;
  for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
      for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        let probe   = pos + vec3f(f32(dx), f32(dy), f32(dz)) * gridRadius;
        let cellIdx = sppmCellIndex(probe, gridRadius);
        var duplicateBucket = false;
        for (var vi = 0u; vi < visitedCellCount; vi = vi + 1u) {
          if (visitedCells[vi] == cellIdx) {
            duplicateBucket = true;
          }
        }
        if (duplicateBucket) { continue; }
        visitedCells[visitedCellCount] = cellIdx;
        visitedCellCount = visitedCellCount + 1u;

        var encodedPhoton = atomicLoad(&sppmCellCounters[cellIdx]);
        var traversed = 0u;
        var cycleSlow = encodedPhoton;
        var cycleFast = encodedPhoton;
        loop {
          if (encodedPhoton == 0u || traversed >= nPhotons) { break; }
          let photonIndex = encodedPhoton - 1u;
          if (photonIndex >= nPhotons ||
              photonIndex >= arrayLength(&sppmPhotonCells)) {
            break;
          }
          let ph = sppmPhotonCells[photonIndex];
          encodedPhoton = ph.nextEncoded;
          traversed = traversed + 1u;
          cycleSlow = sppmNextEncodedHead(cycleSlow, nPhotons);
          cycleFast = sppmNextEncodedHead(
            sppmNextEncodedHead(cycleFast, nPhotons), nPhotons);
          let diff  = ph.position - pos;
          let dist2 = dot(diff, diff);
          let kindMatches = sppmPhotonKind(ph) == gatherKind;
          let mediumMatches = gatherKind == SPPM_PHOTON_KIND_SURFACE ||
            sppmPhotonMediumMatId(ph) == gatherMediumMatId;
          if (dist2 <= r2 && kindMatches && mediumMatches) {
            if (gatherKind == SPPM_PHOTON_KIND_SURFACE) {
              let nDotL = max(dot(normal, -ph.incidentDir), 0.0);
              if (nDotL > 0.0) {
              let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(
                baseColor, roughness, metallic, transmission, etaTOverI,
                normal, clearcoatNormal, wo, -ph.incidentDir,
                clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
                iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
                specularColor, specularIntensity,
                anisotropy, anisotropyRotation, thinFilm, true,
              );
              // Accumulate BRDF-weighted photon flux (no π r² denominator here —
              // it is applied once in the final estimate, not per-photon, which
              // keeps τ in physically-consistent units: [W·sr/m²·sr] = [W/m²]).
              // nDotL is a hemisphere rejection only: the density kernel already integrates projected receiver area.
              phiM = phiM + throughput * brdf * ph.flux;
              M = M + 1.0;
              }
            } else {
              // Volume records live on a 3D measure and carry the phase g that
              // generated them. Matching the packed medium identity prevents a
              // photon from a different participating medium contributing even
              // if the two media overlap spatially.
              let phase = sppmVolumePhase(
                clamp(dot(ph.incidentDir, wo), -1.0, 1.0),
                ph.phaseG,
              );
              phiM = phiM + throughput * phase * ph.flux;
              M = M + 1.0;
            }
          }
          // Valid atomic-exchange publication is acyclic. If a corrupted
          // buffer contains a cycle, Floyd detection stops after each record
          // in that cycle has been considered once.
          if (cycleSlow != 0u && cycleSlow == cycleFast) { break; }
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
  // In spectral mode each photon and the receiver BSDF carry a scalar value at
  // the shared frame hero. Reconstruct only this frame's newly absorbed flux
  // before adding it to persistent RGB tau; accumulated wavelengths must never
  // be relabelled with a later frame's hero. absorbedFluxInvPdf compensates the
  // 50/50 ReSTIR/advanced-estimator schedule when that schedule is active.
  var phiForTau = phiM;
  if (params.spectralEnabled != 0u) {
    phiForTau = heroWavelengthToRgb(heroLambda, luminance(phiM), heroPdf);
  }
  phiForTau = phiForTau * absorbedFluxInvPdf;

  let Nprime  = N + SPPM_ALPHA_WGSL * M;
  let NplusM  = N + M;
  // Guard M=0 ⟹ ratio=1 (no photons this frame → no update, no shrink).
  let ratio   = select(Nprime / NplusM, 1.0, M < 0.5);
  // Surface gather support is a disk: R'² = R²·ratio. Volume support is a
  // sphere: R' = R·cbrt(ratio), hence R'² = R²·ratio^(2/3).
  let radius2Scale = select(
    pow(ratio, 2.0 / 3.0),
    ratio,
    gatherKind == SPPM_PHOTON_KIND_SURFACE,
  );
  let r2prime = r2 * radius2Scale;
  let tauPrime = (tau + phiForTau) * ratio;

  // Write updated per-pixel stats back.
  sppmPixelStats[statsIndex].tau     = tauPrime;
  sppmPixelStats[statsIndex].radius2 = r2prime;
  sppmPixelStats[statsIndex].N       = Nprime;
}

// Read one measure's current cumulative estimate without mutating it. Update
// and readback are deliberately separate: a surface/volume/no-receiver event
// is stochastic, while tau is already normalized by ALL emitted frames. Gating
// readback on the same event would multiply that event probability a second
// time. The kernel therefore updates at most one record, then reads BOTH records
// exactly once on every frame owned by the SPPM estimator.
fn sppmProgressiveEstimateKind(
  pixelIndex: u32,
  gatherKind: u32,
) -> vec3f {
  let nPhotons = sppmStats.photonCount;
  let Ne = f32(sppmStats.frameAccumulated) * f32(nPhotons);
  let pxStats = sppmPixelStats[pixelIndex * 2u + gatherKind];
  if (Ne <= 0.0 || pxStats.radius2 <= 1e-24) { return vec3f(0.0); }
  let kernelMeasure = select(
    (4.0 / 3.0) * PI * pxStats.radius2 * sqrt(pxStats.radius2),
    PI * pxStats.radius2,
    gatherKind == SPPM_PHOTON_KIND_SURFACE,
  );
  return pxStats.tau / (Ne * kernelMeasure);
}

fn sppmCurrentProgressiveEstimate(pixelIndex: u32) -> vec3f {
  return
    sppmProgressiveEstimateKind(pixelIndex, SPPM_PHOTON_KIND_SURFACE) +
    sppmProgressiveEstimateKind(pixelIndex, SPPM_PHOTON_KIND_VOLUME);
}

// Henyey-Greenstein phase density in sr^-1. Kept in the SPPM module so the
// photon-only pipeline and every megakernel composition use the same function.
fn sppmVolumePhase(cosTheta: f32, gRaw: f32) -> f32 {
  let g = clamp(gRaw, -0.999999, 0.999999);
  if (g == 0.0) { return 0.07957747154594767; }
  let absG = abs(g);
  let alignedCos = select(-cosTheta, cosTheta, g > 0.0);
  let oneMinusG = 1.0 - absG;
  let denom = oneMinusG * oneMinusG +
    2.0 * absG * (1.0 - clamp(alignedCos, -1.0, 1.0));
  let numerator = oneMinusG * (1.0 + absG);
  return 0.07957747154594767 * numerator / (denom * sqrt(denom));
}

fn sppmUpdateSurfaceProgressive(
  pixelIndex : u32,
  pos        : vec3f,
  normal     : vec3f,
  clearcoatNormal : vec3f,
  wo         : vec3f,
  baseColor  : vec3f,
  roughness  : f32,
  metallic   : f32,
  transmission : f32,
  etaTOverI : f32,
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
  thinFilm : ThinFilmInterface,
  throughput : vec3f,
  heroLambda : f32,
  heroPdf : f32,
  absorbedFluxInvPdf : f32,
) {
  sppmUpdateProgressiveKind(
    pixelIndex, pos, normal, clearcoatNormal, wo,
    baseColor, roughness, metallic, transmission, etaTOverI,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
    thinFilm,
    throughput, heroLambda, heroPdf, absorbedFluxInvPdf,
    SPPM_PHOTON_KIND_SURFACE, 0u,
  );
}

fn sppmUpdateVolumeProgressive(
  pixelIndex : u32,
  pos : vec3f,
  wo : vec3f,
  mediumMatId : u32,
  throughput : vec3f,
  heroLambda : f32,
  heroPdf : f32,
  absorbedFluxInvPdf : f32,
) {
  sppmUpdateProgressiveKind(
    pixelIndex, pos, vec3f(0.0), vec3f(0.0), wo,
    vec3f(0.0), 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, vec3f(0.0),
    0.0, 1.0, 0.0, 0.0,
    vec3f(1.0), 1.0, 0.0, 0.0,
    bsdfNoThinFilm(),
    throughput, heroLambda, heroPdf, absorbedFluxInvPdf,
    SPPM_PHOTON_KIND_VOLUME, mediumMatId,
  );
}

// Shared insertion/hash helpers above serve the current linked-grid estimator.
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
 * Records store source importance weights without 1/N_photons. The gather
 * applies the sole emitted-count normalization in τ/(N_e π R²).
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
fn sppmSampleDirectionalCone(
  rng: ptr<function, PtRngState>,
  axisIn: vec3f,
  angularDiameter: f32,
) -> vec3f {
  let axis = safe_normalize(axisIn);
  if (angularDiameter <= 0.0) { return axis; }
  let cosHalfAngle = cos(angularDiameter * 0.5);
  let cosTheta = mix(cosHalfAngle, 1.0, rand_f32(rng));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * PI * rand_f32(rng);
  var tangent: vec3f;
  var bitangent: vec3f;
  buildOnb(axis, &tangent, &bitangent);
  return safe_normalize(
    sinTheta * cos(phi) * tangent +
    sinTheta * sin(phi) * bitangent +
    cosTheta * axis
  );
}

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
  let meshBase = meshAreaLightBase(meshIdx);
  return meshAreaLights[meshBase + 3u].w <= 0.5;
}

fn sppmMaterialSigmaA(
  matId: u32,
  mat: DecodedMaterial,
  heroLambda: f32,
) -> vec3f {
  var sigmaA = select(vec3f(0.0), max(mat.sigmaA, vec3f(0.0)), mat.hasSigmaA);
  if (mat.hasSpectralAttenuation && mat.spectralSampleCount > 0u) {
    if (params.spectralEnabled != 0u) {
      sigmaA = vec3f(max(
        sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda)), 0.0,
      ));
    } else {
      sigmaA = max(vec3f(
        sampleMaterialSpectralMu(matId, 0.15),
        sampleMaterialSpectralMu(matId, 0.50),
        sampleMaterialSpectralMu(matId, 0.85),
      ), vec3f(0.0));
    }
  } else if (params.spectralEnabled != 0u) {
    sigmaA = vec3f(spectralRgbFactorAtHero(sigmaA, heroLambda));
  }
  return sigmaA;
}

fn sppmMaterialSigmaS(mat: DecodedMaterial, heroLambda: f32) -> vec3f {
  let sigmaS = max(mat.scatteringRgb, vec3f(0.0));
  return select(
    sigmaS,
    vec3f(spectralRgbFactorAtHero(sigmaS, heroLambda)),
    params.spectralEnabled != 0u,
  );
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
  let photonHero = sppmFrameHeroSample();
  let photonHeroLambda = photonHero.x;

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
  if (hasEnvironmentMap()) {
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
  // Point/spot emitters expose a non-physical, host-authored distance law.
  // Photon density already contributes the physical inverse-square measure,
  // so the deposited flux applies only the relative measure correction.
  var photonUsesPointSpotAttenuation = false;
  var photonCutoffDistance = 0.0;
  var photonDecay = 0.0;
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
      // castShadow:false is sign-encoded as -1-angularDiameter. Selected SPPM
      // sources are shadow-casting, but decode the shared ABI before sampling
      // so both photon and NEE paths consume the exact same authored value.
      let angularDiameterRaw = dDirAD.w;
      let angularDiameter = select(
        angularDiameterRaw,
        -1.0 - angularDiameterRaw,
        angularDiameterRaw < 0.0,
      );
      let sampledTowardLightDir = sppmSampleDirectionalCone(
        &rng, dDirAD.xyz, angularDiameter,
      );
      // Random point on a disk perpendicular to this sampled direction. The
      // disk covers the scene AABB projection for every soft-sun photon.
      let r2d  = sqrt(rand_f32(&rng)) * extent;
      let phi2 = 2.0 * PI * rand_f32(&rng);
      var lt: vec3f; var lb: vec3f;
      buildOnb(sampledTowardLightDir, &lt, &lb);
      let diskPos = r2d * cos(phi2) * lt + r2d * sin(phi2) * lb;
      let sceneCenter = vec3f(sppmStats.sceneCenterX, sppmStats.sceneCenterY, sppmStats.sceneCenterZ);
      photonOrigin = sceneCenter + diskPos + sampledTowardLightDir * extent * 2.0;
      photonDir    = -sampledTowardLightDir;
      // Authored directional irradiance is integrated over the sun cone, just
      // as in production NEE; do not introduce an extra cone-pdf factor.
      // Source weight E × launch area / p_select; gather supplies sole /N_e.
      let diskArea = PI * extent * extent;
      photonFlux   = dIrrMean.rgb * diskArea * lightSelectInvPdf;
      seeded = true;
    }
    current = current + 1u;
  }

  // Point lights.
  for (var pointIdx = 0u; pointIdx < params.pointLightCount; pointIdx = pointIdx + 1u) {
    if (!sppmPointCastsShadow(pointIdx)) { continue; }
    if (current == pick) {
      let pointBase = pointIdx * POINT_LIGHT_VEC4_STRIDE;
      let pointExtra = pointLights[pointBase + 2u];
      photonOrigin = pointLights[pointBase].xyz;
      photonDir    = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
      // Source weight I × 4π / p_select; gather supplies the sole /N_e.
      photonFlux   = pointLights[pointBase + 1u].rgb * (4.0 * PI) * lightSelectInvPdf;
      photonUsesPointSpotAttenuation = true;
      photonCutoffDistance = pointExtra.x;
      photonDecay = pointExtra.y;
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
      let spotExtra = spotLights[spotBase + 3u];
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
      // Uniform-cone weight I × softness × Ω / p_select.
      let softness = smoothstep(cosMin, max(cosInner, cosMin + 1e-6), cosTheta);
      let solidAngle = 2.0 * PI * (1.0 - cosMin);
      photonFlux   = sradW.rgb * solidAngle * softness * lightSelectInvPdf;
      seeded = true;
      photonUsesPointSpotAttenuation = true;
      photonCutoffDistance = spotExtra.x;
      photonDecay = spotExtra.y;
    }
    current = current + 1u;
  }

  // Rect/disc area lights.  Same 4-vec4 record and area conventions as NEE:
  // rshape.w ~= 0 => rect area 4*|u x v|;
  // rshape.w ~= 1 => disc area PI*|u x v|.
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
      if (normalLen > 0.0) {
        let lightNormal = normalRaw / normalLen;
        let isDisc = abs(rshape.w - 1.0) < 0.5;
        let xi1 = rand_f32(&rng);
        let xi2 = rand_f32(&rng);
        var emitPos: vec3f;
        var area: f32;
        if (isDisc) {
          let disc = sppmConcentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));
          emitPos = rpos + ru * disc.x + rv * disc.y;
          area = PI * normalLen;
        } else {
          emitPos = rpos + ru * (xi1 * 2.0 - 1.0) + rv * (xi2 * 2.0 - 1.0);
          area = 4.0 * length(cross(ru, rv));
        }
        if (area > 0.0) {
          let hemi = cosineHemisphereSample(&rng, lightNormal);
          photonOrigin = emitPos;
          photonDir    = hemi.wi;
          photonFlux   = rr * area * PI * lightSelectInvPdf;
          seeded = true;
        }
      }
    }
    current = current + 1u;
  }

  // Mesh-area lights.  Same triangle stream and area convention as NEE.
  for (var meshIdx = 0u; meshIdx < params.meshAreaLightCount; meshIdx = meshIdx + 1u) {
    if (!sppmMeshAreaCastsShadow(meshIdx)) { continue; }
    if (current == pick) {
      let meshBase = meshAreaLightBase(meshIdx);
      let a = meshAreaLights[meshBase].xyz;
      let b = meshAreaLights[meshBase + 1u].xyz;
      let c = meshAreaLights[meshBase + 2u].xyz;
      let e1 = b - a;
      let e2 = c - a;
      let normalRaw = cross(e1, e2);
      let normalLen = length(normalRaw);
      if (normalLen > 0.0) {
        let lightNormal = normalRaw / normalLen;
        let r1 = rand_f32(&rng);
        let r2 = rand_f32(&rng);
        let su = sqrt(r1);
        let uu = 1.0 - su;
        let vv = r2 * su;
        let ww = 1.0 - uu - vv;
        let emitPos = a * uu + b * vv + c * ww;
        let mr = sampleMeshAreaLightRadiance(
          meshIdx, vec3f(uu, vv, ww), emitPos,
        );
        let area = 0.5 * normalLen;
        if (area > 0.0) {
          let hemi = cosineHemisphereSample(&rng, lightNormal);
          photonOrigin = emitPos;
          photonDir    = hemi.wi;
          photonFlux   = mr * area * PI * lightSelectInvPdf;
          seeded = true;
        }
      }
    }
    current = current + 1u;
  }

  // Environment source.  Direction/PDF comes from the same helpers as full-tier
  // NEE; photons are launched from a scene-covering disk perpendicular to wi.
  if (hasEnvironmentMap() && current == pick) {
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
    if (envPdf > 0.0) {
      let extent = sppmStats.sceneExtent;
      let r2d = sqrt(rand_f32(&rng)) * extent;
      let phi = 2.0 * PI * rand_f32(&rng);
      var et: vec3f; var eb: vec3f;
      buildOnb(envDir, &et, &eb);
      let diskPos = r2d * cos(phi) * et + r2d * sin(phi) * eb;
      let sceneCenter = vec3f(sppmStats.sceneCenterX, sppmStats.sceneCenterY, sppmStats.sceneCenterZ);
      photonOrigin = sceneCenter + diskPos + envDir * extent * 2.0;
      photonDir    = -envDir;
      let diskArea = PI * extent * extent;
      photonFlux   = envColor * diskArea * lightSelectInvPdf / envPdf;
      seeded = true;
    }
  }

  if (!seeded) { return; }
  if (params.spectralEnabled != 0u) {
    photonFlux = spectralEmissionAtHero(photonFlux, photonHeroLambda);
  }

  // ── Trace the photon path ──────────────────────────────────────────────────
  var ray  = Ray(photonOrigin + photonDir * 1e-3, photonDir);
  var flux = photonFlux;
  let maxBounces = SPPM_PHOTON_MAX_BOUNCES;
  var photonMediumDepth = 0u;
  var photonMediumMatIds: array<u32, 8>;
  var photonMediumIors: array<f32, 8>;
  var photonMediumSigmaT: array<vec3f, 8>;
  var photonMediumSigmaS: array<vec3f, 8>;
  var photonMediumPhaseG: array<f32, 8>;
  var photonMediumRemainingDistance: array<f32, 8>;
  var photonMediumBoundaryKinds: array<u32, 8>;
  var photonMediumBoundaryIndices: array<u32, 8>;
  var hadDeltaChainEvent = false;

  for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
    if (bounce >= maxBounces) { break; }
    let alphaTraceOrigin = ray.origin;
    var alphaAdvance = 0.0;
    var hit = traceClosest(ray, 1e-4, INFINITY);
    let alphaSurfaceHitLimit = sceneSurfaceHitLimit();
    var alphaSurfaceHitCount = 0u;
    var alphaTraversalValid = true;
    loop {
      if (!hit.didHit) { break; }
      // Observe the final miss after exactly alphaSurfaceHitLimit pass-through
      // surfaces; any further hit is outside the published scene support.
      if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {
        alphaTraversalValid = false;
        break;
      }
      alphaSurfaceHitCount = alphaSurfaceHitCount + 1u;
      if (!alphaTestPassThrough(
        hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, &rng,
      )) { break; }
      let alphaStep = hit.dist + 1e-4;
      alphaAdvance = alphaAdvance + alphaStep;
      ray.origin = ray.origin + ray.direction * alphaStep;
      hit = traceClosest(ray, 1e-4, INFINITY);
    }
    ray.origin = alphaTraceOrigin;
    if (!alphaTraversalValid) { break; }
    if (hit.didHit) { hit.dist = hit.dist + alphaAdvance; }
    if (!hit.didHit) { break; }
    // Point/spot range and decay are properties of the source edge. Photon
    // density already supplies physical inverse-square spreading, so replace
    // that measure exactly once, before this path scatters any further.
    if (bounce == 0u && photonUsesPointSpotAttenuation) {
      flux = flux * pointSpotPathMeasureScale(
        hit.dist, photonCutoffDistance, photonDecay,
      );
      // Zero is the only dead path. Tiny positive flux remains part of the
      // estimator; this is not a Russian-roulette termination site.
      if (max(flux.r, max(flux.g, flux.b)) <= 0.0) { break; }
    }

    // A light may start inside a closed medium and see its back face first.
    // Seed that exact boundary before sampling this first segment; an unowned
    // merged mesh hit is rejected because its entry/exit object is unknowable.
    if (photonMediumDepth == 0u && !hit.frontFace) {
      let inferredMatId = hitMaterialId(hit);
      var inferredMat = decodeMaterial(inferredMatId);
      let inferredThickness = sampleVolumeThicknessTexture(
        inferredMatId, hit.triIndex, hit.baryVW, hit.instanceIndex,
      );
      if (inferredThickness >= 0.0) {
        inferredMat.volumeThickness = max(
          inferredMat.volumeThickness * inferredThickness, 0.0,
        );
        inferredMat.hasVolumeThickness = true;
      }
      let inferredTransmission = clamp(
        inferredMat.transmission * sampleTransmissionTexture(
          inferredMatId, hit.triIndex, hit.baryVW, hit.instanceIndex,
        ),
        0.0, 1.0,
      );
      if (inferredMat.isTranslucent && inferredTransmission > 0.0) {
        let inferredBoundary = mediumBoundaryIdentity(
          hit.triIndex, hit.instanceIndex,
        );
        if (!mediumBoundaryIsValid(inferredBoundary)) { break; }
        let inferredSigmaA = sppmMaterialSigmaA(
          inferredMatId, inferredMat, photonHeroLambda,
        );
        let inferredSigmaS = sppmMaterialSigmaS(
          inferredMat, photonHeroLambda,
        );
        photonMediumMatIds[0u] = inferredMatId;
        photonMediumIors[0u] = max(inferredMat.ior, 1e-4);
        photonMediumSigmaT[0u] = max(
          inferredSigmaA + inferredSigmaS, vec3f(0.0),
        );
        photonMediumSigmaS[0u] = inferredSigmaS;
        photonMediumPhaseG[0u] = clamp(
          inferredMat.scatteringAnisotropy, -0.999999, 0.999999,
        );
        photonMediumRemainingDistance[0u] =
          materialAttenuationDistance(INFINITY, inferredMat);
        photonMediumBoundaryKinds[0u] = inferredBoundary.x;
        photonMediumBoundaryIndices[0u] = inferredBoundary.y;
        photonMediumDepth = 1u;
      }
    }

    if (photonMediumDepth > 0u) {
      let mediumIndex = photonMediumDepth - 1u;
      let photonSigmaT = photonMediumSigmaT[mediumIndex];
      let photonSigmaS = photonMediumSigmaS[mediumIndex];
      let photonHeroSigmaT = select(
        max(photonSigmaT.x, max(photonSigmaT.y, photonSigmaT.z)),
        photonSigmaT.x,
        params.spectralEnabled != 0u,
      );
      if (photonHeroSigmaT > 0.0) {
        let mediumSegmentDistance = min(
          hit.dist, photonMediumRemainingDistance[mediumIndex],
        );
        let xiFlight = rand_f32(&rng);
        let freeFlightDist =
          -log(max(1.0 - xiFlight, 1e-9)) / photonHeroSigmaT;
        if (freeFlightDist < mediumSegmentDistance) {
          // Sample the same hero-channel distance density used by the eye walk.
          // sigma_s * T / p_t leaves photon flux in the correct collision
          // measure; the gather supplies phase [sr^-1] and sphere volume [m^3].
          let transmittance = exp(-photonSigmaT * freeFlightDist);
          let pdfHero = photonHeroSigmaT *
            exp(-photonHeroSigmaT * freeFlightDist);
          flux = flux * photonSigmaS * transmittance / max(pdfHero, 1e-9);
          if (hadDeltaChainEvent &&
              max(flux.r, max(flux.g, flux.b)) > 0.0) {
            let scatterPos = ray.origin + ray.direction * freeFlightDist;
            sppmInsertPhoton(
              photonIdx,
              scatterPos,
              flux,
              ray.direction,
              sppmStats.currentRadius,
              SPPM_PHOTON_KIND_VOLUME,
              photonMediumMatIds[mediumIndex],
              photonMediumPhaseG[mediumIndex],
            );
          }
          // SPPM owns only L-S*-D. A volume event before the first delta is an
          // ordinary photon path; after a delta it has just been deposited.
          break;
        }
        // Reaching the surface already carried hero survival probability.
        // Divide channel transmittance by that probability, exactly as the eye
        // free-flight walk does, rather than multiplying Beer attenuation twice.
        flux = flux * exp(
          -(photonSigmaT - vec3f(photonHeroSigmaT)) * mediumSegmentDistance,
        );
        photonMediumRemainingDistance[mediumIndex] = max(
          photonMediumRemainingDistance[mediumIndex] - mediumSegmentDistance,
          0.0,
        );
      }
      if (max(flux.r, max(flux.g, flux.b)) <= 0.0) { break; }
    }

    let matId = hitMaterialId(hit);
    var mat = decodeMaterial(matId);
    let hp = ray.origin + ray.direction * hit.dist;
    let frontFace = hit.frontFace;
    var normal = select(-hit.normal, hit.normal, frontFace);
    normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex, frontFace);
    normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);
    var clearcoatNormal = applyClearcoatNormalMap(
      matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex,
    );
    var baseColor = mat.baseColor *
      sampleVertexColor(hit.triIndex, hit.baryVW).rgb *
      sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;
    baseColor = baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
    let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
    let roughness = clamp(mat.roughness * ormSample.g, 0.0, 1.0);
    let metallic = clamp(mat.metallic * ormSample.b, 0.0, 1.0);
    let transmission = clamp(
      mat.transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex),
      0.0, 1.0,
    );
    let volumeThicknessSample = sampleVolumeThicknessTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    );
    if (volumeThicknessSample >= 0.0) {
      mat.volumeThickness = max(
        mat.volumeThickness * volumeThicknessSample, 0.0,
      );
      mat.hasVolumeThickness = true;
    }
    mat.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
    mat.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
    mat.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0), vec3f(1.0));
    mat.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
    mat.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
    let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
    if (iridescenceThicknessSample >= 0.0) {
      let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);
      mat.iridescenceThicknessMin = iridescenceThickness;
      mat.iridescenceThicknessMax = iridescenceThickness;
      if (iridescenceThickness <= 0.0) { mat.iridescence = 0.0; }
    }
    mat.specularColor = max(
      mat.specularColor * sampleSpecularColorTexture(
        matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
      ),
      vec3f(0.0),
    );
    mat.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
    var materialIor = mat.ior;
    if (params.spectralEnabled != 0u) {
      baseColor = vec3f(spectralCombinedReflectanceAtHero(
        baseColor,
        mat.baseColor,
        mat.spectralReflCoeffs,
        mat.hasSpectralReflectance,
        photonHeroLambda,
      ));
      mat.sheenColor = vec3f(spectralRgbFactorAtHero(
        mat.sheenColor, photonHeroLambda,
      ));
      mat.specularColor = vec3f(spectralRgbFactorAtHero(
        mat.specularColor, photonHeroLambda,
      ));
      if (mat.dispersionAbbe >= 1.0) {
        materialIor = cauchyIorAtLambda(
          photonHeroLambda, mat.ior, mat.dispersionAbbe,
        );
      }
    }

    // A deposited photon must have crossed at least one canonical production
    // delta event. Direct L-D photons stay exclusively with analytic NEE.
    let diffuseReceiverWeight = (1.0 - metallic) * (1.0 - transmission);
    if (hadDeltaChainEvent && diffuseReceiverWeight > 0.0) {
      if (max(flux.r, max(flux.g, flux.b)) > 0.0) {
        sppmInsertPhoton(
          photonIdx,
          hp,
          flux,
          ray.direction,
          sppmStats.currentRadius,
          SPPM_PHOTON_KIND_SURFACE,
          0u,
          0.0,
        );
      }
      break;
    }
    if (mat.isUnlit) { break; }

    var incidentIor = 1.0;
    if (photonMediumDepth > 0u) {
      incidentIor = photonMediumIors[photonMediumDepth - 1u];
    } else if (!frontFace) {
      incidentIor = max(materialIor, 1e-4);
    }
    var transmittedIor = max(materialIor, 1e-4);
    let hitBoundary = mediumBoundaryIdentity(hit.triIndex, hit.instanceIndex);
    let crossesMedium = mat.isTranslucent && transmission > 0.0;
    if (crossesMedium && !mediumBoundaryIsValid(hitBoundary)) { break; }
    if (!frontFace) {
      transmittedIor = 1.0;
      if (crossesMedium) {
        if (
          photonMediumDepth == 0u ||
          photonMediumMatIds[photonMediumDepth - 1u] != matId ||
          !mediumBoundaryMatches(
            photonMediumBoundaryKinds[photonMediumDepth - 1u],
            photonMediumBoundaryIndices[photonMediumDepth - 1u],
            hitBoundary,
          )
        ) {
          break;
        }
        if (photonMediumDepth > 1u) {
          transmittedIor = photonMediumIors[photonMediumDepth - 2u];
        }
      }
    }
    let etaTOverI = transmittedIor / max(incidentIor, 1e-4);
    let wo = -ray.direction;
    let cosThetaO = max(dot(normal, wo), 0.0);
    let f0Base = materialSpecularF0(
      baseColor, metallic, etaTOverI,
      mat.specularColor, mat.specularIntensity,
    );
    let f0 = iridescenceModifiedF0(
      f0Base, mat.iridescence, mat.iridescenceIor,
      mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, cosThetaO,
    );
    let thinFilm = ThinFilmInterface(
      mat.thinFilmEnabled, matId, mat.thinFilmLayerCountU,
      mat.thinFilmIncidentIor, materialIor, mat.thinFilmAngleDependent,
      frontFace, params.spectralEnabled != 0u, photonHeroLambda, transmission,
    );
    let anisoStrength = materialAnisotropy(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
    let anisoRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);
    let bs = sampleNextBounceDirectionWithClearcoatNormal(
      &rng, ray.direction, hp, hit.normal, normal, clearcoatNormal,
      baseColor, roughness, metallic, transmission, etaTOverI, true,
      materialSpecularFresnelSchlick(
        cosThetaO, f0, metallic, mat.specularIntensity,
      ),
      mat.iridescence, mat.iridescenceIor,
      mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
      mat.specularColor, mat.specularIntensity,
      thinFilm, mat.isTranslucent,
      mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
      mat.sheenColor, anisoStrength, anisoRotation,
    );

    // This production sampler is the path-class boundary. Rough/glossy or
    // diffuse events remain with the ordinary PT estimator and are not mapped.
    if (!bs.sampledIsDelta || bs.sampledEventPdf <= 0.0) { break; }
    flux = flux * bs.throughputMul;
    if (max(flux.r, max(flux.g, flux.b)) <= 0.0) { break; }
    hadDeltaChainEvent = true;

    if (bs.enteredMedium) {
      let sigmaA = sppmMaterialSigmaA(matId, mat, photonHeroLambda);
      let sigmaS = sppmMaterialSigmaS(mat, photonHeroLambda);
      if (photonMediumDepth < 8u) {
        photonMediumMatIds[photonMediumDepth] = matId;
        photonMediumIors[photonMediumDepth] = max(materialIor, 1e-4);
        photonMediumSigmaT[photonMediumDepth] = max(
          sigmaA + sigmaS, vec3f(0.0),
        );
        photonMediumSigmaS[photonMediumDepth] = sigmaS;
        photonMediumPhaseG[photonMediumDepth] = clamp(
          mat.scatteringAnisotropy, -0.999999, 0.999999,
        );
        photonMediumRemainingDistance[photonMediumDepth] =
          materialAttenuationDistance(INFINITY, mat);
        photonMediumBoundaryKinds[photonMediumDepth] = hitBoundary.x;
        photonMediumBoundaryIndices[photonMediumDepth] = hitBoundary.y;
        photonMediumDepth = photonMediumDepth + 1u;
      } else {
        break;
      }
    } else if (bs.exitedMedium) {
      if (
        photonMediumDepth == 0u ||
        photonMediumMatIds[photonMediumDepth - 1u] != matId ||
        !mediumBoundaryMatches(
          photonMediumBoundaryKinds[photonMediumDepth - 1u],
          photonMediumBoundaryIndices[photonMediumDepth - 1u],
          hitBoundary,
        )
      ) {
        break;
      }
      photonMediumDepth = photonMediumDepth - 1u;
    }
    ray.origin = bs.newRayOrigin;
    ray.direction = bs.newRayDir;
  }
}
`;
