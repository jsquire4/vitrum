import { describe, expect, it } from 'vitest';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { SPPM_ALPHA, SPPM_PIXEL_STATS_BYTES_PER_PIXEL } from '../sppmParams.js';
import {
  SPPM_GROUP3_BINDINGS_WGSL,
  SPPM_PHOTON_PASS_WGSL,
} from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';

const SURFACE = 0;
const VOLUME = 1;
const MEDIUM_MASK = 0x7fffffff;

function metadata(kind: number, mediumMatId: number): number {
  return (((kind & 1) << 31) | (mediumMatId & MEDIUM_MASK)) >>> 0;
}

function kindOf(value: number): number {
  return value >>> 31;
}

function mediumOf(value: number): number {
  return value & MEDIUM_MASK;
}

function matches(
  value: number,
  gatherKind: number,
  gatherMediumMatId: number,
): boolean {
  return kindOf(value) === gatherKind &&
    (gatherKind === SURFACE || mediumOf(value) === gatherMediumMatId);
}

interface Stats {
  tau: number;
  radius2: number;
  N: number;
}

function update(stats: Stats, kind: number, M: number, phi: number): Stats {
  const nPrime = stats.N + SPPM_ALPHA * M;
  const ratio = M === 0 ? 1 : nPrime / (stats.N + M);
  return {
    tau: (stats.tau + phi) * ratio,
    radius2: stats.radius2 *
      (kind === SURFACE ? ratio : ratio ** (2 / 3)),
    N: nPrime,
  };
}

describe('SPPM homogeneous-volume production closure', () => {
  it('round-trips photon kind and medium identity without changing record size', () => {
    const surface = metadata(SURFACE, 0);
    const volume = metadata(VOLUME, 12345);
    expect(kindOf(surface)).toBe(SURFACE);
    expect(kindOf(volume)).toBe(VOLUME);
    expect(mediumOf(volume)).toBe(12345);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmPhotonMetadata(kind, mediumMatId)',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmPhotonCells[photonIndex].metadata =',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmPhotonCells[photonIndex].phaseG = phaseG',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain(
      'bitcast<f32>(sppmPhotonMetadata',
    );
  });

  it('filters surface/volume measures and rejects photons from another medium', () => {
    const surface = metadata(SURFACE, 0);
    const fogA = metadata(VOLUME, 7);
    const fogB = metadata(VOLUME, 9);
    expect(matches(surface, SURFACE, 0)).toBe(true);
    expect(matches(fogA, SURFACE, 0)).toBe(false);
    expect(matches(surface, VOLUME, 7)).toBe(false);
    expect(matches(fogA, VOLUME, 7)).toBe(true);
    expect(matches(fogB, VOLUME, 7)).toBe(false);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmPhotonMediumMatId(ph) == gatherMediumMatId',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'ph.phaseG',
    );
  });

  it('uses the normalized sphere kernel and cube-root radius recurrence', () => {
    const r = 0.4;
    const M = 5;
    const initial: Stats = { tau: 0, radius2: r * r, N: 0 };
    const volume = update(initial, VOLUME, M, 3);
    const surface = update(initial, SURFACE, M, 3);
    expect(volume.radius2).toBeCloseTo(
      r * r * SPPM_ALPHA ** (2 / 3),
      14,
    );
    expect(surface.radius2).toBeCloseTo(r * r * SPPM_ALPHA, 14);

    const volumeMeasure = (4 / 3) * Math.PI *
      volume.radius2 * Math.sqrt(volume.radius2);
    const normalizedKernel = 1 / volumeMeasure;
    expect(normalizedKernel * volumeMeasure).toBeCloseTo(1, 14);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('pow(ratio, 1.0 / 3.0)');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'log(4.0 / 3.0) + log(PI) + 3.0 * log(pxStats.radius)',
    );
  });

  it('keeps persistent surface and volume state disjoint across alternating frames', () => {
    const state: [Stats, Stats] = [
      { tau: 0, radius2: 1, N: 0 },
      { tau: 0, radius2: 1, N: 0 },
    ];
    state[SURFACE] = update(state[SURFACE], SURFACE, 2, 4);
    const surfaceAfterFirst = { ...state[SURFACE] };
    state[VOLUME] = update(state[VOLUME], VOLUME, 3, 9);
    expect(state[SURFACE]).toEqual(surfaceAfterFirst);
    const volumeAfterFirst = { ...state[VOLUME] };
    state[SURFACE] = update(state[SURFACE], SURFACE, 1, 2);
    expect(state[VOLUME]).toEqual(volumeAfterFirst);
    expect(state[SURFACE].N).toBeCloseTo(3 * SPPM_ALPHA, 14);
    expect(state[VOLUME].N).toBeCloseTo(3 * SPPM_ALPHA, 14);
    expect(SPPM_PIXEL_STATS_BYTES_PER_PIXEL).toBe(64);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'let statsIndex = pixelIndex * 2u + gatherKind',
    );
  });

  it('publishes both cumulative measures on surface/volume/no-receiver frames without p² attenuation', () => {
    const schedule = ['surface', 'volume', 'none'] as const;
    let surfaceFlux = 0;
    let volumeFlux = 0;
    let alwaysPublishedSum = 0;
    let oldEventGatedSum = 0;
    const frameCount = 30_000;
    for (let frame = 1; frame <= frameCount; frame += 1) {
      const event = schedule[(frame - 1) % schedule.length]!;
      if (event === 'surface') surfaceFlux += 3;
      if (event === 'volume') volumeFlux += 6;
      const surfaceEstimate = surfaceFlux / frame;
      const volumeEstimate = volumeFlux / frame;
      alwaysPublishedSum += surfaceEstimate + volumeEstimate;
      if (event === 'surface') oldEventGatedSum += surfaceEstimate;
      if (event === 'volume') oldEventGatedSum += volumeEstimate;
      if (event === 'none' && frame > 3) {
        expect(surfaceEstimate + volumeEstimate).toBeGreaterThan(0);
      }
    }
    const alwaysPublishedMean = alwaysPublishedSum / frameCount;
    const oldEventGatedMean = oldEventGatedSum / frameCount;
    // E[event flux] = 3/3 + 6/3 = 3. Event-gated readback applies 1/3 again.
    expect(alwaysPublishedMean).toBeCloseTo(3, 3);
    expect(oldEventGatedMean).toBeCloseTo(1, 2);
    expect(oldEventGatedMean / alwaysPublishedMean).toBeCloseTo(1 / 3, 2);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmProgressiveEstimateKind(pixelIndex, SPPM_PHOTON_KIND_SURFACE) +',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'sppmProgressiveEstimateKind(pixelIndex, SPPM_PHOTON_KIND_VOLUME);',
    );
  });

  it('samples photon free flight over the full alpha-skipped segment before surface handling', () => {
    const alphaCursor = SPPM_PHOTON_PASS_WGSL.indexOf(
      'var alphaCursor = ptRayTMin();',
    );
    const cursorAdvance = SPPM_PHOTON_PASS_WGSL.indexOf(
      'alphaCursor = nextAlphaCursor;',
    );
    const sourceAttenuation = SPPM_PHOTON_PASS_WGSL.indexOf(
      'pointSpotPathMeasureScale(',
    );
    const flight = SPPM_PHOTON_PASS_WGSL.indexOf('let freeFlightDist =');
    const surfaceMaterial = SPPM_PHOTON_PASS_WGSL.indexOf(
      'let matId = hitMaterialId(hit)',
    );
    const volumeDeposit = SPPM_PHOTON_PASS_WGSL.indexOf(
      'SPPM_PHOTON_KIND_VOLUME,',
      flight,
    );
    expect(alphaCursor).toBeGreaterThanOrEqual(0);
    expect(alphaCursor).toBeLessThan(cursorAdvance);
    expect(cursorAdvance).toBeLessThan(sourceAttenuation);
    expect(sourceAttenuation).toBeLessThan(flight);
    expect(flight).toBeLessThan(volumeDeposit);
    expect(volumeDeposit).toBeLessThan(surfaceMaterial);
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'if (hadDeltaChainEvent &&',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'flux = flux * photonSigmaS * trueTransmittance / collisionPdf',
    );
  });

  it('uses one eye-path update owner and invokes volume gather at the real collision', () => {
    const collision = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'if (freeFlightDist < attenuationDist)',
    );
    const volumeGather = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'sppmUpdateVolumeProgressive(',
      collision,
    );
    const phaseContinuation = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'let phaseDir = sampleHenyeyGreenstein(',
      collision,
    );
    const surfaceGuard = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'if (!sppmGatherUpdated && sppmReceiverEligible)',
      phaseContinuation,
    );
    expect(collision).toBeGreaterThanOrEqual(0);
    expect(volumeGather).toBeGreaterThan(collision);
    expect(volumeGather).toBeLessThan(phaseContinuation);
    expect(surfaceGuard).toBeGreaterThan(phaseContinuation);
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (sppmActive && !sppmGatherUpdated)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'sppmGatherUpdated = true;',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'sppmRgbRadiance = sppmCurrentProgressiveEstimate(pixelIndex);',
    );
  });

  it('publishes only the implemented bounded volume scope', () => {
    const detail = BACKEND_PROMISE_LEDGER['pt-webgpu']
      .supportDetails.causticStrategies?.['photon-map'];
    expect(detail?.volumeScattering).toBe('native');
    expect(detail?.estimatorScope).toContain('homogeneous-medium collision');
    expect(detail?.estimatorScope).toContain('medium-identity-filtered HG sphere');
    expect(detail?.estimatorScope).toContain('independent progressive state');
  });
});
