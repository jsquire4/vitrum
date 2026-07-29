import { describe, expect, it } from 'vitest';
import {
  SOLAR_ANGULAR_RADIUS,
  solarDiscTexelCoverage,
} from '../solarDiscCoverage.js';

const analyticSolarCapSolidAngle =
  2 * Math.PI * (1 - Math.cos(SOLAR_ANGULAR_RADIUS));

function integratedCoverage(
  coverage: Float64Array,
  width: number,
  height: number,
): number {
  const dTheta = Math.PI / height;
  const dPhi = (2 * Math.PI) / width;
  let integral = 0;
  for (let row = 0; row < height; row += 1) {
    const theta = ((row + 0.5) / height) * Math.PI;
    const measure = dTheta * dPhi * Math.max(Math.sin(theta), 1e-6);
    const rowOffset = row * width;
    for (let x = 0; x < width; x += 1) {
      integral += (coverage[rowOffset + x] ?? 0) * measure;
    }
  }
  return integral;
}

describe('solarDiscTexelCoverage high-resolution overlap quadrature', () => {
  it('fills every intersected longitude at a 4K polar cap without strata holes', () => {
    const width = 4096;
    const height = 2048;
    const coverage = solarDiscTexelCoverage(width, height, [0, 1, 0]);
    let activeRows = 0;
    let occupancyMismatches = 0;
    let largestWithinRowDelta = 0;

    for (let row = 0; row < height; row += 1) {
      const rowOffset = row * width;
      const first = coverage[rowOffset] ?? 0;
      const active = first > 0;
      if (active) activeRows += 1;
      for (let x = 0; x < width; x += 1) {
        const value = coverage[rowOffset + x] ?? 0;
        if ((value > 0) !== active) occupancyMismatches += 1;
        if (active) {
          largestWithinRowDelta = Math.max(
            largestWithinRowDelta,
            Math.abs(value - first),
          );
        }
      }
    }

    expect(occupancyMismatches).toBe(0);
    expect(largestWithinRowDelta).toBe(0);
    expect(activeRows).toBeGreaterThan(1);
    expect(activeRows * width).toBeGreaterThan(4096);
    expect(integratedCoverage(coverage, width, height))
      .toBeCloseTo(analyticSolarCapSolidAngle, 12);
  });

  it('keeps near-pole row occupancy circular-contiguous and conserves cap energy', () => {
    const width = 2048;
    const height = 1024;
    const thetaSun = 0.003;
    const coverage = solarDiscTexelCoverage(width, height, [
      Math.sin(thetaSun),
      Math.cos(thetaSun),
      0,
    ]);
    let activeRows = 0;
    let partialRows = 0;

    for (let row = 0; row < height; row += 1) {
      const rowOffset = row * width;
      let activeTexels = 0;
      let transitions = 0;
      for (let x = 0; x < width; x += 1) {
        const active = (coverage[rowOffset + x] ?? 0) > 0;
        const nextActive =
          (coverage[rowOffset + ((x + 1) % width)] ?? 0) > 0;
        if (active) activeTexels += 1;
        if (active !== nextActive) transitions += 1;
      }
      if (activeTexels === 0) continue;
      activeRows += 1;
      if (activeTexels < width) {
        partialRows += 1;
        expect(transitions).toBe(2);
      } else {
        expect(transitions).toBe(0);
      }
    }

    expect(activeRows).toBeGreaterThan(1);
    expect(partialRows).toBeGreaterThan(0);
    expect(integratedCoverage(coverage, width, height))
      .toBeCloseTo(analyticSolarCapSolidAngle, 12);
  });
});
