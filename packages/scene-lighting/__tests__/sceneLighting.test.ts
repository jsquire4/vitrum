import { describe, expect, it } from 'vitest';
import {
  SUN_INTENSITY,
  computeLightingState,
  getSunIntensity,
  pointIntensityFromLumens,
  rectAreaIntensityFromLumens,
  skyParamsFor,
  worldSunPosition,
} from '../src/index.js';

describe('@vitrum/scene-lighting', () => {
  it('computes stable noon sky parameters and world sun placement', () => {
    const params = skyParamsFor(0.5);

    expect(params.sunPosition).toEqual([0, 0.4, -1]);
    expect(params.turbidity).toBe(2);
    expect(params.rayleigh).toBe(1);
    expect(worldSunPosition(params)).toEqual([0, 4.800000000000001, -12]);
  });

  it('clamps sky time-of-day into the authored dawn/dusk range', () => {
    expect(skyParamsFor(-1)).toEqual(skyParamsFor(0));
    expect(skyParamsFor(2)).toEqual(skyParamsFor(1));
  });

  it('maps time-of-day buckets to the canonical sun intensity table', () => {
    expect(getSunIntensity(0.5)).toBe(SUN_INTENSITY.noon);
    expect(getSunIntensity(0.25)).toBe(SUN_INTENSITY.afternoon);
    expect(getSunIntensity(0.1)).toBe(SUN_INTENSITY.sunset);
    expect(getSunIntensity(0.01)).toBe(SUN_INTENSITY.twilight);
  });

  it('normalizes sun direction and applies day/night intensity policy', () => {
    const day = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: false,
      intensityMultiplier: 2,
    });
    const night = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: true,
      intensityMultiplier: 2,
    });

    const len = Math.hypot(...day.sunDirection);
    expect(len).toBeCloseTo(1, 6);
    expect(day.sunIntensity).toBeCloseTo(SUN_INTENSITY.noon * 2, 6);
    expect(day.skyIrradiance).toBeCloseTo(day.sunIntensity * 0.5, 6);
    expect(night.sunIntensity).toBeCloseTo(SUN_INTENSITY.moonlight * 2, 6);
    expect(night.skyTint).toEqual([0.05, 0.08, 0.15]);
  });

  it('converts practical lumens into point and rectangular area intensities', () => {
    expect(pointIntensityFromLumens(800)).toBeCloseTo(800 / (4 * Math.PI), 6);
    expect(rectAreaIntensityFromLumens(800, 2)).toBeCloseTo(800 / (2 * Math.PI), 6);
  });
});
