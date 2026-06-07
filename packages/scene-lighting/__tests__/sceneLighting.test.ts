/**
 * Unit coverage for @vitrum/scene-lighting — the host-side, backend-agnostic
 * lighting-state primitives.
 *
 * These are pure-math functions, so every assertion is pinned against an
 * INDEPENDENT reference: a value recomputed from first principles inside the
 * test, a physical/analytic identity, or an invariant (range, symmetry,
 * monotonicity). No assertion re-calls the function under test and compares it
 * to itself.
 */
import { describe, expect, it } from 'vitest';
import {
  COLOR_TEMP_HEX,
  PT_SUN_AREA_INTENSITY,
  PT_SUN_DISC_DIAMETER,
  PT_SUN_DISTANCE,
  SUN_ANGULAR_RADIUS,
  SUN_INTENSITY,
  SUN_LIGHT_DISTANCE,
  computeLightingState,
  getSunIntensity,
  pointIntensityFromLumens,
  rectAreaIntensityFromLumens,
  skyParamsFor,
  worldSunPosition,
} from '../src/index.js';
import type { SkyParams } from '../src/index.js';

// ── Independent reference re-implementations ──────────────────────────────────
// Recomputed straight from the documented math so the tests cross-check the
// source rather than mirroring its exact expression tree.

/** Closed-form Preetham sun position + atmospheric params from t∈[0,1]. */
function skyParamsRef(timeOfDay: number) {
  const t = Math.max(0, Math.min(1, timeOfDay));
  const theta = (t - 0.5) * Math.PI;
  const x = Math.sin(theta);
  const y = Math.cos(theta);
  const sunY = Math.max(0.05, y * 0.4);
  const sunZ = -Math.cos(theta) * 0.5 - 0.5;
  const horizonProx = 1 - y;
  return {
    sunPosition: [x, sunY, sunZ] as [number, number, number],
    turbidity: 2 + horizonProx * 6,
    rayleigh: 1 + horizonProx * 2,
  };
}

describe('skyParamsFor — Preetham solar arc', () => {
  it('places the sun at zenith-biased noon with clear-sky atmosphere', () => {
    // theta(0.5) = 0 → x = sin0 = 0, y = cos0 = 1 → sunY = max(.05, .4) = .4,
    // sunZ = -1*.5 - .5 = -1. horizonProx = 0 → turbidity 2, rayleigh 1.
    const p = skyParamsFor(0.5);
    expect(p.sunPosition[0]).toBeCloseTo(0, 12);
    expect(p.sunPosition[1]).toBeCloseTo(0.4, 12);
    expect(p.sunPosition[2]).toBeCloseTo(-1, 12);
    expect(p.turbidity).toBeCloseTo(2, 12);
    expect(p.rayleigh).toBeCloseTo(1, 12);
    // Fixed Mie params — independent of time of day.
    expect(p.mieCoefficient).toBe(0.005);
    expect(p.mieDirectionalG).toBe(0.8);
  });

  it('matches the closed-form reference across the whole arc', () => {
    for (const t of [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1]) {
      const got = skyParamsFor(t);
      const ref = skyParamsRef(t);
      expect(got.sunPosition[0]).toBeCloseTo(ref.sunPosition[0], 12);
      expect(got.sunPosition[1]).toBeCloseTo(ref.sunPosition[1], 12);
      expect(got.sunPosition[2]).toBeCloseTo(ref.sunPosition[2], 12);
      expect(got.turbidity).toBeCloseTo(ref.turbidity, 12);
      expect(got.rayleigh).toBeCloseTo(ref.rayleigh, 12);
    }
  });

  it('keeps the sun above the horizon (sunY floored at 0.05) and behind the panel (sunZ<0)', () => {
    for (let t = 0; t <= 1.0001; t += 0.05) {
      const p = skyParamsFor(t);
      // sunY floor: cos(theta)*0.4 dips to 0 at the horizons, clamp holds 0.05.
      expect(p.sunPosition[1]).toBeGreaterThanOrEqual(0.05 - 1e-12);
      // sunZ = -cos(theta)*0.5 - 0.5 ∈ [-1, -0.5] → always strictly behind.
      expect(p.sunPosition[2]).toBeLessThanOrEqual(-0.5 + 1e-12);
      expect(p.sunPosition[2]).toBeGreaterThanOrEqual(-1 - 1e-12);
    }
  });

  it('is east/west mirror-symmetric about noon (only the x sign flips)', () => {
    for (const d of [0.1, 0.25, 0.4]) {
      const morning = skyParamsFor(0.5 - d);
      const evening = skyParamsFor(0.5 + d);
      expect(morning.sunPosition[0]).toBeCloseTo(-evening.sunPosition[0], 12);
      expect(morning.sunPosition[1]).toBeCloseTo(evening.sunPosition[1], 12);
      expect(morning.sunPosition[2]).toBeCloseTo(evening.sunPosition[2], 12);
      expect(morning.turbidity).toBeCloseTo(evening.turbidity, 12);
      expect(morning.rayleigh).toBeCloseTo(evening.rayleigh, 12);
    }
  });

  it('turbidity and rayleigh rise monotonically from noon toward the horizons', () => {
    const samples = [0.5, 0.4, 0.3, 0.2, 0.1, 0].map(skyParamsFor);
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i].turbidity).toBeGreaterThan(samples[i - 1].turbidity);
      expect(samples[i].rayleigh).toBeGreaterThan(samples[i - 1].rayleigh);
    }
    // Documented ranges: turbidity 2..8, rayleigh 1..3 at the horizons.
    expect(samples.at(-1)!.turbidity).toBeCloseTo(8, 10);
    expect(samples.at(-1)!.rayleigh).toBeCloseTo(3, 12);
  });

  it('clamps t outside [0,1] to the dawn/dusk endpoints', () => {
    expect(skyParamsFor(-5)).toEqual(skyParamsFor(0));
    expect(skyParamsFor(99)).toEqual(skyParamsFor(1));
  });
});

describe('worldSunPosition / SUN_LIGHT_DISTANCE', () => {
  it('scales each sunPosition component by SUN_LIGHT_DISTANCE (12)', () => {
    expect(SUN_LIGHT_DISTANCE).toBe(12);
    const params: SkyParams = {
      sunPosition: [0.5, -0.25, 2],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
    };
    expect(worldSunPosition(params)).toEqual([6, -3, 24]);
  });

  it('preserves the raw (non-unit) sun vector direction — magnitude scales by exactly 12', () => {
    const p = skyParamsFor(0.5);
    const rawLen = Math.hypot(...p.sunPosition);
    const worldLen = Math.hypot(...worldSunPosition(p));
    expect(worldLen).toBeCloseTo(rawLen * SUN_LIGHT_DISTANCE, 10);
  });
});

describe('getSunIntensity — discrete time-of-day buckets', () => {
  it('selects the canonical intensity for each phase', () => {
    expect(getSunIntensity(0.5)).toBe(SUN_INTENSITY.noon);
    expect(getSunIntensity(0.25)).toBe(SUN_INTENSITY.afternoon);
    expect(getSunIntensity(0.1)).toBe(SUN_INTENSITY.sunset);
    expect(getSunIntensity(0.02)).toBe(SUN_INTENSITY.twilight);
  });

  it('pins the exact (strict-inequality) bucket boundaries', () => {
    // Branches use strict < / >, so the lower edge of each window falls THROUGH
    // to the next-brighter bucket. These are the load-bearing edge cases.
    expect(getSunIntensity(0.05)).toBe(SUN_INTENSITY.sunset); // not <0.05 → not twilight
    expect(getSunIntensity(0.15)).toBe(SUN_INTENSITY.afternoon); // not <0.15 → not sunset
    expect(getSunIntensity(0.35)).toBe(SUN_INTENSITY.noon); // not <0.35 → not afternoon
    expect(getSunIntensity(0.65)).toBe(SUN_INTENSITY.noon); // not >0.65 → still noon
    expect(getSunIntensity(0.85)).toBe(SUN_INTENSITY.afternoon); // not >0.85
    expect(getSunIntensity(0.95)).toBe(SUN_INTENSITY.sunset); // not >0.95
  });

  it('is symmetric about noon at the bucket interiors', () => {
    expect(getSunIntensity(0.25)).toBe(getSunIntensity(0.75));
    expect(getSunIntensity(0.1)).toBe(getSunIntensity(0.9));
    expect(getSunIntensity(0.0)).toBe(getSunIntensity(1.0));
  });

  it('clamps out-of-range t to the twilight floor', () => {
    expect(getSunIntensity(-3)).toBe(SUN_INTENSITY.twilight);
    expect(getSunIntensity(7)).toBe(SUN_INTENSITY.twilight);
  });

  it('intensity is non-increasing as the sun moves from noon to twilight', () => {
    const seq = [0.5, 0.3, 0.1, 0.02].map(getSunIntensity);
    for (let i = 1; i < seq.length; i++) {
      expect(seq[i]).toBeLessThan(seq[i - 1]);
    }
  });
});

describe('SUN_INTENSITY table', () => {
  it('encodes post-r155 physical units as pre-r155 multipliers × π', () => {
    // Each value = the documented pre-r155 multiplier times π.
    expect(SUN_INTENSITY.noon).toBeCloseTo(Math.PI * 1.0, 12);
    expect(SUN_INTENSITY.afternoon).toBeCloseTo(Math.PI * 0.7, 12);
    expect(SUN_INTENSITY.sunset).toBeCloseTo(Math.PI * 0.3, 12);
    expect(SUN_INTENSITY.twilight).toBeCloseTo(Math.PI * 0.02, 12);
    expect(SUN_INTENSITY.moonlight).toBeCloseTo(Math.PI * 0.001, 12);
    expect(SUN_INTENSITY.overcast).toBe(0);
  });

  it('is strictly ordered brightest→dimmest (ignoring the overcast=0 special case)', () => {
    expect(SUN_INTENSITY.noon).toBeGreaterThan(SUN_INTENSITY.afternoon);
    expect(SUN_INTENSITY.afternoon).toBeGreaterThan(SUN_INTENSITY.sunset);
    expect(SUN_INTENSITY.sunset).toBeGreaterThan(SUN_INTENSITY.twilight);
    expect(SUN_INTENSITY.twilight).toBeGreaterThan(SUN_INTENSITY.moonlight);
    expect(SUN_INTENSITY.moonlight).toBeGreaterThan(0);
  });
});

describe('lumen → three.js intensity conversions', () => {
  it('point light: candela = lumens / 4π (full-sphere emission)', () => {
    // An 800 lm bulb radiating into 4π sr → 800/(4π) ≈ 63.66 cd.
    expect(pointIntensityFromLumens(800)).toBeCloseTo(63.6619772, 6);
    expect(pointIntensityFromLumens(800)).toBeCloseTo(800 / (4 * Math.PI), 12);
    expect(pointIntensityFromLumens(0)).toBe(0);
  });

  it('point intensity is linear in lumens', () => {
    expect(pointIntensityFromLumens(1600)).toBeCloseTo(2 * pointIntensityFromLumens(800), 12);
  });

  it('rect-area light: cd/m² = lumens / (area·π)', () => {
    // 1000 lm over 0.5 m² → 1000/(0.5π) ≈ 636.62 cd/m².
    expect(rectAreaIntensityFromLumens(1000, 0.5)).toBeCloseTo(636.6197724, 6);
    expect(rectAreaIntensityFromLumens(1000, 0.5)).toBeCloseTo(1000 / (0.5 * Math.PI), 12);
  });

  it('rect-area intensity scales inversely with area (same flux, larger emitter ⇒ dimmer)', () => {
    const small = rectAreaIntensityFromLumens(1000, 1);
    const big = rectAreaIntensityFromLumens(1000, 4);
    expect(big).toBeCloseTo(small / 4, 12);
  });
});

describe('COLOR_TEMP_HEX', () => {
  it('warms (more red, less blue) as color temperature drops', () => {
    const r = (hex: number) => (hex >> 16) & 0xff;
    const b = (hex: number) => hex & 0xff;
    // Cooler temps → higher blue channel; warmer → lower blue.
    expect(b(COLOR_TEMP_HEX.candle)).toBeLessThan(b(COLOR_TEMP_HEX.noonSun));
    expect(b(COLOR_TEMP_HEX.noonSun)).toBeLessThan(b(COLOR_TEMP_HEX.twilight));
    // The warm end pins red at full.
    expect(r(COLOR_TEMP_HEX.candle)).toBe(0xff);
    // The cool/zenith end pulls red back below full.
    expect(r(COLOR_TEMP_HEX.twilight)).toBeLessThan(0xff);
  });
});

describe('computeLightingState — single source of truth for all modes', () => {
  it('normalizes the (non-unit) sky sun vector to a unit direction', () => {
    const sky = skyParamsFor(0.3);
    const s = computeLightingState({ timeOfDay: 0.3, skyParams: sky, isNight: false });
    expect(Math.hypot(...s.sunDirection)).toBeCloseTo(1, 12);
    // Direction must be parallel to the raw sky vector (same sign per component).
    const raw = sky.sunPosition;
    const len = Math.hypot(...raw);
    expect(s.sunDirection[0]).toBeCloseTo(raw[0] / len, 12);
    expect(s.sunDirection[1]).toBeCloseTo(raw[1] / len, 12);
    expect(s.sunDirection[2]).toBeCloseTo(raw[2] / len, 12);
  });

  it('applies the day intensity policy: getSunIntensity(t) × multiplier', () => {
    const s = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: false,
      intensityMultiplier: 3,
    });
    expect(s.sunIntensity).toBeCloseTo(SUN_INTENSITY.noon * 3, 12);
  });

  it('defaults intensityMultiplier to 1 (PT parity)', () => {
    const s = computeLightingState({
      timeOfDay: 0.25,
      skyParams: skyParamsFor(0.25),
      isNight: false,
    });
    expect(s.sunIntensity).toBeCloseTo(SUN_INTENSITY.afternoon, 12);
  });

  it('night mode overrides time-of-day with the moonlight floor', () => {
    const s = computeLightingState({
      timeOfDay: 0.5, // noon-of-day, but night flag wins
      skyParams: skyParamsFor(0.5),
      isNight: true,
      intensityMultiplier: 2,
    });
    expect(s.sunIntensity).toBeCloseTo(SUN_INTENSITY.moonlight * 2, 12);
  });

  it('sky irradiance is half the sun intensity by day, a tiny constant by night', () => {
    const day = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: false,
    });
    expect(day.skyIrradiance).toBeCloseTo(0.5 * day.sunIntensity, 12);

    const night = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: true,
    });
    expect(night.skyIrradiance).toBe(0.02);
  });

  it('night sky tint is the fixed dark-blue dome', () => {
    const night = computeLightingState({
      timeOfDay: 0.7,
      skyParams: skyParamsFor(0.7),
      isNight: true,
    });
    expect(night.skyTint).toEqual([0.05, 0.08, 0.15]);
  });

  it('day sky tint interpolates blue→warm with turbidity', () => {
    // At noon (turbidity 2) the tint hits the cool-blue endpoint.
    const noon = computeLightingState({
      timeOfDay: 0.5,
      skyParams: skyParamsFor(0.5),
      isNight: false,
    });
    expect(noon.skyTint[0]).toBeCloseTo(0.55, 10);
    expect(noon.skyTint[1]).toBeCloseTo(0.75, 10);
    expect(noon.skyTint[2]).toBeCloseTo(1.0, 10);

    // At the horizon (turbidity 8) it hits the warm-red endpoint.
    const dawn = computeLightingState({
      timeOfDay: 0.0,
      skyParams: skyParamsFor(0.0),
      isNight: false,
    });
    expect(dawn.skyTint[0]).toBeCloseTo(1.0, 6);
    expect(dawn.skyTint[1]).toBeCloseTo(0.8, 6);
    expect(dawn.skyTint[2]).toBeCloseTo(0.35, 6);

    // Monotonicity: redder + less blue as you move from noon to horizon.
    expect(dawn.skyTint[0]).toBeGreaterThan(noon.skyTint[0]);
    expect(dawn.skyTint[2]).toBeLessThan(noon.skyTint[2]);
  });

  it('arbitrary turbidity lands on the exact linear-blend tint', () => {
    // turbidity 5 → blend factor (5-2)/6 = 0.5 → midpoint of each channel pair.
    const sky: SkyParams = {
      sunPosition: [0, 1, 0],
      turbidity: 5,
      rayleigh: 2,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
    };
    const s = computeLightingState({ timeOfDay: 0.5, skyParams: sky, isNight: false });
    expect(s.skyTint[0]).toBeCloseTo(0.55 + 0.45 * 0.5, 12); // 0.775
    expect(s.skyTint[1]).toBeCloseTo(0.75 + 0.05 * 0.5, 12); // 0.775
    expect(s.skyTint[2]).toBeCloseTo(1.0 - 0.65 * 0.5, 12); // 0.675
  });

  it('guards a degenerate zero-length sun vector (no NaN/Infinity)', () => {
    const sky: SkyParams = {
      sunPosition: [0, 0, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
    };
    const s = computeLightingState({ timeOfDay: 0.5, skyParams: sky, isNight: false });
    // sunLen falls back to 1, so the direction stays [0,0,0] rather than NaN.
    expect(s.sunDirection.every(Number.isFinite)).toBe(true);
    expect(s.sunDirection).toEqual([0, 0, 0]);
  });
});

describe('PT sun-disc geometry constants', () => {
  it('pins the area-light placement distance', () => {
    expect(PT_SUN_DISTANCE).toBe(10000);
    expect(SUN_ANGULAR_RADIUS).toBeCloseTo(0.00436, 12);
  });

  it('disc diameter subtends 2× the sun angular radius at PT_SUN_DISTANCE', () => {
    // D = 2·d·tan(angRadius) — the angle the full disc spans from the room.
    expect(PT_SUN_DISC_DIAMETER).toBeCloseTo(2 * PT_SUN_DISTANCE * Math.tan(SUN_ANGULAR_RADIUS), 9);
    // Half-angle recovered from the geometry equals SUN_ANGULAR_RADIUS.
    const recoveredAngle = Math.atan(PT_SUN_DISC_DIAMETER / 2 / PT_SUN_DISTANCE);
    expect(recoveredAngle).toBeCloseTo(SUN_ANGULAR_RADIUS, 9);
    // ≈ 87 inches across at distance 10000.
    expect(PT_SUN_DISC_DIAMETER).toBeCloseTo(87.2, 1);
  });

  it('area intensity calibrates near a π-irradiance directional sun', () => {
    expect(PT_SUN_AREA_INTENSITY).toBe(50_000);
    // The documented calibration: I·π·(D/2)²/d² should land in the π ballpark.
    const irradianceProxy =
      (PT_SUN_AREA_INTENSITY * Math.PI * (PT_SUN_DISC_DIAMETER / 2) ** 2) / PT_SUN_DISTANCE ** 2;
    expect(irradianceProxy).toBeGreaterThan(2.5);
    expect(irradianceProxy).toBeLessThan(3.5); // within rounding of π
  });
});
