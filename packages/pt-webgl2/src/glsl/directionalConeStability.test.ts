import { describe, expect, it } from 'vitest';
import * as LightSamplingModule from './shader/sampling/light_sampling_functions.glsl.js';

const LIGHT_SAMPLING = (
  LightSamplingModule as unknown as Record<string, string>
)['light_sampling_functions']!;

function stableConePdfF32(angularDiameter: number): number {
  const diameter = Math.fround(angularDiameter);
  const quarterAngle = Math.fround(diameter * 0.25);
  const sinQuarter = Math.fround(
    quarterAngle < 1e-3 ? quarterAngle : Math.sin(quarterAngle),
  );
  const sinQuarterSquared = Math.fround(sinQuarter * sinQuarter);
  const oneMinusCosHalf = Math.fround(2 * sinQuarterSquared);
  const solidAngle = Math.fround(
    Math.fround(2 * Math.PI) * oneMinusCosHalf,
  );
  return Math.fround(1 / solidAngle);
}

type Vec3 = [number, number, number];
const f32 = Math.fround;

function dotF32(a: Vec3, b: Vec3): number {
  return f32(
    f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) +
    f32(a[2] * b[2]),
  );
}

function crossF32(a: Vec3, b: Vec3): Vec3 {
  return [
    f32(f32(a[1] * b[2]) - f32(a[2] * b[1])),
    f32(f32(a[2] * b[0]) - f32(a[0] * b[2])),
    f32(f32(a[0] * b[1]) - f32(a[1] * b[0])),
  ];
}

function normalizeF32(value: Vec3): Vec3 {
  const length = f32(Math.sqrt(Math.max(0, dotF32(value, value))));
  return value.map((component) => f32(component / length)) as Vec3;
}

function packedAxis(value: Vec3): Vec3 {
  const length = Math.hypot(...value);
  return value.map((component) => f32(component / length)) as Vec3;
}

function sampleDirectionalConeF32(
  sourceAxis: Vec3,
  angularDiameter: number,
  azimuth: number,
): { readonly axis: Vec3; readonly sampled: Vec3 } {
  const axis = normalizeF32(sourceAxis);
  const other: Vec3 = Math.abs(axis[0]) > 0.5 ? [0, 1, 0] : [1, 0, 0];
  const ortho = normalizeF32(crossF32(axis, other));
  const ortho2 = normalizeF32(crossF32(axis, ortho));

  const diameter = f32(angularDiameter);
  const quarterAngle = f32(diameter * 0.25);
  const sinQuarter = f32(
    quarterAngle < 1e-3 ? quarterAngle : Math.sin(quarterAngle),
  );
  const oneMinusCosHalf = f32(2 * f32(sinQuarter * sinQuarter));
  const cosTheta = f32(1 - oneMinusCosHalf);
  const sinTheta = f32(Math.sqrt(Math.max(
    0,
    f32(oneMinusCosHalf * f32(2 - oneMinusCosHalf)),
  )));
  const phi = f32(f32(2 * Math.PI) * f32(azimuth));
  const local: Vec3 = [
    f32(f32(Math.cos(phi)) * sinTheta),
    f32(f32(Math.sin(phi)) * sinTheta),
    cosTheta,
  ];
  const sampled = normalizeF32([0, 1, 2].map((component) =>
    f32(
      f32(f32(ortho2[component]! * local[0]) + f32(ortho[component]! * local[1])) +
      f32(axis[component]! * local[2]),
    ),
  ) as Vec3);
  return { axis, sampled };
}

describe('directional cone float32 stability', () => {
  it('uses the sin-squared identity for both sampling and solid-angle PDF', () => {
    const compact = LIGHT_SAMPLING.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'float sinQuarter = quarterAngle < 1e-3 ? quarterAngle : sin( quarterAngle )',
    );
    expect(compact).toContain(
      'float oneMinusCosHalf = 2.0 * sinQuarterSquared',
    );
    expect(compact).toContain(
      'oneMinusCosTheta * ( 2.0 - oneMinusCosTheta )',
    );
    expect(compact).toContain(
      'float solidAngle = 2.0 * PI * oneMinusCosHalf',
    );
    expect(compact).not.toContain(
      '2.0 * PI * ( 1.0 - cosHalfAngle )',
    );
    expect(compact).not.toContain(
      'mix( cosHalfAngle, 1.0, uv.x )',
    );
  });

  it('keeps narrow supported diameters finite after the old cosine subtraction collapses', () => {
    const diameter = 4e-6;
    const oldOneMinusCos = Math.fround(
      1 - Math.fround(Math.cos(Math.fround(diameter * 0.5))),
    );
    expect(oldOneMinusCos).toBe(0);

    const pdf = stableConePdfF32(diameter);
    const exactSmallAnglePdf = 4 / (Math.PI * diameter * diameter);
    expect(pdf).toBeGreaterThan(0);
    expect(Number.isFinite(pdf)).toBe(true);
    expect(pdf / exactSmallAnglePdf).toBeCloseTo(1, 5);
  });

  it('returns finite positive PDFs across the accepted practical range', () => {
    for (const diameter of [4e-6, 1e-5, 1e-4, 0.01, 1, Math.PI]) {
      const pdf = stableConePdfF32(diameter);
      expect(pdf, `diameter=${diameter}`).toBeGreaterThan(0);
      expect(Number.isFinite(pdf), `diameter=${diameter}`).toBe(true);
    }
  });

  it('retains a world-space rim displacement for generic axes and azimuths', () => {
    const axes: Vec3[] = [
      [0, 1, 0],
      packedAxis([1, 2, 3]),
      packedAxis([-0.37, 0.81, 0.45]),
    ];
    for (const axis of axes) {
      for (const azimuth of [0, 0.125, 0.25, 0.375, 0.5, 0.73]) {
        const { axis: normalizedAxis, sampled } = sampleDirectionalConeF32(
          axis,
          4e-6,
          azimuth,
        );
        expect(
          sampled.some((component, index) => component !== normalizedAxis[index]),
          `axis=${axis.join(',')} azimuth=${azimuth}`,
        ).toBe(true);
      }
    }
  });
});
