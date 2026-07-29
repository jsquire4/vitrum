/**
 * Physical angular radius of the solar disc used by the Preetham sky baker.
 *
 * 0.00436 radians is approximately 0.25 degrees (a 0.5-degree diameter).
 */
export const SOLAR_ANGULAR_RADIUS = 0.00436;

const TWO_PI = 2 * Math.PI;
const POLAR_AXIS_EPSILON = 1e-12;
const MIN_THETA_QUADRATURE_STEPS = 16;
const MAX_THETA_QUADRATURE_STEPS = 512;
const TARGET_THETA_STEP = SOLAR_ANGULAR_RADIUS / 64;

function requireDimension(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function normalizeLongitude(phi: number): number {
  const wrapped = phi % TWO_PI;
  return wrapped < 0 ? wrapped + TWO_PI : wrapped;
}

/**
 * Average solar-cap occupancy for each equirectangular texel.
 *
 * The returned multiplier is defined against the same centre-sampled
 * `sin(theta) * dTheta * dPhi` measure used by the sky baker. Integrating the
 * multipliers with that measure therefore recovers the analytic cap solid
 * angle exactly (up to floating-point roundoff).
 *
 * A fixed set of point samples is insufficient here: at 4K, a polar solar cap
 * intersects more texels than a conventional 4096-sample stratum set and leaves
 * visible holes. This implementation instead integrates the cap's allowed
 * longitude interval into every overlapped texel, using deterministic
 * composite theta quadrature. The exact-pole case has a closed-form row
 * integral and is handled analytically.
 *
 * This is an internal production helper consumed by `preethamSky.ts`; it is
 * exported from its module so its high-resolution occupancy invariant can be
 * tested without allocating and evaluating an entire sky bake.
 */
export function solarDiscTexelCoverage(
  width: number,
  height: number,
  sunDirection: readonly [number, number, number],
): Float64Array {
  requireDimension(width, 'solarDiscTexelCoverage.width');
  requireDimension(height, 'solarDiscTexelCoverage.height');
  const sunLength = Math.hypot(
    sunDirection[0],
    sunDirection[1],
    sunDirection[2],
  );
  if (!Number.isFinite(sunLength) || sunLength <= 0) {
    throw new RangeError(
      'solarDiscTexelCoverage.sunDirection must be a finite non-zero vector',
    );
  }

  const sx = sunDirection[0] / sunLength;
  const sy = sunDirection[1] / sunLength;
  const sz = sunDirection[2] / sunLength;
  const thetaSun = Math.acos(Math.max(-1, Math.min(1, sy)));
  const sinThetaSun = Math.hypot(sx, sz);
  const sunPhi = normalizeLongitude(Math.atan2(sz, sx));
  const cosRadius = Math.cos(SOLAR_ANGULAR_RADIUS);
  const capSolidAngle = TWO_PI * (1 - cosRadius);
  const dTheta = Math.PI / height;
  const dPhi = TWO_PI / width;
  const coverage = new Float64Array(width * height);

  const thetaMin = Math.max(0, thetaSun - SOLAR_ANGULAR_RADIUS);
  const thetaMax = Math.min(Math.PI, thetaSun + SOLAR_ANGULAR_RADIUS);
  const firstRow = Math.max(0, Math.floor(thetaMin / dTheta));
  const lastRow = Math.min(
    height - 1,
    Math.ceil(thetaMax / dTheta) - 1,
  );
  let depositedSolidAngle = 0;

  if (sinThetaSun <= POLAR_AXIS_EPSILON) {
    const northPole = sy >= 0;
    const capMin = northPole ? 0 : Math.PI - SOLAR_ANGULAR_RADIUS;
    const capMax = northPole ? SOLAR_ANGULAR_RADIUS : Math.PI;
    for (let row = firstRow; row <= lastRow; row += 1) {
      const thetaLo = Math.max(row * dTheta, capMin);
      const thetaHi = Math.min((row + 1) * dTheta, capMax);
      if (thetaHi <= thetaLo) continue;
      const texelSolidAngle =
        dPhi * (Math.cos(thetaLo) - Math.cos(thetaHi));
      const rowOffset = row * width;
      for (let x = 0; x < width; x += 1) {
        coverage[rowOffset + x] = texelSolidAngle;
      }
      depositedSolidAngle += texelSolidAngle * width;
    }
  } else {
    const addLongitudeInterval = (
      row: number,
      intervalLo: number,
      intervalHi: number,
      thetaWeight: number,
    ): void => {
      if (intervalHi <= intervalLo) return;
      const firstColumn = Math.max(0, Math.floor(intervalLo / dPhi));
      const lastColumn = Math.min(
        width - 1,
        Math.ceil(intervalHi / dPhi) - 1,
      );
      const rowOffset = row * width;
      for (let x = firstColumn; x <= lastColumn; x += 1) {
        const phiLo = Math.max(intervalLo, x * dPhi);
        const phiHi = Math.min(intervalHi, (x + 1) * dPhi);
        const overlap = Math.max(0, phiHi - phiLo);
        if (overlap <= 0) continue;
        const solidAngle = thetaWeight * overlap;
        coverage[rowOffset + x]! += solidAngle;
        depositedSolidAngle += solidAngle;
      }
    };

    const addPeriodicLongitudeSpan = (
      row: number,
      halfSpan: number,
      thetaWeight: number,
    ): void => {
      if (halfSpan >= Math.PI) {
        addLongitudeInterval(row, 0, TWO_PI, thetaWeight);
        return;
      }
      const start = normalizeLongitude(sunPhi - halfSpan);
      const end = start + 2 * halfSpan;
      addLongitudeInterval(row, start, Math.min(end, TWO_PI), thetaWeight);
      if (end > TWO_PI) {
        addLongitudeInterval(row, 0, end - TWO_PI, thetaWeight);
      }
    };

    for (let row = firstRow; row <= lastRow; row += 1) {
      const thetaLo = Math.max(row * dTheta, thetaMin);
      const thetaHi = Math.min((row + 1) * dTheta, thetaMax);
      if (thetaHi <= thetaLo) continue;
      const thetaSpan = thetaHi - thetaLo;
      const stepCount = Math.max(
        MIN_THETA_QUADRATURE_STEPS,
        Math.min(
          MAX_THETA_QUADRATURE_STEPS,
          Math.ceil(thetaSpan / TARGET_THETA_STEP),
        ),
      );
      const thetaStep = thetaSpan / stepCount;
      for (let sample = 0; sample < stepCount; sample += 1) {
        const theta = thetaLo + (sample + 0.5) * thetaStep;
        const sinTheta = Math.sin(theta);
        const thetaWeight = sinTheta * thetaStep;
        const horizontalTerm = sinTheta * sinThetaSun;
        if (horizontalTerm <= Number.EPSILON) {
          if (Math.cos(theta) * sy >= cosRadius) {
            addPeriodicLongitudeSpan(row, Math.PI, thetaWeight);
          }
          continue;
        }
        const threshold =
          (cosRadius - Math.cos(theta) * sy) / horizontalTerm;
        if (threshold <= -1) {
          addPeriodicLongitudeSpan(row, Math.PI, thetaWeight);
        } else if (threshold < 1) {
          addPeriodicLongitudeSpan(
            row,
            Math.acos(Math.max(-1, Math.min(1, threshold))),
            thetaWeight,
          );
        }
      }
    }
  }

  if (!(depositedSolidAngle > 0) || !Number.isFinite(depositedSolidAngle)) {
    throw new RangeError(
      'solarDiscTexelCoverage could not integrate a finite solar cap',
    );
  }

  // Composite quadrature controls distribution across intersected texels; this
  // final scalar normalization pins the conserved energy to the analytic cap.
  const normalization = capSolidAngle / depositedSolidAngle;
  for (let row = firstRow; row <= lastRow; row += 1) {
    const centerTheta = ((row + 0.5) / height) * Math.PI;
    const integrationMeasure =
      dTheta * dPhi * Math.max(Math.sin(centerTheta), 1e-6);
    const rowOffset = row * width;
    for (let x = 0; x < width; x += 1) {
      const index = rowOffset + x;
      coverage[index] =
        (coverage[index] ?? 0) * normalization / integrationMeasure;
    }
  }

  return coverage;
}
