/** Internal runtime guards shared by the public numerical helpers. */

export function requireFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
  return value;
}

export function requireNonNegative(value: number, label: string): number {
  requireFinite(value, label);
  if (value < 0) throw new RangeError(`${label} must be >= 0`);
  return value;
}

export function requirePositive(value: number, label: string): number {
  requireFinite(value, label);
  if (value <= 0) throw new RangeError(`${label} must be > 0`);
  return value;
}

export function requireInteger(value: number, label: string, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} must be a safe integer in [${min}, ${max}]`);
  }
  return value;
}

export function requireUnitRandom(value: number, label: string): number {
  requireFinite(value, label);
  if (value < 0 || value >= 1) {
    throw new RangeError(`${label} must be in [0, 1)`);
  }
  return value;
}

export function requireFiniteVec3(
  value: readonly [number, number, number],
  label: string,
): readonly [number, number, number] {
  requireFinite(value[0], `${label}[0]`);
  requireFinite(value[1], `${label}[1]`);
  requireFinite(value[2], `${label}[2]`);
  return value;
}

export function finiteOr(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Number(value) : fallback;
}

/** Overflow-safe a/(a+b) for finite non-negative a,b. */
export function normalizedPairFirst(a: number, b: number, bothZero = 0.5): number {
  if (a === 0 && b === 0) return bothZero;
  const scale = Math.max(a, b);
  const as = a / scale;
  const bs = b / scale;
  return as / (as + bs);
}

export function saturatingPositiveMultiply(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  const log = Math.log(a) + Math.log(b);
  if (log >= Math.log(Number.MAX_VALUE)) return Number.MAX_VALUE;
  if (log <= Math.log(Number.MIN_VALUE)) return 0;
  return Math.exp(log);
}
