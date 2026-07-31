/**
 * Canonical binary32 publication helpers for non-negative RGB radiance.
 *
 * JavaScript evaluates arithmetic in binary64, while every renderer consuming
 * these helpers stores the operands or the folded result in binary32.  Building
 * CPU emitter distributions or mutation identities from the binary64 product
 * can therefore disagree with the value the shader actually sees.  These
 * helpers make the storage and multiplication order explicit:
 *
 *   f32(f32(rgb) * f32(scale))
 *
 * Authored zero remains a valid radiometric value. A positive scalar or
 * non-black tuple that disappears completely during binary32 publication is
 * rejected because silently disabling an emitter is not an equivalent value.
 */

export type RadianceRgb = readonly [number, number, number];

function radianceRangeError(label: string, detail: string, value?: unknown): RangeError {
  const suffix = arguments.length >= 3 ? `; received ${String(value)}.` : '.';
  return new RangeError(`${label} ${detail}${suffix}`);
}

/** Snapshot a non-negative scalar into its exact finite binary32 value. */
export function packNonNegativeRadianceScalarF32(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw radianceRangeError(label, 'must be finite and non-negative', value);
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed)) {
    throw radianceRangeError(label, 'must remain finite after Float32 packing', value);
  }
  if (value > 0 && packed === 0) {
    throw radianceRangeError(
      label,
      'must remain positive after Float32 packing when authored positive',
      value,
    );
  }
  return packed === 0 ? 0 : packed;
}

/**
 * Snapshot a non-negative RGB tuple into binary32.
 *
 * Individual positive lanes may underflow when another lane preserves the
 * tuple's non-black identity. A wholly positive-to-black collapse is rejected.
 */
export function packNonNegativeRadianceRgbF32(
  value: RadianceRgb,
  label: string,
): [number, number, number] {
  const packed = value.map((component, index) => {
    if (!Number.isFinite(component) || component < 0) {
      throw radianceRangeError(
        `${label}[${index}]`,
        'must be finite and non-negative',
        component,
      );
    }
    const lane = Math.fround(component);
    if (!Number.isFinite(lane)) {
      throw radianceRangeError(
        `${label}[${index}]`,
        'must remain finite after Float32 packing',
        component,
      );
    }
    return lane === 0 ? 0 : lane;
  }) as [number, number, number];
  if (
    value.some((component) => component > 0) &&
    packed.every((component) => component === 0)
  ) {
    throw radianceRangeError(
      label,
      'must not collapse completely to zero after Float32 packing',
    );
  }
  return packed;
}

/**
 * Canonicalize RGB×scalar radiance in shader evaluation order.
 *
 * The returned `value` and `scale` are the exact operands to publish when the
 * shader multiplies them. `scaled` is the exact folded radiance to publish when
 * the host collapses the pair into one RGB field.
 */
export function packRadianceRgbScaleF32(
  value: RadianceRgb,
  scale: number,
  label: string,
): {
  readonly value: [number, number, number];
  readonly scale: number;
  readonly scaled: [number, number, number];
} {
  const packedValue = packNonNegativeRadianceRgbF32(value, `${label} color`);
  const packedScale = packNonNegativeRadianceScalarF32(scale, `${label} intensity`);
  const scaled = packedValue.map((component) =>
    Math.fround(component * packedScale)
  ) as [number, number, number];
  if (scaled.some((component) => !Number.isFinite(component))) {
    throw radianceRangeError(
      `${label} color×intensity`,
      'must remain finite in Float32',
    );
  }
  if (
    packedScale > 0 &&
    packedValue.some((component) => component > 0) &&
    scaled.every((component) => component === 0)
  ) {
    throw radianceRangeError(
      `positive ${label} color×intensity`,
      'must not underflow completely to zero in Float32',
    );
  }
  return { value: packedValue, scale: packedScale, scaled };
}

/**
 * Canonicalize a lane-wise RGB×RGB radiance modulation in binary32.
 *
 * This is used for CPU-readable emissive-map modulation. A black result caused
 * by disjoint non-zero channels is valid; a result that should have a positive
 * lane but loses every such lane to underflow is rejected.
 */
export function packRadianceRgbProductF32(
  left: RadianceRgb,
  right: RadianceRgb,
  label: string,
): [number, number, number] {
  const packedLeft = packNonNegativeRadianceRgbF32(left, `${label} left`);
  const packedRight = packNonNegativeRadianceRgbF32(right, `${label} right`);
  const product = packedLeft.map((component, index) =>
    Math.fround(component * packedRight[index]!)
  ) as [number, number, number];
  if (product.some((component) => !Number.isFinite(component))) {
    throw radianceRangeError(label, 'must remain finite in Float32');
  }
  if (
    packedLeft.some((component, index) =>
      component > 0 && packedRight[index]! > 0
    ) &&
    product.every((component) => component === 0)
  ) {
    throw radianceRangeError(
      `positive ${label}`,
      'must not underflow completely to zero in Float32',
    );
  }
  return product.map((component) => component === 0 ? 0 : component) as [
    number,
    number,
    number,
  ];
}

/**
 * Multiply two already-published non-negative binary32 radiometric scalars.
 *
 * A product of zero is valid when either operand is zero. Two positive
 * operands may neither overflow nor disappear through binary32 underflow.
 */
export function multiplyNonNegativeRadianceScalarsF32(
  left: number,
  right: number,
  label: string,
): number {
  const product = Math.fround(left * right);
  if (!Number.isFinite(product)) {
    throw radianceRangeError(label, 'must remain finite in Float32');
  }
  if (left > 0 && right > 0 && product === 0) {
    throw radianceRangeError(
      `positive ${label}`,
      'must not underflow to zero in Float32',
    );
  }
  return product === 0 ? 0 : product;
}
