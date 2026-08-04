/**
 * Semantic version of the live ReSTIR reservoir weight representation.
 *
 * Version 1 stores the corrected logarithmic contribution mass
 * `H = log2(W_uncapped * pHat_selected)` in the historical running-weight lane
 * and capped `log2(W)` in the historical linear-weight lane. ReSTIR-GI also
 * stores the selected sample's exact `log2(pHat_native)` in word 26; it never
 * round-trips that density through a bounded linear f32. Keeping all three
 * values logarithmic prevents a tiny normalization or target density from
 * underflowing before it combines with representable radiance.
 */
export const RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1 = 1 as const;

/** Finite log-domain zero shared by the DI/GI reservoir snapshot validators. */
export const RESTIR_RESERVOIR_LOG_ZERO = -Math.fround(
  3.4028234663852886e38,
);
