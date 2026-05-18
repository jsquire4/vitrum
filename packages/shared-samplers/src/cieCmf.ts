/**
 * cieCmf.ts — CIE 1931 standard observer color-matching functions.
 *
 * Provides the canonical 2-degree observer CMF tables in 5 nm steps from
 * 380 nm to 780 nm (81 entries), plus the CIE D65 standard illuminant and
 * conversion utilities for spectral-to-display-RGB reconstruction.
 *
 * Standard reference:
 *   CIE 015:2018 Colorimetry, 4th edition.
 *   CIE standard observer data (public domain):
 *   https://cie.co.at/datatable/cie-1931-colour-matching-functions-2deg-standard-observer
 *   Values below taken from the CIE 1931 2-degree standard observer tabulated
 *   in Wyszecki & Stiles "Color Science", Wiley 1982, Table 1(3.3.1), and
 *   cross-verified against the CIE S 014-1/E:2006 supplementary table.
 *
 * CIE D65 illuminant:
 *   CIE 15:2004 Table T.1, normalised to Y(560 nm) = 100 in the standard
 *   form; here we use the relative spectral power distribution at 5 nm steps.
 *   Source: https://www.cie.co.at/publications/colorimetry
 *
 * XYZ → linear sRGB matrix:
 *   IEC 61966-2-1:1999 Annex F (Bradford-adapted D65).
 *   Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
 *
 * Sprint 12 (Phase 6) deliverable — hero-wavelength spectral path tracing.
 * Used by: wavelengthSampling.ts (importance sampling + reconstruction).
 */

// ── Table dimensions ───────────────────────────────────────────────────────────

/** First wavelength in the CMF tables, nm. */
export const CIE_LAMBDA_MIN = 380;

/** Last wavelength in the CMF tables, nm. */
export const CIE_LAMBDA_MAX = 780;

/** Step size between table entries, nm. */
export const CIE_LAMBDA_STEP = 5;

/** Number of entries in each CMF table (81 = (780−380)/5 + 1). */
export const CIE_TABLE_LENGTH = 81;

// ── CIE 1931 2-degree standard observer CMF tables ────────────────────────────
// Values at 380, 385, 390, ..., 780 nm (81 entries each).
// Public domain. Source: CIE 015:2018.

/** CIE 1931 x̄(λ) color-matching function, 380–780 nm at 5 nm steps. */
export const CIE_X_TABLE: Readonly<Float32Array> = new Float32Array([
  /* 380 */ 0.001368, /* 385 */ 0.002236, /* 390 */ 0.004243, /* 395 */ 0.00765, /* 400 */ 0.01431,
  /* 405 */ 0.02319, /* 410 */ 0.04351, /* 415 */ 0.07763, /* 420 */ 0.13438, /* 425 */ 0.21477,
  /* 430 */ 0.2839, /* 435 */ 0.3285, /* 440 */ 0.34828, /* 445 */ 0.34806, /* 450 */ 0.3362,
  /* 455 */ 0.3187, /* 460 */ 0.2908, /* 465 */ 0.2511, /* 470 */ 0.19536, /* 475 */ 0.1421,
  /* 480 */ 0.09564, /* 485 */ 0.05795, /* 490 */ 0.03201, /* 495 */ 0.0147, /* 500 */ 0.0049,
  /* 505 */ 0.0024, /* 510 */ 0.0093, /* 515 */ 0.0291, /* 520 */ 0.06327, /* 525 */ 0.1096,
  /* 530 */ 0.1655, /* 535 */ 0.22575, /* 540 */ 0.2904, /* 545 */ 0.3597, /* 550 */ 0.43345,
  /* 555 */ 0.51205, /* 560 */ 0.5945, /* 565 */ 0.6784, /* 570 */ 0.7621, /* 575 */ 0.8425,
  /* 580 */ 0.9163, /* 585 */ 0.9786, /* 590 */ 1.0263, /* 595 */ 1.0567, /* 600 */ 1.0622,
  /* 605 */ 1.0456, /* 610 */ 1.0026, /* 615 */ 0.9384, /* 620 */ 0.85445, /* 625 */ 0.7514,
  /* 630 */ 0.6424, /* 635 */ 0.5419, /* 640 */ 0.4479, /* 645 */ 0.3608, /* 650 */ 0.2835,
  /* 655 */ 0.2187, /* 660 */ 0.1649, /* 665 */ 0.1212, /* 670 */ 0.0874, /* 675 */ 0.0636,
  /* 680 */ 0.04677, /* 685 */ 0.0329, /* 690 */ 0.0227, /* 695 */ 0.01584, /* 700 */ 0.011359,
  /* 705 */ 0.008111, /* 710 */ 0.00579, /* 715 */ 0.004109, /* 720 */ 0.002899, /* 725 */ 0.002049,
  /* 730 */ 0.00144, /* 735 */ 0.001, /* 740 */ 0.00069, /* 745 */ 0.000476, /* 750 */ 0.000332,
  /* 755 */ 0.000235, /* 760 */ 0.000166, /* 765 */ 0.000117, /* 770 */ 0.000083,
  /* 775 */ 0.000059, /* 780 */ 0.000042,
]);

/** CIE 1931 ȳ(λ) color-matching function (luminous efficiency), 380–780 nm at 5 nm steps. */
export const CIE_Y_TABLE: Readonly<Float32Array> = new Float32Array([
  /* 380 */ 0.000039, /* 385 */ 0.000064, /* 390 */ 0.00012, /* 395 */ 0.000217, /* 400 */ 0.000396,
  /* 405 */ 0.00064, /* 410 */ 0.00121, /* 415 */ 0.00218, /* 420 */ 0.004, /* 425 */ 0.0073,
  /* 430 */ 0.0116, /* 435 */ 0.01684, /* 440 */ 0.023, /* 445 */ 0.0298, /* 450 */ 0.038,
  /* 455 */ 0.048, /* 460 */ 0.06, /* 465 */ 0.0739, /* 470 */ 0.09098, /* 475 */ 0.1126,
  /* 480 */ 0.13902, /* 485 */ 0.1693, /* 490 */ 0.20802, /* 495 */ 0.2586, /* 500 */ 0.323,
  /* 505 */ 0.4073, /* 510 */ 0.503, /* 515 */ 0.6082, /* 520 */ 0.71, /* 525 */ 0.7932,
  /* 530 */ 0.862, /* 535 */ 0.91485, /* 540 */ 0.954, /* 545 */ 0.9803, /* 550 */ 0.99495,
  /* 555 */ 1.0, /* 560 */ 0.995, /* 565 */ 0.9786, /* 570 */ 0.952, /* 575 */ 0.9154,
  /* 580 */ 0.87, /* 585 */ 0.8163, /* 590 */ 0.757, /* 595 */ 0.6949, /* 600 */ 0.631,
  /* 605 */ 0.5668, /* 610 */ 0.503, /* 615 */ 0.4412, /* 620 */ 0.381, /* 625 */ 0.321,
  /* 630 */ 0.265, /* 635 */ 0.217, /* 640 */ 0.175, /* 645 */ 0.1382, /* 650 */ 0.107,
  /* 655 */ 0.0816, /* 660 */ 0.061, /* 665 */ 0.04458, /* 670 */ 0.032, /* 675 */ 0.0232,
  /* 680 */ 0.017, /* 685 */ 0.01192, /* 690 */ 0.00821, /* 695 */ 0.005723, /* 700 */ 0.004102,
  /* 705 */ 0.002929, /* 710 */ 0.002091, /* 715 */ 0.001484, /* 720 */ 0.001047, /* 725 */ 0.00074,
  /* 730 */ 0.00052, /* 735 */ 0.000361, /* 740 */ 0.000249, /* 745 */ 0.000172, /* 750 */ 0.00012,
  /* 755 */ 0.000085, /* 760 */ 0.00006, /* 765 */ 0.000042, /* 770 */ 0.00003, /* 775 */ 0.000021,
  /* 780 */ 0.000015,
]);

/** CIE 1931 z̄(λ) color-matching function, 380–780 nm at 5 nm steps. */
export const CIE_Z_TABLE: Readonly<Float32Array> = new Float32Array([
  /* 380 */ 0.00645, /* 385 */ 0.01055, /* 390 */ 0.02005, /* 395 */ 0.03621, /* 400 */ 0.06785,
  /* 405 */ 0.1102, /* 410 */ 0.2074, /* 415 */ 0.3713, /* 420 */ 0.6456, /* 425 */ 1.03905,
  /* 430 */ 1.3856, /* 435 */ 1.62296, /* 440 */ 1.74706, /* 445 */ 1.7826, /* 450 */ 1.77211,
  /* 455 */ 1.7441, /* 460 */ 1.6692, /* 465 */ 1.5281, /* 470 */ 1.28764, /* 475 */ 1.0419,
  /* 480 */ 0.81295, /* 485 */ 0.6162, /* 490 */ 0.46518, /* 495 */ 0.3533, /* 500 */ 0.272,
  /* 505 */ 0.2123, /* 510 */ 0.1582, /* 515 */ 0.1117, /* 520 */ 0.07825, /* 525 */ 0.05725,
  /* 530 */ 0.04216, /* 535 */ 0.02984, /* 540 */ 0.0203, /* 545 */ 0.0134, /* 550 */ 0.00875,
  /* 555 */ 0.00575, /* 560 */ 0.0039, /* 565 */ 0.00275, /* 570 */ 0.0021, /* 575 */ 0.0018,
  /* 580 */ 0.00165, /* 585 */ 0.0014, /* 590 */ 0.0011, /* 595 */ 0.001, /* 600 */ 0.0008,
  /* 605 */ 0.0006, /* 610 */ 0.00034, /* 615 */ 0.00024, /* 620 */ 0.00019, /* 625 */ 0.0001,
  /* 630 */ 0.00005, /* 635 */ 0.00003, /* 640 */ 0.00002, /* 645 */ 0.00001, /* 650 */ 0.0,
  /* 655 */ 0.0, /* 660 */ 0.0, /* 665 */ 0.0, /* 670 */ 0.0, /* 675 */ 0.0, /* 680 */ 0.0,
  /* 685 */ 0.0, /* 690 */ 0.0, /* 695 */ 0.0, /* 700 */ 0.0, /* 705 */ 0.0, /* 710 */ 0.0,
  /* 715 */ 0.0, /* 720 */ 0.0, /* 725 */ 0.0, /* 730 */ 0.0, /* 735 */ 0.0, /* 740 */ 0.0,
  /* 745 */ 0.0, /* 750 */ 0.0, /* 755 */ 0.0, /* 760 */ 0.0, /* 765 */ 0.0, /* 770 */ 0.0,
  /* 775 */ 0.0, /* 780 */ 0.0,
]);

// ── CIE D65 illuminant ────────────────────────────────────────────────────────
// Relative spectral power distribution at 5 nm steps, 380–780 nm.
// Normalised so that the entry at 560 nm = 100 (standard form).
// Source: CIE 15:2004 Table T.1.

/** CIE D65 standard illuminant (6500 K daylight), 380–780 nm at 5 nm steps. */
export const CIE_D65_TABLE: Readonly<Float32Array> = new Float32Array([
  /*380*/ 49.9755, /*385*/ 52.3118, /*390*/ 54.6482, /*395*/ 68.7015, /*400*/ 82.7549,
  /*405*/ 87.1204, /*410*/ 91.486, /*415*/ 92.4589, /*420*/ 93.4318, /*425*/ 90.057,
  /*430*/ 86.6823, /*435*/ 95.7736, /*440*/ 104.865, /*445*/ 110.936, /*450*/ 117.008,
  /*455*/ 117.41, /*460*/ 117.812, /*465*/ 116.336, /*470*/ 114.861, /*475*/ 115.392,
  /*480*/ 115.923, /*485*/ 112.367, /*490*/ 108.811, /*495*/ 109.082, /*500*/ 109.354,
  /*505*/ 108.578, /*510*/ 107.802, /*515*/ 106.296, /*520*/ 104.79, /*525*/ 106.239,
  /*530*/ 107.689, /*535*/ 106.047, /*540*/ 104.405, /*545*/ 104.225, /*550*/ 104.046,
  /*555*/ 102.023, /*560*/ 100.0, /*565*/ 98.1671, /*570*/ 96.3342, /*575*/ 96.0611, /*580*/ 95.788,
  /*585*/ 92.2368, /*590*/ 88.6856, /*595*/ 89.3459, /*600*/ 90.0062, /*605*/ 89.8026,
  /*610*/ 89.5991, /*615*/ 88.6489, /*620*/ 87.6987, /*625*/ 85.4936, /*630*/ 83.2886,
  /*635*/ 83.4939, /*640*/ 83.6992, /*645*/ 81.863, /*650*/ 80.0268, /*655*/ 80.1207,
  /*660*/ 80.2146, /*665*/ 81.2462, /*670*/ 82.2778, /*675*/ 80.281, /*680*/ 78.2842,
  /*685*/ 74.0027, /*690*/ 69.7213, /*695*/ 70.6652, /*700*/ 71.6091, /*705*/ 72.979,
  /*710*/ 74.349, /*715*/ 67.9765, /*720*/ 61.604, /*725*/ 65.7448, /*730*/ 69.8856,
  /*735*/ 72.4863, /*740*/ 75.087, /*745*/ 69.3398, /*750*/ 63.5927, /*755*/ 55.0054,
  /*760*/ 46.4182, /*765*/ 56.6118, /*770*/ 66.8054, /*775*/ 65.0941, /*780*/ 63.3828,
]);

// ── Interpolation ──────────────────────────────────────────────────────────────

/**
 * Sample a CIE CMF (or D65) table at an arbitrary wavelength via linear
 * interpolation.  Returns 0 for wavelengths outside [380, 780] nm.
 *
 * @param table   - One of CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE, or CIE_D65_TABLE.
 * @param lambdaNm - Wavelength in nm.
 */
function sampleTable(table: Float32Array, lambdaNm: number): number {
  if (lambdaNm < CIE_LAMBDA_MIN || lambdaNm > CIE_LAMBDA_MAX) return 0;
  const f = (lambdaNm - CIE_LAMBDA_MIN) / CIE_LAMBDA_STEP;
  const lo = Math.floor(f);
  const hi = Math.min(lo + 1, CIE_TABLE_LENGTH - 1);
  const t = f - lo;
  const vlo = table[lo] ?? 0;
  const vhi = table[hi] ?? 0;
  return vlo + t * (vhi - vlo);
}

/**
 * Sample the CIE 1931 standard observer CMF at a given wavelength via linear
 * interpolation between the 5 nm table entries.  Returns [x, y, z] = [0, 0, 0]
 * for wavelengths outside [380, 780] nm.
 *
 * @param lambdaNm - Wavelength in nm.
 * @returns [x̄(λ), ȳ(λ), z̄(λ)] tristimulus weight values.
 */
export function sampleCMF(lambdaNm: number): readonly [number, number, number] {
  return [
    sampleTable(CIE_X_TABLE, lambdaNm),
    sampleTable(CIE_Y_TABLE, lambdaNm),
    sampleTable(CIE_Z_TABLE, lambdaNm),
  ];
}

// ── XYZ → linear sRGB conversion ──────────────────────────────────────────────
// IEC 61966-2-1:1999 D65-adapted matrix (Bradford).
// Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
//
//   R =  3.2404542·X − 1.5371385·Y − 0.4985314·Z
//   G = −0.9692660·X + 1.8760108·Y + 0.0415560·Z
//   B =  0.0556434·X − 0.2040259·Y + 1.0572252·Z

/**
 * Convert CIE XYZ (D65 white point) to linear sRGB using the standard
 * Bradford-adapted D65 matrix (IEC 61966-2-1:1999).
 *
 * Output components may fall outside [0, 1] for colors outside the sRGB gamut;
 * callers should clamp as appropriate.
 *
 * @param x - CIE X tristimulus value.
 * @param y - CIE Y tristimulus value.
 * @param z - CIE Z tristimulus value.
 * @returns [r, g, b] in linear sRGB.
 */
export function xyzToLinearSRGB(
  x: number,
  y: number,
  z: number,
): readonly [number, number, number] {
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}
