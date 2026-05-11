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
  /* 380 */ 0.001368, /* 385 */ 0.002236, /* 390 */ 0.004243, /* 395 */ 0.007650,
  /* 400 */ 0.014310, /* 405 */ 0.023190, /* 410 */ 0.043510, /* 415 */ 0.077630,
  /* 420 */ 0.134380, /* 425 */ 0.214770, /* 430 */ 0.283900, /* 435 */ 0.328500,
  /* 440 */ 0.348280, /* 445 */ 0.348060, /* 450 */ 0.336200, /* 455 */ 0.318700,
  /* 460 */ 0.290800, /* 465 */ 0.251100, /* 470 */ 0.195360, /* 475 */ 0.142100,
  /* 480 */ 0.095640, /* 485 */ 0.057950, /* 490 */ 0.032010, /* 495 */ 0.014700,
  /* 500 */ 0.004900, /* 505 */ 0.002400, /* 510 */ 0.009300, /* 515 */ 0.029100,
  /* 520 */ 0.063270, /* 525 */ 0.109600, /* 530 */ 0.165500, /* 535 */ 0.225750,
  /* 540 */ 0.290400, /* 545 */ 0.359700, /* 550 */ 0.433450, /* 555 */ 0.512050,
  /* 560 */ 0.594500, /* 565 */ 0.678400, /* 570 */ 0.762100, /* 575 */ 0.842500,
  /* 580 */ 0.916300, /* 585 */ 0.978600, /* 590 */ 1.026300, /* 595 */ 1.056700,
  /* 600 */ 1.062200, /* 605 */ 1.045600, /* 610 */ 1.002600, /* 615 */ 0.938400,
  /* 620 */ 0.854450, /* 625 */ 0.751400, /* 630 */ 0.642400, /* 635 */ 0.541900,
  /* 640 */ 0.447900, /* 645 */ 0.360800, /* 650 */ 0.283500, /* 655 */ 0.218700,
  /* 660 */ 0.164900, /* 665 */ 0.121200, /* 670 */ 0.087400, /* 675 */ 0.063600,
  /* 680 */ 0.046770, /* 685 */ 0.032900, /* 690 */ 0.022700, /* 695 */ 0.015840,
  /* 700 */ 0.011359, /* 705 */ 0.008111, /* 710 */ 0.005790, /* 715 */ 0.004109,
  /* 720 */ 0.002899, /* 725 */ 0.002049, /* 730 */ 0.001440, /* 735 */ 0.001000,
  /* 740 */ 0.000690, /* 745 */ 0.000476, /* 750 */ 0.000332, /* 755 */ 0.000235,
  /* 760 */ 0.000166, /* 765 */ 0.000117, /* 770 */ 0.000083, /* 775 */ 0.000059,
  /* 780 */ 0.000042,
]);

/** CIE 1931 ȳ(λ) color-matching function (luminous efficiency), 380–780 nm at 5 nm steps. */
export const CIE_Y_TABLE: Readonly<Float32Array> = new Float32Array([
  /* 380 */ 0.000039, /* 385 */ 0.000064, /* 390 */ 0.000120, /* 395 */ 0.000217,
  /* 400 */ 0.000396, /* 405 */ 0.000640, /* 410 */ 0.001210, /* 415 */ 0.002180,
  /* 420 */ 0.004000, /* 425 */ 0.007300, /* 430 */ 0.011600, /* 435 */ 0.016840,
  /* 440 */ 0.023000, /* 445 */ 0.029800, /* 450 */ 0.038000, /* 455 */ 0.048000,
  /* 460 */ 0.060000, /* 465 */ 0.073900, /* 470 */ 0.090980, /* 475 */ 0.112600,
  /* 480 */ 0.139020, /* 485 */ 0.169300, /* 490 */ 0.208020, /* 495 */ 0.258600,
  /* 500 */ 0.323000, /* 505 */ 0.407300, /* 510 */ 0.503000, /* 515 */ 0.608200,
  /* 520 */ 0.710000, /* 525 */ 0.793200, /* 530 */ 0.862000, /* 535 */ 0.914850,
  /* 540 */ 0.954000, /* 545 */ 0.980300, /* 550 */ 0.994950, /* 555 */ 1.000000,
  /* 560 */ 0.995000, /* 565 */ 0.978600, /* 570 */ 0.952000, /* 575 */ 0.915400,
  /* 580 */ 0.870000, /* 585 */ 0.816300, /* 590 */ 0.757000, /* 595 */ 0.694900,
  /* 600 */ 0.631000, /* 605 */ 0.566800, /* 610 */ 0.503000, /* 615 */ 0.441200,
  /* 620 */ 0.381000, /* 625 */ 0.321000, /* 630 */ 0.265000, /* 635 */ 0.217000,
  /* 640 */ 0.175000, /* 645 */ 0.138200, /* 650 */ 0.107000, /* 655 */ 0.081600,
  /* 660 */ 0.061000, /* 665 */ 0.044580, /* 670 */ 0.032000, /* 675 */ 0.023200,
  /* 680 */ 0.017000, /* 685 */ 0.011920, /* 690 */ 0.008210, /* 695 */ 0.005723,
  /* 700 */ 0.004102, /* 705 */ 0.002929, /* 710 */ 0.002091, /* 715 */ 0.001484,
  /* 720 */ 0.001047, /* 725 */ 0.000740, /* 730 */ 0.000520, /* 735 */ 0.000361,
  /* 740 */ 0.000249, /* 745 */ 0.000172, /* 750 */ 0.000120, /* 755 */ 0.000085,
  /* 760 */ 0.000060, /* 765 */ 0.000042, /* 770 */ 0.000030, /* 775 */ 0.000021,
  /* 780 */ 0.000015,
]);

/** CIE 1931 z̄(λ) color-matching function, 380–780 nm at 5 nm steps. */
export const CIE_Z_TABLE: Readonly<Float32Array> = new Float32Array([
  /* 380 */ 0.006450, /* 385 */ 0.010550, /* 390 */ 0.020050, /* 395 */ 0.036210,
  /* 400 */ 0.067850, /* 405 */ 0.110200, /* 410 */ 0.207400, /* 415 */ 0.371300,
  /* 420 */ 0.645600, /* 425 */ 1.039050, /* 430 */ 1.385600, /* 435 */ 1.622960,
  /* 440 */ 1.747060, /* 445 */ 1.782600, /* 450 */ 1.772110, /* 455 */ 1.744100,
  /* 460 */ 1.669200, /* 465 */ 1.528100, /* 470 */ 1.287640, /* 475 */ 1.041900,
  /* 480 */ 0.812950, /* 485 */ 0.616200, /* 490 */ 0.465180, /* 495 */ 0.353300,
  /* 500 */ 0.272000, /* 505 */ 0.212300, /* 510 */ 0.158200, /* 515 */ 0.111700,
  /* 520 */ 0.078250, /* 525 */ 0.057250, /* 530 */ 0.042160, /* 535 */ 0.029840,
  /* 540 */ 0.020300, /* 545 */ 0.013400, /* 550 */ 0.008750, /* 555 */ 0.005750,
  /* 560 */ 0.003900, /* 565 */ 0.002750, /* 570 */ 0.002100, /* 575 */ 0.001800,
  /* 580 */ 0.001650, /* 585 */ 0.001400, /* 590 */ 0.001100, /* 595 */ 0.001000,
  /* 600 */ 0.000800, /* 605 */ 0.000600, /* 610 */ 0.000340, /* 615 */ 0.000240,
  /* 620 */ 0.000190, /* 625 */ 0.000100, /* 630 */ 0.000050, /* 635 */ 0.000030,
  /* 640 */ 0.000020, /* 645 */ 0.000010, /* 650 */ 0.000000, /* 655 */ 0.000000,
  /* 660 */ 0.000000, /* 665 */ 0.000000, /* 670 */ 0.000000, /* 675 */ 0.000000,
  /* 680 */ 0.000000, /* 685 */ 0.000000, /* 690 */ 0.000000, /* 695 */ 0.000000,
  /* 700 */ 0.000000, /* 705 */ 0.000000, /* 710 */ 0.000000, /* 715 */ 0.000000,
  /* 720 */ 0.000000, /* 725 */ 0.000000, /* 730 */ 0.000000, /* 735 */ 0.000000,
  /* 740 */ 0.000000, /* 745 */ 0.000000, /* 750 */ 0.000000, /* 755 */ 0.000000,
  /* 760 */ 0.000000, /* 765 */ 0.000000, /* 770 */ 0.000000, /* 775 */ 0.000000,
  /* 780 */ 0.000000,
]);

// ── CIE D65 illuminant ────────────────────────────────────────────────────────
// Relative spectral power distribution at 5 nm steps, 380–780 nm.
// Normalised so that the entry at 560 nm = 100 (standard form).
// Source: CIE 15:2004 Table T.1.

/** CIE D65 standard illuminant (6500 K daylight), 380–780 nm at 5 nm steps. */
export const CIE_D65_TABLE: Readonly<Float32Array> = new Float32Array([
  /*380*/  49.9755, /*385*/  52.3118, /*390*/  54.6482, /*395*/  68.7015,
  /*400*/  82.7549, /*405*/  87.1204, /*410*/  91.4860, /*415*/  92.4589,
  /*420*/  93.4318, /*425*/  90.0570, /*430*/  86.6823, /*435*/  95.7736,
  /*440*/ 104.8650, /*445*/ 110.9360, /*450*/ 117.0080, /*455*/ 117.4100,
  /*460*/ 117.8120, /*465*/ 116.3360, /*470*/ 114.8610, /*475*/ 115.3920,
  /*480*/ 115.9230, /*485*/ 112.3670, /*490*/ 108.8110, /*495*/ 109.0820,
  /*500*/ 109.3540, /*505*/ 108.5780, /*510*/ 107.8020, /*515*/ 106.2960,
  /*520*/ 104.7900, /*525*/ 106.2390, /*530*/ 107.6890, /*535*/ 106.0470,
  /*540*/ 104.4050, /*545*/ 104.2250, /*550*/ 104.0460, /*555*/ 102.0230,
  /*560*/ 100.0000, /*565*/  98.1671, /*570*/  96.3342, /*575*/  96.0611,
  /*580*/  95.7880, /*585*/  92.2368, /*590*/  88.6856, /*595*/  89.3459,
  /*600*/  90.0062, /*605*/  89.8026, /*610*/  89.5991, /*615*/  88.6489,
  /*620*/  87.6987, /*625*/  85.4936, /*630*/  83.2886, /*635*/  83.4939,
  /*640*/  83.6992, /*645*/  81.8630, /*650*/  80.0268, /*655*/  80.1207,
  /*660*/  80.2146, /*665*/  81.2462, /*670*/  82.2778, /*675*/  80.2810,
  /*680*/  78.2842, /*685*/  74.0027, /*690*/  69.7213, /*695*/  70.6652,
  /*700*/  71.6091, /*705*/  72.9790, /*710*/  74.3490, /*715*/  67.9765,
  /*720*/  61.6040, /*725*/  65.7448, /*730*/  69.8856, /*735*/  72.4863,
  /*740*/  75.0870, /*745*/  69.3398, /*750*/  63.5927, /*755*/  55.0054,
  /*760*/  46.4182, /*765*/  56.6118, /*770*/  66.8054, /*775*/  65.0941,
  /*780*/  63.3828,
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
    -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
     0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}
