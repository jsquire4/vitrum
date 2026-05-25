// spectral_accumulator.glsl.js — Sprint 12 hero-wavelength spectral accumulator.
//
// Implements CIE 1931 2-degree standard observer CMF sampling and XYZ → linear
// sRGB conversion for the hero-wavelength spectral path tracing kernel.
//
// GLSL mirror of @vitrum/shared-samplers/src/cieCmf.ts and wavelengthSampling.ts.
//
// Technique: Fascione et al. "Hero Wavelength Spectral Sampling", EGSR 2015.
//
// At each path termination the scalar throughput at the hero wavelength is
// converted to an RGB radiance contribution via:
//   XYZ = [x̄(λ), ȳ(λ), z̄(λ)] × throughput / (pdfLambda × ∫Y dλ)
//   RGB = M_XYZ_to_sRGB × XYZ
// where M_XYZ_to_sRGB is the Bradford-adapted D65 matrix (IEC 61966-2-1:1999).
//
// CMF tables are uploaded as uniform arrays (81 entries × 3 channels).
// The Y-CMF CDF (82 entries) is used for importance-sampling sampleHeroWavelength.
//
// New uniforms (added to PhysicalPathTracingMaterial.js in this sprint):
//   uCmfX[81]      — CIE x̄(λ) table (380–780 nm at 5 nm steps)
//   uCmfY[81]      — CIE ȳ(λ) table
//   uCmfZ[81]      — CIE z̄(λ) table
//   uYCmfCdf[82]   — normalised CDF of ȳ(λ) for importance sampling
//   uYCmfIntegral  — ∫ Y dλ (nm), ≈ 106.857; PDF normalisation constant
//
// References:
//   CIE 015:2018 Colorimetry, 4th edition.
//   IEC 61966-2-1:1999 Annex F (Bradford-adapted D65 XYZ→sRGB matrix).
//   plan/sprint-12-pt-fork-patch.md §2.4 (spectral accumulator spec).

export const spectral_accumulator = /* glsl */`

	// Sprint 12 CMF uniform arrays.
	// Populated by the host from @vitrum/shared-samplers CIE_X/Y/Z_TABLE.
	// 81 entries, 380–780 nm at 5 nm steps.
	uniform float uCmfX[81];
	uniform float uCmfY[81];
	uniform float uCmfZ[81];

	// CMF CDFs for hero wavelength importance sampling (82 entries each).
	// uXCmfCdf, uYCmfCdf, uZCmfCdf — each starts at 0 and ends at 1.
	// Wilkie 2015 §3.3: one-sample MIS across all three strategies gives
	// balanced chromatic coverage at low SPP (Y-only sampling collapses
	// blue and most red because Y(λ) is heavily concentrated near 555 nm).
	uniform float uXCmfCdf[82];
	uniform float uYCmfCdf[82];
	uniform float uZCmfCdf[82];

	// Integrals of X/Y/Z CMFs over [380, 780] nm. By CIE 1931 normalisation
	// these are all equal to ≈ 106.857 (chromaticity of equal-energy white
	// is (1/3, 1/3, 1/3)) — but the host uploads them independently so the
	// shader does not encode the convention.
	uniform float uXCmfIntegral;
	uniform float uYCmfIntegral;
	uniform float uZCmfIntegral;

	// Non-zero enables experimental hero-wavelength RGB reconstruction. The
	// default preview path stays RGB-stable because single-wavelength display
	// has very high chroma variance at low SPP.
	uniform int uSpectralRendering;

	// ── CMF linear interpolation ───────────────────────────────────────────────

	// Linear interpolation helper for a 81-entry CMF table at 5 nm steps.
	// Returns 0 for wavelengths outside [380, 780] nm (mirror of sampleTable in cieCmf.ts).
	float sampleCmfTable81( float table[81], float lambda ) {
		if ( lambda < 380.0 || lambda > 780.0 ) return 0.0;
		float f = ( lambda - 380.0 ) / 5.0;
		int lo = int( f );
		int hi = min( lo + 1, 80 );
		float t = f - float( lo );
		return table[ lo ] + t * ( table[ hi ] - table[ lo ] );
	}

	// Sample CIE x̄(λ) at an arbitrary wavelength (linear interpolation).
	float sampleCmfX( float lambda ) { return sampleCmfTable81( uCmfX, lambda ); }

	// Sample CIE ȳ(λ) at an arbitrary wavelength (linear interpolation).
	float sampleCmfY( float lambda ) { return sampleCmfTable81( uCmfY, lambda ); }

	// Sample CIE z̄(λ) at an arbitrary wavelength (linear interpolation).
	float sampleCmfZ( float lambda ) { return sampleCmfTable81( uCmfZ, lambda ); }

	// ── Hero wavelength importance sampling ────────────────────────────────────

	// Wilkie 2015 §3.3 CMF strategy index — used for both per-strategy CDF
	// dispatch in the MIS sampler and for the mixture pdf evaluation.
	const int CMF_STRATEGY_X = 0;
	const int CMF_STRATEGY_Y = 1;
	const int CMF_STRATEGY_Z = 2;

	// Linear interpolation helper for any of the 81-entry CMF tables, given
	// the table-relative segment index 'lo' and segment fraction 't'.
	// Returns CMF value at the interpolated wavelength.
	float cmfAtSegment( float table[81], int lo, float t ) {
		float vLo = table[ lo ];
		float vHi = ( lo < 80 ) ? table[ lo + 1 ] : 0.0;
		return vLo + t * ( vHi - vLo );
	}

	// Inverse-CDF sampling helper. Given a uniform u and a 82-entry CDF
	// (starting at 0, ending at 1), returns the wavelength in [380, 780] nm
	// and writes the CDF segment index + fraction into the out parameters.
	// Caller can then look up CMF values at the sampled lambda via cmfAtSegment.
	float sampleCmfCdfInverse( float u, float cdf[82], out int outLo, out float outT ) {
		float uClamped = clamp( u, 0.0, 1.0 - 1e-7 );

		int lo = 0;
		int hi = 80;
		for ( int iter = 0; iter < 7; iter ++ ) {  // 2^7 = 128 > 82 entries
			int mid = ( lo + hi ) / 2;
			if ( cdf[ mid + 1 ] <= uClamped ) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}

		float cdfLo = cdf[ lo ];
		float cdfHi = cdf[ lo + 1 ];
		float t = ( cdfHi > cdfLo ) ? ( uClamped - cdfLo ) / ( cdfHi - cdfLo ) : 0.0;
		float lambda = clamp( float( 380 + lo * 5 ) + t * 5.0, 380.0, 780.0 );

		outLo = lo;
		outT = t;
		return lambda;
	}

	// Mixture pdf evaluated at λ — used by both legacy single-strategy and
	// MIS multi-strategy samplers as the Monte Carlo weight denominator.
	// pdf_mis(λ) = (X(λ)/∫X + Y(λ)/∫Y + Z(λ)/∫Z) / 3   (balance heuristic)
	float misMixturePdf( int lo, float t ) {
		float x = cmfAtSegment( uCmfX, lo, t );
		float y = cmfAtSegment( uCmfY, lo, t );
		float z = cmfAtSegment( uCmfZ, lo, t );
		float pX = ( uXCmfIntegral > 0.0 ) ? x / uXCmfIntegral : 0.0;
		float pY = ( uYCmfIntegral > 0.0 ) ? y / uYCmfIntegral : 0.0;
		float pZ = ( uZCmfIntegral > 0.0 ) ? z / uZCmfIntegral : 0.0;
		return ( pX + pY + pZ ) / 3.0;
	}

	// Sample a hero wavelength from pdf(λ) ∝ Y(λ) (luminous efficiency).
	// GLSL mirror of @vitrum/shared-samplers/src/wavelengthSampling.ts::sampleHeroWavelength.
	// Kept for backward compat — production sampler is sampleHeroWavelengthMIS.
	float sampleHeroWavelength( float u, out float pdf ) {
		int lo;
		float t;
		float lambda = sampleCmfCdfInverse( u, uYCmfCdf, lo, t );
		float yAtLambda = cmfAtSegment( uCmfY, lo, t );
		pdf = ( uYCmfIntegral > 0.0 ) ? yAtLambda / uYCmfIntegral : 0.0;
		return lambda;
	}

	// One-sample MIS hero wavelength sampler across X, Y, Z CMFs (Wilkie 2015 §3.3).
	// GLSL mirror of @vitrum/shared-samplers/src/wavelengthSampling.ts::sampleHeroWavelengthMIS.
	//
	// uStrategy picks one of {X, Y, Z} with probability 1/3 each;
	// uLambda   inverse-CDF-samples within the chosen strategy.
	// The returned pdf is the MIXTURE pdf (balance heuristic), not the
	// per-strategy pdf — this is the correct denominator for the MC estimator.
	float sampleHeroWavelengthMIS( float uStrategy, float uLambda, out float pdf ) {
		float s = clamp( uStrategy, 0.0, 1.0 - 1e-7 );
		int lo;
		float t;
		float lambda;
		if ( s < 1.0 / 3.0 ) {
			lambda = sampleCmfCdfInverse( uLambda, uXCmfCdf, lo, t );
		} else if ( s < 2.0 / 3.0 ) {
			lambda = sampleCmfCdfInverse( uLambda, uYCmfCdf, lo, t );
		} else {
			lambda = sampleCmfCdfInverse( uLambda, uZCmfCdf, lo, t );
		}
		pdf = misMixturePdf( lo, t );
		return lambda;
	}

	// ── Spectral → RGB accumulator ─────────────────────────────────────────────

	// Convert a hero-wavelength path result to linear sRGB.
// For path with hero wavelength lambda, scalar throughput, and wavelength PDF:
//   XYZ = [x̄(λ), ȳ(λ), z̄(λ)] × throughput / (pdfLambda × ∫Y dλ)
	//   RGB = M_D65 × XYZ
	//
	// The Bradford-adapted D65 XYZ → linear sRGB matrix (IEC 61966-2-1:1999):
	//   R =  3.2404542·X − 1.5371385·Y − 0.4985314·Z
	//   G = −0.9692660·X + 1.8760108·Y + 0.0415560·Z
	//   B =  0.0556434·X − 0.2040259·Y + 1.0572252·Z
	//
	// GLSL mirror of @vitrum/shared-samplers/src/wavelengthSampling.ts::wavelengthToRGB.
	vec3 wavelengthToRGB( float lambda, float throughput, float pdfLambda ) {
		if ( uSpectralRendering == 0 ) return vec3( throughput );
		if ( pdfLambda <= 0.0 ) return vec3( 0.0 );

		float x = sampleCmfX( lambda );
		float y = sampleCmfY( lambda );
		float z = sampleCmfZ( lambda );

		float weight = throughput / max( pdfLambda * uYCmfIntegral, 1e-6 );
		vec3 xyz = vec3( x, y, z ) * weight;

		// XYZ → linear sRGB (Bradford-adapted D65 matrix, IEC 61966-2-1:1999)
		vec3 rgb;
		rgb.r =  3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
		rgb.g = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
		rgb.b =  0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;

		return rgb;
	}

`;
