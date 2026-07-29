// References
// - https://jcgt.org/published/0009/04/01/
// - Code from https://www.shadertoy.com/view/WtGyDm

// functions to generate multi-dimensions variables of the same functions
// to support 1, 2, 3, and 4 dimensional sobol sampling.
function generateSobolFunctionVariants( dim = 1 ) {

	let type = 'uint';
	if ( dim > 1 ) {

		type = 'uvec' + dim;

	}

	return /* glsl */`
		${ type } sobolReverseBits( ${ type } x ) {

			x = ( ( ( x & 0xaaaaaaaau ) >> 1 ) | ( ( x & 0x55555555u ) << 1 ) );
			x = ( ( ( x & 0xccccccccu ) >> 2 ) | ( ( x & 0x33333333u ) << 2 ) );
			x = ( ( ( x & 0xf0f0f0f0u ) >> 4 ) | ( ( x & 0x0f0f0f0fu ) << 4 ) );
			x = ( ( ( x & 0xff00ff00u ) >> 8 ) | ( ( x & 0x00ff00ffu ) << 8 ) );
			return ( ( x >> 16 ) | ( x << 16 ) );

		}

		${ type } sobolHashCombine( uint seed, ${ type } v ) {

			return seed ^ ( v + ${ type }( ( seed << 6 ) + ( seed >> 2 ) ) );

		}

		${ type } sobolLaineKarrasPermutation( ${ type } x, ${ type } seed ) {

			x += seed;
			x ^= x * 0x6c50b47cu;
			x ^= x * 0xb82f1e52u;
			x ^= x * 0xc7afe638u;
			x ^= x * 0x8d22f6e6u;
			return x;

		}

		${ type } nestedUniformScrambleBase2( ${ type } x, ${ type } seed ) {

			x = sobolLaineKarrasPermutation( x, seed );
			x = sobolReverseBits( x );
			return x;

		}
	`;

}

function generateSobolSampleFunctions( dim = 1 ) {

	let utype = 'uint';
	let vtype = 'float';
	let num = '';
	let components = '.r';
	let combineValues = '1u';
	if ( dim > 1 ) {

		utype = 'uvec' + dim;
		vtype = 'vec' + dim;
		num = dim + '';
		if ( dim === 2 ) {

			components = '.rg';
			combineValues = 'uvec2( 1u, 2u )';

		} else if ( dim === 3 ) {

			components = '.rgb';
			combineValues = 'uvec3( 1u, 2u, 3u )';

		} else {

			components = '';
			combineValues = 'uvec4( 1u, 2u, 3u, 4u )';

		}

	}

	return /* glsl */`

		${ vtype } sobol${ num }( int effect ) {

			uint seed = sobolGetSeed( sobolBounceIndex, uint( effect ) );
			uint index = sobolPathIndex;

			uint shuffle_seed = sobolHashCombine( seed, 0u );
			uint shuffled_index = nestedUniformScrambleBase2( sobolReverseBits( index ), shuffle_seed );
			${ vtype } sobol_pt = sobolGetTexturePoint( shuffled_index )${ components };
			${ utype } result = ${ utype }( sobol_pt * 16777216.0 );

			${ utype } seed2 = sobolHashCombine( seed, ${ combineValues } );
			result = nestedUniformScrambleBase2( result, seed2 );

			return SOBOL_FACTOR * ${ vtype }( result >> 8 );

		}
	`;

}

export const sobol_common = /* glsl */`

	// Utils
	const float SOBOL_FACTOR = 1.0 / 16777216.0;
	const uint SOBOL_MAX_POINTS = 256u * 256u;

	${ generateSobolFunctionVariants( 1 ) }
	${ generateSobolFunctionVariants( 2 ) }
	${ generateSobolFunctionVariants( 3 ) }
	${ generateSobolFunctionVariants( 4 ) }

	uint sobolHash( uint x ) {

		// finalizer from murmurhash3
		x ^= x >> 16;
		x *= 0x85ebca6bu;
		x ^= x >> 13;
		x *= 0xc2b2ae35u;
		x ^= x >> 16;
		return x;

	}

`;

export const sobol_functions = /* glsl */`

	// Seeds
	uniform sampler2D sobolTexture;
	uint sobolPixelIndex = 0u;
	uint sobolPathIndex = 0u;
	uint sobolBounceIndex = 0u;

	uint sobolGetSeed( uint bounce, uint effect ) {

		return sobolHash(
			sobolHashCombine(
				sobolHashCombine(
					sobolHash( bounce ),
					sobolPixelIndex
				),
				effect
			)
		);

	}

	vec4 sobolGetTexturePoint( uint index ) {

		if ( index >= SOBOL_MAX_POINTS ) {

			index = index % SOBOL_MAX_POINTS;

		}

		uvec2 dim = uvec2( textureSize( sobolTexture, 0 ).xy );
		uint y = index / dim.x;
		uint x = index - y * dim.x;
		vec2 uv = ( vec2( x, y ) + 0.5 ) / vec2( dim );
		return texture( sobolTexture, uv );

	}

	${ generateSobolSampleFunctions( 1 ) }
	${ generateSobolSampleFunctions( 2 ) }
	${ generateSobolSampleFunctions( 3 ) }
	${ generateSobolSampleFunctions( 4 ) }

`;
