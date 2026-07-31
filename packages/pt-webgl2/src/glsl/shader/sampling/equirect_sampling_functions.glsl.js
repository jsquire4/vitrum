export const equirect_functions = /* glsl */ `

	// Canonical source-grid UV shared with pt-webgpu and walkaround:
	// u = atan2(z,x)/(2PI)+0.5 (wrapped), v = acos(y)/PI. CPU payload row zero
	// is the north-pole row, so this deliberately does not use the GL-oriented
	// y flip in the generic equirect camera helpers.
	vec2 equirectEnvironmentUv( vec3 direction ) {

		vec3 n = vitrumNormalizeVec3(
			direction,
			vec3( 0.0, 1.0, 0.0 )
		);
		float horizontalScale = max( abs( n.x ), abs( n.z ) );
		float u = horizontalScale > 0.0
			? fract( atan( n.z, n.x ) / ( 2.0 * PI ) + 0.5 )
			: 0.5;
		float v = clamp( acos( clamp( n.y, -1.0, 1.0 ) ) / PI, 0.0, 0.99999994 );
		return vec2( u, v );

	}

	// Keep every environment-radiance scale in finite binary32.
	bool equirectScaleIsRepresentable( vec3 value, float scale ) {

		if (
			any( isnan( value ) ) ||
			any( isinf( value ) ) ||
			any( lessThan( value, vec3( 0.0 ) ) ) ||
			! ( scale >= 0.0 ) ||
			isnan( scale ) ||
			isinf( scale )
		) {

			return false;

		}
		vec3 scaled = value * scale;
		return ! any( isnan( scaled ) ) && ! any( isinf( scaled ) );

	}

	vec3 finiteEquirectScaledColor( vec3 value, float scale ) {

		return equirectScaleIsRepresentable( value, scale )
			? value * scale
			: vec3( 0.0 );

	}

	// Canonical receiver-side product:
	// HDRI texel * global environment intensity * material environment intensity.
	// Keeping the historical order preserves ordinary binary32 results while the
	// staged guards make an unrepresentable combined radiance deterministic zero.
	vec3 finiteEquirectRadiance( vec3 color, float materialIntensity ) {

		return finiteEquirectScaledColor(
			finiteEquirectScaledColor( color, environmentIntensity ),
			materialIntensity
		);

	}

	// Fail closed before legacy callers apply the global intensity themselves.
	// This protects low-level/external GPU textures that have no host-inspectable
	// pixel payload and keeps camera-background multiplication finite.
	vec3 finiteEquirectEnvironmentColor( vec3 color ) {

		return equirectScaleIsRepresentable( color, environmentIntensity )
			? color
			: vec3( 0.0 );

	}

	float finiteEquirectPdf( float density ) {

		return density > 0.0 && ! isnan( density ) && ! isinf( density )
			? density
			: 0.0;

	}

	// Samples the given environment map in the given world direction.
	vec3 sampleEquirectColor( sampler2D envMap, vec3 direction ) {

		return finiteEquirectEnvironmentColor(
			texture2D( envMap, equirectEnvironmentUv( direction ) ).rgb
		);

	}

	float equirectCdfXi( float xi ) {

		return clamp( xi, 0.0, 0.99999994 );

	}

	int equirectMarginalRow( float xi, ivec2 resolution ) {

		int lo = 0;
		int hi = resolution.y - 1;
		for ( int step = 0; step < 32; step ++ ) {

			if ( lo >= hi ) break;
			int mid = ( lo + hi ) / 2;
			float cdf = texelFetch(
				envMapInfo.distributionWeights,
				ivec2( 0, mid ),
				0
			).g;
			if ( cdf <= xi ) {

				lo = mid + 1;

			} else {

				hi = mid;

			}

		}
		return lo;

	}

	int equirectConditionalColumn( float xi, int row, ivec2 resolution ) {

		int lo = 0;
		int hi = resolution.x - 1;
		for ( int step = 0; step < 32; step ++ ) {

			if ( lo >= hi ) break;
			int mid = ( lo + hi ) / 2;
			float cdf = texelFetch(
				envMapInfo.distributionWeights,
				ivec2( mid, row ),
				0
			).r;
			if ( cdf <= xi ) {

				lo = mid + 1;

			} else {

				hi = mid;

			}

		}
		return lo;

	}

	// Samples the environment in a supplied direction and returns the exact
	// density of the realized uploaded Float32-CDF proposal for that cell.
	float sampleEquirect( vec3 direction, inout vec3 color ) {

		float totalSum = envMapInfo.totalSum;
		if ( ! ( totalSum > 0.0 ) || isnan( totalSum ) || isinf( totalSum ) ) {

			color = vec3( 0.0 );
			return 0.0;

		}

		ivec2 resolution = textureSize( envMapInfo.map, 0 );
		vec2 uv = equirectEnvironmentUv( direction );
		ivec2 texel = clamp(
			ivec2( floor( uv * vec2( resolution ) ) ),
			ivec2( 0 ),
			resolution - ivec2( 1 )
		);
		vec4 packed = texelFetch( envMapInfo.map, texel, 0 );
		color = finiteEquirectEnvironmentColor( packed.rgb );

		return finiteEquirectPdf( packed.a );

	}

	// Invert the uploaded forward CDFs. The normalized residual left inside each
	// selected interval remains uniform, so the same two random variates sample
	// continuously and uniformly in solid angle inside the selected cell.
	float sampleEquirectProbability( vec2 r, inout vec3 color, inout vec3 direction ) {

		ivec2 resolution = textureSize( envMapInfo.map, 0 );
		if (
			! ( envMapInfo.totalSum > 0.0 ) ||
			isnan( envMapInfo.totalSum ) ||
			isinf( envMapInfo.totalSum ) ||
			resolution.x <= 0 ||
			resolution.y <= 0
		) {

			color = vec3( 0.0 );
			direction = vec3( 0.0, 1.0, 0.0 );
			return 0.0;

		}

		float rowXi = equirectCdfXi( r.x );
		int row = equirectMarginalRow( rowXi, resolution );
		float rowCdf = texelFetch(
			envMapInfo.distributionWeights,
			ivec2( 0, row ),
			0
		).g;
		float rowPrior = row == 0 ? 0.0 : texelFetch(
			envMapInfo.distributionWeights,
			ivec2( 0, row - 1 ),
			0
		).g;
		float rowWidth = rowCdf - rowPrior;
		if ( rowWidth <= 0.0 ) return 0.0;
		float rowResidual = clamp( ( rowXi - rowPrior ) / rowWidth, 0.0, 1.0 );

		float columnXi = equirectCdfXi( r.y );
		int column = equirectConditionalColumn( columnXi, row, resolution );
		float columnCdf = texelFetch(
			envMapInfo.distributionWeights,
			ivec2( column, row ),
			0
		).r;
		float columnPrior = column == 0 ? 0.0 : texelFetch(
			envMapInfo.distributionWeights,
			ivec2( column - 1, row ),
			0
		).r;
		float columnWidth = columnCdf - columnPrior;
		if ( columnWidth <= 0.0 ) return 0.0;
		float columnResidual = clamp(
			( columnXi - columnPrior ) / columnWidth,
			0.0,
			1.0
		);

		float theta0 = float( row ) * PI / float( resolution.y );
		float theta1 = float( row + 1 ) * PI / float( resolution.y );
		float cosTheta = mix( cos( theta0 ), cos( theta1 ), rowResidual );
		float sinTheta = sqrt( max( 0.0, 1.0 - cosTheta * cosTheta ) );
		float u = ( float( column ) + columnResidual ) / float( resolution.x );
		float phi = ( u - 0.5 ) * 2.0 * PI;
		direction = vec3(
			cos( phi ) * sinTheta,
			cosTheta,
			sin( phi ) * sinTheta
		);

		vec4 packed = texelFetch( envMapInfo.map, ivec2( column, row ), 0 );
		color = finiteEquirectEnvironmentColor( packed.rgb );
		return finiteEquirectPdf( packed.a );

	}
`;
