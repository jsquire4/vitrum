import {
	MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
	MATERIAL_PIXELS,
	MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
	MATERIAL_WRAP_TEXEL_OFFSET,
} from './materialStride.js';

/** @public — dynamic-access test-load-bearing; accessed via namespace import in untestedMaterialMaps.test.ts */
export const material_struct = /* glsl */ `

	struct Material {

		vec3 color;
		int map;

		float metalness;
		int metalnessMap;

		float roughness;
		int roughnessMap;

		float ior;
		float transmission;
		int transmissionMap;

		float emissiveIntensity;
		vec3 emissive;
		int emissiveMap;

		int normalMap;
		vec2 normalScale;

		float clearcoat;
		int clearcoatMap;
		int clearcoatNormalMap;
		vec2 clearcoatNormalScale;
		float clearcoatRoughness;
		int clearcoatRoughnessMap;

		int iridescenceMap;
		int iridescenceThicknessMap;
		float iridescence;
		float iridescenceIor;
		float iridescenceThicknessMinimum;
		float iridescenceThicknessMaximum;

		vec3 specularColor;
		int specularColorMap;

		float specularIntensity;
		int specularIntensityMap;
		bool thinFilm;
		float anisotropy;
		float anisotropyRotation;
		int anisotropyMap;

		vec3 attenuationColor;
		float attenuationDistance;
		float thickness;
		int thicknessMap;

		int alphaMap;

		bool castShadow;
		float opacity;
		float alphaTest;

		float side;
		bool matte;

		float sheen;
		vec3 sheenColor;
		int sheenColorMap;
		float sheenRoughness;
		int sheenRoughnessMap;

		bool vertexColors;
		bool flatShading;
		bool transparent;
		bool unlit;
		bool meshEmitterCastShadowDisabled;
		bool fogVolume;
		uint flags;
		float sssSigmaT;
		float sssAnisotropyG;
		vec3 sssSigmaS;
		float dispersionStrength;
		float thinFilmEnabled;
		float thinFilmLayerCount;
		float thinFilmIncidentIor;
		bool thinFilmAngleDependent;
		bool hasSpectralAttenuation;
		vec3 frontLayerTransmission;
		float frontLayerRoughness;
		bool hasFrontLayer;
		int frontLayerNormalMap;
		vec2 frontLayerNormalScale;
		mat3 frontLayerNormalMapTransform;
		vec4 frontLayerNormalMapWrap;
		float frontLayerNormalTexCoord;
		vec3 backLayerTransmission;
		float backLayerRoughness;
		bool hasBackLayer;
		int backLayerNormalMap;
		vec2 backLayerNormalScale;
		mat3 backLayerNormalMapTransform;
		vec4 backLayerNormalMapWrap;
		float backLayerNormalTexCoord;
		vec3 spectralReflectanceCoeffs;
		bool hasSpectralReflectance;

		// D3 — reserved-field consumption: ambient-occlusion / baked light / bump maps
		// + per-material env-IBL scale. Layout: texels 85..92 (after the 30 transform
		// texels 55..84), plus alphaMapTransform at 93..94. aoMap modulates albedo
		// (with the geometry-occlusion caveat),
		// lightMap adds baked irradiance at camera hits, bumpMap perturbs the normal by
		// its height gradient, envMapIntensity scales this material's IBL contribution.
		int aoMap;
		int lightMap;
		int bumpMap;
		float aoMapIntensity;
		float lightMapIntensity;
		float bumpScale;
		float envMapIntensity;

		// UV-set selector bitmask (texel 86.a). Bit k set = map k samples uv1
		// (ATTR_UV1) instead of uv0 (ATTR_UV). Bit assignments: see materialStride.js.
		// Stored as a float; decoded in readMaterialInfo via uint(round(...)).
		uint uvTexCoordMask;

		mat3 mapTransform;
		mat3 metalnessMapTransform;
		mat3 roughnessMapTransform;
		mat3 transmissionMapTransform;
		mat3 emissiveMapTransform;
		mat3 normalMapTransform;
		mat3 clearcoatMapTransform;
		mat3 clearcoatNormalMapTransform;
		mat3 clearcoatRoughnessMapTransform;
		mat3 sheenColorMapTransform;
		mat3 sheenRoughnessMapTransform;
		mat3 iridescenceMapTransform;
		mat3 iridescenceThicknessMapTransform;
		mat3 specularColorMapTransform;
		mat3 specularIntensityMapTransform;
		mat3 alphaMapTransform;
		mat3 anisotropyMapTransform;
		mat3 thicknessMapTransform;

		vec4 mapWrap;
		vec4 metalnessMapWrap;
		vec4 roughnessMapWrap;
		vec4 transmissionMapWrap;
		vec4 emissiveMapWrap;
		vec4 normalMapWrap;
		vec4 alphaMapWrap;
		vec4 clearcoatMapWrap;
		vec4 clearcoatRoughnessMapWrap;
		vec4 clearcoatNormalMapWrap;
		vec4 sheenColorMapWrap;
		vec4 sheenRoughnessMapWrap;
		vec4 iridescenceMapWrap;
		vec4 iridescenceThicknessMapWrap;
		vec4 specularColorMapWrap;
		vec4 specularIntensityMapWrap;
		vec4 aoMapWrap;
		vec4 lightMapWrap;
		vec4 bumpMapWrap;
		vec4 anisotropyMapWrap;
		vec4 thicknessMapWrap;

		// D3 — transforms for the new maps (texels 87/89/91, 2 texels per mat3 —
		// see readMaterialInfo). Identity when the corresponding map id == -1.
		mat3 aoMapTransform;
		mat3 lightMapTransform;
		mat3 bumpMapTransform;

	};

	float wrapMaterialTextureCoord( float coord, float mode ) {

		int m = int( round( mode ) );
		if ( m == 1 ) {

			return min( clamp( coord, 0.0, 1.0 ), 0.999999 );

		}

		if ( m == 2 ) {

			float period = coord - 2.0 * floor( coord * 0.5 );
			float mirrored = period <= 1.0 ? period : 2.0 - period;
			return min( max( mirrored, 0.0 ), 0.999999 );

		}

		return fract( coord );

	}

	vec2 wrapMaterialTextureUv( vec2 uv, vec2 wrapMode ) {

		return vec2(
			wrapMaterialTextureCoord( uv.x, wrapMode.x ),
			wrapMaterialTextureCoord( uv.y, wrapMode.y )
		);

	}

	int wrapMaterialTextureIndex( int coord, int size, float mode ) {

		int m = int( round( mode ) );
		if ( m == 1 ) {

			return clamp( coord, 0, size - 1 );

		}

		if ( m == 2 ) {

			int period = max( size * 2, 1 );
			int c = coord - period * int( floor( float( coord ) / float( period ) ) );
			return c < size ? c : period - 1 - c;

		}

		return coord - size * int( floor( float( coord ) / float( size ) ) );

	}

	bool materialTextureUsesLinearFilter( vec4 samplerPolicy, bool minifying ) {

		int packed = int( round( samplerPolicy.w ) );
		int magFilter = packed - ( packed / 2 ) * 2;
		int minFilter = packed / 2;
		return ( minifying ? minFilter : magFilter ) == 1;

	}

	float materialTextureRawLod( vec2 uv, vec2 baseSize ) {

		vec2 dx = dFdx( uv * baseSize );
		vec2 dy = dFdy( uv * baseSize );
		float rho = max( length( dx ), length( dy ) );
		return max( log2( max( rho, 1e-8 ) ), 0.0 );

	}

	vec4 sampleMaterialTextureNearestLevel( sampler2DArray tex, vec2 uv, int layer, vec4 samplerPolicy, int level ) {

		ivec2 size = textureSize( tex, level ).xy;
		ivec2 p = ivec2( floor( uv * vec2( size ) ) );
		int x = wrapMaterialTextureIndex( p.x, size.x, samplerPolicy.x );
		int y = wrapMaterialTextureIndex( p.y, size.y, samplerPolicy.y );
		return texelFetch( tex, ivec3( x, y, layer ), level );

	}

	vec4 sampleMaterialTextureLinearLevel( sampler2DArray tex, vec2 uv, int layer, vec4 samplerPolicy, int level ) {

		ivec2 size = textureSize( tex, level ).xy;
		vec2 p = uv * vec2( size ) - vec2( 0.5 );
		ivec2 p0 = ivec2( floor( p ) );
		vec2 f = fract( p );
		int x0 = wrapMaterialTextureIndex( p0.x, size.x, samplerPolicy.x );
		int y0 = wrapMaterialTextureIndex( p0.y, size.y, samplerPolicy.y );
		int x1 = wrapMaterialTextureIndex( p0.x + 1, size.x, samplerPolicy.x );
		int y1 = wrapMaterialTextureIndex( p0.y + 1, size.y, samplerPolicy.y );

		vec4 c00 = texelFetch( tex, ivec3( x0, y0, layer ), level );
		vec4 c10 = texelFetch( tex, ivec3( x1, y0, layer ), level );
		vec4 c01 = texelFetch( tex, ivec3( x0, y1, layer ), level );
		vec4 c11 = texelFetch( tex, ivec3( x1, y1, layer ), level );
		return mix( mix( c00, c10, f.x ), mix( c01, c11, f.x ), f.y );

	}

	vec4 sampleMaterialTextureLevel(
		sampler2DArray tex,
		vec2 uv,
		int layer,
		vec4 samplerPolicy,
		int level,
		bool linearFilter
	) {

		return linearFilter
			? sampleMaterialTextureLinearLevel( tex, uv, layer, samplerPolicy, level )
			: sampleMaterialTextureNearestLevel( tex, uv, layer, samplerPolicy, level );

	}

	vec4 sampleMaterialTexture( sampler2DArray tex, vec2 uv, int layer, vec4 samplerPolicy ) {

		if ( layer < 0 ) {

			return vec4( 1.0 );

		}

		ivec2 baseSizeI = textureSize( tex, 0 ).xy;
		vec2 baseSize = vec2( baseSizeI );
		float rawLod = materialTextureRawLod( uv, baseSize );
		bool minifying = rawLod > 0.0;
		bool linearFilter = materialTextureUsesLinearFilter( samplerPolicy, minifying );
		int mipFilter = int( round( samplerPolicy.z ) );
		int maxLevel = max( 0, int( floor( log2( float( max( baseSizeI.x, baseSizeI.y ) ) ) ) ) );

		if ( mipFilter == 0 || maxLevel == 0 ) {

			return sampleMaterialTextureLevel( tex, uv, layer, samplerPolicy, 0, linearFilter );

		}

		if ( mipFilter == 1 ) {

			int level = clamp( int( floor( rawLod + 0.5 ) ), 0, maxLevel );
			return sampleMaterialTextureLevel( tex, uv, layer, samplerPolicy, level, linearFilter );

		}

		float clampedLod = clamp( rawLod, 0.0, float( maxLevel ) );
		int level0 = int( floor( clampedLod ) );
		int level1 = min( level0 + 1, maxLevel );
		float t = clampedLod - float( level0 );
		vec4 a = sampleMaterialTextureLevel( tex, uv, layer, samplerPolicy, level0, linearFilter );
		vec4 b = sampleMaterialTextureLevel( tex, uv, layer, samplerPolicy, level1, linearFilter );
		return mix( a, b, t );


	}

	mat3 readTextureTransform( sampler2D tex, uint index ) {

		mat3 textureTransform;

		vec4 row1 = texelFetch1D( tex, index );
		vec4 row2 = texelFetch1D( tex, index + 1u );

		textureTransform[0] = vec3(row1.r, row2.r, 0.0);
		textureTransform[1] = vec3(row1.g, row2.g, 0.0);
		textureTransform[2] = vec3(row1.b, row2.b, 1.0);

		return textureTransform;

	}

	Material readMaterialInfo( sampler2D tex, uint index ) {

		// D3 — stride bumped 85 → ${MATERIAL_PIXELS} (single-sourced from materialStride.js; the
		// packer imports the same constant): texels 85/86 carry ao/light/bump map
		// ids + scalars + envMapIntensity; texels 87..92 carry their 3 transforms;
		// texels 93/94 carry alphaMapTransform; texels 95/96 carry anisotropyMapTransform;
		// texel 97 carries volume thickness + thicknessMap; texels 98/99 carry
		// thicknessMapTransform; texels ${MATERIAL_WRAP_TEXEL_OFFSET}..${MATERIAL_WRAP_TEXEL_OFFSET + 20}
		// carry per-map sampler policy; texel ${MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET}
		// carries per-material Jakob-Hanika spectral reflectance coefficients.
		uint i = index * ${MATERIAL_PIXELS}u;

		vec4 s0 = texelFetch1D( tex, i + 0u );
		vec4 s1 = texelFetch1D( tex, i + 1u );
		vec4 s2 = texelFetch1D( tex, i + 2u );
		vec4 s3 = texelFetch1D( tex, i + 3u );
		vec4 s4 = texelFetch1D( tex, i + 4u );
		vec4 s5 = texelFetch1D( tex, i + 5u );
		vec4 s6 = texelFetch1D( tex, i + 6u );
		vec4 s7 = texelFetch1D( tex, i + 7u );
		vec4 s8 = texelFetch1D( tex, i + 8u );
		vec4 s9 = texelFetch1D( tex, i + 9u );
		vec4 s10 = texelFetch1D( tex, i + 10u );
		vec4 s11 = texelFetch1D( tex, i + 11u );
		vec4 s12 = texelFetch1D( tex, i + 12u );
		vec4 s13 = texelFetch1D( tex, i + 13u );
		vec4 s14 = texelFetch1D( tex, i + 14u );
		vec4 s15 = texelFetch1D( tex, i + 15u );
		vec4 s16 = texelFetch1D( tex, i + 16u );
		vec4 s17 = texelFetch1D( tex, i + 17u );
		vec4 s18 = texelFetch1D( tex, i + 18u );
		vec4 s19 = texelFetch1D( tex, i + 19u );
		vec4 s22 = texelFetch1D( tex, i + 97u );
		vec4 sSpectralReflectance = texelFetch1D( tex, i + ${MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET}u );

		Material m;
		m.color = s0.rgb;
		m.map = int( round( s0.a ) );

		m.metalness = s1.r;
		m.metalnessMap = int( round( s1.g ) );
		m.roughness = s1.b;
		m.roughnessMap = int( round( s1.a ) );

		m.ior = s2.r;
		m.transmission = s2.g;
		m.transmissionMap = int( round( s2.b ) );
		m.emissiveIntensity = s2.a;

		m.emissive = s3.rgb;
		m.emissiveMap = int( round( s3.a ) );

		m.normalMap = int( round( s4.r ) );
		m.normalScale = s4.gb;

		m.clearcoat = s4.a;
		m.clearcoatMap = int( round( s5.r ) );
		m.clearcoatRoughness = s5.g;
		m.clearcoatRoughnessMap = int( round( s5.b ) );
		m.clearcoatNormalMap = int( round( s5.a ) );
		m.clearcoatNormalScale = s6.rg;
		m.anisotropyMap = int( round( s6.b ) );

		m.sheen = s6.a;
		m.sheenColor = s7.rgb;
		m.sheenColorMap = int( round( s7.a ) );
		m.sheenRoughness = s8.r;
		m.sheenRoughnessMap = int( round( s8.g ) );

		m.iridescenceMap = int( round( s8.b ) );
		m.iridescenceThicknessMap = int( round( s8.a ) );
		m.iridescence = s9.r;
		m.iridescenceIor = s9.g;
		m.iridescenceThicknessMinimum = s9.b;
		m.iridescenceThicknessMaximum = s9.a;

		m.specularColor = s10.rgb;
		m.specularColorMap = int( round( s10.a ) );

		m.specularIntensity = s11.r;
		m.specularIntensityMap = int( round( s11.g ) );
		m.thinFilm = bool( s11.b );
		m.anisotropy = clamp( s11.a, 0.0, 1.0 );

		m.attenuationColor = s12.rgb;
		m.attenuationDistance = s12.a;
		m.thickness = max( s22.r, 0.0 );
		m.thicknessMap = int( round( s22.g ) );

		m.alphaMap = int( round( s13.r ) );

		m.opacity = s13.g;
		m.alphaTest = s13.b;
		m.side = s13.a;

		m.matte = bool( s14.r );
		m.castShadow = bool( s14.g );
		m.vertexColors = bool( int( s14.b ) & 1 );
		m.flatShading = bool( int( s14.b ) & 2 );
		m.fogVolume = bool( int( s14.b ) & 4 );
		uint packedFlags = uint( round( s14.a ) );
		m.transparent = bool( packedFlags & 1u );
		m.unlit = bool( packedFlags & 0x20u );
		m.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
		m.flags = packedFlags;
		m.sssSigmaT = s15.r;
		m.sssAnisotropyG = s15.g;
		m.dispersionStrength = s15.b;
		m.thinFilmEnabled = s15.a;
		m.sssSigmaS = s16.rgb;
		m.thinFilmLayerCount = s16.a;
		m.thinFilmIncidentIor = max( s17.r, 1.0 );
		m.thinFilmAngleDependent = s17.g > 0.5;
		m.anisotropyRotation = s17.b;
		uint featureFlags = uint( round( s17.a ) );
		m.hasSpectralAttenuation = bool( featureFlags & 1u );
		m.hasFrontLayer = bool( featureFlags & 2u );
		m.hasBackLayer = bool( featureFlags & 4u );
		m.frontLayerTransmission = s18.rgb;
		m.frontLayerRoughness = s18.a;
		m.backLayerTransmission = s19.rgb;
		m.backLayerRoughness = s19.a;
		m.spectralReflectanceCoeffs = sSpectralReflectance.xyz;
		m.hasSpectralReflectance = sSpectralReflectance.w > 0.5;
		vec4 layerNormal = texelFetch1D( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET}u );
		m.frontLayerNormalMap = int( round( layerNormal.r ) );
		m.frontLayerNormalScale = vec2( layerNormal.g );
		m.backLayerNormalMap = int( round( layerNormal.b ) );
		m.backLayerNormalScale = vec2( layerNormal.a );
		m.frontLayerNormalMapTransform = m.frontLayerNormalMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 1}u );
		m.backLayerNormalMapTransform = m.backLayerNormalMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 3}u );
		m.frontLayerNormalMapWrap = texelFetch1D( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 5}u );
		m.backLayerNormalMapWrap = texelFetch1D( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 6}u );
		vec4 layerNormalUv = texelFetch1D( tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 7}u );
		m.frontLayerNormalTexCoord = layerNormalUv.r;
		m.backLayerNormalTexCoord = layerNormalUv.g;

		// D3 — texels 85/86: ao/light/bump map ids + scalars + envMapIntensity.
		vec4 s20 = texelFetch1D( tex, i + 85u );
		vec4 s21 = texelFetch1D( tex, i + 86u );
		m.aoMap = int( round( s20.r ) );
		m.lightMap = int( round( s20.g ) );
		m.bumpMap = int( round( s20.b ) );
		m.envMapIntensity = s20.a;
		m.aoMapIntensity = s21.r;
		m.lightMapIntensity = s21.g;
		m.bumpScale = s21.b;
		// UV-set bitmask (was pad). Bit k set = map k samples uv1 (ATTR_UV1).
		m.uvTexCoordMask = uint( round( s21.a ) );

		uint firstTextureTransformIdx = i + 55u;

		// mat3( 1.0 ) is an identity matrix
		m.mapTransform = m.map == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx );
		m.metalnessMapTransform = m.metalnessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 2u );
		m.roughnessMapTransform = m.roughnessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 4u );
		m.transmissionMapTransform = m.transmissionMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 6u );
		m.emissiveMapTransform = m.emissiveMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 8u );
		m.normalMapTransform = m.normalMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 10u );
		m.clearcoatMapTransform = m.clearcoatMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 12u );
		m.clearcoatNormalMapTransform = m.clearcoatNormalMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 14u );
		m.clearcoatRoughnessMapTransform = m.clearcoatRoughnessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 16u );
		m.sheenColorMapTransform = m.sheenColorMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 18u );
		m.sheenRoughnessMapTransform = m.sheenRoughnessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 20u );
		m.iridescenceMapTransform = m.iridescenceMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 22u );
		m.iridescenceThicknessMapTransform = m.iridescenceThicknessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 24u );
		m.specularColorMapTransform = m.specularColorMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 26u );
		m.specularIntensityMapTransform = m.specularIntensityMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, firstTextureTransformIdx + 28u );
		m.alphaMapTransform = m.alphaMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 93u );
		m.anisotropyMapTransform = m.anisotropyMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 95u );
		m.thicknessMapTransform = m.thicknessMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 98u );

		// D3 — ao/light/bump transforms at texels 87/89/91 (2 texels per mat3).
		m.aoMapTransform = m.aoMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 87u );
		m.lightMapTransform = m.lightMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 89u );
		m.bumpMapTransform = m.bumpMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 91u );

		// Texture sampler policy: .x/.y wrapS/wrapT (0 repeat, 1 clamp, 2 mirror),
		// .z mip filter (0 none, 1 nearest, 2 linear), .w packed mag/min filter.
		m.mapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 0}u );
		m.metalnessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 1}u );
		m.roughnessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 2}u );
		m.transmissionMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 3}u );
		m.emissiveMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 4}u );
		m.normalMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 5}u );
		m.alphaMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 6}u );
		m.clearcoatMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 7}u );
		m.clearcoatRoughnessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 8}u );
		m.clearcoatNormalMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 9}u );
		m.sheenColorMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 10}u );
		m.sheenRoughnessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 11}u );
		m.iridescenceMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 12}u );
		m.iridescenceThicknessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 13}u );
		m.specularColorMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 14}u );
		m.specularIntensityMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 15}u );
		m.aoMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 16}u );
		m.lightMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 17}u );
		m.bumpMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 18}u );
		m.anisotropyMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 19}u );
		m.thicknessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 20}u );

		return m;

	}

`;
