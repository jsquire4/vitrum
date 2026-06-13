import { MATERIAL_PIXELS } from './materialStride.js';

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

		vec3 attenuationColor;
		float attenuationDistance;

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
		bool fogVolume;
		uint flags;
		float sssSigmaT;
		float sssAnisotropyG;
		vec3 sssAlbedo;
		float dispersionStrength;
		float thinFilmEnabled;
		float thinFilmLayerCount;
		float thinFilmIncidentIor;
		bool thinFilmAngleDependent;
		bool hasSpectralAttenuation;
		vec3 frontLayerTransmission;
		float frontLayerRoughness;
		bool hasFrontLayer;
		vec3 backLayerTransmission;
		float backLayerRoughness;
		bool hasBackLayer;

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

		// D3 — transforms for the new maps (texels 87/89/91, 2 texels per mat3 —
		// see readMaterialInfo). Identity when the corresponding map id == -1.
		mat3 aoMapTransform;
		mat3 lightMapTransform;
		mat3 bumpMapTransform;

	};

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

		// D3 — stride bumped 85 → 95 (single-sourced from materialStride.js; the
		// packer imports the same constant): texels 85/86 carry ao/light/bump map
		// ids + scalars + envMapIntensity; texels 87..92 carry their 3 transforms;
		// texels 93/94 carry alphaMapTransform.
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

		m.attenuationColor = s12.rgb;
		m.attenuationDistance = s12.a;

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
		m.flags = packedFlags;
		m.sssSigmaT = s15.r;
		m.sssAnisotropyG = s15.g;
		m.dispersionStrength = s15.b;
		m.thinFilmEnabled = s15.a;
		m.sssAlbedo = s16.rgb;
		m.thinFilmLayerCount = s16.a;
		m.thinFilmIncidentIor = max( s17.r, 1.0 );
		m.thinFilmAngleDependent = s17.g > 0.5;
		uint featureFlags = uint( round( s17.a ) );
		m.hasSpectralAttenuation = bool( featureFlags & 1u );
		m.hasFrontLayer = bool( featureFlags & 2u );
		m.hasBackLayer = bool( featureFlags & 4u );
		m.frontLayerTransmission = s18.rgb;
		m.frontLayerRoughness = s18.a;
		m.backLayerTransmission = s19.rgb;
		m.backLayerRoughness = s19.a;

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

		// D3 — ao/light/bump transforms at texels 87/89/91 (2 texels per mat3).
		m.aoMapTransform = m.aoMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 87u );
		m.lightMapTransform = m.lightMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 89u );
		m.bumpMapTransform = m.bumpMap == - 1 ? mat3( 1.0 ) : readTextureTransform( tex, i + 91u );

		return m;

	}

`;
