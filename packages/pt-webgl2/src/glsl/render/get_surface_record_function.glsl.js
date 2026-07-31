
import {
	MATERIAL_ALPHA_TRANSFORM_TEXEL,
	MATERIAL_ANISOTROPY_TRANSFORM_TEXEL,
	MATERIAL_AO_TRANSFORM_TEXEL,
	MATERIAL_BUMP_TRANSFORM_TEXEL,
	MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
	MATERIAL_LIGHTMAP_TRANSFORM_TEXEL,
	MATERIAL_THICKNESS_TRANSFORM_TEXEL,
	MATERIAL_TRANSFORM_TEXEL,
	MATERIAL_WRAP_TEXEL_OFFSET,
} from '../shader/structs/materialStride.js';

/** @public — dynamic-access test-load-bearing; accessed via namespace import in wsl-gpu/scripts and untestedMaterialMaps.test.ts */
export const get_surface_record_function = /* glsl */`

	#define SKIP_SURFACE 0
	#define HIT_SURFACE 1
	// materialLodDepth controls the optional bounce-depth threshold beyond which
	// texture fetches are replaced by flat material constants. The host default is
	// 0, which disables LOD and preserves highest-fidelity texture sampling at every
	// bounce; positive values opt into the performance approximation.
	uniform int materialLodDepth;

	mat3 getBasisFromNormalAndTangent( vec3 normal, vec4 tangentSample ) {

		if ( length( normal ) <= 1e-6 ) {

			return getBasisFromNormal( vec3( 0.0, 0.0, 1.0 ) );

		}

		vec3 n = normalize( normal );
		if ( length( tangentSample.xyz ) <= 1e-6 ) {

			return getBasisFromNormal( n );

		}

		vec3 tangent = tangentSample.xyz - n * dot( tangentSample.xyz, n );
		if ( length( tangent ) <= 1e-6 ) {

			return getBasisFromNormal( n );

		}

		tangent = normalize( tangent );
		float tangentHandedness = tangentSample.w < 0.0 ? -1.0 : 1.0;
		vec3 bitangent = cross( n, tangent ) * tangentHandedness;
		if ( length( bitangent ) <= 1e-6 ) {

			return getBasisFromNormal( n );

		}

		return mat3( tangent, normalize( bitangent ), n );

	}

	int getSurfaceRecord(
		uint materialIndex, SurfaceHit surfaceHit, sampler2DArray attributesArray,
		float accumulatedRoughness, int pathDepth, float heroWavelength,
		bool stochasticOpacityAlreadyAccepted,
		inout SurfaceRecord surf
	) {
		Material material;
		readMaterialInfo( materials, materialIndex, material );

		vec4 vertexColor = textureSampleBarycoord( attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz );

		// Barycentrically interpolate the exact dense attribute layer packed for
		// this map slot. The CPU layout maps arbitrary authored texCoord ids here.
		#define MAP_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
		// Texture metadata is intentionally decoded only in the branch that samples
		// that map. Keeping 21 mat3 transforms and 21 vec4 policies out of Material
		// avoids copying an enormous value through every path-bounce helper.
		#define MAP_TRANSFORM(offset) readMaterialMapTransform( materials, materialIndex, offset )
		#define MAP_POLICY(offset) readMaterialMapPolicy( materials, materialIndex, offset )
		#define MAP_SAMPLE(layer,transformOffset,policyOffset,uvCoord) sampleMappedMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord )
		#define MAP_SRGB_SAMPLE(layer,transformOffset,policyOffset,uvCoord) sampleMappedSrgbMaterialTexture( materials, textures, materialIndex, layer, transformOffset, policyOffset, uvCoord )
		#define MAP_RADIANCE_SAMPLE(layer,transformOffset,policyOffset,uvCoord) sampleMappedMaterialTexture( materials, materialRadianceTextures, materialIndex, layer, transformOffset, policyOffset, uvCoord )

		// Optional material LOD by depth. When pathDepth > materialLodDepth, skip
		// texture fetches and use flat material constants. materialLodDepth == 0
		// disables LOD (textures at all depths), which is the host default.
		bool useTextures = ( materialLodDepth == 0 ) || ( pathDepth <= materialLodDepth );

		// albedo (baseColorMap = bit 0)
		vec4 albedo = vec4( material.color, material.opacity );
		vec3 albedoModulation = vec3( 1.0 );
		if ( useTextures && material.map != - 1 ) {

			vec4 baseColorSample = MAP_SRGB_SAMPLE(
				material.map, ${MATERIAL_TRANSFORM_TEXEL.baseColorMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 0}u, MAP_UV( 0u )
			);
			albedo *= baseColorSample;
			albedoModulation *= baseColorSample.rgb;

		}

		if ( material.vertexColors ) {

			albedo *= vertexColor;
			albedoModulation *= vertexColor.rgb;

		}

		// alphaMap (bit 6)
		if ( useTextures && material.alphaMap != - 1 ) {

			albedo.a *= MAP_SAMPLE(
				material.alphaMap, ${MATERIAL_ALPHA_TRANSFORM_TEXEL}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 6}u, MAP_UV( 6u )
			).x;

		}

		// D3 — aoMap (glTF occlusionTexture, R channel, bit 16): modulate albedo by
		// mix(1, ao, aoMapIntensity). CAVEAT (documented biased semantics, mirrors
		// pt-webgpu sampleAoFactor): a path tracer integrates real occlusion, so a
		// baked AO term double-darkens crevices — aoMapIntensity is the artist dial
		// (1 matches the raster look; 0 disables).
		if ( useTextures && material.aoMap != - 1 ) {

			float ao = MAP_SAMPLE(
				material.aoMap, ${MATERIAL_AO_TRANSFORM_TEXEL}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 16}u, MAP_UV( 16u )
			).r;
			float aoFactor = clamp( mix( 1.0, ao, material.aoMapIntensity ), 0.0, 1.0 );
			albedo.rgb *= aoFactor;
			albedoModulation *= aoFactor;

		}

		// possibly skip this sample if it's transparent, alpha test is enabled, or we hit the wrong material side
		// and it's single sided.
		// - alpha test is disabled when it === 0
		// - the material sidedness test is complicated because we want light to pass through the back side but still
		// be able to see the front side. This boolean checks if the side we hit is the front side on the first ray
		// and we're rendering the other then we skip it. Do the opposite on subsequent bounces to get incoming light.
		float alphaTest = material.alphaTest;
		bool useAlphaTest = alphaTest != 0.0;
		if (
			// material sidedness
			material.side != 0.0 && surfaceHit.side != material.side

			// alpha test
			|| useAlphaTest && albedo.a < alphaTest

			// opacity
			|| ! stochasticOpacityAlreadyAccepted && material.transparent && ! useAlphaTest && albedo.a < rand( 3 )
		) {

			return SKIP_SURFACE;

		}

		// fetch the interpolated smooth normal
		vec3 normal = normalize( textureSampleBarycoord(
			attributesArray,
			ATTR_NORMAL,
			surfaceHit.barycoord,
			surfaceHit.faceIndices.xyz
		).xyz );

		// roughness (roughnessMap = bit 2)
		float roughness = material.roughness;
		if ( useTextures && material.roughnessMap != - 1 ) {

			roughness *= MAP_SAMPLE(
				material.roughnessMap, ${MATERIAL_TRANSFORM_TEXEL.roughnessMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 2}u, MAP_UV( 2u )
			).g;

		}

		// metalness (metallicMap = bit 1)
		float metalness = material.metalness;
		if ( useTextures && material.metalnessMap != - 1 ) {

			metalness *= MAP_SAMPLE(
				material.metalnessMap, ${MATERIAL_TRANSFORM_TEXEL.metallicMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 1}u, MAP_UV( 1u )
			).b;

		}

		// emission (emissiveMap = bit 4)
		vec3 emission = vitrumFiniteNonNegativeRadianceProduct(
			material.emissive, vec3( material.emissiveIntensity )
		);
		if ( useTextures && material.emissiveMap != - 1 ) {

			emission = vitrumFiniteNonNegativeRadianceProduct(
				emission,
				MAP_RADIANCE_SAMPLE(
					material.emissiveMap, ${MATERIAL_TRANSFORM_TEXEL.emissiveMap}u,
					${MATERIAL_WRAP_TEXEL_OFFSET + 4}u, MAP_UV( 4u )
				).xyz
			).xyz;

		}

		// D3 — lightMap (bit 17): baked OUTGOING radiance (linear), added at camera-visible
		// hits ONLY (pathDepth == 0; matches pt-webgpu's emissive-on-hit semantics).
		// It never enters NEE/MIS (it is not in the lights texture), and adding it
		// at indirect depths would double-count the live lights the bake encodes.
		if ( useTextures && material.lightMap != - 1 && pathDepth == 0 ) {

			emission = vitrumFiniteNonNegativeRadianceSum(
				emission,
				vitrumFiniteNonNegativeRadianceProduct(
					vec3( material.lightMapIntensity ),
					MAP_RADIANCE_SAMPLE(
						material.lightMap, ${MATERIAL_LIGHTMAP_TRANSFORM_TEXEL}u,
						${MATERIAL_WRAP_TEXEL_OFFSET + 17}u, MAP_UV( 17u )
					).rgb
				)
			);

		}

		// transmission (transmissionMap = bit 3)
		float transmission = material.transmission;
		if ( useTextures && material.transmissionMap != - 1 ) {

			transmission *= MAP_SAMPLE(
				material.transmissionMap, ${MATERIAL_TRANSFORM_TEXEL.transmissionMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 3}u, MAP_UV( 3u )
			).r;

		}

		// KHR_materials_volume thicknessTexture (bit 20): G channel scales
		// thicknessFactor. When present (or when scalar thickness > 0), the
		// Beer-Lambert path length is clamped to this authored local thickness.
		float attenuationThickness = material.thickness;
		bool hasAttenuationThickness = material.thickness > 0.0 || material.thicknessMap != - 1;
		if ( useTextures && material.thicknessMap != - 1 ) {

			attenuationThickness *= MAP_SAMPLE(
				material.thicknessMap, ${MATERIAL_THICKNESS_TRANSFORM_TEXEL}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 20}u, MAP_UV( 20u )
			).g;

		}

		vec3 baseNormal = normal;
		// frontFace is used to determine transmissive properties and per-face layer selection.
		bool frontFaceHit = surfaceHit.side == 1.0 || transmission == 0.0;
		bool hasFaceLayer = frontFaceHit ? material.hasFrontLayer : material.hasBackLayer;
		vec3 layerTransmission = frontFaceHit ? material.frontLayerTransmission : material.backLayerTransmission;
		float layerRoughness = frontFaceHit ? material.frontLayerRoughness : material.backLayerRoughness;

		int activeNormalMap = material.normalMap;
		uint activeNormalMapTransformOffset = ${MATERIAL_TRANSFORM_TEXEL.normalMap}u;
		vec2 activeNormalScale = material.normalScale;
		uint activeNormalMapPolicyOffset = ${MATERIAL_WRAP_TEXEL_OFFSET + 5}u;
		int activeNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );
		vec2 activeNormalUv = textureSampleBarycoord( attributesArray, activeNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
		if ( hasFaceLayer && frontFaceHit && material.frontLayerNormalMap != - 1 ) {
			activeNormalMap = material.frontLayerNormalMap;
			activeNormalMapTransformOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 1}u;
			activeNormalScale = material.frontLayerNormalScale;
			activeNormalMapPolicyOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 5}u;
			activeNormalUvLayer = int( round( material.frontLayerNormalTexCoord ) );
			activeNormalUv = textureSampleBarycoord( attributesArray, activeNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
		} else if ( hasFaceLayer && ! frontFaceHit && material.backLayerNormalMap != - 1 ) {
			activeNormalMap = material.backLayerNormalMap;
			activeNormalMapTransformOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 3}u;
			activeNormalScale = material.backLayerNormalScale;
			activeNormalMapPolicyOffset = ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 6}u;
			activeNormalUvLayer = int( round( material.backLayerNormalTexCoord ) );
			activeNormalUv = textureSampleBarycoord( attributesArray, activeNormalUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
		}

		// Sprint 4: P3 — when !useTextures, skip TBN tangent-space transform
		// (avoids tangent attribute fetch) and use the smooth geometric normal directly.
		// normalMap = bit 5; per-face layer normals use their own UV payload.
		if ( useTextures && activeNormalMap != - 1 ) {

			vec4 tangentSample = textureSampleBarycoord(
				attributesArray,
				ATTR_TANGENT,
				surfaceHit.barycoord,
				surfaceHit.faceIndices.xyz
			);

			mat3 vTBN = getBasisFromSelectedUv(
				bvh.position, attributesArray, activeNormalUvLayer,
				surfaceHit.faceIndices.xyz, normal, tangentSample
			);
			vec3 texNormal = MAP_SAMPLE(
				activeNormalMap,
				activeNormalMapTransformOffset,
				activeNormalMapPolicyOffset,
				activeNormalUv
			).xyz * 2.0 - 1.0;
			texNormal.xy *= activeNormalScale;
			vec3 mappedNormal = vTBN * texNormal;
			if ( length( mappedNormal ) > 1e-6 ) normal = normalize( mappedNormal );

		}

		// D3 — bumpMap (bit 18): height-field normal perturbation (Blinn 1978),
		// forward differences in UV space (no screen derivatives on secondary rays).
		// The packed policy records each map's true source extent inside the atlas;
		// derive one logical source-texel step independently on each axis.
		// Applied AFTER the normal map so the two compose.
		if ( useTextures && material.bumpMap != - 1 ) {

			vec4 tangentSample = textureSampleBarycoord(
				attributesArray,
				ATTR_TANGENT,
				surfaceHit.barycoord,
				surfaceHit.faceIndices.xyz
			);

			int bumpUvLayer = readMaterialMapUvLayer( materials, materialIndex, 18u );
			mat3 bumpBasis = getBasisFromSelectedUv(
				bvh.position, attributesArray, bumpUvLayer,
				surfaceHit.faceIndices.xyz, normal, tangentSample
			);
			vec3 uvPrime = MAP_TRANSFORM( ${MATERIAL_BUMP_TRANSFORM_TEXEL}u ) * vec3(
				textureSampleBarycoord( attributesArray, bumpUvLayer, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy,
				1
			);
			vec4 bumpMapPolicy = MAP_POLICY( ${MATERIAL_WRAP_TEXEL_OFFSET + 18}u );
			vec2 bumpTexel = 1.0 / vec2(
				materialTextureSourceSize( textures, bumpMapPolicy, 0 )
			);
			float hC = sampleMaterialTexture( textures, uvPrime.xy, material.bumpMap, bumpMapPolicy ).r;
			float hU = sampleMaterialTexture(
				textures,
				uvPrime.xy + vec2( bumpTexel.x, 0.0 ),
					material.bumpMap,
					bumpMapPolicy
				).r;
			float hV = sampleMaterialTexture(
				textures,
				uvPrime.xy + vec2( 0.0, bumpTexel.y ),
					material.bumpMap,
					bumpMapPolicy
				).r;
			float dhdu = ( hU - hC ) / bumpTexel.x;
			float dhdv = ( hV - hC ) / bumpTexel.y;
				vec3 tangent = bumpBasis[ 0 ];
				vec3 bitangent = bumpBasis[ 1 ];
				vec3 perturbed = normal - material.bumpScale * ( dhdu * tangent + dhdv * bitangent );
				if ( length( perturbed ) > 1e-6 ) {

					normal = normalize( perturbed );

				}

		}

		normal *= surfaceHit.side;

		// clearcoat (clearcoatMap = bit 7)
		float clearcoat = material.clearcoat;
		if ( useTextures && material.clearcoatMap != - 1 ) {

			clearcoat *= MAP_SAMPLE(
				material.clearcoatMap, ${MATERIAL_TRANSFORM_TEXEL.clearcoatMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 7}u, MAP_UV( 7u )
			).r;

		}

		// clearcoatRoughness (clearcoatRoughnessMap = bit 8)
		float clearcoatRoughness = material.clearcoatRoughness;
		if ( useTextures && material.clearcoatRoughnessMap != - 1 ) {

			clearcoatRoughness *= MAP_SAMPLE(
				material.clearcoatRoughnessMap, ${MATERIAL_TRANSFORM_TEXEL.clearcoatRoughnessMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 8}u, MAP_UV( 8u )
			).g;

		}

		// clearcoatNormal (clearcoatNormalMap = bit 9)
		vec3 clearcoatNormal = baseNormal;
		if ( useTextures && material.clearcoatNormalMap != - 1 ) {

			vec4 tangentSample = textureSampleBarycoord(
				attributesArray,
				ATTR_TANGENT,
				surfaceHit.barycoord,
				surfaceHit.faceIndices.xyz
			);

			int clearcoatNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 9u );
			mat3 vTBN = getBasisFromSelectedUv(
				bvh.position, attributesArray, clearcoatNormalUvLayer,
				surfaceHit.faceIndices.xyz, clearcoatNormal, tangentSample
			);
			vec2 clearcoatNormalUv = textureSampleBarycoord(
				attributesArray, clearcoatNormalUvLayer,
				surfaceHit.barycoord, surfaceHit.faceIndices.xyz
			).xy;
			vec3 texNormal = MAP_SAMPLE(
				material.clearcoatNormalMap, ${MATERIAL_TRANSFORM_TEXEL.clearcoatNormalMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 9}u, clearcoatNormalUv
			).xyz * 2.0 - 1.0;
			texNormal.xy *= material.clearcoatNormalScale;
			vec3 mappedClearcoatNormal = vTBN * texNormal;
			if ( length( mappedClearcoatNormal ) > 1e-6 ) {
				clearcoatNormal = normalize( mappedClearcoatNormal );
			}

		}

		clearcoatNormal *= surfaceHit.side;

		// sheenColor (sheenColorMap = bit 10)
		vec3 sheenColor = material.sheenColor;
		if ( useTextures && material.sheenColorMap != - 1 ) {

			sheenColor *= MAP_SRGB_SAMPLE(
				material.sheenColorMap, ${MATERIAL_TRANSFORM_TEXEL.sheenColorMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 10}u, MAP_UV( 10u )
			).rgb;

		}

		// sheenRoughness (sheenRoughnessMap = bit 11)
		float sheenRoughness = material.sheenRoughness;
		if ( useTextures && material.sheenRoughnessMap != - 1 ) {

			sheenRoughness *= MAP_SAMPLE(
				material.sheenRoughnessMap, ${MATERIAL_TRANSFORM_TEXEL.sheenRoughnessMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 11}u, MAP_UV( 11u )
			).a;

		}

		// iridescence (iridescenceMap = bit 12)
		float iridescence = material.iridescence;
		if ( useTextures && material.iridescenceMap != - 1 ) {

			iridescence *= MAP_SAMPLE(
				material.iridescenceMap, ${MATERIAL_TRANSFORM_TEXEL.iridescenceMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 12}u, MAP_UV( 12u )
			).r;

		}

		// iridescence thickness (iridescenceThicknessMap = bit 13)
		float iridescenceThickness = material.iridescenceThicknessMaximum;
		if ( useTextures && material.iridescenceThicknessMap != - 1 ) {

			float iridescenceThicknessSampled = MAP_SAMPLE(
				material.iridescenceThicknessMap, ${MATERIAL_TRANSFORM_TEXEL.iridescenceThicknessMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 13}u, MAP_UV( 13u )
			).g;
			iridescenceThickness = mix( material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, iridescenceThicknessSampled );

		}

		iridescence = iridescenceThickness == 0.0 ? 0.0 : iridescence;

		// specular color (specularColorMap = bit 14)
		vec3 specularColor = material.specularColor;
		if ( useTextures && material.specularColorMap != - 1 ) {

			specularColor *= MAP_SRGB_SAMPLE(
				material.specularColorMap, ${MATERIAL_TRANSFORM_TEXEL.specularColorMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 14}u, MAP_UV( 14u )
			).rgb;

		}

		// specular intensity (specularIntensityMap = bit 15)
		float specularIntensity = material.specularIntensity;
		if ( useTextures && material.specularIntensityMap != - 1 ) {

			specularIntensity *= MAP_SAMPLE(
				material.specularIntensityMap, ${MATERIAL_TRANSFORM_TEXEL.specularIntensityMap}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 15}u, MAP_UV( 15u )
			).a;

		}

		// anisotropyMap (bit 19): KHR_materials_anisotropy stores tangent direction
		// in RG ([0,1] -> [-1,1]) and strength in B. Mirrors pt-webgpu's
		// materialAnisotropy/materialAnisotropyRotation accessors.
		float anisotropy = clamp( material.anisotropy, 0.0, 1.0 );
		float anisotropyRotation = material.anisotropyRotation;
		if ( useTextures && material.anisotropyMap != - 1 ) {

			vec3 anisotropyTexel = MAP_SAMPLE(
				material.anisotropyMap, ${MATERIAL_ANISOTROPY_TRANSFORM_TEXEL}u,
				${MATERIAL_WRAP_TEXEL_OFFSET + 19}u, MAP_UV( 19u )
			).rgb;
			vec2 rg = anisotropyTexel.rg * 2.0 - vec2( 1.0 );
			anisotropy *= anisotropyTexel.b;
			anisotropyRotation += atan( rg.y, rg.x );

		}

		layerTransmission = clamp( layerTransmission, vec3( 0.0 ), vec3( 1.0 ) );
		if ( hasFaceLayer && layerRoughness >= 0.0 ) {
			roughness = clamp( layerRoughness, 0.0, 1.0 );
		}

		surf.volumeParticle = false;

		surf.faceNormal = surfaceHit.faceNormal;
		surf.normal = normal;

		vec3 surfaceColor = albedo.rgb;
		if ( uSpectralRendering == 1 && material.hasSpectralReflectance ) {
			float baseReflectance = evalSpectrum( material.spectralReflectanceCoeffs, heroWavelength );
			float modulation = heroScalarFromRgb( albedoModulation, heroWavelength );
			surfaceColor = vec3( baseReflectance * modulation );
		}

		surf.metalness = metalness;
		surf.color = surfaceColor;
		surf.rgbColor = albedo.rgb;
		surf.emission = emission;

		surf.ior = material.ior;
		surf.transmission = transmission;
		surf.thinFilm = material.thinFilm;
		surf.thinFilmEnabled = material.thinFilmEnabled;
		surf.thinFilmLayerCount = material.thinFilmLayerCount;
		surf.thinFilmIncidentIor = material.thinFilmIncidentIor;
		surf.thinFilmAngleDependent = material.thinFilmAngleDependent;
		surf.dispersionStrength = material.dispersionStrength;
		surf.sssSigmaT = material.sssSigmaT;
		surf.sssAnisotropyG = material.sssAnisotropyG;
		surf.sssSigmaS = material.sssSigmaS;
		surf.hasSpectralAttenuation = material.hasSpectralAttenuation;
		surf.activeLayerTransmission = hasFaceLayer ? layerTransmission : vec3( 1.0 );
		surf.hasActiveLayer = hasFaceLayer;
		surf.materialIndex = materialIndex;
		surf.attenuationColor = material.attenuationColor;
		surf.attenuationDistance = material.attenuationDistance;
		surf.attenuationThickness = max( attenuationThickness, 0.0 );
		surf.hasAttenuationThickness = hasAttenuationThickness;

		surf.clearcoatNormal = clearcoatNormal;
		surf.clearcoat = clearcoat;

		surf.sheen = material.sheen;
		surf.sheenColor = sheenColor;

		surf.iridescence = iridescence;
		surf.iridescenceIor = material.iridescenceIor;
		surf.iridescenceThickness = iridescenceThickness;

		surf.specularColor = specularColor;
		surf.specularIntensity = specularIntensity;
		surf.anisotropy = clamp( anisotropy, 0.0, 1.0 );
		surf.anisotropyRotation = anisotropyRotation;
		surf.spectralReflectanceCoeffs = material.spectralReflectanceCoeffs;
		surf.hasSpectralReflectance = material.hasSpectralReflectance;
		surf.envMapIntensity = max( material.envMapIntensity, 0.0 );

		// apply perceptual roughness factor from gltf. sheen perceptual roughness is
		// applied by its brdf function
		// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#microfacet-surfaces
		surf.roughness = roughness * roughness;
		surf.clearcoatRoughness = clearcoatRoughness * clearcoatRoughness;
		surf.sheenRoughness = sheenRoughness;

		// frontFace is used to determine transmissive properties and PDF. If no transmission is used
		// then we can just always assume this is a front face.
		surf.frontFace = frontFaceHit;
		surf.eta = material.thinFilm || surf.frontFace ? 1.0 / material.ior : material.ior;
		surf.f0 = iorRatioToF0( surf.eta );

		// Compute the filtered roughness value to use during specular reflection computations.
		// The accumulated roughness value is scaled by a user setting and a "magic value" of 5.0.
		// If we're exiting something transmissive then scale the factor down significantly so we can retain
		// sharp internal reflections
		surf.filteredRoughness = applyFilteredGlossy( surf.roughness, accumulatedRoughness );
		surf.filteredClearcoatRoughness = applyFilteredGlossy( surf.clearcoatRoughness, accumulatedRoughness );

		// get the normal frames
		vec4 bsdfTangentSample = textureSampleBarycoord(
			attributesArray,
			ATTR_TANGENT,
			surfaceHit.barycoord,
			surfaceHit.faceIndices.xyz
		);
		int bsdfBasisUvLayer = material.anisotropyMap != - 1
			? readMaterialMapUvLayer( materials, materialIndex, 19u )
			: ATTR_UV;
		surf.normalBasis = getBasisFromSelectedUv(
			bvh.position, attributesArray, bsdfBasisUvLayer,
			surfaceHit.faceIndices.xyz, surf.normal, bsdfTangentSample
		);

		int clearcoatBasisUvLayer = material.clearcoatNormalMap != - 1
			? readMaterialMapUvLayer( materials, materialIndex, 9u )
			: ATTR_UV;
		surf.clearcoatBasis = getBasisFromSelectedUv(
			bvh.position, attributesArray, clearcoatBasisUvLayer,
			surfaceHit.faceIndices.xyz, surf.clearcoatNormal, bsdfTangentSample
		);

		// Sprint 4: P1 — lobeMask bitfield.
		// Gates optional BSDF lobes so downstream bsdfEval skips zero-weight math.
		// Diffuse (bit 0): present when not fully metallic and non-transmissive path is active.
		// Specular (bit 1): always present.
		// Sheen (bit 2), clearcoat (bit 3), iridescence (bit 4), transmission (bit 5).
		surf.lobeMask = 0u;
		if ( surf.roughness > 0.0 || surf.metalness < 1.0 ) surf.lobeMask |= 1u;  // diffuse
		surf.lobeMask |= 2u;                                                        // specular always
    if ( surf.sheen > 0.0 )         surf.lobeMask |= 4u;
    if ( surf.clearcoat > 0.0 )     surf.lobeMask |= 8u;
    if ( surf.iridescence > 0.0 )   surf.lobeMask |= 16u;
    if ( surf.transmission > 0.0 )  surf.lobeMask |= 32u;

		#undef MAP_SAMPLE
		#undef MAP_SRGB_SAMPLE
		#undef MAP_RADIANCE_SAMPLE
		#undef MAP_POLICY
		#undef MAP_TRANSFORM
		#undef MAP_UV

		return HIT_SURFACE;

	}

	int getSurfaceRecord(
		uint materialIndex, SurfaceHit surfaceHit, sampler2DArray attributesArray,
		float accumulatedRoughness, int pathDepth, float heroWavelength,
		inout SurfaceRecord surf
	) {

		return getSurfaceRecord(
			materialIndex, surfaceHit, attributesArray,
			accumulatedRoughness, pathDepth, heroWavelength,
			false,
			surf
		);

	}
`;
