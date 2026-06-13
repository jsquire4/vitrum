
/** @public — dynamic-access test-load-bearing; accessed via namespace import in wsl-gpu/scripts and untestedMaterialMaps.test.ts */
export const get_surface_record_function = /* glsl */`

	#define SKIP_SURFACE 0
	#define HIT_SURFACE 1
	// materialLodDepth controls the optional bounce-depth threshold beyond which
	// texture fetches are replaced by flat material constants. The host default is
	// 0, which disables LOD and preserves highest-fidelity texture sampling at every
	// bounce; positive values opt into the performance approximation.
	uniform int materialLodDepth;

	int getSurfaceRecord(
		Material material, uint materialIndex, SurfaceHit surfaceHit, sampler2DArray attributesArray,
		float accumulatedRoughness, int pathDepth,
		inout SurfaceRecord surf
	) {

		if ( material.fogVolume ) {

			vec3 normal = vec3( 0, 0, 1 );

			SurfaceRecord fogSurface;
			fogSurface.volumeParticle = true;
			fogSurface.color = material.color;
			fogSurface.emission = material.emissiveIntensity * material.emissive;
			fogSurface.normal = normal;
			fogSurface.faceNormal = normal;
			fogSurface.clearcoatNormal = normal;
			// Sprint 4: fog particle — only diffuse lobe; never liteMode.
			fogSurface.lobeMask = 1u;
			fogSurface.liteMode = false;

			surf = fogSurface;
			return HIT_SURFACE;

		}

		// uv coord for textures (uv0 = ATTR_UV; uv1 = ATTR_UV1, falls back to uv0)
		vec2 uv = textureSampleBarycoord( attributesArray, ATTR_UV, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
		vec2 uv1 = textureSampleBarycoord( attributesArray, ATTR_UV1, surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy;
		vec4 vertexColor = textureSampleBarycoord( attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz );

		// Inline helper: select the correct UV channel for map bit k.
		// Returns uv1 when bit k is set in material.uvTexCoordMask, else uv.
		#define MAP_UV(bit) ( ( ( material.uvTexCoordMask >> (bit) ) & 1u ) != 0u ? uv1 : uv )

		// Optional material LOD by depth. When pathDepth > materialLodDepth, skip
		// texture fetches and use flat material constants. materialLodDepth == 0
		// disables LOD (textures at all depths), which is the host default.
		bool useTextures = ( materialLodDepth == 0 ) || ( pathDepth <= materialLodDepth );

		// albedo (baseColorMap = bit 0)
		vec4 albedo = vec4( material.color, material.opacity );
		if ( useTextures && material.map != - 1 ) {

			vec3 uvPrime = material.mapTransform * vec3( MAP_UV( 0u ), 1 );
			albedo *= sampleMaterialTexture( textures, uvPrime.xy, material.map, material.mapWrap );

		}

		if ( material.vertexColors ) {

			albedo *= vertexColor;

		}

		// alphaMap (bit 6)
		if ( useTextures && material.alphaMap != - 1 ) {

			vec3 uvPrime = material.alphaMapTransform * vec3( MAP_UV( 6u ), 1 );
			albedo.a *= sampleMaterialTexture( textures, uvPrime.xy, material.alphaMap, material.alphaMapWrap ).x;

		}

		// D3 — aoMap (glTF occlusionTexture, R channel, bit 16): modulate albedo by
		// mix(1, ao, aoMapIntensity). CAVEAT (documented biased semantics, mirrors
		// pt-webgpu sampleAoFactor): a path tracer integrates real occlusion, so a
		// baked AO term double-darkens crevices — aoMapIntensity is the artist dial
		// (1 matches the raster look; 0 disables).
		if ( useTextures && material.aoMap != - 1 ) {

			vec3 uvPrime = material.aoMapTransform * vec3( MAP_UV( 16u ), 1 );
			float ao = sampleMaterialTexture( textures, uvPrime.xy, material.aoMap, material.aoMapWrap ).r;
			albedo.rgb *= clamp( mix( 1.0, ao, material.aoMapIntensity ), 0.0, 1.0 );

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
			|| material.transparent && ! useAlphaTest && albedo.a < rand( 3 )
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

			vec3 uvPrime = material.roughnessMapTransform * vec3( MAP_UV( 2u ), 1 );
			roughness *= sampleMaterialTexture( textures, uvPrime.xy, material.roughnessMap, material.roughnessMapWrap ).g;

		}

		// metalness (metallicMap = bit 1)
		float metalness = material.metalness;
		if ( useTextures && material.metalnessMap != - 1 ) {

			vec3 uvPrime = material.metalnessMapTransform * vec3( MAP_UV( 1u ), 1 );
			metalness *= sampleMaterialTexture( textures, uvPrime.xy, material.metalnessMap, material.metalnessMapWrap ).b;

		}

		// emission (emissiveMap = bit 4)
		vec3 emission = material.emissiveIntensity * material.emissive;
		if ( useTextures && material.emissiveMap != - 1 ) {

			vec3 uvPrime = material.emissiveMapTransform * vec3( MAP_UV( 4u ), 1 );
			emission *= sampleMaterialTexture( textures, uvPrime.xy, material.emissiveMap, material.emissiveMapWrap ).xyz;

		}

		// D3 — lightMap (bit 17): baked OUTGOING radiance (linear), added at camera-visible
		// hits ONLY (pathDepth == 0; matches pt-webgpu's emissive-on-hit semantics).
		// It never enters NEE/MIS (it is not in the lights texture), and adding it
		// at indirect depths would double-count the live lights the bake encodes.
		if ( useTextures && material.lightMap != - 1 && pathDepth == 0 ) {

			vec3 uvPrime = material.lightMapTransform * vec3( MAP_UV( 17u ), 1 );
			emission += material.lightMapIntensity *
				sampleMaterialTexture( textures, uvPrime.xy, material.lightMap, material.lightMapWrap ).rgb;

		}

		// transmission (transmissionMap = bit 3)
		float transmission = material.transmission;
		if ( useTextures && material.transmissionMap != - 1 ) {

			vec3 uvPrime = material.transmissionMapTransform * vec3( MAP_UV( 3u ), 1 );
			transmission *= sampleMaterialTexture( textures, uvPrime.xy, material.transmissionMap, material.transmissionMapWrap ).r;

		}

		// KHR_materials_volume thicknessTexture (bit 20): G channel scales
		// thicknessFactor. When present (or when scalar thickness > 0), the
		// Beer-Lambert path length is clamped to this authored local thickness.
		float attenuationThickness = material.thickness;
		bool hasAttenuationThickness = material.thickness > 0.0 || material.thicknessMap != - 1;
		if ( useTextures && material.thicknessMap != - 1 ) {

			vec3 uvPrime = material.thicknessMapTransform * vec3( MAP_UV( 20u ), 1 );
			attenuationThickness *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.thicknessMap,
				material.thicknessMapWrap
			).g;

		}

		// normal
		if ( material.flatShading ) {

			// if we're rendering a flat shaded object then use the face normals - the face normal
			// is provided based on the side the ray hits the mesh so flip it to align with the
			// interpolated vertex normals.
			normal = surfaceHit.faceNormal * surfaceHit.side;

		}

		vec3 baseNormal = normal;
		// Sprint 4: P3 — when !useTextures, skip TBN tangent-space transform
		// (avoids tangent attribute fetch) and use the smooth geometric normal directly.
		// normalMap = bit 5
		if ( useTextures && material.normalMap != - 1 ) {

			vec4 tangentSample = textureSampleBarycoord(
				attributesArray,
				ATTR_TANGENT,
				surfaceHit.barycoord,
				surfaceHit.faceIndices.xyz
			);

			// some provided tangents can be malformed (0, 0, 0) causing the normal to be degenerate
			// resulting in NaNs and slow path tracing.
			if ( length( tangentSample.xyz ) > 0.0 ) {

				vec3 tangent = normalize( tangentSample.xyz );
				float tangentHandedness = tangentSample.w < 0.0 ? -1.0 : 1.0;
				vec3 bitangent = normalize( cross( normal, tangent ) * tangentHandedness );
				mat3 vTBN = mat3( tangent, bitangent, normal );

				vec3 uvPrime = material.normalMapTransform * vec3( MAP_UV( 5u ), 1 );
				vec3 texNormal =
					sampleMaterialTexture( textures, uvPrime.xy, material.normalMap, material.normalMapWrap ).xyz * 2.0 - 1.0;
				texNormal.xy *= material.normalScale;
				normal = vTBN * texNormal;

			}

		}

		// D3 — bumpMap (bit 18): height-field normal perturbation (Blinn 1978), central
		// differences in UV space (no screen derivatives on secondary rays; mirrors
		// pt-webgpu applyBumpMap's fixed 1/512 step + n - scale·(dh/du·T + dh/dv·B)).
		// Applied AFTER the normal map so the two compose.
		if ( useTextures && material.bumpMap != - 1 ) {

			vec4 tangentSample = textureSampleBarycoord(
				attributesArray,
				ATTR_TANGENT,
				surfaceHit.barycoord,
				surfaceHit.faceIndices.xyz
			);

			if ( length( tangentSample.xyz ) > 0.0 ) {

				vec3 uvPrime = material.bumpMapTransform * vec3( MAP_UV( 18u ), 1 );
				float du = 1.0 / 512.0;
				float hC = sampleMaterialTexture( textures, uvPrime.xy, material.bumpMap, material.bumpMapWrap ).r;
				float hU = sampleMaterialTexture(
					textures,
					uvPrime.xy + vec2( du, 0.0 ),
					material.bumpMap,
					material.bumpMapWrap
				).r;
				float hV = sampleMaterialTexture(
					textures,
					uvPrime.xy + vec2( 0.0, du ),
					material.bumpMap,
					material.bumpMapWrap
				).r;
				float dhdu = ( hU - hC ) / du;
				float dhdv = ( hV - hC ) / du;
				vec3 tangent = normalize( tangentSample.xyz );
				float tangentHandedness = tangentSample.w < 0.0 ? -1.0 : 1.0;
				vec3 bitangent = normalize( cross( normal, tangent ) * tangentHandedness );
				vec3 perturbed = normal - material.bumpScale * ( dhdu * tangent + dhdv * bitangent );
				if ( length( perturbed ) > 1e-6 ) {

					normal = normalize( perturbed );

				}

			}

		}

		normal *= surfaceHit.side;

		// clearcoat (clearcoatMap = bit 7)
		float clearcoat = material.clearcoat;
		if ( useTextures && material.clearcoatMap != - 1 ) {

			vec3 uvPrime = material.clearcoatMapTransform * vec3( MAP_UV( 7u ), 1 );
			clearcoat *= sampleMaterialTexture( textures, uvPrime.xy, material.clearcoatMap, material.clearcoatMapWrap ).r;

		}

		// clearcoatRoughness (clearcoatRoughnessMap = bit 8)
		float clearcoatRoughness = material.clearcoatRoughness;
		if ( useTextures && material.clearcoatRoughnessMap != - 1 ) {

			vec3 uvPrime = material.clearcoatRoughnessMapTransform * vec3( MAP_UV( 8u ), 1 );
			clearcoatRoughness *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.clearcoatRoughnessMap,
				material.clearcoatRoughnessMapWrap
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

			// some provided tangents can be malformed (0, 0, 0) causing the normal to be degenerate
			// resulting in NaNs and slow path tracing.
			if ( length( tangentSample.xyz ) > 0.0 ) {

				vec3 tangent = normalize( tangentSample.xyz );
				float tangentHandedness = tangentSample.w < 0.0 ? -1.0 : 1.0;
				vec3 bitangent = normalize( cross( clearcoatNormal, tangent ) * tangentHandedness );
				mat3 vTBN = mat3( tangent, bitangent, clearcoatNormal );

				vec3 uvPrime = material.clearcoatNormalMapTransform * vec3( MAP_UV( 9u ), 1 );
				vec3 texNormal = sampleMaterialTexture(
					textures,
					uvPrime.xy,
					material.clearcoatNormalMap,
					material.clearcoatNormalMapWrap
				).xyz * 2.0 - 1.0;
				texNormal.xy *= material.clearcoatNormalScale;
				clearcoatNormal = vTBN * texNormal;

			}

		}

		clearcoatNormal *= surfaceHit.side;

		// sheenColor (sheenColorMap = bit 10)
		vec3 sheenColor = material.sheenColor;
		if ( useTextures && material.sheenColorMap != - 1 ) {

			vec3 uvPrime = material.sheenColorMapTransform * vec3( MAP_UV( 10u ), 1 );
			sheenColor *= sampleMaterialTexture( textures, uvPrime.xy, material.sheenColorMap, material.sheenColorMapWrap ).rgb;

		}

		// sheenRoughness (sheenRoughnessMap = bit 11)
		float sheenRoughness = material.sheenRoughness;
		if ( useTextures && material.sheenRoughnessMap != - 1 ) {

			vec3 uvPrime = material.sheenRoughnessMapTransform * vec3( MAP_UV( 11u ), 1 );
			sheenRoughness *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.sheenRoughnessMap,
				material.sheenRoughnessMapWrap
			).a;

		}

		// iridescence (iridescenceMap = bit 12)
		float iridescence = material.iridescence;
		if ( useTextures && material.iridescenceMap != - 1 ) {

			vec3 uvPrime = material.iridescenceMapTransform * vec3( MAP_UV( 12u ), 1 );
			iridescence *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.iridescenceMap,
				material.iridescenceMapWrap
			).r;

		}

		// iridescence thickness (iridescenceThicknessMap = bit 13)
		float iridescenceThickness = material.iridescenceThicknessMaximum;
		if ( useTextures && material.iridescenceThicknessMap != - 1 ) {

			vec3 uvPrime = material.iridescenceThicknessMapTransform * vec3( MAP_UV( 13u ), 1 );
			float iridescenceThicknessSampled = sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.iridescenceThicknessMap,
				material.iridescenceThicknessMapWrap
			).g;
			iridescenceThickness = mix( material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, iridescenceThicknessSampled );

		}

		iridescence = iridescenceThickness == 0.0 ? 0.0 : iridescence;

		// specular color (specularColorMap = bit 14)
		vec3 specularColor = material.specularColor;
		if ( useTextures && material.specularColorMap != - 1 ) {

			vec3 uvPrime = material.specularColorMapTransform * vec3( MAP_UV( 14u ), 1 );
			specularColor *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.specularColorMap,
				material.specularColorMapWrap
			).rgb;

		}

		// specular intensity (specularIntensityMap = bit 15)
		float specularIntensity = material.specularIntensity;
		if ( useTextures && material.specularIntensityMap != - 1 ) {

			vec3 uvPrime = material.specularIntensityMapTransform * vec3( MAP_UV( 15u ), 1 );
			specularIntensity *= sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.specularIntensityMap,
				material.specularIntensityMapWrap
			).a;

		}

		// anisotropyMap (bit 19): KHR_materials_anisotropy stores tangent direction
		// in RG ([0,1] -> [-1,1]) and strength in B. Mirrors pt-webgpu's
		// materialAnisotropy/materialAnisotropyRotation accessors.
		float anisotropy = clamp( material.anisotropy, 0.0, 1.0 );
		float anisotropyRotation = material.anisotropyRotation;
		if ( useTextures && material.anisotropyMap != - 1 ) {

			vec3 uvPrime = material.anisotropyMapTransform * vec3( MAP_UV( 19u ), 1 );
			vec3 anisotropyTexel = sampleMaterialTexture(
				textures,
				uvPrime.xy,
				material.anisotropyMap,
				material.anisotropyMapWrap
			).rgb;
			vec2 rg = anisotropyTexel.rg * 2.0 - vec2( 1.0 );
			anisotropy *= anisotropyTexel.b;
			anisotropyRotation += atan( rg.y, rg.x );

		}

		// frontFace is used to determine transmissive properties and per-face layer selection.
		bool frontFaceHit = surfaceHit.side == 1.0 || transmission == 0.0;
		bool hasFaceLayer = frontFaceHit ? material.hasFrontLayer : material.hasBackLayer;
		vec3 layerTransmission = frontFaceHit ? material.frontLayerTransmission : material.backLayerTransmission;
		float layerRoughness = frontFaceHit ? material.frontLayerRoughness : material.backLayerRoughness;
		layerTransmission = clamp( layerTransmission, vec3( 0.0 ), vec3( 1.0 ) );
		if ( hasFaceLayer && layerRoughness >= 0.0 ) {
			roughness = clamp( layerRoughness, 0.0, 1.0 );
		}

		surf.volumeParticle = false;

		surf.faceNormal = surfaceHit.faceNormal;
		surf.normal = normal;

		surf.metalness = metalness;
		surf.color = albedo.rgb;
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
		surf.sssAlbedo = material.sssAlbedo;
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
		surf.normalBasis = getBasisFromNormal( surf.normal );
		surf.normalInvBasis = inverse( surf.normalBasis );

		surf.clearcoatBasis = getBasisFromNormal( surf.clearcoatNormal );
		surf.clearcoatInvBasis = inverse( surf.clearcoatBasis );

		// Sprint 4: P1 — lobeMask bitfield.
		// Gates optional BSDF lobes so downstream bsdfEval skips zero-weight math.
		// Diffuse (bit 0): present when not fully metallic and non-transmissive path is active.
		// Specular (bit 1): always present.
		// Sheen (bit 2), clearcoat (bit 3), iridescence (bit 4), transmission (bit 5).
		surf.lobeMask = 0u;
		if ( surf.roughness > 0.0 || surf.metalness < 1.0 ) surf.lobeMask |= 1u;  // diffuse
		surf.lobeMask |= 2u;                                                        // specular always
		if ( surf.sheen > 0.001 )       surf.lobeMask |= 4u;
		if ( surf.clearcoat > 0.001 )   surf.lobeMask |= 8u;
		if ( surf.iridescence > 0.001 ) surf.lobeMask |= 16u;
		if ( surf.transmission > 0.001 ) surf.lobeMask |= 32u;

		// Sprint 4: P2 — lite BSDF for indirect bounces.
		// At depth > 1 (second bounce and beyond), skip sheen/clearcoat/iridescence
		// and replace multiscatter GGX with single-scatter. Transmission is always
		// kept (visually important at all depths for glass).
		// liteMode is overridden to false when lobeMask has no optional lobes, as a
		// cheap no-op guard — the real override is via forceFullBSDF (lobeMask = 0xFF).
		surf.liteMode = ( pathDepth > 1 );

		return HIT_SURFACE;

	}
`;
