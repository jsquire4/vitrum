/**
 * Bounded general-BDPT light-subpath construction for the native WebGL2 PT.
 *
 * One 8x8 RGBA32F texture stores up to eight vertices:
 *   row 0: position.xyz | kind (0=ordinary, 1=delta, 3=invalid)
 *   row 1: shading normal / emitter axis | forward directional density
 *   row 2: throughput.rgb | reverse directional density
 *   row 3: direction toward predecessor | material id / endpoint sentinel
 *   row 4: surface barycentric/side payload or endpoint attenuation payload
 *   row 5: medium-stack count | material ids 0..2
 *   row 6: medium-stack material ids 3..6
 *   row 7: medium-stack material id 7 | scattering-medium id |
 *          exact surface face index as two u16 float words
 *
 * Columns are built sequentially with ping-pong render targets. Extension k
 * patches k-1 row 0 (the sampled event's real delta classification). Its
 * swapped reverse BSDF times the intermediate incoming-edge distance factor
 * belongs to k-2 row 2; this is the reverse density that becomes known only
 * after the successor exists.
 *
 * References: Veach 1997, chapter 10; PBRT-v4, section 16.3.
 */
export const bdpt_light_subpath = /* glsl */`

	const float BDPT_KIND_LIGHT = 0.0;
	const float BDPT_KIND_DELTA = 1.0;
	const float BDPT_KIND_INVALID = 3.0;

	const float BDPT_LV_AREA_EMITTER_MATID = -2.0;
	const float BDPT_LV_POINT_EMITTER_MATID = -4.0;
	const float BDPT_LV_SPOT_EMITTER_MATID = -5.0;
	const float BDPT_LV_MEDIUM_MATID = -7.0;
	const float BDPT_LV_DIRECTIONAL_EMITTER_MATID = -8.0;
	const float BDPT_LV_ENVIRONMENT_EMITTER_MATID = -9.0;

	bool bdptFiniteFloat( float value ) {
		return ! isnan( value ) && ! isinf( value );
	}

	bool bdptFiniteVec3( vec3 value ) {
		return ! any( isnan( value ) ) && ! any( isinf( value ) );
	}

	bool bdptFiniteVec4( vec4 value ) {
		return ! any( isnan( value ) ) && ! any( isinf( value ) );
	}

		bool bdptKnownEndpointSentinel( float sentinel ) {
		return
			sentinel == BDPT_LV_AREA_EMITTER_MATID ||
			sentinel == BDPT_LV_POINT_EMITTER_MATID ||
			sentinel == BDPT_LV_SPOT_EMITTER_MATID ||
			sentinel == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
			sentinel == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
		}

		vec2 bdptPackFaceIndexWords( uint faceIndex ) {
			return vec2(
				float( faceIndex & 0xffffu ),
				float( faceIndex >> 16u )
			);
		}

		bool bdptStoredVertexRowsValid(
		vec4 v0, vec4 v1, vec4 v2, vec4 v3,
		vec4 v4, vec4 v5, vec4 v6, vec4 v7
	) {
		bool knownKind =
			v0.w == BDPT_KIND_LIGHT ||
			v0.w == BDPT_KIND_DELTA;
		bool finiteRows =
			bdptFiniteVec4( v0 ) && bdptFiniteVec4( v1 ) &&
			bdptFiniteVec4( v2 ) && bdptFiniteVec4( v3 ) &&
			bdptFiniteVec4( v4 ) && bdptFiniteVec4( v5 ) &&
			bdptFiniteVec4( v6 ) && bdptFiniteVec4( v7 );
		bool finiteDensities =
			v1.w > 0.0 && v2.w >= 0.0 &&
			all( greaterThanEqual( v2.xyz, vec3( 0.0 ) ) );
			bool stackCountValid =
				v5.x >= 0.0 &&
				v5.x <= float( MEDIUM_STACK_CAPACITY ) &&
				v5.x == round( v5.x );
			bool exactSurfaceFaceValid =
				v3.w < 0.0 || meshLightSourceFaceWordsValid( v7.zw );
			return knownKind && finiteRows && finiteDensities &&
				stackCountValid && exactSurfaceFaceValid;
	}

        float bdptConnectionInverseDistanceSquared( vec3 posX, vec3 posY ) {
		return vitrumInverseSquareDistance( posY - posX );
	}

		vec4 bdptSurfacePayload( SurfaceHit hit ) {
			return vec4(
				hit.barycoord.x, hit.barycoord.y, hit.side, 0.0
			);
		}

	        bool bdptLoadSurfaceRecord(
			float materialId,
			vec4 payload,
			vec2 faceIndexWords,
			float heroWavelength,
			out SurfaceRecord surf
		) {
			if ( ! meshLightSourceFaceWordsValid( faceIndexWords ) ) return false;
			uint triIndex = meshLightSourceFaceIndex( faceIndexWords );
			uvec3 indices = uTexelFetch1D( bvh.index, triIndex ).xyz;
		vec3 p0 = texelFetch1D( bvh.position, indices.x ).xyz;
		vec3 p1 = texelFetch1D( bvh.position, indices.y ).xyz;
		vec3 p2 = texelFetch1D( bvh.position, indices.z ).xyz;
		VitrumAreaVectorMeasure faceMeasure =
			vitrumMeasureAreaVector( p1 - p0, p2 - p0, 0.5 );
		if ( ! faceMeasure.valid ) return false;

		SurfaceHit hit;
			hit.faceIndices = uvec4( indices, triIndex );
			hit.barycoord = vec3(
				payload.x, payload.y, max( 0.0, 1.0 - payload.x - payload.y )
			);
			hit.faceNormal = faceMeasure.normal;
			hit.side = payload.z < 0.0 ? -1.0 : 1.0;
		hit.dist = 0.0;
		uint matIdx = uint( max( floor( materialId + 0.5 ), 0.0 ) );
		// The stored hit already passed stochastic opacity when the path was built.
		return getSurfaceRecord(
			matIdx, hit, attributesArray, 0.0, 0, heroWavelength, true, surf
		) == HIT_SURFACE;
	}

	// Compatibility overload used by the connection kernel; the exact geometric
	// normal is reconstructed from the stored triangle rather than the fallback.
		bool bdptLoadSurfaceRecord(
			float materialId,
			vec4 payload,
			vec2 faceIndexWords,
			vec3 fallbackFaceNormal,
			float heroWavelength,
			out SurfaceRecord surf
		) {
	                return bdptLoadSurfaceRecord(
				materialId, payload, faceIndexWords, heroWavelength, surf
			);
	        }

        void bdptPackMediumStack(
                const in MediumStack stack,
                float scatteringMediumId,
                out vec4 row5,
                out vec4 row6,
                out vec4 row7
        ) {
                // Preserve outer-to-inner order exactly; leaveMedium requires
                // the current innermost material at count-1 on unpack.
                row5 = vec4( float( stack.count ), -1.0, -1.0, -1.0 );
                row6 = vec4( -1.0 );
	                row7 = vec4( -1.0, scatteringMediumId, -1.0, -1.0 );
                for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
                        if ( i >= stack.count ) break;
                        float materialId = float( stack.materialIds[ i ] );
                        if ( i == 0 ) row5.y = materialId;
                        else if ( i == 1 ) row5.z = materialId;
                        else if ( i == 2 ) row5.w = materialId;
                        else if ( i == 3 ) row6.x = materialId;
                        else if ( i == 4 ) row6.y = materialId;
                        else if ( i == 5 ) row6.z = materialId;
                        else if ( i == 6 ) row6.w = materialId;
                        else row7.x = materialId;
                }
        }

        bool bdptUnpackMediumStack(
                vec4 row5,
                vec4 row6,
                vec4 row7,
                out MediumStack stack,
                inout FogMaterial fog
        ) {
                initMediumStack( stack );
                if (
                        ! bdptFiniteVec4( row5 ) ||
                        ! bdptFiniteVec4( row6 ) ||
                        ! bdptFiniteVec4( row7 )
                ) return false;
                int count = int( round( row5.x ) );
                if (
                        count < 0 ||
                        count > MEDIUM_STACK_CAPACITY ||
                        row5.x != float( count )
                ) return false;
                float packedIds[ MEDIUM_STACK_CAPACITY ];
                packedIds[ 0 ] = row5.y;
                packedIds[ 1 ] = row5.z;
                packedIds[ 2 ] = row5.w;
                packedIds[ 3 ] = row6.x;
                packedIds[ 4 ] = row6.y;
                packedIds[ 5 ] = row6.z;
                packedIds[ 6 ] = row6.w;
                packedIds[ 7 ] = row7.x;
                for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
                        if ( i >= count ) break;
                        if ( packedIds[ i ] < 0.0 ) return false;
                        stack.materialIds[ i ] = uint( round( packedIds[ i ] ) );
                        stack.count ++;
                }
                // The final packed id remains the top/innermost medium. Do not
                // sort or deduplicate: repeated material ids are valid nesting.
                refreshMediumFromStack( stack, materials, fog );
                return true;
        }

        void writeBdptInvalidVertex(
                out vec4 v0, out vec4 v1, out vec4 v2,
                out vec4 v3, out vec4 v4, out vec4 v5,
                out vec4 v6, out vec4 v7
	) {
		v0 = vec4( 0.0, 0.0, 0.0, BDPT_KIND_INVALID );
		v1 = vec4( 0.0 );
		v2 = vec4( 0.0 );
		v3 = vec4( 0.0, 0.0, 0.0, BDPT_LV_AREA_EMITTER_MATID );
		v4 = vec4( 0.0 );
                v5 = vec4( 0.0, -1.0, 0.0, 0.0 );
                v6 = vec4( -1.0 );
                v7 = vec4( -1.0 );
	}

        float bdptAnalyticEmitterPower( uint index ) {
                return finitePositiveLightPower(
                        readLightInfo( lights.tex, index ).power
                );
        }

        float bdptMeshEmitterPower( uint index ) {
                return finitePositiveLightPower(
                        readMeshTriLight( uMeshLights, index ).power
                );
	}

	float bdptEmitterLogPower( float power ) {
		return power > 0.0 && ! isnan( power ) && ! isinf( power )
			? log2( power )
			: - INFINITY;
	}

	float bdptAnalyticEmitterLogPower( uint index ) {
		return bdptEmitterLogPower( bdptAnalyticEmitterPower( index ) );
	}

	float bdptMeshEmitterLogPower( uint index ) {
		return bdptEmitterLogPower( bdptMeshEmitterPower( index ) );
	}

	float bdptEnvironmentEmitterLogPower() {
		if (
			! ( environmentIntensity > 0.0 ) ||
			! ( envMapInfo.totalSum > 0.0 ) ||
			isnan( environmentIntensity ) || isinf( environmentIntensity ) ||
			isnan( envMapInfo.totalSum ) || isinf( envMapInfo.totalSum )
		) return - INFINITY;
		return log2( environmentIntensity ) + log2( envMapInfo.totalSum );
	}

	float bdptEmitterMaxLogPower() {
		float maxLogPower = bdptEnvironmentEmitterLogPower();
		for ( uint i = 0u; i < lights.count; i ++ ) {
			maxLogPower = max( maxLogPower, bdptAnalyticEmitterLogPower( i ) );
		}
		for ( uint i = 0u; i < uMeshLightCount; i ++ ) {
			maxLogPower = max( maxLogPower, bdptMeshEmitterLogPower( i ) );
		}
		return maxLogPower;
	}

	float bdptEmitterScaledWeight( float logPower, float maxLogPower ) {
		return logPower > - INFINITY && maxLogPower > - INFINITY
			? exp2( logPower - maxLogPower )
			: 0.0;
	}

	float bdptTotalEmitterScaledWeight( float maxLogPower ) {
		if ( ! ( maxLogPower > - INFINITY ) ) return 0.0;
		float total = bdptEmitterScaledWeight(
			bdptEnvironmentEmitterLogPower(), maxLogPower
		);
		for ( uint i = 0u; i < lights.count; i ++ ) {
			total += bdptEmitterScaledWeight(
				bdptAnalyticEmitterLogPower( i ), maxLogPower
			);
		}
		for ( uint i = 0u; i < uMeshLightCount; i ++ ) {
			total += bdptEmitterScaledWeight(
				bdptMeshEmitterLogPower( i ), maxLogPower
			);
		}
		return total;
	}

	float bdptEmitterDiscretePdf( float logPower ) {
		float maxLogPower = bdptEmitterMaxLogPower();
		float total = bdptTotalEmitterScaledWeight( maxLogPower );
		return total > 0.0
			? bdptEmitterScaledWeight( logPower, maxLogPower ) / total
			: 0.0;
	}

        // The main eye pass does not include the candidate-pass direct-light
        // module, so keep the same two-slot distant-family denominator here for
        // environment escape MIS. Finite emitters are owned by BDPT c=0.
        float bdptDirectionalNeePower() {
                float total = 0.0;
                for ( uint i = 0u; i < lights.count; i ++ ) {
                        Light light = readLightInfo( lights.tex, i );
                        if ( light.type == DIR_LIGHT_TYPE ) {
                                total += finitePositiveLightPower( light.power );
                        }
                }
                return total;
        }

        float bdptDistantNeeDenom() {
                float directionalSlot = bdptDirectionalNeePower() > 0.0 ? 1.0 : 0.0;
                float environmentSlot =
                        envMapInfo.totalSum > 0.0 && environmentIntensity > 0.0 ? 1.0 : 0.0;
                return directionalSlot + environmentSlot;
        }

	bool bdptSampleAreaAnalytic(
		Light light, vec2 uv,
		out vec3 pos, out vec3 normal, out vec3 radiance, out float pdfArea
	) {
                if (
                        ( light.type != RECT_AREA_LIGHT_TYPE && light.type != CIRC_AREA_LIGHT_TYPE ) ||
                        light.area <= 0.0
		) return false;
		if ( light.type == RECT_AREA_LIGHT_TYPE ) {
			pos = light.position + light.u * ( uv.x - 0.5 ) + light.v * ( uv.y - 0.5 );
		} else {
			float radius = 0.5 * sqrt( max( uv.x, 0.0 ) );
			float phi = 2.0 * PI * uv.y;
			pos = light.position + radius * ( light.u * cos( phi ) + light.v * sin( phi ) );
		}
		VitrumAreaVectorMeasure areaMeasure =
			vitrumMeasureAreaVector(
				light.u, light.v,
				light.type == CIRC_AREA_LIGHT_TYPE ? PI / 4.0 : 1.0
			);
		if ( ! areaMeasure.valid ) return false;
		normal = areaMeasure.normal;
		radiance = light.color * light.intensity;
                pdfArea = 1.0 / areaMeasure.area;
		return any( greaterThan( radiance, vec3( 0.0 ) ) );
	}

	bool bdptSampleMeshArea(
		MeshTriLight tri, vec2 uv,
		out vec3 pos, out vec3 normal, out vec3 radiance, out float pdfArea
	) {
                if ( tri.area <= 0.0 || tri.power <= 0.0 ) return false;
		float su = sqrt( max( uv.x, 0.0 ) );
		float b0 = 1.0 - su;
		float b1 = uv.y * su;
		float b2 = 1.0 - b0 - b1;
		pos = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * b2;
		VitrumAreaVectorMeasure areaMeasure = vitrumMeasureAreaVector(
			tri.v1 - tri.v0, tri.v2 - tri.v0, 0.5
		);
		if ( ! areaMeasure.valid ) return false;
		normal = areaMeasure.normal;
		radiance = tri.radiance;
                pdfArea = 1.0 / areaMeasure.area;
		return any( greaterThan( radiance, vec3( 0.0 ) ) );
	}

	vec3 bdptDiskLaunch( vec3 towardSource, vec2 uv ) {
		float radius = max( uBdptSceneRadius, 1.175494351e-38 );
		float r = radius * sqrt( max( uv.x, 0.0 ) );
		float phi = 2.0 * PI * uv.y;
		vec3 sourceDirection =
			vitrumNormalizeVec3( towardSource, vec3( 0.0 ) );
		mat3 basis = getBasisFromNormal( sourceDirection );
		return uBdptSceneCenter + sourceDirection * radius +
			basis[ 0 ] * ( r * cos( phi ) ) + basis[ 1 ] * ( r * sin( phi ) );
	}

	void bdptWriteEndpoint(
		vec3 pos, vec3 axis, vec3 data, vec3 radiance,
		float pdfPosition, float sentinel, float kind,
                vec4 payload,
                out vec4 v0, out vec4 v1, out vec4 v2,
                out vec4 v3, out vec4 v4, out vec4 v5,
                out vec4 v6, out vec4 v7
	) {
		bool requiresAxis = sentinel != BDPT_LV_POINT_EMITTER_MATID;
		if (
			! bdptFiniteVec3( pos ) ||
			! bdptFiniteVec3( axis ) ||
			! bdptFiniteVec3( data ) ||
			! bdptFiniteVec3( radiance ) ||
			! bdptFiniteVec4( payload ) ||
			! bdptFiniteFloat( pdfPosition ) ||
			! bdptKnownEndpointSentinel( sentinel ) ||
			( kind != BDPT_KIND_LIGHT && kind != BDPT_KIND_DELTA ) ||
			( requiresAxis && ! vitrumFiniteNonZeroVec3( axis ) ) ||
			any( lessThan( radiance, vec3( 0.0 ) ) ) ||
			! ( pdfPosition > 0.0 )
		) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		v0 = vec4( pos, kind );
		v1 = vec4( axis, pdfPosition );
		v2 = vec4( radiance / pdfPosition, 0.0 );
                v3 = vec4( data, sentinel );
                v4 = payload;
                MediumStack emptyStack;
                initMediumStack( emptyStack );
                bdptPackMediumStack( emptyStack, -1.0, v5, v6, v7 );
		if ( ! bdptStoredVertexRowsValid( v0, v1, v2, v3, v4, v5, v6, v7 ) ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
		}
	}

        void bdptWriteBounce0(
                out vec4 v0, out vec4 v1, out vec4 v2,
                out vec4 v3, out vec4 v4, out vec4 v5,
                out vec4 v6, out vec4 v7
	) {
		float maxLogPower = bdptEmitterMaxLogPower();
		float totalScaledWeight =
			bdptTotalEmitterScaledWeight( maxLogPower );
		if ( totalScaledWeight <= 0.0 ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}

		float pick = rand( 50 ) * totalScaledWeight;
		float cumulative = 0.0;
		float selectedScaledWeight = 0.0;
		int selectedFamily = -1; // 0 analytic, 1 mesh, 2 environment
		uint selectedIndex = 0u;
		for ( uint i = 0u; i < lights.count; i ++ ) {
			float p = bdptEmitterScaledWeight(
				bdptAnalyticEmitterLogPower( i ), maxLogPower
			);
			cumulative += p;
			if ( selectedFamily < 0 && p > 0.0 && pick < cumulative ) {
				selectedFamily = 0;
				selectedIndex = i;
				selectedScaledWeight = p;
			}
		}
		for ( uint i = 0u; i < uMeshLightCount; i ++ ) {
			float p = bdptEmitterScaledWeight(
				bdptMeshEmitterLogPower( i ), maxLogPower
			);
			cumulative += p;
			if ( selectedFamily < 0 && p > 0.0 && pick < cumulative ) {
				selectedFamily = 1;
				selectedIndex = i;
				selectedScaledWeight = p;
			}
		}
		float environmentScaledWeight = bdptEmitterScaledWeight(
			bdptEnvironmentEmitterLogPower(), maxLogPower
		);
		if ( selectedFamily < 0 && environmentScaledWeight > 0.0 ) {
			selectedFamily = 2;
			selectedScaledWeight = environmentScaledWeight;
		}
		if ( selectedFamily < 0 || selectedScaledWeight <= 0.0 ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		float discretePdf = selectedScaledWeight / totalScaledWeight;

		if ( selectedFamily == 1 ) {
			MeshTriLight tri = readMeshTriLight( uMeshLights, selectedIndex );
			vec3 pos; vec3 normal; vec3 radiance; float pdfArea;
			if ( ! bdptSampleMeshArea( tri, rand2( 51 ), pos, normal, radiance, pdfArea ) ) {
				writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
			}
			bdptWriteEndpoint(
				pos, normal, normal, radiance, discretePdf * pdfArea,
				BDPT_LV_AREA_EMITTER_MATID, BDPT_KIND_LIGHT,
					// Mesh-area emission sidedness is owned by the referenced
					// material and shared by NEE, forward hits, and BDPT. The
					// final two lanes retain the exact target face as two u16
					// words for c=0 visibility.
					vec4(
						tri.castShadowDisabled,
						tri.twoSided,
						tri.sourceFaceWords.x,
						tri.sourceFaceWords.y
					),
				v0, v1, v2, v3, v4, v5, v6, v7
			);
			return;
		}

			if ( selectedFamily == 2 ) {
			vec3 envColor = vec3( 0.0 );
			vec3 envLocalDir = vec3( 0.0, 1.0, 0.0 );
			float directionPdf = sampleEquirectProbability( rand2( 51 ), envColor, envLocalDir );
			vec3 towardSource = vitrumNormalizeVec3(
				invEnvRotation3x3 * envLocalDir, vec3( 0.0 )
			);
                        float radius = max( uBdptSceneRadius, 1.175494351e-38 );
                        VitrumAreaVectorMeasure launchArea = vitrumMeasureAreaVector(
                                vec3( radius, 0.0, 0.0 ),
                                vec3( 0.0, radius, 0.0 ),
                                PI
                        );
                        float pdfPosition = launchArea.valid
                                ? discretePdf / launchArea.area
                                : 0.0;
                        float neePdf = directionPdf / bdptDistantNeeDenom();
                        bdptWriteEndpoint(
                                bdptDiskLaunch( towardSource, rand2( 52 ) ), towardSource,
                                - towardSource, finiteEquirectRadiance( envColor, 1.0 ),
                                pdfPosition, BDPT_LV_ENVIRONMENT_EMITTER_MATID, BDPT_KIND_LIGHT,
                                vec4( 0.0, directionPdf, neePdf, 0.0 ),
				v0, v1, v2, v3, v4, v5, v6, v7
			);
			return;
		}

		Light light = readLightInfo( lights.tex, selectedIndex );
		vec3 radiance = light.color * light.intensity;
		if ( light.type == RECT_AREA_LIGHT_TYPE || light.type == CIRC_AREA_LIGHT_TYPE ) {
			vec3 pos; vec3 normal; vec3 areaRadiance; float pdfArea;
			if ( ! bdptSampleAreaAnalytic( light, rand2( 51 ), pos, normal, areaRadiance, pdfArea ) ) {
				writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
			}
				bdptWriteEndpoint(
					pos, normal, normal, areaRadiance, discretePdf * pdfArea,
					BDPT_LV_AREA_EMITTER_MATID, BDPT_KIND_LIGHT,
					// Core analytic rect/disc emitters are one-sided along
					// cross(u,v), matching ordinary NEE and forward-hit MIS.
					// z=-1 is the explicit no-target sentinel that distinguishes
					// analytic area roots from mesh-area roots.
					vec4( light.castShadowDisabled, 0.0, -1.0, 0.0 ),
					v0, v1, v2, v3, v4, v5, v6, v7
				);
			} else if ( light.type == POINT_LIGHT_TYPE ) {
				bdptWriteEndpoint(
					light.position, vec3( 0.0 ), vec3( 0.0 ), radiance, discretePdf,
					// Position is singular, but the emitted direction is finite.
					// Do not mark the root as a delta scattering event: doing so
					// asymmetrically removes the valid c=0↔c=1 MIS transition.
					BDPT_LV_POINT_EMITTER_MATID, BDPT_KIND_LIGHT,
				vec4( light.castShadowDisabled, light.distance, light.decay, 0.0 ),
				v0, v1, v2, v3, v4, v5, v6, v7
			);
		} else if ( light.type == SPOT_LIGHT_TYPE ) {
			vec3 backAxis = vitrumNormalizeVec3(
				cross( light.u, light.v ), vec3( 0.0 )
			);
			vec3 emitAxis = - backAxis;
				bdptWriteEndpoint(
					light.position, emitAxis, vec3( light.penumbraCos, 0.0, 0.0 ), radiance,
					discretePdf, BDPT_LV_SPOT_EMITTER_MATID, BDPT_KIND_LIGHT,
				vec4( light.castShadowDisabled, light.distance, light.decay, light.coneCos ),
				v0, v1, v2, v3, v4, v5, v6, v7
			);
		} else if ( light.type == DIR_LIGHT_TYPE ) {
			float directionPdf = 1.0;
			vec3 towardSource = sampleDirectionalCone(
				vitrumNormalizeVec3( light.u, vec3( 0.0 ) ),
				light.angularDiameter, rand2( 51 ), directionPdf
			);
				float radius = max( uBdptSceneRadius, 1.175494351e-38 );
                        VitrumAreaVectorMeasure launchArea = vitrumMeasureAreaVector(
                                vec3( radius, 0.0, 0.0 ),
                                vec3( 0.0, radius, 0.0 ),
                                PI
                        );
                        float pdfPosition = launchArea.valid
                                ? discretePdf / launchArea.area
                                : 0.0;
                        bool isDelta = light.angularDiameter <= 0.0;
			// Directional RGB is irradiance. For a soft cone, carry p_dir here so
			// the extension's division by p_dir preserves authored irradiance.
                        vec3 weightedRadiance = radiance * ( isDelta ? 1.0 : directionPdf );
                        float directionalNeePdf =
                                bdptAnalyticEmitterPower( selectedIndex ) /
                                bdptDirectionalNeePower() *
                                directionPdf /
                                bdptDistantNeeDenom();
                        bdptWriteEndpoint(
                                bdptDiskLaunch( towardSource, rand2( 52 ) ), towardSource,
                                - towardSource, weightedRadiance, pdfPosition,
                                BDPT_LV_DIRECTIONAL_EMITTER_MATID,
                                // The singular direction is an endpoint measure, not a
                                // delta scattering vertex.  Keeping the root connectable
                                // lets s=1 NEE compete with s>=2 launch-disk strategies.
                                BDPT_KIND_LIGHT,
                                vec4(
                                        light.castShadowDisabled,
                                        directionPdf,
                                        directionalNeePdf,
                                        isDelta ? 1.0 : 0.0
                                ),
				v0, v1, v2, v3, v4, v5, v6, v7
			);
		} else {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
		}
	}

        bool bdptSampledSurfaceEventIsDelta(
                SurfaceRecord surf, vec3 wo, ScatterRecord rec
        ) {
                return rec.sampledDelta;
        }

	bool bdptSampleEndpointDirection(
		float sentinel, vec3 axis, vec3 endpointData, vec4 payload, int seedBase,
		out vec3 direction, out float pdf, out vec3 throughput
	) {
			throughput = vec3( 1.0 );
			if ( sentinel == BDPT_LV_AREA_EMITTER_MATID ) {
				if ( ! vitrumFiniteNonZeroVec3( axis ) ) return false;
				vec3 endpointNormal =
					vitrumNormalizeVec3( axis, vec3( 0.0 ) );
				bool twoSided = payload.y > 0.5;
				if ( twoSided && rand( seedBase + 2 ) < 0.5 ) {
					endpointNormal = - endpointNormal;
				}
				direction = sampleHemisphere( endpointNormal, rand2( seedBase ) );
				float cosTheta = abs( dot(
					vitrumNormalizeVec3( axis, vec3( 0.0 ) ),
					direction
				) );
				pdf = cosTheta / ( twoSided ? 2.0 * PI : PI );
				throughput = vec3( cosTheta );
		} else if ( sentinel == BDPT_LV_POINT_EMITTER_MATID ) {
			direction = sampleSphere( rand2( seedBase ) );
			pdf = 1.0 / ( 4.0 * PI );
		} else if ( sentinel == BDPT_LV_SPOT_EMITTER_MATID ) {
			if ( ! vitrumFiniteNonZeroVec3( axis ) ) return false;
			float halfAngle = acos( clamp( payload.w, -1.0, 1.0 ) );
			vec3 endpointAxis =
				vitrumNormalizeVec3( axis, vec3( 0.0 ) );
			direction = sampleDirectionalCone(
				endpointAxis, 2.0 * halfAngle, rand2( seedBase ), pdf
			);
			float angleCos = dot( endpointAxis, direction );
			throughput = vec3( getSpotAttenuation( payload.w, endpointData.x, angleCos ) );
		} else if (
			sentinel == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
			sentinel == BDPT_LV_ENVIRONMENT_EMITTER_MATID
		) {
			if ( ! vitrumFiniteNonZeroVec3( endpointData ) ) return false;
			direction =
				vitrumNormalizeVec3( endpointData, vec3( 0.0 ) );
			pdf = payload.y;
		} else {
			return false;
		}
                return pdf > 0.0 &&
			bdptFiniteFloat( pdf ) &&
			vitrumFiniteNonZeroVec3( direction ) &&
			bdptFiniteVec3( throughput ) &&
			all( greaterThanEqual( throughput, vec3( 0.0 ) ) );
	}

	void writeLightSubpathVertex(
		int vertexCol,
		int maxLightBounces,
                sampler2D lightPathTex,
                const in FogMaterial initialFogMat,
                const in MediumStack initialMediumStack,
                float heroWavelength,
                out vec4 v0, out vec4 v1, out vec4 v2,
                out vec4 v3, out vec4 v4, out vec4 v5,
                out vec4 v6, out vec4 v7,
                out vec4 predecessor0, out vec4 predecessor2
	) {
		predecessor0 = vec4( 0.0 );
		predecessor2 = vec4( 0.0 );
		if ( vertexCol < 0 || vertexCol >= maxLightBounces || vertexCol >= 8 ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
		}
		if ( vertexCol == 0 ) {
			bdptWriteBounce0( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
		}

		int prevCol = vertexCol - 1;
		vec4 p0 = texelFetch( lightPathTex, ivec2( prevCol, 0 ), 0 );
		vec4 p1 = texelFetch( lightPathTex, ivec2( prevCol, 1 ), 0 );
		vec4 p2 = texelFetch( lightPathTex, ivec2( prevCol, 2 ), 0 );
		vec4 p3 = texelFetch( lightPathTex, ivec2( prevCol, 3 ), 0 );
                vec4 p4 = texelFetch( lightPathTex, ivec2( prevCol, 4 ), 0 );
		vec4 p5 = texelFetch( lightPathTex, ivec2( prevCol, 5 ), 0 );
		vec4 p6 = texelFetch( lightPathTex, ivec2( prevCol, 6 ), 0 );
		vec4 p7 = texelFetch( lightPathTex, ivec2( prevCol, 7 ), 0 );
		predecessor0 = p0;
		// The reverse-density patch belongs to vertexCol-2, not prevCol.
		// Preserve that target vertex's throughput xyz while replacing only its
		// newly-known reverse density in w. Copying p2 here corrupts the endpoint
		// throughput as soon as column 2 is built (and each earlier light vertex
		// on subsequent extensions).
		if ( vertexCol >= 2 ) {
			predecessor2 = texelFetch(
				lightPathTex, ivec2( vertexCol - 2, 2 ), 0
			);
		}
		if ( p0.w == BDPT_KIND_INVALID ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
                }
		if ( ! bdptStoredVertexRowsValid( p0, p1, p2, p3, p4, p5, p6, p7 ) ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}

                FogMaterial fogMat = initialFogMat;
                MediumStack mediumStack;
                if ( ! bdptUnpackMediumStack( p5, p6, p7, mediumStack, fogMat ) ) {
                        writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
                        return;
                }

		vec3 scatterDir = vec3( 0.0 );
		float scatterPdf = 0.0;
		vec3 scatterThroughput = vec3( 0.0 );
		bool eventDelta = false;
		float prevMatId = p3.w;
		int seedBase = 53 + vertexCol * 5;
		if ( prevMatId < 0.0 && prevMatId != BDPT_LV_MEDIUM_MATID ) {
			if ( ! bdptSampleEndpointDirection(
				prevMatId, p1.xyz, p3.xyz, p4, seedBase,
				scatterDir, scatterPdf, scatterThroughput
			) ) {
				writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
			}
			eventDelta = p0.w == BDPT_KIND_DELTA;
		} else {
                        SurfaceRecord prevSurf;
                        if ( prevMatId == BDPT_LV_MEDIUM_MATID ) {
                                if ( p7.y < 0.0 ) {
                                        writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
                                }
                                FogMaterial scatteringFog = readFogMaterialInfo(
                                        materials, uint( round( p7.y ) )
                                );
                                scatteringFog.fogVolume = true;
                                setFogSurfaceRecord( scatteringFog, prevSurf );
			} else if ( ! bdptLoadSurfaceRecord(
				prevMatId, p4, p7.zw, heroWavelength, prevSurf
			) ) {
				writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
			}
			if ( ! vitrumFiniteNonZeroVec3( p3.xyz ) ) {
				writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
				return;
			}
			vec3 directionToPredecessor =
				vitrumNormalizeVec3( p3.xyz, vec3( 0.0 ) );
			ScatterRecord rec = bsdfSample(
				directionToPredecessor, prevSurf, heroWavelength
			);
			scatterDir = rec.direction;
			scatterPdf = rec.pdf;
			scatterThroughput = rec.throughput;
			eventDelta = bdptSampledSurfaceEventIsDelta(
				prevSurf, directionToPredecessor, rec
			);
		}
                if (
			! ( scatterPdf > 0.0 ) ||
			! bdptFiniteFloat( scatterPdf ) ||
			! vitrumFiniteNonZeroVec3( scatterDir ) ||
			! bdptFiniteVec3( scatterThroughput ) ||
			any( lessThan( scatterThroughput, vec3( 0.0 ) ) )
		) {
                        writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
                }

		vec3 unitScatterDirection =
			vitrumNormalizeVec3( scatterDir, vec3( 0.0 ) );
                vec3 rayOrigin = stepRayOrigin(
			p0.xyz, vec3( 0.0 ), unitScatterDirection, 0.0
		);
                #if FEATURE_FOG
                if ( prevMatId < 0.0 && prevMatId != BDPT_LV_MEDIUM_MATID ) {
                        // Column 1 is the first segment out of the sampled emitter.
                        // Reconstruct its medium at the actual launch point and along
                        // the sampled launch ray. A world-origin probe is wrong for
                        // translated volumes and for an emitter on a volume boundary.
                        vec3 endpointLaunchDirection = unitScatterDirection;
                        vec3 endpointLaunchOrigin = stepRayOrigin(
				p0.xyz, vec3( 0.0 ), endpointLaunchDirection, 0.0
			);
                        if ( ! bvhBuildMediumStack(
                                endpointLaunchOrigin,
                                - endpointLaunchDirection,
                                materialIndexAttribute,
                                materials,
                                mediumStack,
                                fogMat
                        ) ) {
                                writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
                                return;
                        }
                }
                #endif
                if ( prevMatId >= 0.0 ) {
			SurfaceRecord offsetSurf;
			if ( bdptLoadSurfaceRecord(
				prevMatId, p4, p7.zw, heroWavelength, offsetSurf
			) ) {
				float side = dot( scatterDir, offsetSurf.faceNormal ) < 0.0 ? -1.0 : 1.0;
				rayOrigin = stepRayOrigin(
					p0.xyz,
					vec3( 0.0 ),
					offsetSurf.faceNormal * side,
					0.0
				);
			}
		}
		if ( ! bdptFiniteVec3( rayOrigin ) ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		Ray scatterRay;
		scatterRay.origin = rayOrigin;
		scatterRay.direction = unitScatterDirection;

		SurfaceHit hit;
		SurfaceRecord newSurf;
		bool foundVertex = false;
                bool mediumVertex = false;
                float scatteringMediumId = -1.0;
                uint newMatIdx = 0u;
                vec3 newPos = vec3( 0.0 );
                float segmentSurvival = 1.0;
                vec3 segmentRatioWeight = vec3( 1.0 );
                float forwardCollisionDensity = 1.0;
                float reverseCollisionDensity = 1.0;
                if ( prevMatId == BDPT_LV_MEDIUM_MATID && p7.y >= 0.0 ) {
                        reverseCollisionDensity = max(
                                readFogMaterialInfo(
                                        materials, uint( round( p7.y ) )
                                ).opacity,
                                0.0
                        );
                }
                for ( int traversal = 0; traversal < 32; traversal ++ ) {
			int hitType = traceScene( scatterRay, fogMat, hit );
			if ( hitType == NO_HIT ) break;
			if ( ! bdptFiniteFloat( hit.dist ) || hit.dist < 0.0 ) break;
                        if ( fogMat.fogVolume ) {
                                float sigmaT = max( fogMat.opacity, 0.0 );
                                float mediumDistance = max( hit.dist, 0.0 );
                                float survival = exp( - sigmaT * mediumDistance );
                                segmentSurvival *= survival;
                                segmentRatioWeight *= fogFreeFlightRatioWeight(
                                        materials,
                                        fogMat,
                                        mediumDistance,
                                        heroWavelength
                                );
                                if ( hitType == FOG_HIT ) forwardCollisionDensity = sigmaT;
			}
			newPos = scatterRay.origin + scatterRay.direction * hit.dist;
			if ( ! bdptFiniteVec3( newPos ) ) break;
                        if ( hitType == FOG_HIT ) {
                                setFogSurfaceRecord( fogMat, newSurf );
                                mediumVertex = true;
                                scatteringMediumId = float( fogMat.materialIndex );
                                foundVertex = true;
                                break;
			}

			newMatIdx = uTexelFetch1D( materialIndexAttribute, hit.faceIndices.w ).r;
			MaterialControl control;
                        readMaterialControl( materials, newMatIdx, control );
                        if ( control.fogVolume ) {
                                bool stackValid = hit.side == 1.0
                                        ? enterMedium(
                                                mediumStack, newMatIdx,
                                                materials, fogMat
                                        )
                                        : leaveMedium(
                                                mediumStack, newMatIdx,
                                                materials, fogMat
                                        );
                                if ( ! stackValid ) break;
                                scatterRay.origin = stepRayOrigin(
					scatterRay.origin, scatterRay.direction, - hit.faceNormal, hit.dist
				);
				continue;
			}
			int status = getSurfaceRecord(
				newMatIdx, hit, attributesArray, 0.0, vertexCol, heroWavelength, newSurf
			);
			if ( status == SKIP_SURFACE ) {
				scatterRay.origin = stepRayOrigin(
					scatterRay.origin, scatterRay.direction, - hit.faceNormal, hit.dist
				);
				continue;
			}
			foundVertex = true;
			break;
		}
                float segmentForwardDensity = segmentSurvival * forwardCollisionDensity;
                float segmentReverseDensity = segmentSurvival * reverseCollisionDensity;
                if (
			! foundVertex ||
			! ( segmentForwardDensity > 0.0 ) ||
			! bdptFiniteFloat( segmentForwardDensity ) ||
			! ( segmentReverseDensity >= 0.0 ) ||
			! bdptFiniteFloat( segmentReverseDensity ) ||
			! bdptFiniteVec3( segmentRatioWeight ) ||
			any( lessThan( segmentRatioWeight, vec3( 0.0 ) ) )
		) {
                        writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 ); return;
                }

		float edgeDistance = vitrumLengthVec3( newPos - p0.xyz );
		if ( ! ( edgeDistance > 0.0 ) ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
			if ( prevMatId == BDPT_LV_POINT_EMITTER_MATID || prevMatId == BDPT_LV_SPOT_EMITTER_MATID ) {
				scatterThroughput *= getDistanceAttenuation( edgeDistance, p4.y, p4.z );
			}
			vec3 incomingPathThroughput = p2.xyz;
			if ( prevMatId == BDPT_LV_ENVIRONMENT_EMITTER_MATID ) {
				// Match direct-environment NEE: the first real receiver's
				// per-material environment scale is a radiance factor, not a PDF.
				// Apply it to the already globally-scaled root throughput through
				// the canonical guard before the extension product is formed.
				incomingPathThroughput = finiteEquirectScaledColor(
					incomingPathThroughput, newSurf.envMapIntensity
				);
			}
                        vec3 newThroughput =
                                incomingPathThroughput * segmentRatioWeight *
                                scatterThroughput / scatterPdf;
                float edgePdf = scatterPdf * segmentForwardDensity;
                float reverseScatterPdf = scatterPdf;
                if ( prevMatId >= 0.0 && ! eventDelta ) {
                        SurfaceRecord reverseSurf;
                        if ( bdptLoadSurfaceRecord(
				prevMatId, p4, p7.zw, heroWavelength, reverseSurf
			) ) {
                                vec3 reverseColor;
                                reverseScatterPdf = bsdfResult(
                                        unitScatterDirection,
					vitrumNormalizeVec3( p3.xyz, vec3( 0.0 ) ),
					reverseSurf,
                                        heroWavelength, reverseColor
                                );
                        }
                }
		if (
			! ( reverseScatterPdf >= 0.0 ) ||
			! bdptFiniteFloat( reverseScatterPdf )
		) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		float predecessorReverseDensity = reverseScatterPdf * p2.w;
		if (
			! ( predecessorReverseDensity >= 0.0 ) ||
			! bdptFiniteFloat( predecessorReverseDensity )
		) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
                // row2.w on a newly created vertex carries the intermediate
                // reverse distance/collision density of its incoming edge.  One
                // column later the successor supplies the swapped BSDF at the
                // current vertex; their product belongs to the vertex BEFORE
                // current (PBRT RandomWalk: prev.pdfRev =
                // vertex.ConvertDensity(pdfRevAtVertex, prev)).  The main pass
                // therefore routes predecessor2 to vertexCol-2, while the delta
                // marker above still belongs to current at vertexCol-1.
		vec3 edgeToPredecessor = p0.xyz - newPos;
		if (
			! vitrumFiniteNonZeroVec3( edgeToPredecessor ) ||
			( ! mediumVertex && ! vitrumFiniteNonZeroVec3( newSurf.normal ) )
		) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		vec3 woToPrev =
			vitrumNormalizeVec3( edgeToPredecessor, vec3( 0.0 ) );
		vec3 newNormal = mediumVertex
			? vec3( 0.0 )
			: vitrumNormalizeVec3( newSurf.normal, vec3( 0.0 ) );
		v0 = vec4( newPos, BDPT_KIND_LIGHT );
		v1 = vec4( newNormal, edgePdf );
                v2 = vec4( newThroughput, segmentReverseDensity );
                v3 = vec4( woToPrev, mediumVertex ? BDPT_LV_MEDIUM_MATID : float( newMatIdx ) );
                v4 = mediumVertex ? vec4( 0.0 ) : bdptSurfacePayload( hit );
	                bdptPackMediumStack(
	                        mediumStack, scatteringMediumId, v5, v6, v7
	                );
			if ( ! mediumVertex ) {
				v7.zw = bdptPackFaceIndexWords( hit.faceIndices.w );
			}
		if ( ! bdptStoredVertexRowsValid( v0, v1, v2, v3, v4, v5, v6, v7 ) ) {
			writeBdptInvalidVertex( v0, v1, v2, v3, v4, v5, v6, v7 );
			return;
		}
		// Publish predecessor patches only after the new record is proven valid,
		// so a failed extension cannot corrupt an otherwise usable prefix.
                predecessor0.w =
			eventDelta ? BDPT_KIND_DELTA : BDPT_KIND_LIGHT;
                predecessor2.w = predecessorReverseDensity;
	}

`;
