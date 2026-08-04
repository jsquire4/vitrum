export const inside_fog_volume_function = /* glsl */`

bool opticalLexicographicLess( vec3 a, vec3 b ) {
	if ( a.x != b.x ) return a.x < b.x;
	if ( a.y != b.y ) return a.y < b.y;
	return a.z < b.z;
}

// Canonical endpoint ordering makes the projected edge value exactly
// antisymmetric when adjacent triangles request the same shared edge in the
// opposite direction. No triangle-dependent coordinate scale participates.
float opticalProjectedEdgeFunction( vec2 a, vec2 b ) {
	if ( all( equal( a, b ) ) ) return 0.0;
	bool ordered =
		a.x < b.x || ( a.x == b.x && a.y < b.y );
	vec2 first = ordered ? a : b;
	vec2 second = ordered ? b : a;
	float canonical =
		first.x * second.y - first.y * second.x;
	return ordered ? canonical : - canonical;
}

float opticalRayParameterAtDelta(
	vec3 rayDirection, vec3 delta, out bool valid
) {
	float deltaScale = max(
		abs( delta.x ), max( abs( delta.y ), abs( delta.z ) )
	);
	float directionScale = max(
		abs( rayDirection.x ),
		max( abs( rayDirection.y ), abs( rayDirection.z ) )
	);
	if (
		! ( directionScale > 0.0 ) || isinf( directionScale ) ||
		isnan( deltaScale ) || isinf( deltaScale )
	) {
		valid = false;
		return -1.0;
	}
	if ( ! ( deltaScale > 0.0 ) ) {
		valid = true;
		return 0.0;
	}
	vec3 scaledDirection = rayDirection / directionScale;
	float denominator = dot( scaledDirection, scaledDirection );
	float result =
		( dot( delta / deltaScale, scaledDirection ) / denominator ) *
		( deltaScale / directionScale );
	valid = ! isnan( result ) && ! isinf( result );
	return result;
}

// Evaluate t from a shared edge using only its canonically ordered endpoints
// and the ray. Adjacent faces may have arbitrarily different third vertices
// and still produce a bit-identical group key.
float opticalSharedEdgeDistance(
	vec3 rawP0, vec3 rawP1,
	vec3 worldP0, vec3 worldP1,
	vec2 projectedP0, vec2 projectedP1,
	vec3 rayDirection,
	out vec3 canonicalPoint,
	out bool valid
) {
	canonicalPoint = vec3( 0.0 );
	vec3 p0 = rawP0;
	vec3 p1 = rawP1;
	vec3 representedP0 = worldP0;
	vec3 representedP1 = worldP1;
	vec2 projected0 = projectedP0;
	vec2 projected1 = projectedP1;
	if ( opticalLexicographicLess( p1, p0 ) ) {
		vec3 swapPoint = p0;
		p0 = p1;
		p1 = swapPoint;
		vec3 swapRepresentedPoint = representedP0;
		representedP0 = representedP1;
		representedP1 = swapRepresentedPoint;
		vec2 swapProjected = projected0;
		projected0 = projected1;
		projected1 = swapProjected;
	}
	vec2 projectedDelta = projected1 - projected0;
	float edgeParameter;
	if ( abs( projectedDelta.x ) >= abs( projectedDelta.y ) ) {
		if ( projectedDelta.x == 0.0 ) {
			valid = false;
			return -1.0;
		}
		edgeParameter = - projected0.x / projectedDelta.x;
	} else {
		if ( projectedDelta.y == 0.0 ) {
			valid = false;
			return -1.0;
		}
		edgeParameter = - projected0.y / projectedDelta.y;
	}
	if ( isnan( edgeParameter ) || isinf( edgeParameter ) ) {
		valid = false;
		return -1.0;
	}
	canonicalPoint = representedP0 +
		edgeParameter * ( representedP1 - representedP0 );
	return opticalRayParameterAtDelta(
		rayDirection,
		p0 + edgeParameter * ( p1 - p0 ),
		valid
	);
}

// Inclusive watertight projected-edge solve. Edge and vertex candidates are
// intentionally all retained: the range traversal groups exact f32-equal t
// values by validated boundary identity and resolves the complete side set.
bool intersectsOpticalBoundaryTriangle(
	vec3 rayOrigin, vec3 rayDirection, vec3 a, vec3 b, vec3 c,
	out vec3 barycoord, out vec3 norm, out vec3 point,
	out float dist, out float side,
	out bool invalidFeature
) {
	invalidFeature = false;
	point = vec3( 0.0 );
	vec3 rawA = a - rayOrigin;
	vec3 rawB = b - rayOrigin;
	vec3 rawC = c - rayOrigin;
	if (
		any( isnan( rawA ) ) || any( isinf( rawA ) ) ||
		any( isnan( rawB ) ) || any( isinf( rawB ) ) ||
		any( isnan( rawC ) ) || any( isinf( rawC ) ) ||
		any( isnan( rayDirection ) ) || any( isinf( rayDirection ) )
	) return false;

	float directionScale = max(
		abs( rayDirection.x ),
		max( abs( rayDirection.y ), abs( rayDirection.z ) )
	);
	if ( ! ( directionScale > 0.0 ) || isinf( directionScale ) ) return false;
	vec3 scaledDirection = rayDirection / directionScale;
	vec3 absDirection = abs( scaledDirection );
	int kz = absDirection.x > absDirection.y
		? ( absDirection.x > absDirection.z ? 0 : 2 )
		: ( absDirection.y > absDirection.z ? 1 : 2 );
	if ( ! ( absDirection[ kz ] > 0.0 ) ) return false;
	int kx = kz == 2 ? 0 : kz + 1;
	int ky = kx == 2 ? 0 : kx + 1;
	if ( scaledDirection[ kz ] < 0.0 ) {
		int swapAxis = kx;
		kx = ky;
		ky = swapAxis;
	}
	float shearX = scaledDirection[ kx ] / scaledDirection[ kz ];
	float shearY = scaledDirection[ ky ] / scaledDirection[ kz ];
	float shearZ = 1.0 / scaledDirection[ kz ];
	vec2 projectedA = vec2(
		rawA[ kx ] - shearX * rawA[ kz ],
		rawA[ ky ] - shearY * rawA[ kz ]
	);
	vec2 projectedB = vec2(
		rawB[ kx ] - shearX * rawB[ kz ],
		rawB[ ky ] - shearY * rawB[ kz ]
	);
	vec2 projectedC = vec2(
		rawC[ kx ] - shearX * rawC[ kz ],
		rawC[ ky ] - shearY * rawC[ kz ]
	);
	if (
		any( isnan( projectedA ) ) || any( isinf( projectedA ) ) ||
		any( isnan( projectedB ) ) || any( isinf( projectedB ) ) ||
		any( isnan( projectedC ) ) || any( isinf( projectedC ) )
	) return false;
	float u = opticalProjectedEdgeFunction( projectedB, projectedC );
	float v = opticalProjectedEdgeFunction( projectedC, projectedA );
	float w = opticalProjectedEdgeFunction( projectedA, projectedB );
	float determinant = u + v + w;
	if ( determinant == 0.0 || isnan( determinant ) || isinf( determinant ) ) {
		return false;
	}
	if ( determinant < 0.0 ) {
		u = - u;
		v = - v;
		w = - w;
		determinant = - determinant;
	}
	if ( u < 0.0 || v < 0.0 || w < 0.0 ) return false;

	float az = rawA[ kz ] * shearZ;
	float bz = rawB[ kz ] * shearZ;
	float cz = rawC[ kz ] * shearZ;
	float scaledT = u * az + v * bz + w * cz;
	if ( scaledT < 0.0 || isnan( scaledT ) || isinf( scaledT ) ) return false;
	float rayT = ( scaledT / determinant ) / directionScale;

	// Canonicalize feature hits. Two zero edge functions identify a shared
	// vertex; one identifies a shared edge. Interior hits keep the ordinary
	// scaled solve above.
	bool zeroU = u == 0.0;
	bool zeroV = v == 0.0;
	bool zeroW = w == 0.0;
	bool featureDistanceValid = true;
	vec3 canonicalPoint = vec3( 0.0 );
	if ( zeroV && zeroW ) {
		canonicalPoint = a;
		rayT = opticalRayParameterAtDelta(
			rayDirection, rawA, featureDistanceValid
		);
	} else if ( zeroW && zeroU ) {
		canonicalPoint = b;
		rayT = opticalRayParameterAtDelta(
			rayDirection, rawB, featureDistanceValid
		);
	} else if ( zeroU && zeroV ) {
		canonicalPoint = c;
		rayT = opticalRayParameterAtDelta(
			rayDirection, rawC, featureDistanceValid
		);
	} else if ( zeroU || zeroV || zeroW ) {
		if ( zeroU ) {
			rayT = opticalSharedEdgeDistance(
				rawB, rawC, b, c, projectedB, projectedC,
				rayDirection, canonicalPoint, featureDistanceValid
			);
		} else if ( zeroV ) {
			rayT = opticalSharedEdgeDistance(
				rawC, rawA, c, a, projectedC, projectedA,
				rayDirection, canonicalPoint, featureDistanceValid
			);
		} else {
			rayT = opticalSharedEdgeDistance(
				rawA, rawB, a, b, projectedA, projectedB,
				rayDirection, canonicalPoint, featureDistanceValid
			);
		}
	}
	if ( ! featureDistanceValid ) {
		invalidFeature = true;
		return false;
	}
	if ( isnan( rayT ) || isinf( rayT ) || rayT < 0.0 ) return false;

	vec3 edge0 = b - a;
	vec3 edge1 = c - a;
	float edgeScale = max(
		max( abs( edge0.x ), max( abs( edge0.y ), abs( edge0.z ) ) ),
		max( abs( edge1.x ), max( abs( edge1.y ), abs( edge1.z ) ) )
	);
	if ( ! ( edgeScale > 0.0 ) || isinf( edgeScale ) ) return false;
	vec3 scaledNormal = cross( edge0 / edgeScale, edge1 / edgeScale );
	float normalScale = max(
		abs( scaledNormal.x ),
		max( abs( scaledNormal.y ), abs( scaledNormal.z ) )
	);
	if ( ! ( normalScale > 0.0 ) || isinf( normalScale ) ) return false;
	vec3 normalDirection = scaledNormal / normalScale;
	vec3 unitNormal = normalDirection / length( normalDirection );
	float sideDeterminant = - dot( scaledDirection, unitNormal );
	if ( sideDeterminant == 0.0 || isnan( sideDeterminant ) ) return false;

	side = sideDeterminant > 0.0 ? 1.0 : -1.0;
	if ( side == 0.0 ) return false;
	norm = side * unitNormal;
	barycoord = vec3( u, v, w ) / determinant;
	if ( ! zeroU && ! zeroV && ! zeroW ) {
		// Anchor the interior point at one represented vertex. In particular, a
		// coordinate that is constant over the triangle has zero deltas and cannot
		// drift onto a one-ULP adjacent layer through origin + direction * t.
		canonicalPoint = a +
			barycoord.y * ( b - a ) +
			barycoord.z * ( c - a );
	}
	if ( any( isnan( canonicalPoint ) ) || any( isinf( canonicalPoint ) ) ) {
		invalidFeature = true;
		return false;
	}
	point = canonicalPoint;
	dist = rayT;
	return true;
}

bool opticalBitwiseEqualVec3( vec3 a, vec3 b ) {
	uvec3 canonicalA = uvec3(
		a.x == 0.0 ? 0u : floatBitsToUint( a.x ),
		a.y == 0.0 ? 0u : floatBitsToUint( a.y ),
		a.z == 0.0 ? 0u : floatBitsToUint( a.z )
	);
	uvec3 canonicalB = uvec3(
		b.x == 0.0 ? 0u : floatBitsToUint( b.x ),
		b.y == 0.0 ? 0u : floatBitsToUint( b.y ),
		b.z == 0.0 ? 0u : floatBitsToUint( b.z )
	);
	return all( equal( canonicalA, canonicalB ) );
}

bool opticalTriangleContainsVertex(
	vec3 a, vec3 b, vec3 c, vec3 vertex
) {
	return
		opticalBitwiseEqualVec3( a, vertex ) ||
		opticalBitwiseEqualVec3( b, vertex ) ||
		opticalBitwiseEqualVec3( c, vertex );
}

bool opticalTriangleIsIncidentToSourceFeature(
	uint candidateBoundaryId,
	uint candidatePrimitiveInstanceId,
	vec3 a, vec3 b, vec3 c,
	uint sourceFeatureKind,
	uint sourceBoundaryId,
	uint sourcePrimitiveInstanceId,
	vec3 sourceFeatureA,
	vec3 sourceFeatureB
) {
	if (
		sourceFeatureKind < 2u ||
		candidateBoundaryId != sourceBoundaryId ||
		candidatePrimitiveInstanceId == 0u ||
		candidatePrimitiveInstanceId != sourcePrimitiveInstanceId
	) return false;
	bool containsA = opticalTriangleContainsVertex(
		a, b, c, sourceFeatureA
	);
	if ( sourceFeatureKind == 3u ) return containsA;
	return containsA && opticalTriangleContainsVertex(
		a, b, c, sourceFeatureB
	);
}

void setExactRaySourceFeature(
	inout Ray ray,
	vec3 exactOrigin,
	uint sourceFaceIndex,
	vec3 sourceBarycoord
) {
	setExactRayRange( ray, exactOrigin, sourceFaceIndex );
	uvec4 identity = uTexelFetch1D(
		materialIndexAttribute, sourceFaceIndex
	);
	ray.sourceBoundaryId = identity.g;
	ray.sourcePrimitiveInstanceId = identity.b;
	uvec3 indices = uTexelFetch1D( bvh.index, sourceFaceIndex ).xyz;
	vec3 a = texelFetch1D( bvh.position, indices.x ).rgb;
	vec3 b = texelFetch1D( bvh.position, indices.y ).rgb;
	vec3 c = texelFetch1D( bvh.position, indices.z ).rgb;
	bool zeroU = sourceBarycoord.x == 0.0;
	bool zeroV = sourceBarycoord.y == 0.0;
	bool zeroW = sourceBarycoord.z == 0.0;
	if ( zeroV && zeroW ) {
		ray.sourceFeatureKind = 3u;
		ray.sourceFeatureA = a;
	} else if ( zeroW && zeroU ) {
		ray.sourceFeatureKind = 3u;
		ray.sourceFeatureA = b;
	} else if ( zeroU && zeroV ) {
		ray.sourceFeatureKind = 3u;
		ray.sourceFeatureA = c;
	} else if ( zeroU || zeroV || zeroW ) {
		ray.sourceFeatureKind = 2u;
		vec3 edgeA = zeroU ? b : zeroV ? c : a;
		vec3 edgeB = zeroU ? c : zeroV ? a : b;
		if ( opticalLexicographicLess( edgeB, edgeA ) ) {
			vec3 swapPoint = edgeA;
			edgeA = edgeB;
			edgeB = swapPoint;
		}
		ray.sourceFeatureA = edgeA;
		ray.sourceFeatureB = edgeB;
	}
}

// Re-solve the accepted source triangle with the canonical edge equations so
// the outgoing token uses exact zero flags even when the ordinary shading BVH
// traversal supplied a noncanonical barycentric payload.
bool canonicalizeSelectedSurfaceHit(
	Ray ray, inout SurfaceHit hit
) {
	uvec3 indices = uTexelFetch1D( bvh.index, hit.faceIndices.w ).xyz;
	vec3 a = texelFetch1D( bvh.position, indices.x ).rgb;
	vec3 b = texelFetch1D( bvh.position, indices.y ).rgb;
	vec3 c = texelFetch1D( bvh.position, indices.z ).rgb;
	vec3 canonicalBarycoord;
	vec3 canonicalNormal;
	vec3 canonicalPoint;
	float canonicalDistance;
	float canonicalSide;
	bool invalidFeature;
	bool accepted = intersectsOpticalBoundaryTriangle(
		ray.origin, ray.direction, a, b, c,
		canonicalBarycoord, canonicalNormal, canonicalPoint,
		canonicalDistance, canonicalSide, invalidFeature
	);
	if ( ! accepted || invalidFeature ) return false;
	hit.faceIndices = uvec4( indices, hit.faceIndices.w );
	hit.barycoord = canonicalBarycoord;
	hit.faceNormal = canonicalNormal;
	hit.point = canonicalPoint;
	hit.side = canonicalSide;
	hit.dist = canonicalDistance;
	return true;
}

bool setExactRayRangeFromSurfaceHit(
	inout Ray ray, SurfaceHit sourceHit
) {
	if ( ! canonicalizeSelectedSurfaceHit( ray, sourceHit ) ) return false;
	vec3 exactOrigin = sourceHit.point;
	if ( any( isnan( exactOrigin ) ) || any( isinf( exactOrigin ) ) ) {
		return false;
	}
	setExactRaySourceFeature(
		ray, exactOrigin, sourceHit.faceIndices.w, sourceHit.barycoord
	);
	return true;
}

bool intersectWatertightRangeTriangles(
	sampler2D positionAttr,
	usampler2D indexAttr,
	usampler2D materialIndexAttribute,
	sampler2D materials,
	uint offset,
	uint count,
	vec3 rayOrigin,
	vec3 rayDirection,
	float minimumDistanceExclusive,
	uint ignoredFaceIndex,
	uint sourceFeatureKind,
	uint sourceBoundaryId,
	uint sourcePrimitiveInstanceId,
	vec3 sourceFeatureA,
	vec3 sourceFeatureB,
	bool opticalOnly,
	bool allowDistinctOpaqueTie,
	inout float closestDistance,
	inout uvec4 faceIndices,
	inout vec3 faceNormal,
	inout vec3 barycoord,
	inout vec3 point,
	inout float side,
	inout float dist,
	inout uint componentId,
	inout uint primitiveInstanceId,
	inout uint materialIndex,
	inout int frontCount,
	inout int backCount,
	inout bool invalidIdentity
) {
	bool found = false;
	for ( uint i = offset, end = offset + count; i < end; i ++ ) {
		uvec4 identity = uTexelFetch1D( materialIndexAttribute, i );
		uint localMaterialIndex = identity.r;
		uint localComponentId = identity.g;
		uint localPrimitiveInstanceId = identity.b;
		MaterialControl control;
		readMaterialControl( materials, localMaterialIndex, control );
		if ( control.opticalVolume != ( localComponentId != 0u ) ) {
			invalidIdentity = true;
			continue;
		}
		if ( opticalOnly && ! control.opticalVolume ) continue;

		uvec3 indices = uTexelFetch1D( indexAttr, i ).xyz;
		vec3 a = texelFetch1D( positionAttr, indices.x ).rgb;
		vec3 b = texelFetch1D( positionAttr, indices.y ).rgb;
		vec3 c = texelFetch1D( positionAttr, indices.z ).rgb;
		if (
			i == ignoredFaceIndex ||
			opticalTriangleIsIncidentToSourceFeature(
				identity.g, identity.b, a, b, c,
				sourceFeatureKind, sourceBoundaryId,
				sourcePrimitiveInstanceId,
				sourceFeatureA, sourceFeatureB
			)
		) continue;
		vec3 localBarycoord;
		vec3 localNormal;
		vec3 localPoint;
		float localDist;
		float localSide;
		bool invalidFeature;
		bool intersects = intersectsOpticalBoundaryTriangle(
			rayOrigin, rayDirection, a, b, c,
			localBarycoord, localNormal, localPoint, localDist, localSide,
			invalidFeature
		);
		if ( invalidFeature ) {
			invalidIdentity = true;
			continue;
		}
		if (
			! intersects ||
			! ( localDist > minimumDistanceExclusive ) ||
			localDist > closestDistance
		) continue;

		if ( localDist < closestDistance ) {
			found = true;
			closestDistance = localDist;
			faceIndices = uvec4( indices, i );
			faceNormal = localNormal;
			barycoord = localBarycoord;
			point = localPoint;
			side = localSide;
			dist = localDist;
			componentId = localComponentId;
			primitiveInstanceId = localPrimitiveInstanceId;
			materialIndex = localMaterialIndex;
			frontCount = localSide > 0.0 ? 1 : 0;
			backCount = localSide < 0.0 ? 1 : 0;
		} else if ( localDist == closestDistance ) {
			found = true;
			bool distinctIdentity =
				localComponentId != componentId ||
				localPrimitiveInstanceId != primitiveInstanceId;
			bool incompatibleBulkMaterial =
				localComponentId != 0u && localMaterialIndex != materialIndex;
			bool invalidDistinctIdentity = distinctIdentity;
			if ( distinctIdentity && allowDistinctOpaqueTie ) {
				MaterialControl selectedControl;
				readMaterialControl( materials, materialIndex, selectedControl );
				bool selectedTransmissive =
					selectedControl.opticalVolume || selectedControl.thinFilm;
				bool localTransmissive =
					control.opticalVolume || control.thinFilm;
				invalidDistinctIdentity =
					selectedTransmissive || localTransmissive;
			}
			if ( invalidDistinctIdentity || incompatibleBulkMaterial ) {
				invalidIdentity = true;
				continue;
			}
			if ( localComponentId != 0u ) {
				if ( localSide > 0.0 ) {
					if ( frontCount == 2147483647 ) invalidIdentity = true;
					else frontCount ++;
				} else {
					if ( backCount == 2147483647 ) invalidIdentity = true;
					else backCount ++;
				}
			}
			if ( i < faceIndices.w ) {
				faceIndices = uvec4( indices, i );
				faceNormal = localNormal;
				barycoord = localBarycoord;
				point = localPoint;
				side = localSide;
				materialIndex = localMaterialIndex;
			}
		}
	}
	return found;
}

#define\
	bvhIntersectWatertightRangeFirstHit(\
		bvh, rayOrigin, rayDirection, minimumDistanceExclusive, ignoredFaceIndex,\
		sourceFeatureKind, sourceBoundaryId, sourcePrimitiveInstanceId,\
		sourceFeatureA, sourceFeatureB,\
		opticalOnly, allowDistinctOpaqueTie, materialIndexAttribute, materials, faceIndices, faceNormal,\
		barycoord, point, side, dist, componentId, primitiveInstanceId, materialIndex, frontCount, backCount,\
		invalidIdentity\
	)\
	_bvhIntersectWatertightRangeFirstHit(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		rayOrigin, rayDirection, minimumDistanceExclusive, ignoredFaceIndex,\
		sourceFeatureKind, sourceBoundaryId, sourcePrimitiveInstanceId,\
		sourceFeatureA, sourceFeatureB,\
		opticalOnly, allowDistinctOpaqueTie, materialIndexAttribute, materials, faceIndices, faceNormal,\
		barycoord, point, side, dist, componentId, primitiveInstanceId, materialIndex, frontCount, backCount,\
		invalidIdentity\
	)

bool _bvhIntersectWatertightRangeFirstHit(
	sampler2D bvhPosition,
	usampler2D bvhIndex,
	sampler2D bvhBounds,
	usampler2D bvhContents,
	vec3 rayOrigin,
	vec3 rayDirection,
	float minimumDistanceExclusive,
	uint ignoredFaceIndex,
	uint sourceFeatureKind,
	uint sourceBoundaryId,
	uint sourcePrimitiveInstanceId,
	vec3 sourceFeatureA,
	vec3 sourceFeatureB,
	bool opticalOnly,
	bool allowDistinctOpaqueTie,
	usampler2D materialIndexAttribute,
	sampler2D materials,
	inout uvec4 faceIndices,
	inout vec3 faceNormal,
	inout vec3 barycoord,
	inout vec3 point,
	inout float side,
	inout float dist,
	out uint componentId,
	out uint primitiveInstanceId,
	out uint materialIndex,
	out int frontCount,
	out int backCount,
	out bool invalidIdentity
) {
	int pointer = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;
	float triangleDistance = INFINITY;
	bool found = false;
	componentId = 0u;
	primitiveInstanceId = 0u;
	materialIndex = 0u;
	frontCount = 0;
	backCount = 0;
	invalidIdentity = false;
	while ( pointer > -1 && pointer < BVH_STACK_DEPTH ) {
		uint nodeIndex = stack[ pointer ];
		pointer --;
		float boundsDistance;
		if (
			! intersectsBVHNodeBounds(
				rayOrigin, rayDirection, bvhBounds, nodeIndex, boundsDistance
			) ||
			boundsDistance > triangleDistance
		) continue;

		uvec2 info = uTexelFetch1D( bvhContents, nodeIndex ).xy;
		bool leaf = bool( info.x & 0xffff0000u );
		if ( leaf ) {
			found = intersectWatertightRangeTriangles(
				bvhPosition, bvhIndex,
				materialIndexAttribute, materials,
				info.y, info.x & 0x0000ffffu,
				rayOrigin, rayDirection, minimumDistanceExclusive,
				ignoredFaceIndex,
				sourceFeatureKind, sourceBoundaryId,
				sourcePrimitiveInstanceId,
				sourceFeatureA, sourceFeatureB,
				opticalOnly, allowDistinctOpaqueTie,
				triangleDistance, faceIndices, faceNormal,
				barycoord, point, side, dist, componentId, primitiveInstanceId, materialIndex,
				frontCount, backCount, invalidIdentity
			) || found;
		} else {
			if ( pointer + 2 >= BVH_STACK_DEPTH ) {
				invalidIdentity = true;
				return false;
			}
			uint left = nodeIndex + 1u;
			uint splitAxis = info.x & 0x0000ffffu;
			uint right = nodeIndex + info.y;
			bool leftFirst = rayDirection[ splitAxis ] >= 0.0;
			uint first = leftFirst ? left : right;
			uint second = leftFirst ? right : left;
			pointer ++;
			stack[ pointer ] = second;
			pointer ++;
			stack[ pointer ] = first;
		}
	}
	if ( found && componentId != 0u ) {
		if ( frontCount > 0 && backCount > 0 ) {
			if ( frontCount == backCount ) {
				// Balanced opposite signs are a silhouette tangent: the ray does
				// not change side and the caller advances only its exact min-t.
				side = 0.0;
			} else {
				invalidIdentity = true;
			}
		} else if ( frontCount > 0 ) {
			side = 1.0;
		} else if ( backCount > 0 ) {
			side = -1.0;
		} else {
			invalidIdentity = true;
		}
	}
	return found;
}

// The first segment of an advanced optical path must use the same inclusive
// watertight solve as exact continuations. At an exact equal-t group, triangles
// belonging to one represented primitive are one boundary candidate. Distinct
// represented transmissive candidates are ambiguous and fail closed; distinct
// opaque candidates retain deterministic lowest-face selection.
bool bvhIntersectCanonicalInitialFirstHit(
	Ray ray, inout SurfaceHit surfaceHit, out bool invalidRange
) {
	invalidRange = false;
	float minimumDistanceExclusive = 0.0;
	uint queryLimit = uSceneTriangleCount + 1u;
	if ( queryLimit == 0u ) {
		invalidRange = true;
		return false;
	}
	for ( uint queryIndex = 0u; queryIndex < queryLimit; queryIndex ++ ) {
		uint componentId;
		uint primitiveInstanceId;
		uint materialIndex;
		int frontCount;
		int backCount;
		bool invalidIdentity;
		bool found = bvhIntersectWatertightRangeFirstHit(
			bvh, ray.origin, ray.direction, minimumDistanceExclusive,
			0xffffffffu,
			0u, 0u, 0u, vec3( 0.0 ), vec3( 0.0 ),
			false, true,
			materialIndexAttribute, materials,
			surfaceHit.faceIndices, surfaceHit.faceNormal,
			surfaceHit.barycoord, surfaceHit.point,
			surfaceHit.side, surfaceHit.dist,
			componentId, primitiveInstanceId, materialIndex, frontCount, backCount,
			invalidIdentity
		);
		if ( invalidIdentity ) {
			invalidRange = true;
			return false;
		}
		if ( ! found ) return false;
		if (
			isnan( surfaceHit.dist ) || isinf( surfaceHit.dist ) ||
			! ( surfaceHit.dist > minimumDistanceExclusive )
		) {
			invalidRange = true;
			return false;
		}
		if ( componentId != 0u && surfaceHit.side == 0.0 ) {
			minimumDistanceExclusive = surfaceHit.dist;
			continue;
		}
		return true;
	}
	invalidRange = true;
	return false;
}

// Canonical continuation traversal used after accepted bulk or compound-sheet
// transmission. It keeps the geometric anchor fixed, excludes only the source
// face, advances by a strict exact-t range, and skips balanced optical tangents.
bool bvhIntersectExactRangeFirstHit(
	Ray ray, inout SurfaceHit surfaceHit, out bool invalidRange
) {
	invalidRange = false;
	float minimumDistanceExclusive = ray.minimumDistanceExclusive;
	uint queryLimit = uSceneTriangleCount + 1u;
	if ( queryLimit == 0u ) {
		invalidRange = true;
		return false;
	}
	for ( uint queryIndex = 0u; queryIndex < queryLimit; queryIndex ++ ) {
		uint componentId;
		uint primitiveInstanceId;
		uint materialIndex;
		int frontCount;
		int backCount;
		bool invalidIdentity;
		bool found = bvhIntersectWatertightRangeFirstHit(
			bvh, ray.origin, ray.direction, minimumDistanceExclusive,
			ray.ignoredFaceIndex,
			ray.sourceFeatureKind, ray.sourceBoundaryId,
			ray.sourcePrimitiveInstanceId,
			ray.sourceFeatureA, ray.sourceFeatureB,
			false, false,
			materialIndexAttribute, materials,
			surfaceHit.faceIndices, surfaceHit.faceNormal,
			surfaceHit.barycoord, surfaceHit.point,
			surfaceHit.side, surfaceHit.dist,
			componentId, primitiveInstanceId, materialIndex, frontCount, backCount,
			invalidIdentity
		);
		if ( invalidIdentity ) {
			invalidRange = true;
			return false;
		}
		if ( ! found ) return false;
		if (
			isnan( surfaceHit.dist ) || isinf( surfaceHit.dist ) ||
			! ( surfaceHit.dist > minimumDistanceExclusive )
		) {
			invalidRange = true;
			return false;
		}
		if ( componentId != 0u && surfaceHit.side == 0.0 ) {
			minimumDistanceExclusive = surfaceHit.dist;
			continue;
		}
		return true;
	}
	invalidRange = true;
	return false;
}

// Classify the medium occupied by the first transport segment using a scan
// from its exact launch point toward infinity. Front faces push a temporary
// LIFO and matching backs pop it. An unmatched back means the launch started
// inside that component; such components arrive inner-to-outer and are reversed
// into the live outer-to-inner MediumStack. No synthetic outside point or
// positional epsilon participates in classification.
#if ADVANCED_OPTICAL_TRANSPORT
bool bvhBuildMediumStack(
	vec3 rayOrigin, vec3 rayDirection,
	usampler2D materialIndexAttribute, sampler2D materials,
	sampler2DArray attributesArray,
	out MediumStack stack,
	inout FogMaterial material
) {
	initMediumStack( stack );
	initFogMaterial( material );
	if ( uSceneTriangleCount == 0u ) return true;
	if (
		any( isnan( rayOrigin ) ) || any( isinf( rayOrigin ) ) ||
		any( isnan( rayDirection ) ) || any( isinf( rayDirection ) ) ||
		! vitrumFiniteNonZeroVec3( rayDirection )
	) return false;
	vec3 walkDirection = vitrumNormalizeVec3( rayDirection, vec3( 0.0 ) );
	if ( ! vitrumFiniteNonZeroVec3( walkDirection ) ) return false;

	uint pendingComponentIds[ MEDIUM_STACK_CAPACITY ];
	uint pendingMaterialIds[ MEDIUM_STACK_CAPACITY ];
	int pendingCount = 0;
	uint containingMaterialIds[ MEDIUM_STACK_CAPACITY ];
	uint containingComponentIds[ MEDIUM_STACK_CAPACITY ];
	uint containingHasThickness[ MEDIUM_STACK_CAPACITY ];
	float containingThickness[ MEDIUM_STACK_CAPACITY ];
	int containingCount = 0;
	for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
		pendingComponentIds[ i ] = 0u;
		pendingMaterialIds[ i ] = 0u;
		containingMaterialIds[ i ] = 0u;
		containingComponentIds[ i ] = 0u;
		containingHasThickness[ i ] = 0u;
		containingThickness[ i ] = 0.0;
	}

	// t=0 is exclusive. A launch exactly on a boundary is therefore classified
	// on the side occupied immediately by the supplied transport direction.
	float minimumDistanceExclusive = 0.0;
	uint queryLimit = uSceneTriangleCount + 1u;
	if ( queryLimit == 0u ) return false;
	for ( uint queryIndex = 0u; queryIndex < queryLimit; queryIndex ++ ) {
		SurfaceHit boundaryHit;
		boundaryHit.faceIndices = uvec4( 0u );
		boundaryHit.faceNormal = vec3( 0.0, 0.0, 1.0 );
		boundaryHit.barycoord = vec3( 0.0 );
		boundaryHit.point = vec3( 0.0 );
		boundaryHit.side = 0.0;
		boundaryHit.dist = INFINITY;
		uint boundaryComponentId;
		uint boundaryPrimitiveInstanceId;
		uint boundaryMaterialIndex;
		int frontCount;
		int backCount;
		bool invalidBoundaryIdentity;
		bool hit = bvhIntersectWatertightRangeFirstHit(
			bvh, rayOrigin, walkDirection, minimumDistanceExclusive,
			0xffffffffu,
			0u, 0u, 0u, vec3( 0.0 ), vec3( 0.0 ),
			true, false,
			materialIndexAttribute, materials,
			boundaryHit.faceIndices, boundaryHit.faceNormal,
			boundaryHit.barycoord, boundaryHit.point,
			boundaryHit.side, boundaryHit.dist,
			boundaryComponentId, boundaryPrimitiveInstanceId, boundaryMaterialIndex,
			frontCount, backCount, invalidBoundaryIdentity
		);
		if ( invalidBoundaryIdentity ) return false;
		if ( ! hit ) {
			if ( pendingCount != 0 ) return false;
			for ( int outputIndex = 0; outputIndex < MEDIUM_STACK_CAPACITY; outputIndex ++ ) {
				if ( outputIndex >= containingCount ) break;
				int sourceIndex = containingCount - 1 - outputIndex;
				if ( ! enterMedium(
					stack,
					containingComponentIds[ sourceIndex ],
					containingMaterialIds[ sourceIndex ],
					containingHasThickness[ sourceIndex ] != 0u,
					containingThickness[ sourceIndex ],
					materials,
					material
				) ) return false;
			}
			return true;
		}
		if (
			isnan( boundaryHit.dist ) || isinf( boundaryHit.dist ) ||
			! ( boundaryHit.dist > minimumDistanceExclusive ) ||
			boundaryComponentId == 0u
		) return false;
		if ( queryIndex >= uSceneTriangleCount ) return false;

		// Balanced signs are one exact silhouette tangent and do not mutate
		// either stack. The exact t is still consumed monotonically.
		if ( boundaryHit.side == 0.0 ) {
			minimumDistanceExclusive = boundaryHit.dist;
			continue;
		}
		if ( boundaryHit.side == 1.0 ) {
			if ( pendingCount >= MEDIUM_STACK_CAPACITY ) return false;
			pendingComponentIds[ pendingCount ] = boundaryComponentId;
			pendingMaterialIds[ pendingCount ] = boundaryMaterialIndex;
			pendingCount ++;
		} else if ( boundaryHit.side == -1.0 ) {
			if ( pendingCount > 0 ) {
				int top = pendingCount - 1;
				if (
					pendingComponentIds[ top ] != boundaryComponentId ||
					pendingMaterialIds[ top ] != boundaryMaterialIndex
				) return false;
				pendingComponentIds[ top ] = 0u;
				pendingMaterialIds[ top ] = 0u;
				pendingCount = top;
			} else {
				if ( containingCount >= MEDIUM_STACK_CAPACITY ) return false;
				Material boundaryMaterial;
				readMaterialInfo(
					materials, boundaryMaterialIndex, boundaryMaterial
				);
				bool boundaryHasThickness;
				float boundaryThickness = evaluateAttenuationThickness(
					boundaryMaterialIndex,
					boundaryMaterial,
					boundaryHit,
					attributesArray,
					0,
					boundaryHasThickness
				);
				containingMaterialIds[ containingCount ] =
					boundaryMaterialIndex;
				containingComponentIds[ containingCount ] =
					boundaryComponentId;
				containingHasThickness[ containingCount ] =
					boundaryHasThickness ? 1u : 0u;
				containingThickness[ containingCount ] =
					max( boundaryThickness, 0.0 );
				containingCount ++;
			}
		} else {
			return false;
		}
		minimumDistanceExclusive = boundaryHit.dist;
	}
	return false;
}
#endif

`;
