// Ported from three-mesh-bvh (gkjohnson/three-mesh-bvh),
// src/webgl/glsl/bvh_ray_functions.glsl.js. MIT License, (c) Garrett Johnson.
// Vitrum's triangle solve separates angular, barycentric, and ray-parameter
// tolerances and equilibrates coordinates for scale-independent classification.
// See CREDITS.md.

/**
 * Set of shader functions used for interacting with the packed BVH in a shader and sampling
 * VertexAttributeTextures. Provides ray intersection functions. See
 * [src/webgl/glsl](https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgl/glsl)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.bvh_ray_functions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
export const bvh_ray_functions = /* glsl */`

#define TRI_INTERSECT_ANGULAR_EPSILON 1e-7
#define TRI_INTERSECT_BARYCENTRIC_EPSILON 1e-6

// Raycasting
bool intersectsBounds( vec3 rayOrigin, vec3 rayDirection, vec3 boundsMin, vec3 boundsMax, out float dist ) {

	// Evaluate each slab explicitly. The common reciprocal-vector form produces
	// 0 * Inf = NaN when an axis-parallel ray starts exactly on a slab, which can
	// turn a real hit into a driver-dependent miss. Division is used only after a
	// non-zero direction check, so boundary numerators remain an exact zero.
	if (
		any( isnan( rayOrigin ) ) || any( isinf( rayOrigin ) ) ||
		any( isnan( rayDirection ) ) || any( isinf( rayDirection ) ) ||
		any( isnan( boundsMin ) ) || any( isinf( boundsMin ) ) ||
		any( isnan( boundsMax ) ) || any( isinf( boundsMax ) ) ||
		any( greaterThan( boundsMin, boundsMax ) )
	) {
		dist = 0.0;
		return false;
	}

	float nearDistance = 0.0;
	float farDistance = INFINITY;
	for ( int axis = 0; axis < 3; axis ++ ) {

		float origin = rayOrigin[ axis ];
		float direction = rayDirection[ axis ];
		float slabMin = boundsMin[ axis ];
		float slabMax = boundsMax[ axis ];
		if ( direction == 0.0 ) {

			if ( origin < slabMin || origin > slabMax ) {
				dist = 0.0;
				return false;
			}
			continue;

		}

		float first = ( slabMin - origin ) / direction;
		float second = ( slabMax - origin ) / direction;
		float axisNear = min( first, second );
		float axisFar = max( first, second );
		nearDistance = max( nearDistance, axisNear );
		farDistance = min( farDistance, axisFar );
		if ( farDistance < nearDistance ) {
			dist = 0.0;
			return false;
		}

	}

	dist = nearDistance;
	return true;

}

bool intersectsTriangle(
	vec3 rayOrigin, vec3 rayDirection, vec3 a, vec3 b, vec3 c,
	out vec3 barycoord, out vec3 norm, out float dist, out float side
) {

	// https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
	vec3 edge1 = b - a;
	vec3 edge2 = c - a;
	float edgeScale = max(
		max( max( abs( edge1.x ), abs( edge1.y ) ), abs( edge1.z ) ),
		max( max( abs( edge2.x ), abs( edge2.y ) ), abs( edge2.z ) )
	);
	float directionScale = max(
		abs( rayDirection.x ),
		max( abs( rayDirection.y ), abs( rayDirection.z ) )
	);
	if (
		! ( edgeScale > 0.0 ) || edgeScale > 3.402823e38 ||
		! ( directionScale > 0.0 ) || directionScale > 3.402823e38
	) {
		return false;
	}
	vec3 scaledEdge1 = edge1 / edgeScale;
	vec3 scaledEdge2 = edge2 / edgeScale;
	vec3 scaledDirection = rayDirection / directionScale;
	vec3 scaledAO = ( rayOrigin - a ) / edgeScale;
	vec3 scaledNormal = cross( scaledEdge1, scaledEdge2 );
	float normalScale = max(
		abs( scaledNormal.x ),
		max( abs( scaledNormal.y ), abs( scaledNormal.z ) )
	);
	if ( ! ( normalScale > 0.0 ) || normalScale > 3.402823e38 ) {
		return false;
	}
	vec3 normalDirection = scaledNormal / normalScale;
	float normalLength = length( normalDirection );
	vec3 directionUnit = scaledDirection / length( scaledDirection );
	float det = - dot( scaledDirection, scaledNormal );
	float angularDeterminant = abs(
		dot( directionUnit, normalDirection / normalLength )
	);
	if (
		! ( angularDeterminant > TRI_INTERSECT_ANGULAR_EPSILON ) ||
		det == 0.0
	) {
		return false;
	}
	float invdet = 1.0 / det;
	vec3 DAO = cross( scaledAO, scaledDirection );

	vec4 uvt;
	uvt.x = dot( scaledEdge2, DAO ) * invdet;
	uvt.y = - dot( scaledEdge1, DAO ) * invdet;
	uvt.z =
		dot( scaledAO, scaledNormal ) * invdet *
		edgeScale / directionScale;
	uvt.w = 1.0 - uvt.x - uvt.y;

	// set the hit information
	barycoord = uvt.wxy; // arranged in A, B, C order
	dist = uvt.z;
	side = sign( det );
	norm = side * normalDirection / normalLength;

	bool finiteHit =
		all( lessThanEqual( abs( uvt ), vec4( 3.402823e38 ) ) );
	return finiteHit &&
		all( greaterThanEqual(
			uvt.xyw,
			vec3( - TRI_INTERSECT_BARYCENTRIC_EPSILON )
		) ) &&
		uvt.z >= 0.0;

}

bool intersectTriangles(
	// geometry info and triangle range
	sampler2D positionAttr, usampler2D indexAttr, uint offset, uint count,

	// ray
	vec3 rayOrigin, vec3 rayDirection,

	// outputs
	inout float minDistance, inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout float dist
) {

	bool found = false;
	vec3 localBarycoord, localNormal;
	float localDist, localSide;
	for ( uint i = offset, l = offset + count; i < l; i ++ ) {

		uvec3 indices = uTexelFetch1D( indexAttr, i ).xyz;
		vec3 a = texelFetch1D( positionAttr, indices.x ).rgb;
		vec3 b = texelFetch1D( positionAttr, indices.y ).rgb;
		vec3 c = texelFetch1D( positionAttr, indices.z ).rgb;

		if (
			intersectsTriangle( rayOrigin, rayDirection, a, b, c, localBarycoord, localNormal, localDist, localSide )
			&& localDist < minDistance
		) {

			found = true;
			minDistance = localDist;

			faceIndices = uvec4( indices.xyz, i );
			faceNormal = localNormal;

			side = localSide;
			barycoord = localBarycoord;
			dist = localDist;

		}

	}

	return found;

}

bool intersectsBVHNodeBounds( vec3 rayOrigin, vec3 rayDirection, sampler2D bvhBounds, uint currNodeIndex, out float dist ) {

	uint cni2 = currNodeIndex * 2u;
	vec3 boundsMin = texelFetch1D( bvhBounds, cni2 ).xyz;
	vec3 boundsMax = texelFetch1D( bvhBounds, cni2 + 1u ).xyz;
	return intersectsBounds( rayOrigin, rayDirection, boundsMin, boundsMax, dist );

}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhIntersectFirstHit(\
		bvh,\
		rayOrigin, rayDirection, faceIndices, faceNormal, barycoord, side, dist\
	)\
	_bvhIntersectFirstHit(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		rayOrigin, rayDirection, faceIndices, faceNormal, barycoord, side, dist\
	)

bool _bvhIntersectFirstHit(
	// bvh info
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,

	// ray
	vec3 rayOrigin, vec3 rayDirection,

	// output variables split into separate variables due to output precision
	inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout float dist
) {

	// stack needs to be twice as long as the deepest tree we expect because
	// we push both the left and right child onto the stack every traversal
	int pointer = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

	float triangleDistance = INFINITY;
	bool found = false;
	while ( pointer > - 1 && pointer < BVH_STACK_DEPTH ) {

		uint currNodeIndex = stack[ pointer ];
		pointer --;

		// check if we intersect the current bounds
		float boundsHitDistance;
		if (
			! intersectsBVHNodeBounds( rayOrigin, rayDirection, bvh_bvhBounds, currNodeIndex, boundsHitDistance )
			|| boundsHitDistance > triangleDistance
		) {

			continue;

		}

		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, currNodeIndex ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;

			found = intersectTriangles(
				bvh_position, bvh_index, offset, count,
				rayOrigin, rayDirection, triangleDistance,
				faceIndices, faceNormal, barycoord, side, dist
			) || found;

} else {

// Reserve both stack slots before writing them. A valid packed BVH is
// bounded by BVH_STACK_DEPTH, but this guard also turns malformed or
// cyclic texture data into a conservative miss instead of an
// out-of-bounds local-array write (undefined GLSL behaviour).
if ( pointer + 2 >= BVH_STACK_DEPTH ) {

continue;

}

uint leftIndex = currNodeIndex + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = currNodeIndex + boundsInfo.y;

			bool leftToRight = rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;

			// set c2 in the stack so we traverse it later. We need to keep track of a pointer in
			// the stack while we traverse. The second pointer added is the one that will be
			// traversed first
			pointer ++;
			stack[ pointer ] = c2;

			pointer ++;
			stack[ pointer ] = c1;

		}

	}

	return found;

}
`;
