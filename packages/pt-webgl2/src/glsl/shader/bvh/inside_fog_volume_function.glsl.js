import { MATERIAL_PIXELS } from '../structs/materialStride.js';

export const inside_fog_volume_function = /* glsl */`

#ifndef FOG_CHECK_ITERATIONS
#define FOG_CHECK_ITERATIONS 64
#endif

// returns whether the given material is a fog material or not
bool isMaterialFogVolume( sampler2D materials, uint materialIndex ) {

	const uint MATERIAL_PIXELS = ${MATERIAL_PIXELS}u;
	uint i = materialIndex * uint( MATERIAL_PIXELS );
	vec4 s14 = texelFetch1D( materials, i + 14u );
	return bool( int( s14.b ) & 4 );

}

// Reconstruct every closed participating medium that contains rayOrigin.
// Along a ray launched from an interior point, enclosing shells are encountered
// as back faces from inner to outer; the stored stack reverses that order so the
// innermost medium is on top. The first front face belongs to a volume that does
// not contain the origin and terminates the enclosure scan.
bool bvhBuildMediumStack(
        vec3 rayOrigin, vec3 rayDirection,
        usampler2D materialIndexAttribute, sampler2D materials,
        out MediumStack stack,
        inout FogMaterial material
) {

        initMediumStack( stack );
        material.fogVolume = false;
        uint containing[ MEDIUM_STACK_CAPACITY ];
        int containingCount = 0;

        for ( int i = 0; i < FOG_CHECK_ITERATIONS; i ++ ) {

		// find nearest hit
		uvec4 faceIndices = uvec4( 0u );
		vec3 faceNormal = vec3( 0.0, 0.0, 1.0 );
		vec3 barycoord = vec3( 0.0 );
		float side = 1.0;
		float dist = 0.0;
		bool hit = bvhIntersectFirstHit( bvh, rayOrigin, rayDirection, faceIndices, faceNormal, barycoord, side, dist );
		if ( hit ) {

                        // Record containing shells. A front face is the first
                        // boundary of a non-containing shell along this ray.
                        uint materialIndex = uTexelFetch1D( materialIndexAttribute, faceIndices.w ).r;
                        if ( isMaterialFogVolume( materials, materialIndex ) ) {
                                if ( side == 1.0 ) break;
                                if ( containingCount >= MEDIUM_STACK_CAPACITY ) return false;
                                containing[ containingCount ] = materialIndex;
                                containingCount ++;
                        }

                        // move the ray forward
                        rayOrigin = stepRayOrigin( rayOrigin, rayDirection, - faceNormal, dist );

                } else {
                        break;
                }

        }

        for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
                if ( i >= containingCount ) break;
                uint materialIndex = containing[ containingCount - 1 - i ];
                if ( ! enterMedium( stack, materialIndex, materials, material ) ) return false;
        }
        refreshMediumFromStack( stack, materials, material );
        return true;

}

// Compatibility wrapper for callers that only need the top medium.
bool bvhIntersectFogVolumeHit(
        vec3 rayOrigin, vec3 rayDirection,
        usampler2D materialIndexAttribute, sampler2D materials,
        inout FogMaterial material
) {
        MediumStack stack;
        bool valid = bvhBuildMediumStack(
                rayOrigin, rayDirection,
                materialIndexAttribute, materials,
                stack, material
        );
        return valid && material.fogVolume;

}

`;
