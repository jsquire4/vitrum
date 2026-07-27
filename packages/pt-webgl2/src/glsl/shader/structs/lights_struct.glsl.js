export const lights_struct = /* glsl */`

	#define RECT_AREA_LIGHT_TYPE 0
	#define CIRC_AREA_LIGHT_TYPE 1
	#define SPOT_LIGHT_TYPE 2
	#define DIR_LIGHT_TYPE 3
	#define POINT_LIGHT_TYPE 4
	// B4 — mesh-area triangle light (NEE). Packed separately in uMeshLights, but the
	// type id shares the LightRecord.type space so directLightContribution can branch.
	#define TRI_AREA_LIGHT_TYPE 5

	struct LightsInfo {

		sampler2D tex;
		uint count;

	};

	struct Light {

		vec3 position;
		int type;

		vec3 color;
		float intensity;

		vec3 u;
		vec3 v;
		float area;
		float power;

		// spot light fields
		float decay;
		float distance;
		float coneCos;
		float penumbraCos;
			// SHADOW-01 — s5.g (the former IES padding slot) carries 1.0 when the
			// emitter set castShadow:false (0.0 default). Packed for EVERY light kind
			// by lightsTexture.ts; consumed by directLightContribution to skip the
			// NEE shadow test for that light.
			float castShadowDisabled;
			// DirectionalEmitter.angularDiameter in radians. Packed in s5.b for
			// directional lights; 0 keeps the historical delta/hard-sun path.
			float angularDiameter;

	};

	Light readLightInfo( sampler2D tex, uint index ) {

		uint i = index * 6u;

		vec4 s0 = texelFetch1D( tex, i + 0u );
		vec4 s1 = texelFetch1D( tex, i + 1u );
		vec4 s2 = texelFetch1D( tex, i + 2u );
		vec4 s3 = texelFetch1D( tex, i + 3u );

		Light l;
		l.position = s0.rgb;
		l.type = int( round( s0.a ) );

		l.color = s1.rgb;
		l.intensity = s1.a;

		l.u = s2.rgb;
		l.power = s2.a;
		l.v = s3.rgb;
		l.area = s3.a;

		// SHADOW-01 — s5.g carries castShadowDisabled for EVERY light kind, so s5
		// is fetched unconditionally (one extra texel fetch; the grid always has
		// 6 texels per light).
			vec4 s5 = texelFetch1D( tex, i + 5u );
			l.castShadowDisabled = s5.g;
			l.angularDiameter = l.type == DIR_LIGHT_TYPE ? max( s5.b, 0.0 ) : 0.0;

		if ( l.type == SPOT_LIGHT_TYPE || l.type == POINT_LIGHT_TYPE ) {

			vec4 s4 = texelFetch1D( tex, i + 4u );
			l.decay = s4.g;
			l.distance = s4.b;
			l.coneCos = s4.a;

			l.penumbraCos = s5.r;

		} else {

			l.decay = 0.0;
			l.distance = 0.0;

			l.coneCos = 0.0;
			l.penumbraCos = 0.0;

		}

		return l;

	}

`;
