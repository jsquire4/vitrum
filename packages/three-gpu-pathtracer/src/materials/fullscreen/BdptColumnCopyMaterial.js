import { NoBlending } from 'three';
import { MaterialBase } from '../MaterialBase.js';

/**
 * Copies one column from a BDPT light-subpath scratch texture into the
 * accumulation target (GPU-only — ANGLE often cannot read float FBOs).
 */
export class BdptColumnCopyMaterial extends MaterialBase {

	constructor() {

		super( {

			blending: NoBlending,

			uniforms: {
				uSrc: { value: null },
				uCol: { value: 0 },
			},

			vertexShader: /* glsl */`
				varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader: /* glsl */`
				uniform sampler2D uSrc;
				uniform int uCol;
				void main() {
					if ( int( gl_FragCoord.x ) != uCol ) {
						discard;
					}
					gl_FragColor = texelFetch( uSrc, ivec2( uCol, int( gl_FragCoord.y ) ), 0 );
				}`,

		} );

	}

}
