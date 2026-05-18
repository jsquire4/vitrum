/**
 * Optional fullscreen bilateral-ish blur + ACES tonemap for preview (not SVGF-class).
 * Keeps the fork canvas as unbiased HDR accumulation; this draws a separate canvas.
 */

import * as THREE from 'three';

const BILATERAL_VS = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}
`;

const BILATERAL_FS = /* glsl */ `
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uSigmaSpatial;
uniform float uSigmaRange;
varying vec2 vUv;

vec3 tonemapACES( vec3 x ) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp( ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e ), 0.0, 1.0 );
}

float lum( vec3 c ) {
  return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
}

void main() {
  vec4 tc = texture2D( uTex, vUv );
  vec3 c0 = tc.rgb / max( tc.a, 1e-6 );
  float L0 = lum( c0 );
  vec3 sum = vec3( 0.0 );
  float wsum = 0.0;
  for ( int dy = - 2; dy <= 2; dy ++ ) {
    for ( int dx = - 2; dx <= 2; dx ++ ) {

      vec2 o = vec2( float( dx ), float( dy ) ) * uTexel;
      vec4 t = texture2D( uTex, vUv + o );
      vec3 c = t.rgb / max( t.a, 1e-6 );
      float ws = exp( - float( dx * dx + dy * dy ) / ( 2.0 * uSigmaSpatial * uSigmaSpatial + 1e-6 ) );
      float wr = exp( - abs( lum( c ) - L0 ) / ( 2.0 * uSigmaRange * uSigmaRange + 1e-6 ) );
      float w = ws * wr;
      sum += c * w;
      wsum += w;

    }
  }

  vec3 outc = sum / max( wsum, 1e-6 );
  gl_FragColor = vec4( tonemapACES( outc ), 1.0 );
}
`;

export type DenoiseDisplayMode = 'raw' | 'bilateral' | 'oidn' | 'wgsl' | 'svgf';

export class BilateralPreviewCanvas {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  readonly mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  readonly canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: false,
      antialias: false,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(1);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: null },
        uTexel: { value: new THREE.Vector2(1, 1) },
        uSigmaSpatial: { value: 1.25 },
        uSigmaRange: { value: 0.12 },
      },
      vertexShader: BILATERAL_VS,
      fragmentShader: BILATERAL_FS,
      depthTest: false,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    this.scene.add(this.mesh);
  }

  render(tex: THREE.Texture, width: number, height: number): void {
    const { uniforms } = this.mesh.material;
    const uTex = uniforms['uTex'];
    const uTexel = uniforms['uTexel'];
    if (uTex == null || uTexel == null) throw new Error('BilateralPreviewCanvas: shader uniforms missing');
    uTex.value = tex;
    uTexel.value.set(1 / Math.max(width, 1), 1 / Math.max(height, 1));
    this.renderer.setSize(width, height, false);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.renderer.dispose();
  }
}

/** Simple Reinhard + γ preview for OIDN float RGB output on a 2D canvas. */
export function writeTonemappedRgbToCanvas(canvas: HTMLCanvasElement, rgb: Float32Array, w: number, h: number): void {
  canvas.style.display = 'block';
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (ctx == null) return;
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const reinhard = (x: number): number => x / (1 + Math.max(x, 0));
  const toByte = (x: number): number => {
    const rh = reinhard(x);
    return Math.min(255, Math.max(0, Math.floor(255 * rh ** (1 / 2.2))));
  };
  for (let i = 0; i < w * h; i += 1) {
    const j = i * 4;
    d[j] = toByte(rgb[i * 3] ?? 0);
    d[j + 1] = toByte(rgb[i * 3 + 1] ?? 0);
    d[j + 2] = toByte(rgb[i * 3 + 2] ?? 0);
    d[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
