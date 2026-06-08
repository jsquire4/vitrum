/**
 * Per-tile variance estimation from the HDR accumulation texture (sum/count in RGBA).
 * Drives optional repeat factors (1–4) for additive accumulation — noisy tiles receive
 * extra path samples within the same scheduler round without freezing unbiased averages.
 */

import type { Texture as THREE_Texture, WebGLRenderer } from 'three';
import {
  FloatType,
  Mesh,
  NearestFilter,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Vector2,
  WebGLRenderTarget,
} from 'three';
import { ZERO_SAMPLE_COUNT_EPSILON } from '../../accumulationSampleEpsilon.js';

const TILE_VARIANCE_VS = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}
`;

const TILE_VARIANCE_FS = /* glsl */ `
precision highp float;
precision highp int;

uniform sampler2D uAccum;
uniform vec2 uTexSize;
uniform vec2 uTiles;

varying vec2 vUv;

float lum( vec3 c ) {
  return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
}

void main() {
  vec2 tileIds = floor( vUv * uTiles - vec2( 1e-4 ) );
  vec2 tileIdx = clamp( tileIds, vec2( 0.0 ), uTiles - vec2( 1.0 ) );

  vec2 tileSize = uTexSize / max( uTiles, vec2( 1.0 ) );
  vec2 base = tileIdx * tileSize;

  float mean = 0.0;
  float meanSq = 0.0;
  const int STEP = 4;
  float n = 0.0;

  for ( int dy = 0; dy < STEP; dy ++ ) {
    for ( int dx = 0; dx < STEP; dx ++ ) {

      vec2 p = base + ( vec2( float( dx ), float( dy ) ) + vec2( 0.5 ) ) / vec2( float( STEP ) ) * tileSize;
      ivec2 ip = ivec2( clamp( floor( p ), vec2( 0.0 ), uTexSize - vec2( 1.0 ) ) );
      vec4 t = texelFetch( uAccum, ip, 0 );
      float cnt = t.a;
      float L = cnt > ${ZERO_SAMPLE_COUNT_EPSILON} ? lum( t.rgb / cnt ) : 0.0;
      mean += L;
      meanSq += L * L;
      n += 1.0;

    }
  }

  mean /= max( n, 1.0 );
  float variance = max( 0.0, meanSq / max( n, 1.0 ) - mean * mean );
  gl_FragColor = vec4( variance, mean, 0.0, 1.0 );
}
`;

/** Maximum tile grid edge supported by the variance scratch target. */
export const MAX_TILE_GRID = 8 as const;

export class TileVariancePass {
  readonly rt: WebGLRenderTarget;
  readonly scene = new Scene();
  readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  readonly mesh: Mesh<PlaneGeometry, ShaderMaterial>;

  constructor(maxTiles: number) {
    const dim = Math.max(1, Math.min(MAX_TILE_GRID, maxTiles));
    this.rt = new WebGLRenderTarget(dim, dim, {
      format: RGBAFormat,
      type: FloatType,
      depthBuffer: false,
      stencilBuffer: false,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
    });
    const mat = new ShaderMaterial({
      uniforms: {
        uAccum: { value: null },
        uTexSize: { value: new Vector2(1, 1) },
        uTiles: { value: new Vector2(1, 1) },
      },
      vertexShader: TILE_VARIANCE_VS,
      fragmentShader: TILE_VARIANCE_FS,
      depthTest: false,
      depthWrite: false,
    });
    this.mesh = new Mesh(new PlaneGeometry(2, 2), mat);
    this.scene.add(this.mesh);
  }

  run(
    renderer: WebGLRenderer,
    accumTexture: THREE_Texture,
    texWidth: number,
    texHeight: number,
    tilesX: number,
    tilesY: number,
  ): void {
    const mat = this.mesh.material;
    mat.uniforms['uAccum']!.value = accumTexture;
    mat.uniforms['uTexSize']!.value.set(texWidth, texHeight);
    mat.uniforms['uTiles']!.value.set(tilesX, tilesY);
    const og = renderer.getRenderTarget();
    this.rt.setSize(tilesX, tilesY);
    renderer.setRenderTarget(this.rt);
    renderer.clear(true, true, true);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(og);
  }

  dispose(): void {
    this.rt.dispose();
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

/**
 * Maps readRenderTargetPixels row index `py` (bottom row = 0) into the same linear
 * tile index order used by PathTracingRenderer.tileRepeatFactors (`ty * tilesX + px`).
 */
export function linearTileIndexFromVarianceReadPixelsPy(
  py: number,
  px: number,
  tilesX: number,
  tilesY: number,
): number {
  const tyTopDown = tilesY - 1 - py;
  return tyTopDown * tilesX + px;
}

function classifyVariances(variances: Float32Array, tileCount: number, out: Uint8Array): void {
  if (tileCount <= 0) return;
  const sorted = new Float32Array(tileCount);
  sorted.set(variances.subarray(0, tileCount));
  sorted.sort();

  const q = (p: number): number => {
    const idx = Math.min(tileCount - 1, Math.max(0, Math.floor(p * (tileCount - 1))));
    return sorted[idx] ?? 0;
  };

  const q50 = q(0.5);
  const q75 = q(0.75);
  const hi = Math.max(q75 * 1.25, q50 * 1.5 + 1e-8);

  for (let i = 0; i < tileCount; i += 1) {
    const v = variances[i] ?? 0;
    if (v >= hi * 1.1) out[i] = 4;
    else if (v >= hi * 0.65) out[i] = 3;
    else if (v >= q50 * 1.2) out[i] = 2;
    else out[i] = 1;
  }
}

/**
 * Renders per-tile luminance variance into an internal Float RT, reads back,
 * and fills `factorsOut` with repeat counts in [1, 4].
 */
export function computeAdaptiveTileRepeatFactors(
  pass: TileVariancePass,
  renderer: WebGLRenderer,
  accumTexture: THREE_Texture,
  texWidth: number,
  texHeight: number,
  tilesX: number,
  tilesY: number,
  factorsOut: Uint8Array,
): void {
  const tileCount = tilesX * tilesY;
  if (tileCount <= 0 || factorsOut.length < tileCount) return;

  pass.run(renderer, accumTexture, texWidth, texHeight, tilesX, tilesY);

  const pixelCount = tilesX * tilesY * 4;
  const buf = new Float32Array(pixelCount);
  renderer.readRenderTargetPixels(pass.rt, 0, 0, tilesX, tilesY, buf);

  const variances = new Float32Array(tileCount);
  for (let py = 0; py < tilesY; py += 1) {
    for (let px = 0; px < tilesX; px += 1) {
      const src = (py * tilesX + px) * 4;
      variances[linearTileIndexFromVarianceReadPixelsPy(py, px, tilesX, tilesY)] = buf[src] ?? 0;
    }
  }

  classifyVariances(variances, tileCount, factorsOut);
}
