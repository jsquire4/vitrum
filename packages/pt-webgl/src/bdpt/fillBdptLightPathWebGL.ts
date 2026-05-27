/**
 * CPU fill of BdptLightPathBuffer texture (bounce 0) for WebGL2 hosts.
 * Until the fork's dedicated light-subpath draw pass is wired, this unblocks
 * cornell-box `?vitrumBdpt=1` and reference captures.
 */

import type { Scene } from '@vitrum/core';
import {
  DataTexture,
  FloatType,
  NearestFilter,
  RGBAFormat,
  type Texture,
  type WebGLRenderer,
} from 'three';
import {
  BDPT_KIND_INVALID,
  BDPT_KIND_LIGHT,
  type BdptBounce0Vertex,
  sampleBdptBounce0FromScene,
} from './bdptSceneEmittersCpu.js';

/** Row-major RGBA32F bytes matching fork `writeLightSubpathVertex` bounce-0 layout. */
export function packBdptLightPathColumnsWebGL(
  width: number,
  bounce0: BdptBounce0Vertex | null,
): Float32Array {
  const data = new Float32Array(width * 4 * 3);
  for (let col = 0; col < width; col += 1) {
    data[col * 4 + 3] = BDPT_KIND_INVALID;
  }
  if (bounce0 == null) return data;
  const col = 0;
  const o0 = col * 4;
  const o1 = width * 4 + col * 4;
  const o2 = width * 8 + col * 4;
  data[o0 + 0] = bounce0.emitPos[0];
  data[o0 + 1] = bounce0.emitPos[1];
  data[o0 + 2] = bounce0.emitPos[2];
  data[o0 + 3] = BDPT_KIND_LIGHT;
  data[o1 + 0] = bounce0.emitNormal[0];
  data[o1 + 1] = bounce0.emitNormal[1];
  data[o1 + 2] = bounce0.emitNormal[2];
  data[o1 + 3] = bounce0.pdfJoint;
  data[o2 + 0] = bounce0.throughput[0];
  data[o2 + 1] = bounce0.throughput[1];
  data[o2 + 2] = bounce0.throughput[2];
  data[o2 + 3] = bounce0.pdfHemi;
  return data;
}

export function fillBdptLightPathWebGL(
  renderer: WebGLRenderer,
  texture: Texture,
  maxLightBounces: number,
  scene: Scene,
  frameSeed: number,
): void {
  const width = maxLightBounces;
  const sample = sampleBdptBounce0FromScene(scene, frameSeed);
  const data = packBdptLightPathColumnsWebGL(width, sample);
  uploadLightPathTexture(renderer, texture, width, data);
}

function uploadLightPathTexture(
  renderer: WebGLRenderer,
  texture: Texture,
  width: number,
  data: Float32Array,
): void {
  const prevRt = renderer.getRenderTarget();
  const src = new DataTexture(data, width, 3, RGBAFormat, FloatType);
  src.minFilter = NearestFilter;
  src.magFilter = NearestFilter;
  src.needsUpdate = true;
  try {
    renderer.initTexture(src);
    renderer.initTexture(texture);
    if (typeof renderer.copyTextureToTexture === 'function') {
      renderer.copyTextureToTexture(src, texture);
      return;
    }

    const gl = renderer.getContext() as WebGL2RenderingContext;
    const props = renderer.properties.get(texture) as { __webglTexture?: WebGLTexture };
    const webglTexture = props.__webglTexture;
    if (webglTexture == null) {
      throw new Error('[fillBdptLightPathWebGL] failed to allocate light-path texture');
    }
    renderer.state.activeTexture(gl.TEXTURE0);
    renderer.state.bindTexture(gl.TEXTURE_2D, webglTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, 3, gl.RGBA, gl.FLOAT, data);
  } finally {
    renderer.setRenderTarget(prevRt);
    src.dispose();
  }
}
