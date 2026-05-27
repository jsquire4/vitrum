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
  sampleBdptBounce0FromScene,
} from './bdptSceneEmittersCpu.js';

export function fillBdptLightPathWebGL(
  renderer: WebGLRenderer,
  texture: Texture,
  maxLightBounces: number,
  scene: Scene,
  frameSeed: number,
): void {
  const width = maxLightBounces;
  const data = new Float32Array(width * 4 * 3);
  for (let col = 0; col < width; col += 1) {
    const o0 = col * 4;
    data[o0 + 3] = BDPT_KIND_INVALID;
  }

  const sample = sampleBdptBounce0FromScene(scene, frameSeed);
  if (sample != null) {
    const col = 0;
    const o0 = col * 4;
    const o1 = width * 4 + col * 4;
    const o2 = width * 8 + col * 4;
    data[o0 + 0] = sample.emitPos[0];
    data[o0 + 1] = sample.emitPos[1];
    data[o0 + 2] = sample.emitPos[2];
    data[o0 + 3] = BDPT_KIND_LIGHT;
    data[o1 + 0] = sample.emitNormal[0];
    data[o1 + 1] = sample.emitNormal[1];
    data[o1 + 2] = sample.emitNormal[2];
    data[o1 + 3] = sample.pdfJoint;
    data[o2 + 0] = sample.throughput[0];
    data[o2 + 1] = sample.throughput[1];
    data[o2 + 2] = sample.throughput[2];
    data[o2 + 3] = sample.pdfHemi;
  }

  uploadLightPathTexture(renderer, texture, width, data);
}

function uploadLightPathTexture(
  renderer: WebGLRenderer,
  texture: Texture,
  width: number,
  data: Float32Array,
): void {
  const src = new DataTexture(data, width, 3, RGBAFormat, FloatType);
  src.minFilter = NearestFilter;
  src.magFilter = NearestFilter;
  src.needsUpdate = true;
  renderer.initTexture(src);
  renderer.initTexture(texture);
  if (typeof renderer.copyTextureToTexture === 'function') {
    renderer.copyTextureToTexture(src, texture);
    src.dispose();
    return;
  }

  const gl = renderer.getContext() as WebGL2RenderingContext;
  const props = renderer.properties.get(texture) as { __webglTexture?: WebGLTexture };
  const webglTexture = props.__webglTexture;
  if (webglTexture == null) {
    src.dispose();
    throw new Error('[fillBdptLightPathWebGL] failed to allocate light-path texture');
  }
  const prevRt = renderer.getRenderTarget();
  renderer.state.activeTexture(gl.TEXTURE0);
  renderer.state.bindTexture(gl.TEXTURE_2D, webglTexture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, 3, gl.RGBA, gl.FLOAT, data);
  renderer.setRenderTarget(prevRt);
  src.dispose();
}
