// GlResources — the GL resource owner (plan/three-removal/02-gl-framework.md §5, §6).
// Analog of pt-webgpu's GpuResources / the fork's PathTracingRenderer state. It owns:
//   - the accumulation FBO + RGBA32F color texture (+ optional MRT gNormalDepth/gAlbedo),
//   - the Regime-2 ping-pong blend pair (allocated lazily only when needed),
//   - the PT GlProgram (built from composeTraceGlsl + FULLSCREEN_VERT),
//   - the Regime-2 BlendMaterial composite quad program,
//   - the std140 FrameParams UBO (WS5 frameParamsPacker target),
//   - a FullscreenQuad.
//
// The per-sample draw (drawAccumStep) replaces the fork's renderTask generator: bind the
// accum FBO + MRT draw buffers, set the blend regime, bind scene textures + params UBO, and
// draw the fullscreen triangle (fork PathTracingRenderer.js:144-167, §6 of plan 02).

import type { TraceFeatures, AccumRegime } from '../featureTypes.js';
import { featureDefines } from '../featureTypes.js';
import type { UploadedSceneTextures } from '../scene/sceneTextures.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { GlProgram } from './glProgram.js';
import { FullscreenQuad, FULLSCREEN_VERT } from './fullscreenQuad.js';
import {
  createRenderTarget,
  bindRenderTarget,
  clearRenderTarget,
  type RenderTarget,
} from './framebuffer.js';
import { setBlendForRegime } from './blend.js';
import { probeGlCaps, type GlCaps } from './glCaps.js';
import {
  CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE,
  X_CMF_CDF, Y_CMF_CDF, Z_CMF_CDF,
  X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';

// H2 — spectral CMF upload tables (constant; precomputed Float32 copies so the
// per-frame upload allocates nothing). The spectral path importance-samples the
// hero wavelength against these CIE 1931 tables/CDFs and reconstructs RGB via the
// integrals; without them every uniform is 0 → wavelengthPdf=0 → black. The CDFs
// are Float64Array in @vitrum/shared-samplers (length 82); GL needs Float32.
const CMF_X_F32 = Float32Array.from(CIE_X_TABLE);
const CMF_Y_F32 = Float32Array.from(CIE_Y_TABLE);
const CMF_Z_F32 = Float32Array.from(CIE_Z_TABLE);
const CMF_XCDF_F32 = Float32Array.from(X_CMF_CDF);
const CMF_YCDF_F32 = Float32Array.from(Y_CMF_CDF);
const CMF_ZCDF_F32 = Float32Array.from(Z_CMF_CDF);

/**
 * Per-frame INDIVIDUAL uniforms the (verbatim-copied) fork GLSL reads — it declares
 * `uniform vec2 resolution; uniform int bounces; uniform mat4 cameraWorldMatrix; ...`,
 * NOT a FrameParams UBO. Verified on a real driver: a UBO bind alone renders black
 * (camera matrices read as zero). The engine computes these each frame.
 */
export interface FrameUniforms {
  readonly resolution: readonly [number, number];
  readonly bounces: number;
  readonly transmissiveBounces: number;
  readonly filterGlossyFactor: number;
  readonly radianceClamp: number;
  readonly cameraWorldMatrix: Float32Array; // inverse(viewMatrix)
  readonly invProjectionMatrix: Float32Array; // inverse(projMatrix)
  readonly environmentIntensity: number;
  readonly environmentRotation: Float32Array; // mat4
  readonly spectralEnabled: boolean;
  readonly causticStrategy: number; // 0=none, 1=manifold-nee, 2=photon-map
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
}

/**
 * The Regime-2 composite quad fragment (verbatim port of the fork's BlendMaterial.js:31-59,
 * GL_FragColor → pc_fragColor). Lerps target1/target2 by `opacity = 1/(samples+1)` with
 * alpha-weighted compositing, written into the ping-pong pair.
 */
const BLEND_FRAG = `
in vec2 vUv;
uniform float opacity;
uniform sampler2D target1;
uniform sampler2D target2;
void main() {
  vec4 color1 = texture(target1, vUv);
  vec4 color2 = texture(target2, vUv);
  float invOpacity = 1.0 - opacity;
  float totalAlpha = color1.a * invOpacity + color2.a * opacity;
  if (color1.a != 0.0 || color2.a != 0.0) {
    pc_fragColor.rgb = color1.rgb * (invOpacity * color1.a / totalAlpha)
                     + color2.rgb * (opacity * color2.a / totalAlpha);
    pc_fragColor.a = totalAlpha;
  } else {
    pc_fragColor = vec4(0.0);
  }
}
`;

export class GlResources {
  readonly #gl: WebGL2RenderingContext;
  readonly #caps: GlCaps;
  /** Whether MRT aux g-buffers (gNormalDepth/gAlbedo) are allocated + written. */
  readonly #auxBuffers: boolean;
  /** OES_draw_buffers_indexed — per-draw-buffer blend state, so the colour blend runs
   *  on attachment 0 while the aux g-buffer attachments overwrite (no blend). null when
   *  unavailable (the g-buffer is then best-effort; the colour path is always correct). */
  readonly #drawBuffersIndexed: { disableiOES(target: number, index: number): void } | null;
  readonly #quad: FullscreenQuad;

  /** The PT accumulation target (RGBA32F primary + optional MRT aux). */
  #accum: RenderTarget | null = null;
  /** Regime-2 ping-pong blend pair — allocated lazily on first alpha-composite step. */
  #blend: [RenderTarget, RenderTarget] | null = null;
  /** Which slot of the blend pair currently holds the readable result. */
  #blendReadIndex = 0;

  #ptProgram: GlProgram | null = null;
  #blendProgram: GlProgram | null = null;

  #accumWidth = 0;
  #accumHeight = 0;
  /** Samples already accumulated since the last clearAccum() — drives opacity 1/(N+1). */
  #samples = 0;

  /**
   * @param supportsAuxBuffers host policy for the MRT g-buffers; intersected here with the
   * device's MAX_DRAW_BUFFERS so a 1-MRT device degrades to attachment-0-only (plan 02 §4).
   */
  constructor(gl: WebGL2RenderingContext, supportsAuxBuffers: boolean) {
    this.#gl = gl;
    this.#caps = probeGlCaps(gl);
    this.#auxBuffers = supportsAuxBuffers && this.#caps.maxDrawBuffers >= 3;
    const dbi = gl.getExtension('OES_draw_buffers_indexed') as { disableiOES?: unknown } | null;
    this.#drawBuffersIndexed =
      dbi != null && typeof dbi.disableiOES === 'function'
        ? (dbi as { disableiOES(target: number, index: number): void })
        : null;
    this.#quad = new FullscreenQuad(gl);
  }

  /** Probed device capabilities (regime selection, MRT/sampler budgets). */
  get caps(): GlCaps {
    return this.#caps;
  }

  /**
   * Ensure the accumulation resources match w×h, reallocating on a size change.
   * Returns `recreated` — true when targets were (re)built, so the caller resets its
   * sample counter (fork PathTracingRenderer.setSize → reset, :358-374).
   */
  ensureAccumResources(w: number, h: number): boolean {
    if (w === this.#accumWidth && h === this.#accumHeight && this.#accum != null) return false;
    this.#destroyTargets();
    this.#accum = createRenderTarget(this.#gl, w, h, this.#auxBuffers);
    this.#accumWidth = w;
    this.#accumHeight = h;
    this.#samples = 0;
    return true;
  }

  /** Build the PT program from `composeTraceGlsl(features)` once (idempotent). */
  ensureProgram(features: TraceFeatures): void {
    this.#ptProgram ??= new GlProgram(
      this.#gl,
      FULLSCREEN_VERT,
      composeTraceGlsl(features),
      featureDefines(features),
    );
  }

  /** The PT program (null before ensureProgram) — for the host to set uniforms/defines. */
  get ptProgram(): GlProgram | null {
    return this.#ptProgram;
  }

  /** Clear all accumulation targets to (0,0,0,0) + reset the sample counter (fork reset()). */
  clearAccum(): void {
    if (this.#accum != null) clearRenderTarget(this.#gl, this.#accum);
    if (this.#blend != null) {
      clearRenderTarget(this.#gl, this.#blend[0]);
      clearRenderTarget(this.#gl, this.#blend[1]);
    }
    this.#blendReadIndex = 0;
    this.#samples = 0;
  }

  /**
   * One accumulation step (plan 02 §6). Binds the accum FBO + MRT draw buffers, sets the
   * blend regime, uploads the running-average opacity + seed, binds the scene textures and
   * the params UBO, and draws the fullscreen triangle. For the alpha-composite regime it then
   * runs the BlendMaterial composite into the ping-pong pair.
   */
  drawAccumStep(
    scene: UploadedSceneTextures,
    regime: AccumRegime,
    seed: number,
    frame: FrameUniforms,
  ): void {
    const gl = this.#gl;
    if (this.#accum == null) throw new Error('pt-webgl2: drawAccumStep before ensureAccumResources');
    if (this.#ptProgram == null) throw new Error('pt-webgl2: drawAccumStep before ensureProgram');
    const prog = this.#ptProgram;

    bindRenderTarget(gl, this.#accum);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    setBlendForRegime(gl, regime, this.#samples);
    // The MRT g-buffer (gNormalDepth@1, gAlbedo@2) holds sample-invariant primary-hit
    // data and must NOT accumulate: WebGL2 weights each draw buffer by its OWN output
    // alpha, and gNormalDepth packs linear depth (>1) in alpha, so the colour-blend
    // recurrence diverges → NaN. Disable blend on JUST the aux attachments (per-draw-buffer
    // state) so they overwrite (latest sample = correct, primary-hit is deterministic)
    // while the colour attachment keeps its EXACT running-average blend untouched. No
    // extension → the g-buffer is best-effort (the colour path is always correct).
    if (this.#auxBuffers && this.#drawBuffersIndexed != null) {
      this.#drawBuffersIndexed.disableiOES(gl.BLEND, 1);
      this.#drawBuffersIndexed.disableiOES(gl.BLEND, 2);
    }

    prog.use();
    // The copied fork GLSL reads INDIVIDUAL uniforms (no FrameParams UBO).
    prog.setInt('seed', seed);
    prog.setFloat('opacity', 1 / (this.#samples + 1));
    prog.setVec2('resolution', frame.resolution[0], frame.resolution[1]);
    prog.setInt('bounces', frame.bounces);
    prog.setInt('transmissiveBounces', frame.transmissiveBounces);
    prog.setFloat('filterGlossyFactor', frame.filterGlossyFactor);
    prog.setFloat('uRadianceClamp', frame.radianceClamp);
    prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
    prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);
    prog.setFloat('environmentIntensity', frame.environmentIntensity);
    prog.setMat4('environmentRotation', frame.environmentRotation);
    prog.setInt('uSpectralRendering', frame.spectralEnabled ? 1 : 0);
    if (frame.spectralEnabled) {
      // H2 FIX (2026-06-09): upload the CIE CMF tables + CDFs + integrals. Before
      // this, GlProgram.setFloatArray had ZERO callers, so uCmfX/Y/Z, the three
      // CDFs, and the integrals all defaulted to 0 → wavelengthPdf=0 →
      // wavelengthToRGB() returned vec3(0) → `spectral: true` rendered BLACK.
      // Constant data, cheap re-upload; gated so non-spectral frames skip it.
      // (u_jakobCoeffs / iorCauchy stay at their flat-spectrum / no-dispersion
      // defaults — those refine spectral reflectance colour, not black-vs-lit.)
      prog.setFloatArray('uCmfX', CMF_X_F32);
      prog.setFloatArray('uCmfY', CMF_Y_F32);
      prog.setFloatArray('uCmfZ', CMF_Z_F32);
      prog.setFloatArray('uXCmfCdf', CMF_XCDF_F32);
      prog.setFloatArray('uYCmfCdf', CMF_YCDF_F32);
      prog.setFloatArray('uZCmfCdf', CMF_ZCDF_F32);
      prog.setFloat('uXCmfIntegral', X_CMF_INTEGRAL);
      prog.setFloat('uYCmfIntegral', Y_CMF_INTEGRAL);
      prog.setFloat('uZCmfIntegral', Z_CMF_INTEGRAL);
    }
    prog.setInt('uCausticStrategy', frame.causticStrategy);
    prog.setFloat('uMneeMaxIterations', frame.mneeMaxIterations);
    prog.setFloat('uMneeMaxChainLength', frame.mneeMaxChainLength);
    this.#bindSceneTextures(prog, scene);
    this.#quad.draw(gl);

    if (regime === 'alpha-composite') this.#compositeBlendStep();

    this.#samples += 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** The readable accumulation result texture (ping-pong slot for Regime 2, else primary). */
  resultTexture(): WebGLTexture | null {
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      return (this.#blendReadIndex === 0 ? a : b).color;
    }
    return this.#accum?.color ?? null;
  }

  /** The packed normal+depth g-buffer (null when MRT disabled). */
  get normalDepthTex(): WebGLTexture | null {
    return this.#accum?.normalDepth ?? null;
  }

  /** The albedo g-buffer (null when MRT disabled). */
  get albedoTex(): WebGLTexture | null {
    return this.#accum?.albedo ?? null;
  }

  dispose(): void {
    const gl = this.#gl;
    this.#destroyTargets();
    this.#ptProgram?.dispose();
    this.#ptProgram = null;
    this.#blendProgram?.dispose();
    this.#blendProgram = null;
    this.#quad.dispose(gl);
  }

  // ----- internals -------------------------------------------------------------------------

  /** Composite the latest PT sample into the ping-pong pair (Regime 2; fork :156-165). */
  #compositeBlendStep(): void {
    const gl = this.#gl;
    const accum = this.#accum;
    if (accum == null) return;
    this.#ensureBlendPair();
    const blend = this.#blend;
    if (blend == null) return;
    const [slot0, slot1] = blend;
    const readTarget = this.#blendReadIndex === 0 ? slot0 : slot1;
    const writeTarget = this.#blendReadIndex === 0 ? slot1 : slot0;

    const blendProg = this.#ensureBlendProgram();
    bindRenderTarget(gl, writeTarget);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    gl.disable(gl.BLEND); // the composite math is done in-shader, not by fixed-function blend.

    blendProg.use();
    blendProg.setFloat('opacity', 1 / (this.#samples + 1));
    // target1 = prior accumulated result; target2 = the just-rendered PT sample.
    blendProg.bindTexture('target1', readTarget.color);
    blendProg.bindTexture('target2', accum.color);
    this.#quad.draw(gl);

    this.#blendReadIndex = 1 - this.#blendReadIndex;
  }

  #ensureBlendProgram(): GlProgram {
    this.#blendProgram ??= new GlProgram(this.#gl, FULLSCREEN_VERT, BLEND_FRAG, {});
    return this.#blendProgram;
  }

  #ensureBlendPair(): void {
    if (this.#blend != null) return;
    this.#blend = [
      createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false),
      createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false),
    ];
    clearRenderTarget(this.#gl, this.#blend[0]);
    clearRenderTarget(this.#gl, this.#blend[1]);
    this.#blendReadIndex = 0;
  }

  /** Bind the scene texture bundle to the PT program's samplers (plan 04 §4 binding remap). */
  #bindSceneTextures(prog: GlProgram, scene: UploadedSceneTextures): void {
    const gl = this.#gl;
    // BVH struct samplers (BVH { usampler2D index; sampler2D position; sampler2D bvhBounds;
    // usampler2D bvhContents; } — bvh_struct_definitions.glsl).
    prog.bindTexture('bvh.index', scene.bvhIndex);
    prog.bindTexture('bvh.position', scene.bvhPosition);
    prog.bindTexture('bvh.bvhBounds', scene.bvhBounds);
    prog.bindTexture('bvh.bvhContents', scene.bvhContents);
    prog.bindTexture('materialIndexAttribute', scene.materialIndex);
    prog.bindTexture('materials', scene.materials);
    prog.bindTexture('attributesArray', scene.attributesArray, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('lights.tex', scene.lights); // LightsInfo { sampler2D tex; uint count; }
    // H1 FIX (2026-06-09): upload the `lights.count` uint — without it the field
    // defaults to 0u and the ENTIRE analytic-light system is inert (the NEE gate
    // `rand(5) < count/lightsDenom`, the forward light-hit loop `i < lights.count`,
    // and the BDPT light-subpath all see zero lights). Only `mesh-area` emitters
    // lit anything, via the emissive fold. This restores point/spot/rect-area/
    // circ-area/directional NEE. The packed lights texture was already uploaded;
    // only its count uniform was missing.
    prog.setUint('lights.count', scene.lightCount);
    // Every OPTIONAL sampler the fork GLSL declares must reference a valid texture of
    // the matching type — an unbound sampler defaults to unit 0 and collides with a
    // different-typed sampler there (GL_INVALID_OPERATION → black). bindTexture no-ops
    // for inactive samplers, so binding a dummy where there's no scene data is safe.
    const d2d = this.#dummyTex2D();
    const d2a = this.#dummyTex2DArray();
    prog.bindTexture('iesProfiles', scene.iesProfiles ?? d2a, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('textures', scene.textures2DArray ?? d2a, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('backgroundMap', d2d);
    prog.bindTexture('sobolTexture', d2d);
    prog.bindTexture('stratifiedTexture', d2d);
    prog.bindTexture('stratifiedOffsetTexture', d2d);
    // EquirectHdrInfo { sampler2D marginalWeights; sampler2D conditionalWeights; sampler2D map; float totalSum; }
    prog.bindTexture('envMapInfo.map', scene.envMap ?? d2d);
    prog.bindTexture('envMapInfo.marginalWeights', scene.envMarginal ?? d2d);
    prog.bindTexture('envMapInfo.conditionalWeights', scene.envConditional ?? d2d);
    // The env-sampling GLSL early-outs on `envMapInfo.totalSum == 0.0`; bind the
    // scalar so a present environment is actually sampled (0 when absent → correct
    // no-env early-out). Without this the env textures upload but never light.
    prog.setFloat('envMapInfo.totalSum', scene.envTotalSum);
  }

  #dummy2dTex: WebGLTexture | null = null;
  #dummy2dArrTex: WebGLTexture | null = null;

  #dummyTex2D(): WebGLTexture {
    const gl = this.#gl;
    if (this.#dummy2dTex == null) {
      const t = gl.createTexture();
      if (t == null) throw new Error('pt-webgl2: failed to create dummy 2D texture');
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.#dummy2dTex = t;
    }
    return this.#dummy2dTex;
  }

  #dummyTex2DArray(): WebGLTexture {
    const gl = this.#gl;
    if (this.#dummy2dArrTex == null) {
      const t = gl.createTexture();
      if (t == null) throw new Error('pt-webgl2: failed to create dummy 2D-array texture');
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.#dummy2dArrTex = t;
    }
    return this.#dummy2dArrTex;
  }

  #destroyTargets(): void {
    this.#accum?.destroy();
    this.#accum = null;
    if (this.#blend != null) {
      this.#blend[0].destroy();
      this.#blend[1].destroy();
      this.#blend = null;
    }
    this.#accumWidth = 0;
    this.#accumHeight = 0;
    this.#blendReadIndex = 0;
  }
}
