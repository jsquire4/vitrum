# WS2 — The raw-WebGL2 render framework (`src/gl/`)

> This replaces ~2,764 LOC of THREE `WebGLRenderer`-bound glue. The fork's `PathTracingRenderer.js` is *already* a fragment-shader accumulation design — we re-express the SAME operations (FBO ping-pong, fullscreen quad, additive/alpha/normal blend, per-tile scissor) in raw WebGL2. The novel piece is the **program builder** (replicating THREE's `ShaderMaterial` `#define`-recompile + uniform aliasing + auto-GLSL3), since we no longer have `ShaderMaterial`.

## 1. File inventory

```
src/gl/
  glResources.ts      # class GlResources — owns FBOs/textures/UBO/programs (analog of pt-webgpu GpuResources)
  programBuilder.ts   # GlProgram — compile/link, #define injection, uniform/sampler binding, recompile-on-define-change
  framebuffer.ts      # FBO + attachment helpers (RGBA32F, MRT, ping-pong pairs)
  textures.ts         # GL texture create/upload helpers (RGBA32F/RG32UI/HalfFloat/sampler2DArray) for WS3 packers
  fullscreenQuad.ts   # the 1-triangle/quad VAO + draw helper
  blend.ts            # the 3 accumulation blend regimes (additive / alpha-composite / normal-average)
  glCaps.ts           # capability probe: EXT_color_buffer_float, EXT_float_blend, MAX_DRAW_BUFFERS, texunits
```

## 2. `GlProgram` — the program builder (replaces `MaterialBase` + `ShaderMaterial`)

Verified behaviors to replicate (from `MaterialBase.js`, all 71 lines, + the GLSL3 auto-promotion finding):
1. **Uniform-as-property aliasing** — `material.bounces = 10` writes `uniforms.bounces.value`. → We use an explicit `set(name, value)` + a cached `name → WebGLUniformLocation` map.
2. **`needsUpdate` → `'recompilation'` event** — any `#define` change relinks the program and (per the fork comment `PhysicalPathTracingMaterial.js:43-44`) resets the accumulator. → `setDefine(name, value)` is change-gated; a real change marks `#dirty = true`; `use()` relinks if dirty.
3. **GLSL3 is implicit in THREE** (auto from `precision highp isampler2D` + `layout(location=N) out`). → We emit `#version 300 es` manually and declare `layout(location = 0) out vec4 pc_fragColor;` ourselves (THREE auto-injected location 0; the fork shader assumes it).

```ts
export class GlProgram {
  #gl: WebGL2RenderingContext;
  #vertSrc: string; #fragSrcBody: string;        // body WITHOUT #version/#define preamble
  #defines = new Map<string, number>();          // FEATURE_MIS=1, RANDOM_TYPE=2, ... (04-glsl-kernels §1)
  #program: WebGLProgram | null = null;
  #uniformLoc = new Map<string, WebGLUniformLocation | null>();
  #samplerUnit = new Map<string, number>();      // name → texture unit (assigned at link)
  #dirty = true;

  constructor(gl: WebGL2RenderingContext, vertSrc: string, fragSrcBody: string, defines: Record<string, number>) {
    this.#gl = gl; this.#vertSrc = vertSrc; this.#fragSrcBody = fragSrcBody;
    for (const [k, v] of Object.entries(defines)) this.#defines.set(k, v);
  }

  setDefine(name: string, value: number): boolean {   // returns true on actual change (change-gated, like MaterialBase:43-69)
    if (this.#defines.get(name) === value) return false;
    this.#defines.set(name, value); this.#dirty = true; return true;
  }

  #preamble(): string {
    let s = '#version 300 es\n';
    for (const [k, v] of this.#defines) s += `#define ${k} ${v}\n`;
    return s;
  }

  #relink(): void {
    const gl = this.#gl;
    if (this.#program) gl.deleteProgram(this.#program);
    const pre = this.#preamble();
    // fragment: #version + defines + `layout(location=0) out vec4 pc_fragColor;` + body
    const frag = `${pre}precision highp float;\nprecision highp int;\nlayout(location = 0) out vec4 pc_fragColor;\n${this.#fragSrcBody}`;
    const vert = `${pre}${this.#vertSrc}`;
    this.#program = linkProgram(gl, vert, frag);   // compileShader + attach + linkProgram + getProgramInfoLog throw
    this.#uniformLoc.clear(); this.#samplerUnit.clear();
    // discover uniforms + assign sampler units (sampler2D/usampler2D/sampler2DArray → sequential units)
    assignSamplerUnits(gl, this.#program, this.#uniformLoc, this.#samplerUnit);
    this.#dirty = false;
  }

  use(): void { if (this.#dirty || !this.#program) this.#relink(); this.#gl.useProgram(this.#program!); }
  setFloat(name: string, v: number): void { const l = this.#loc(name); if (l) this.#gl.uniform1f(l, v); }
  setInt(name: string, v: number): void   { const l = this.#loc(name); if (l) this.#gl.uniform1i(l, v); }
  setVec2/3/4, setMat4, setFloatArray ...  // 1:1 with the fork's uniform types (Vector2/3/4, Matrix4, Float32Array(81/82))
  bindTexture(name: string, tex: WebGLTexture, target = gl.TEXTURE_2D): void {
    const unit = this.#samplerUnit.get(name); if (unit == null) return;
    this.#gl.activeTexture(this.#gl.TEXTURE0 + unit); this.#gl.bindTexture(target, tex);
    this.#gl.uniform1i(this.#loc(name)!, unit);
  }
  #loc(name: string) { if (!this.#uniformLoc.has(name)) this.#uniformLoc.set(name, this.#gl.getUniformLocation(this.#program!, name)); return this.#uniformLoc.get(name) ?? null; }
}
```
The full set of `set*`/`bind*` methods is dictated by the fork uniform schema in [`04`](./04-glsl-kernels.md) §1 (≈55 uniforms across 11 groups). Recompiles must NOT fire every frame (they reset the accumulator) — that's why `setDefine` is change-gated and the runtime-toggled defines (`FEATURE_DOF`, `FEATURE_FOG`, `FEATURE_BACKGROUND_MAP`, `FEATURE_MIS`, `FEATURE_ADDITIVE_ACCUM`, `CAMERA_TYPE`) are set only on actual state change.

## 3. The three accumulation regimes (`blend.ts`) — verbatim from `PathTracingRenderer.js:18-53`

A WebGL2 path tracer accumulates by **blending each new sample into a float render target**. The fork has three regimes; we reproduce each as GL blend state:

```ts
// Regime 1 — ADDITIVE (FEATURE_ADDITIVE_ACCUM): buffer = SUM(rgb)/COUNT(alpha). Host clears to 0 first.
//   Needs EXT_float_blend. The unbiased HDR path used by the BDPT/spectral additive accumulation.
function setAdditive(gl) { gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE); }
//   (material opacity forced to 1; frag writes premultiplied sample, alpha=1 per sample → COUNT)

// Regime 2 — ALPHA-COMPOSITE (when !EXT_float_blend OR backgroundAlpha != 1): PT renders with NO blend into
//   _primaryTarget, then a BlendMaterial fullscreen quad composites into a ping-pong pair with opacity = 1/(samples+1).
function setNoBlend(gl) { gl.disable(gl.BLEND); }   // PT pass
//   blend quad: gl.enable(BLEND); blendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA); blend-quad alpha uniform = 1/(samples+1)

// Regime 3 — NORMAL running-average (default, float-blend present): material blends with source-alpha lerp,
//   opacity = 1/(samples+1).
function setNormal(gl) { gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); }
//   the PT frag's pc_fragColor.a = 1; the running mean emerges from opacity = 1/(samples+1).
```
Regime selection (verified `WebGLPathTracer.js:470-473` + `:10-14`): `needsAlphaComposite = backgroundAlpha !== 1 || !gl.getExtension('EXT_float_blend')`. Default S0–S2 path is **Regime 3** when `EXT_color_buffer_float` + `EXT_float_blend` are present (probe in `glCaps.ts`); fall back to Regime 2 otherwise. Regime 1 is used only for additive features (S3).

## 4. Render targets (`framebuffer.ts`) — verbatim formats from `PathTracingRenderer.js:271-290`

All accumulation RTs are **`RGBA32F`** (`format RGBAFormat, type FloatType`) with **`NEAREST`** min/mag. We allocate:
- `primaryTarget` — RGBA32F, the PT accumulation target.
- `blendTargets[0..1]` — RGBA32F ping-pong pair (Regime 2 only).
- MRT extras (when `supportsAuxBuffers`): `gNormalDepth` (RGBA32F, location 1) + `gAlbedo` (RGBA32F, location 2). The fork declares these as MRT outputs (`PhysicalPathTracingMaterial.js:244-245`); we attach them as `COLOR_ATTACHMENT1/2` and call `gl.drawBuffers([COLOR_ATTACHMENT0, ...1, ...2])`. On a non-MRT device, attach only attachment 0 and `drawBuffers([COLOR_ATTACHMENT0])` — the fork comment confirms locations 1/2 are "harmlessly ignored" then.

`reset()` clears all three RTs to `(0,0,0,0)` (verbatim `PathTracingRenderer.js:400-433`): bind each FBO, `gl.clearColor(0,0,0,0); gl.clear(COLOR_BUFFER_BIT)`.

## 5. `GlResources` — the resource owner (analog of pt-webgpu `GpuResources`)

```ts
export class GlResources {
  // owned GL objects (null until ensured) — analog of GpuResources' textures/buffers/pipelines
  accumFbo: WebGLFramebuffer | null = null;
  accumTex: WebGLTexture | null = null;
  blendFbo: [WebGLFramebuffer, WebGLFramebuffer] | null = null;
  blendTex: [WebGLTexture, WebGLTexture] | null = null;
  normalDepthTex: WebGLTexture | null = null;  albedoTex: WebGLTexture | null = null;
  accumWidth = 0; accumHeight = 0;
  ptProgram: GlProgram | null = null;          // the megakernel program (composeTraceGlsl)
  blendProgram: GlProgram | null = null;       // the BlendMaterial quad (Regime 2)
  displayProgram: GlProgram | null = null;     // ClampedInterpolation tonemap blit (final → host)
  paramsUbo: WebGLBuffer | null = null;        // std140 FrameParams UBO (WS5 frameParamsPacker target)
  #quad: FullscreenQuad;

  constructor(gl: WebGL2RenderingContext, traceTier: 'full'|'lite', bdpt: boolean) { /* probe caps, build quad */ }

  ensureAccumResources(w: number, h: number): boolean {   // returns `recreated` — caller resets sample counter
    if (w === this.accumWidth && h === this.accumHeight && this.accumFbo) return false;
    this.#destroyTargets(); /* alloc RGBA32F primary + (MRT aux) + (blend pair); */ this.accumWidth = w; this.accumHeight = h;
    return true;
  }
  ensureProgram(features: TraceFeatures): void { this.ptProgram ??= new GlProgram(gl, FULLSCREEN_VERT, composeTraceGlsl(features), DEFINES_FROM(features)); }
  clearAccum(): void { /* bind accumFbo + (blend pair) → clearColor(0,0,0,0)+clear, per PathTracingRenderer.reset */ }
  drawAccumStep(scene: UploadedSceneTextures, regime: Regime, seed: number): void { /* §6 */ }
  blitToDisplay(target: WebGLTexture, quality: FrameQualitySettings): WebGLTexture { /* tonemap pass → return display tex */ }
  dispose(): void { /* deleteFramebuffer/Texture/Buffer/Program for all owned objects */ }
  #destroyTargets(): void { ... }
}
```

## 6. The per-sample draw (`drawAccumStep`) — the inner loop (replaces `renderTask` generator)

The fork tiles (`tiles=3×3` default, per-tile scissor) to keep each draw short. For S0 we can render the full frame in one draw (no tiling); add tiling in WS5 if needed for long frames. One accumulation step:
```ts
drawAccumStep(scene, regime, seed) {
  const gl = this.gl; const prog = this.ptProgram!;
  bindFbo(gl, this.accumFbo, this.mrtDrawBuffers);
  gl.viewport(0, 0, this.accumWidth, this.accumHeight);
  setBlendForRegime(gl, regime, this.#samples);   // §3 — opacity/blend per regime
  prog.use();
  prog.setInt('seed', seed); prog.setFloat('opacity', 1 / (this.#samples + 1));  // running-average (Regime 3)
  bindAllSceneTextures(prog, scene);              // WS3 — bvh*, materials, attributesArray, lights, envMapInfo, ...
  bindParamsUbo(gl, prog, this.paramsUbo);        // WS5 — FrameParams std140 block
  this.#quad.draw(gl);                            // fullscreen triangle → fragment kernel per pixel
  this.#samples++;
}
```
Regime 2 adds the post blend-quad pass into the ping-pong pair (verbatim `PathTracingRenderer.js:158-165`). The readable result is `Regime 2 ? blendTex[1] : accumTex` (mirror `get target()` `:208-212`).

## 7. WS2 done-when
- A trivial `composeTraceGlsl` that outputs a constant color renders to `accumTex` and reads back nonzero (the program builder + FBO + quad path works end-to-end on the capture host).
- `setDefine` relinks; an unchanged `setDefine` does not (no per-frame recompile).
- All three blend regimes verified: additive sums; Regime 3 converges to a running mean over N draws (unit-test the math on a constant-shaded quad).
- `EXT_float_blend` absent → Regime 2 path selected automatically.
