/**
 * OptionalSubsystemBindingState — manages the GPU placeholder and real-buffer
 * binding state for THREE optional subsystems that share a single
 * `hybridLayers` bind group:
 *
 *  1. **DDGI** (`_irrTex` / `_visTex` / `_placeholderUBO`) — irradiance +
 *     visibility atlases. `null` → 1×1 placeholder textures from
 *     {@link FrameResources.ddgi}. Placeholder UBO is cached so
 *     `setInputs(null)` never allocates a fresh `Float32Array(16)` when DDGI
 *     is disabled (HOT-1 fix, 2026-05-16 sweep).
 *
 *  2. **RC** (`_rcCascade0` / `_rcParamsBuffer` / `_rcCascade0Placeholder` /
 *     `_rcParamsPlaceholder`) — cascade-0 buffer + packed RCParams uniform.
 *     `null` → 16-byte storage placeholder + 64-byte UBO placeholder with
 *     `enabled = 0u` (W8 Phase 3, 2026-05-18).
 *
 *  3. **PPG** (`_ppgPlaceholder`) — shared 16-byte zeroed storage placeholder
 *     for the three sTree/dTree/dTreeOffsets slots when PPG is disabled (W9,
 *     2026-05-18). The placeholder is never dereferenced when PPG is off
 *     because gi-ris guards every dTree descent on `ubo.ppgEnabled == 1`.
 *
 * Extracted from {@link WalkaroundGPUPipeline} in the 2026-05-18 refactor
 * sweep. Public surface: `setInputs` called from `HybridLayeredStage`;
 * `buildBindGroup` called once per frame from `renderFrame`.
 */

import {
  buildHybridLayersBindGroup,
  buildShadeHybridLayersBindGroup,
  buildTransparentOitBindGroup,
  type NrcHybridLayerBindings,
} from './bindGroupBuilders.js';
import { RC_PARAMS_BYTE_SIZE } from '../rc/rcParamsLayout.generated.js';
import {
  buildDDGIPlaceholderUBO,
  type FrameResources,
} from './resourceManager.js';
import type { BGLCache } from '../bglTypes.js';
import type { PipelineSubsystem } from './PipelineSubsystem.js';
import type { GpuMemoryExternalSections } from './gpuMemoryEstimate.js';
import { cachedBindGroup, type PipelineResourceCache } from './PipelineResourceCache.js';
import { DDGI_GRID_UBO_BYTES } from '../ddgi/ddgiGridUbo.js';

interface DDGISetInputs {
  irradianceTex: GPUTexture;
  visibilityTex: GPUTexture;
  gridParams: ArrayBuffer;
}

/** W8 Phase 3 (2026-05-18) — RC cascade-0 + params handed to the shade
 *  pass alongside the DDGI atlas (same bind group, slots 4-5). */
interface RCSetInputs {
  /** Raw cascade-0 GPUBuffer from RCSubsystem (probeX·probeY·probeZ·rays × vec4f). */
  cascade0Buffer: GPUBuffer;
  /** Packed RCParams uniform bytes (64 bytes) — see packRCParams in HybridEngineRC.ts. */
  paramsBytes: ArrayBuffer;
}

export class OptionalSubsystemBindingState implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  /** DDGI inputs (layered hybrid). Null → placeholder textures. */
  private _irrTex: GPUTexture | null = null;
  private _visTex: GPUTexture | null = null;
  /** Cached DDGI placeholder UBO — reused by setInputs(null) so we don't
   *  allocate a fresh Float32Array(16) every frame when DDGI is disabled.
   *  Populated lazily on first setInputs(null) call. */
  private _placeholderUBO: Float32Array<ArrayBuffer> | null = null;

  /** W8 Phase 3 — RC cascade-0 buffer (host-supplied via setRCInputs) or
   *  the 16-byte placeholder created at first setRCInputs(null). */
  private _rcCascade0: GPUBuffer | null = null;
  /** Placeholder buffers for the RC slot when RC is disabled. WebGPU does
   *  not allow null bindings on a layout entry, so we create a 16-byte
   *  storage placeholder + a 64-byte UBO placeholder (enabled = 0u) on
   *  first need. Both live for the engine's lifetime; dispose() releases. */
  private _rcCascade0Placeholder: GPUBuffer | null = null;
  private _rcParamsPlaceholder:   GPUBuffer | null = null;
  /** Real RC params buffer when RC is enabled (host-allocated 64 bytes;
   *  contents rewritten by setRCInputs each frame). Null when RC disabled. */
  private _rcParamsBuffer: GPUBuffer | null = null;

  /** W9 guided sampling — shared 16-byte zeroed placeholder for the three PPG
   *  tree storage-buffer slots (sTree / dTree / dTreeOffsets) when PPG is
   *  disabled (or before the PPG buffers are allocated). WebGPU forbids null
   *  bindings, so a single read-only-storage placeholder backs all three. It
   *  is never dereferenced when PPG is off because gi-ris guards every dTree
   *  descent on `ubo.ppgEnabled == 1`. Owned here; dispose() releases it. */
  private _ppgPlaceholder: GPUBuffer | null = null;

  constructor(device: GPUDevice) {
    this._device = device;
  }

  /**
   * Bind a DDGI atlas. Pass `null` to revert to placeholder (no-DDGI
   * fallback). The shade pass continues to render with hardcoded sky color
   * when isDDGIWired() returns false (dimsX ≤ 1 in the placeholder UBO).
   *
   * Caller (HybridLayeredStage) provides:
   *  - irradianceTex / visibilityTex: GPUTexture with TEXTURE_BINDING usage.
   *  - gridParams: 64-byte ArrayBuffer matching the WGSL DDGIGridUniform
   *    layout (origin vec3 + spacing f32 + dims vec3u + pad u32 +
   *    irradianceAtlasW/H + visibilityAtlasW/H).
   */
  setInputs(inputs: DDGISetInputs | null, frameResources: FrameResources): void {
    if (inputs === null) {
      // Restore placeholder UBO (dims=1×1×1) so shade.wgsl's
      // isDDGIWired() check returns false and Lo_ddgi drops to zero.
      // Cache the placeholder to avoid allocating Float32Array(16) every
      // frame when DDGI is disabled (HOT-1 fix).
      const placeholder = this._placeholderUBO ?? buildDDGIPlaceholderUBO();
      this._device.queue.writeBuffer(
        frameResources.ddgi.ddgiUboBuffer,
        0,
        placeholder.buffer,
      );
      this._placeholderUBO = placeholder;
      this._irrTex = null;
      this._visTex = null;
    } else {
      if (
        !(inputs.gridParams instanceof ArrayBuffer) ||
        inputs.gridParams.byteLength !== DDGI_GRID_UBO_BYTES
      ) {
        throw new RangeError(
          `DDGI gridParams must be exactly ${DDGI_GRID_UBO_BYTES} bytes.`,
        );
      }
      this._device.queue.writeBuffer(
        frameResources.ddgi.ddgiUboBuffer,
        0,
        inputs.gridParams,
      );
      this._irrTex = inputs.irradianceTex;
      this._visTex = inputs.visibilityTex;
    }
  }

  /**
   * Bind RC cascade-0 inputs from {@link RCSubsystem}. Pass `null` when RC
   * is disabled or the subsystem has no scene yet — the bind group then
   * uses 16-byte / 64-byte placeholder buffers (the rcParams placeholder
   * has `enabled = 0u` so `sampleCascadeC0` short-circuits to vec3f(0)).
   *
   * Per-frame: when `inputs != null`, host code passes the cascade-0
   * GPUBuffer + a fresh `paramsBytes` ArrayBuffer (64 bytes) capturing the
   * current rcWeight + probe geometry. The buffer contents are written
   * via `device.queue.writeBuffer`.
   */
  setRCInputs(inputs: RCSetInputs | null): void {
    if (inputs === null) {
      this._rcCascade0 = null;
      if (this._rcParamsBuffer) {
        this._rcParamsBuffer.destroy();
        this._rcParamsBuffer = null;
      }
      // Placeholders are allocated lazily at first buildBindGroup() so we
      // don't allocate GPU buffers in setRCInputs(null) called before the
      // pipeline exists.
      return;
    }
    this._rcCascade0 = inputs.cascade0Buffer;
    if (this._rcParamsBuffer === null) {
      this._rcParamsBuffer = this._device.createBuffer({
        label: 'rc-params-ubo',
        size: RC_PARAMS_BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this._device.queue.writeBuffer(this._rcParamsBuffer, 0, inputs.paramsBytes);
  }

  private _ensureRCPlaceholders(): { cascade0: GPUBuffer; params: GPUBuffer } {
    if (this._rcCascade0Placeholder === null) {
      this._rcCascade0Placeholder = this._device.createBuffer({
        label: 'rc-cascade0-placeholder',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(this._rcCascade0Placeholder.getMappedRange()).set([0, 0, 0, 0]);
      this._rcCascade0Placeholder.unmap();
    }
    if (this._rcParamsPlaceholder === null) {
      this._rcParamsPlaceholder = this._device.createBuffer({
        label: 'rc-params-placeholder',
        size: RC_PARAMS_BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      // All-zero contents — including the `enabled` field at byte offset 28
      // (uint32 word index 7) so sampleCascadeC0 short-circuits.
      new Uint8Array(this._rcParamsPlaceholder.getMappedRange()).fill(0);
      this._rcParamsPlaceholder.unmap();
    }
    return { cascade0: this._rcCascade0Placeholder, params: this._rcParamsPlaceholder };
  }

  /** Lazily create the shared PPG read-only-storage placeholder (16 bytes,
   *  zeroed). Returned for any of the three PPG slots whose real buffer is
   *  absent (PPG disabled). */
  private _ensurePpgPlaceholder(): GPUBuffer {
    if (this._ppgPlaceholder === null) {
      this._ppgPlaceholder = this._device.createBuffer({
        label: 'ppg-tree-placeholder',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint8Array(this._ppgPlaceholder.getMappedRange()).fill(0);
      this._ppgPlaceholder.unmap();
    }
    return this._ppgPlaceholder;
  }

  /**
   * Build the hybrid-layers bind group for the current frame, falling back
   * to the placeholder textures when no host-supplied DDGI atlases have
   * been bound. Delegates to {@link buildHybridLayersBindGroup} so the
   * caller layout (binding slots 0..5) stays in one place.
   */
  buildBindGroup(
    device: GPUDevice,
    bglCache: BGLCache,
    frameResources: FrameResources,
    resourceCache?: PipelineResourceCache,
    nrcBindings?: NrcHybridLayerBindings,
  ): GPUBindGroup {
    const rcPh = this._ensureRCPlaceholders();
    // W9 — bind the real PPG tree buffers when PPG is enabled (FrameResources.ppg
    // is populated by allocatePPGResources), else the shared 16-byte placeholder.
    const ppgPh = this._ensurePpgPlaceholder();
    const ppg = frameResources.ppg;
    const rcCascade0Buffer = this._rcCascade0 ?? rcPh.cascade0;
    const rcParamsBuffer = this._rcParamsBuffer ?? rcPh.params;
    const ppgQueryArenaBuffer = ('queryArenaBuf' in ppg) ? ppg.queryArenaBuf : ppgPh;
    const irrTex = this._irrTex ?? frameResources.ddgi.ddgiPlaceholderRgba16f;
    const visTex = this._visTex ?? frameResources.ddgi.ddgiPlaceholderVisRgba16f;
    const build = (): GPUBindGroup => buildHybridLayersBindGroup(device, bglCache, {
      ddgiIrrTex:             this._irrTex,
      ddgiVisTex:             this._visTex,
      ddgiPlaceholderRgba16f: frameResources.ddgi.ddgiPlaceholderRgba16f,
      ddgiPlaceholderVisRgba16f: frameResources.ddgi.ddgiPlaceholderVisRgba16f,
      ddgiUboBuffer:          frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
      ppgQueryArenaBuffer,
      ...(nrcBindings ? { nrc: nrcBindings } : {}),
    }, resourceCache);
    return cachedBindGroup(resourceCache, 'per-frame:hybrid-layers', [
      irrTex,
      visTex,
      frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
      ppgQueryArenaBuffer,
      ...(nrcBindings
        ? [nrcBindings.inferenceArenaBuffer, nrcBindings.runtimeArenaBuffer, nrcBindings.configBuffer]
        : []),
    ], build);
  }

  buildShadeBindGroup(
    device: GPUDevice,
    bglCache: BGLCache,
    frameResources: FrameResources,
    resourceCache?: PipelineResourceCache,
  ): GPUBindGroup {
    const rcPh = this._ensureRCPlaceholders();
    const rcCascade0Buffer = this._rcCascade0 ?? rcPh.cascade0;
    const rcParamsBuffer = this._rcParamsBuffer ?? rcPh.params;
    const irrTex = this._irrTex ?? frameResources.ddgi.ddgiPlaceholderRgba16f;
    const visTex = this._visTex ?? frameResources.ddgi.ddgiPlaceholderVisRgba16f;
    // The shade hybrid-layers bind group binds bindings 0-5 only (no PPG
    // trees), so we pass the narrowed ShadeHybridLayersResources shape — no
    // more rcCascade0Buffer PPG-filler to satisfy an over-wide type (D5-7).
    const build = (): GPUBindGroup => buildShadeHybridLayersBindGroup(device, bglCache, {
      ddgiIrrTex:             this._irrTex,
      ddgiVisTex:             this._visTex,
      ddgiPlaceholderRgba16f: frameResources.ddgi.ddgiPlaceholderRgba16f,
      ddgiPlaceholderVisRgba16f: frameResources.ddgi.ddgiPlaceholderVisRgba16f,
      ddgiUboBuffer:          frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
    }, resourceCache);
    return cachedBindGroup(resourceCache, 'per-frame:shade-hybrid-layers', [
      irrTex,
      visTex,
      frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
    ], build);
  }

  buildTransparentOitBindGroup(
    device: GPUDevice,
    bglCache: BGLCache,
    frameResources: FrameResources,
    background: GPUTexture,
    output: GPUTexture,
    resourceCache?: PipelineResourceCache,
  ): GPUBindGroup {
    const rcPh = this._ensureRCPlaceholders();
    const rcCascade0Buffer = this._rcCascade0 ?? rcPh.cascade0;
    const rcParamsBuffer = this._rcParamsBuffer ?? rcPh.params;
    const irrTex = this._irrTex ?? frameResources.ddgi.ddgiPlaceholderRgba16f;
    const visTex = this._visTex ?? frameResources.ddgi.ddgiPlaceholderVisRgba16f;
    const build = (): GPUBindGroup => buildTransparentOitBindGroup(
      device,
      bglCache,
      resourceCache?.textureView(irrTex) ?? irrTex.createView(),
      resourceCache?.textureView(visTex) ?? visTex.createView(),
      frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
      resourceCache?.textureView(background) ?? background.createView(),
      resourceCache?.textureView(output) ?? output.createView(),
    );
    return cachedBindGroup(resourceCache, 'pass:transparent-oit', [
      irrTex,
      visTex,
      frameResources.ddgi.ddgiUboBuffer,
      rcCascade0Buffer,
      rcParamsBuffer,
      background,
      output,
    ], build);
  }

  gpuMemorySections(): GpuMemoryExternalSections {
    const section: Record<string, unknown> = {};
    const add = (name: string, resource: unknown): void => {
      if (resource != null) section[name] = resource;
    };

    add('rcCascade0Placeholder', this._rcCascade0Placeholder);
    add('rcParamsPlaceholder', this._rcParamsPlaceholder);
    add('rcParamsBuffer', this._rcParamsBuffer);
    add('ppgTreePlaceholder', this._ppgPlaceholder);

    return Object.keys(section).length === 0
      ? {}
      : { hybridBindingState: section };
  }

  /** Release held atlas references (the host owns those GPUTextures; we only
   *  drop our references). The cached placeholder UBO is a plain
   *  Float32Array — no GPU resource to release. RC placeholder + params
   *  GPUBuffers ARE owned here and get destroyed. */
  dispose(): void {
    this._irrTex = null;
    this._visTex = null;
    this._placeholderUBO = null;
    this._rcCascade0 = null;
    if (this._rcParamsBuffer) { this._rcParamsBuffer.destroy(); this._rcParamsBuffer = null; }
    if (this._rcCascade0Placeholder) { this._rcCascade0Placeholder.destroy(); this._rcCascade0Placeholder = null; }
    if (this._rcParamsPlaceholder) { this._rcParamsPlaceholder.destroy(); this._rcParamsPlaceholder = null; }
    if (this._ppgPlaceholder) { this._ppgPlaceholder.destroy(); this._ppgPlaceholder = null; }
  }
}
