/**
 * PPG training — extra @group(3) storage for path-guiding record pass.
 *
 * Binds after DDGI slots 0–3. Fragment is injected into SHADE_WGSL only when
 * `ppgEnabled` at pipeline init (see pipelineCompiler).
 *
 * Records one training sample per shaded pixel (up to buffer capacity) using
 * an atomic slot counter. `ppgUpdateKernel` consumes non-zero-luminance slots.
 */

export const SHADE_PPG_TRAIN_STRUCTS = /* wgsl */`
struct PPGTrainPathSample {
  worldPos:    vec3f,
  _pad:        f32,
  incidentDir: vec3f,
  _pad2:       f32,
  radiance:    vec3f,
  _pad3:       f32,
};
@group(3) @binding(4) var<storage, read_write> ppgTrainSamples: array<PPGTrainPathSample>;
@group(3) @binding(5) var<storage, read_write> ppgTrainHead: array<atomic<u32>>;
`;

/** Inserted immediately before hdrColorOut store in shadeMain (after `combined` is final). */
export const SHADE_PPG_TRAIN_RECORD = /* wgsl */`
  let ppgI = atomicAdd(&ppgTrainHead[0], 1u);
  let ppgLim = arrayLength(&ppgTrainSamples);
  if (ppgI < ppgLim) {
    ppgTrainSamples[ppgI].worldPos = pos;
    ppgTrainSamples[ppgI]._pad = 0.0;
    ppgTrainSamples[ppgI].incidentDir = wo;
    ppgTrainSamples[ppgI]._pad2 = 0.0;
    ppgTrainSamples[ppgI].radiance = combined;
    ppgTrainSamples[ppgI]._pad3 = 0.0;
  }
`;

/** Inject PPG structs/bindings after DDGI grid UBO line. */
export function injectPpgBindingsIntoShadeWgsl(shadeWgsl: string): string {
  const anchor = '@group(3) @binding(3) var<uniform> ddgiGrid: DDGIGridUBO;';
  if (!shadeWgsl.includes(anchor)) {
    throw new Error('[shade PPG] anchor for DDGI grid UBO not found');
  }
  return shadeWgsl.replace(anchor, `${anchor}\n${SHADE_PPG_TRAIN_STRUCTS}`);
}

/** Append atomic training record before final hdr store. */
export function injectPpgRecordBeforeHdrStore(shadeWgsl: string): string {
  const needle = 'textureStore(hdrColorOut, gid.xy, vec4f(combined, 1.0));';
  if (!shadeWgsl.includes(needle)) {
    throw new Error('[shade PPG] hdrColorOut store anchor not found');
  }
  return shadeWgsl.replace(
    needle,
    `${SHADE_PPG_TRAIN_RECORD}\n  ${needle}`,
  );
}
