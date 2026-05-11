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

/** Marker comment used to locate the PPG train-binding injection point. */
const TRAIN_BINDINGS_MARKER = '// @@PPG_TRAIN_BINDINGS_INSERT@@';
/** Marker comment used to locate the PPG training-record injection point. */
const RECORD_MARKER = '// @@PPG_RECORD_INSERT@@';

/** Inject PPG structs/bindings into shade.wgsl at the
 *  `// @@PPG_TRAIN_BINDINGS_INSERT@@` marker. */
export function injectPpgBindingsIntoShadeWgsl(shadeWgsl: string): string {
  if (!shadeWgsl.includes(TRAIN_BINDINGS_MARKER)) {
    throw new Error(
      `[shade PPG] expected marker "${TRAIN_BINDINGS_MARKER}" not found in shade.wgsl — ` +
        `re-add it after the DDGIGridUBO @group(3) binding so PPG train bindings can attach.`,
    );
  }
  return shadeWgsl.replace(
    TRAIN_BINDINGS_MARKER,
    `${TRAIN_BINDINGS_MARKER}\n${SHADE_PPG_TRAIN_STRUCTS}`,
  );
}

/** Append atomic training record at the `// @@PPG_RECORD_INSERT@@` marker. */
export function injectPpgRecordBeforeHdrStore(shadeWgsl: string): string {
  if (!shadeWgsl.includes(RECORD_MARKER)) {
    throw new Error(
      `[shade PPG] expected marker "${RECORD_MARKER}" not found in shade.wgsl — ` +
        `re-add it immediately before the final textureStore(hdrColorOut, ...) call.`,
    );
  }
  return shadeWgsl.replace(
    RECORD_MARKER,
    `${SHADE_PPG_TRAIN_RECORD}\n  ${RECORD_MARKER}`,
  );
}
