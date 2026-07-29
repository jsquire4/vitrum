/**
 * tensorDimSolver.ts — PURE per-layer tensor-dimension solver for the U-Net
 * inference graph.
 *
 * Extracted from InferenceGraph (Task 4.5 Theme I) so the forward-shape
 * simulation is a stand-alone pure function: given a {@link UNetSpec} and the
 * frame (W, H) it returns the `(H, W, C)` of every named tensor in the graph.
 *
 * This is the ONLY place the graph's tensor shapes are derived. It has no GPU
 * dependency; InferenceGraph calls {@link preflightTensorDims} ONCE at
 * `initialize()` and stores the result, then re-uses the stored map every frame
 * (the perf fix: the previous code recomputed this map per-layer per-frame
 * inside the dispatch loop). The arithmetic here is byte-identical to the
 * pre-extraction inline simulation — pinned by
 * `__tests__/inferenceGraphDimsOnce.test.ts`.
 */

import type { UNetSpec, LayerSpec, LayerKind } from './unetArchitecture.js';

export interface TensorDims {
  H: number;
  W: number;
  C: number;
}

function defaultConv2dPadding(kH: number, kW: number): number {
  return kH === 3 && kW === 3 ? 1 : 0;
}

/**
 * Compute tensor dimensions for every named tensor in the graph by simulating
 * the forward pass. Seeds the three input tensors (noisyColor / albedo /
 * normals) at H×W×3 then propagates through every layer.
 *
 * Pure: no device, no allocation, no side effects beyond the returned map.
 */
function computeTensorDimsUnchecked(spec: UNetSpec, W: number, H: number): Map<string, TensorDims> {
  const dims = new Map<string, TensorDims>();

  // Seed the three input tensors.
  dims.set('noisyColor', { H, W, C: 3 });
  dims.set('albedo',     { H, W, C: 3 });
  dims.set('normals',    { H, W, C: 3 });

  for (const layer of spec.layers) {
    const inDims = layer.inputs.length > 0 ? dims.get(layer.inputs[0]!) : undefined;
    if (!inDims && layer.kind !== 'inputPack') continue;

    let outH = inDims?.H ?? H;
    let outW = inDims?.W ?? W;
    let outC = layer.params.outC;

    switch (layer.kind) {
      case 'inputPack':
        // All three inputs are H×W×3; output is H×W×9.
        dims.set(layer.output, { H, W, C: 9 });
        continue;

      case 'conv2d': {
        const kH = layer.params.kH ?? 3;
        const kW = layer.params.kW ?? 3;
        const s  = layer.params.stride ?? 1;
        const p  = layer.params.padding ?? defaultConv2dPadding(kH, kW);
        outH = Math.floor((outH + 2 * p - kH) / s) + 1;
        outW = Math.floor((outW + 2 * p - kW) / s) + 1;
        break;
      }

      case 'transposedConv2d': {
        const kH = layer.params.kH ?? 2;
        const kW = layer.params.kW ?? 2;
        const s = layer.params.stride ?? 2;
        const p = layer.params.padding ?? 0;
        const d = layer.params.dilation ?? 1;
        const op = layer.params.outputPadding ?? 0;
        outH = (outH - 1) * s - 2 * p + d * (kH - 1) + op + 1;
        outW = (outW - 1) * s - 2 * p + d * (kW - 1) + op + 1;
        break;
      }

      case 'relu':
        // Same dims as input; in-place conceptually.
        outC = inDims!.C;
        break;

      case 'skipAdd':
        // Both inputs must have identical dims (validated by validateSkipShapes).
        outC = inDims!.C;
        break;

    }

    dims.set(layer.output, { H: outH, W: outW, C: outC });
  }

  return dims;
}
export function computeTensorDims(spec: UNetSpec, W: number, H: number): Map<string, TensorDims> {
  if (!Number.isSafeInteger(W) || !Number.isSafeInteger(H) || W <= 0 || H <= 0) {
    throw new RangeError(
      `[InferenceGraph] width and height must be positive safe integers; got ${W}x${H}`,
    );
  }
  if (spec.layers.length === 0) throw new Error('[InferenceGraph] U-Net spec must contain at least one layer');
  if (spec.inputChannels !== 9 || spec.outputChannels !== 3) {
    throw new Error(
      `[InferenceGraph] U-Net contract requires 9 input and 3 output channels; got ` +
      `${spec.inputChannels} and ${spec.outputChannels}`,
    );
  }

  const knownChannels = new Map<string, number>([
    ['noisyColor', 3], ['albedo', 3], ['normals', 3],
  ]);
  const layerNames = new Set<string>();
  let derivedParamCount = 0;

  for (const layer of spec.layers) {
    if (layer.name.length === 0 || layerNames.has(layer.name)) {
      throw new Error(`[InferenceGraph] duplicate or empty layer name '${layer.name}'`);
    }
    layerNames.add(layer.name);
    if (layer.output.length === 0 || knownChannels.has(layer.output)) {
      throw new Error(`[InferenceGraph] duplicate, reserved, or empty output tensor '${layer.output}'`);
    }

    const expectedArity = layer.kind === 'inputPack' ? 3 : layer.kind === 'skipAdd' ? 2 : 1;
    if (layer.inputs.length !== expectedArity) {
      throw new Error(
        `[InferenceGraph] ${layer.kind} layer '${layer.name}' must have exactly ` +
        `${expectedArity} input(s), got ${layer.inputs.length}`,
      );
    }
    const inputChannels = layer.inputs.map(name => {
      const channels = knownChannels.get(name);
      if (channels == null) {
        throw new Error(
          `[InferenceGraph] layer '${layer.name}' references missing or not-yet-produced input '${name}'`,
        );
      }
      return channels;
    });
    for (const [name, value] of [['inC', layer.params.inC], ['outC', layer.params.outC]] as const) {
      if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`[InferenceGraph] layer '${layer.name}' has invalid ${name}=${value}`);
      }
    }

    const primaryChannels = inputChannels[0]!;
    let outputChannels = layer.params.outC;
    if (layer.kind === 'inputPack') {
      const totalChannels = inputChannels.reduce((sum, value) => sum + value, 0);
      if (
        layer.params.inC !== totalChannels ||
        layer.params.outC !== spec.inputChannels ||
        layer.weightLayout !== 'none'
      ) {
        throw new Error(
          `[InferenceGraph] inputPack layer '${layer.name}' must pack ` +
          `${totalChannels} channels into ${spec.inputChannels} without weights`,
        );
      }
      outputChannels = totalChannels;
    } else if (layer.kind === 'conv2d' || layer.kind === 'transposedConv2d') {
      requireInputChannels(layer, primaryChannels);
      requireWeightLayout(layer, layer.kind === 'conv2d' ? 'OIKW' : 'IOKW');
      const kH = positiveLayerInteger(layer, 'kH', layer.params.kH ?? (layer.kind === 'conv2d' ? 3 : 2));
      const kW = positiveLayerInteger(layer, 'kW', layer.params.kW ?? (layer.kind === 'conv2d' ? 3 : 2));
      const stride = positiveLayerInteger(
        layer, 'stride', layer.params.stride ?? (layer.kind === 'conv2d' ? 1 : 2),
      );
      nonNegativeLayerInteger(
        layer, 'padding',
        layer.params.padding ?? (layer.kind === 'conv2d' ? defaultConv2dPadding(kH, kW) : 0),
      );
      if (layer.kind === 'transposedConv2d') {
        positiveLayerInteger(layer, 'dilation', layer.params.dilation ?? 1);
        const outputPadding = nonNegativeLayerInteger(
          layer, 'outputPadding', layer.params.outputPadding ?? 0,
        );
        if (outputPadding >= stride) {
          throw new RangeError(
            `[InferenceGraph] layer '${layer.name}' requires outputPadding < stride; ` +
            `got ${outputPadding} >= ${stride}`,
          );
        }
      } else if (layer.params.dilation !== undefined || layer.params.outputPadding !== undefined) {
        throw new Error(
          `[InferenceGraph] conv2d layer '${layer.name}' has transposed-convolution-only parameters`,
        );
      }
      derivedParamCount += layer.params.inC * layer.params.outC * kH * kW + layer.params.outC;
    } else {
      requireParameterlessLayer(layer, primaryChannels);
      if (layer.kind === 'skipAdd' && inputChannels[1] !== primaryChannels) {
        throw new Error(
          `[InferenceGraph] skipAdd layer '${layer.name}' channel mismatch: ` +
          `${primaryChannels} vs ${inputChannels[1]}`,
        );
      }
      outputChannels = primaryChannels;
    }
    knownChannels.set(layer.output, outputChannels);
  }

  if (derivedParamCount !== spec.paramCount) {
    throw new Error(
      `[InferenceGraph] spec.paramCount=${spec.paramCount} does not match derived ${derivedParamCount}`,
    );
  }
  const dims = computeTensorDimsUnchecked(spec, W, H);
  for (const layer of spec.layers) {
    const output = dims.get(layer.output);
    if (output == null) throw new Error(`[InferenceGraph] layer '${layer.name}' did not produce '${layer.output}'`);
    assertTensorDims(layer.output, output.H, output.W, output.C);
  }
  validateSkipShapes(spec, dims);
  const denoised = dims.get('denoised');
  if (
    denoised == null || denoised.H !== H || denoised.W !== W ||
    denoised.C !== spec.outputChannels ||
    spec.layers[spec.layers.length - 1]!.output !== 'denoised'
  ) {
    throw new Error(
      `[InferenceGraph] final graph layer must produce 'denoised' as ` +
      `[${H}×${W}×${spec.outputChannels}]`,
    );
  }
  return dims;
}

function positiveLayerInteger(
  layer: LayerSpec,
  name: 'kH' | 'kW' | 'stride' | 'dilation',
  value: number,
): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`[InferenceGraph] layer '${layer.name}' has invalid ${name}=${value}`);
  }
  return value;
}

function nonNegativeLayerInteger(
  layer: LayerSpec,
  name: 'padding' | 'outputPadding',
  value: number,
): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`[InferenceGraph] layer '${layer.name}' has invalid ${name}=${value}`);
  }
  return value;
}

function requireInputChannels(layer: LayerSpec, actual: number): void {
  if (layer.params.inC !== actual) {
    throw new Error(
      `[InferenceGraph] layer '${layer.name}' declares inC=${layer.params.inC}, input has C=${actual}`,
    );
  }
}

function requireWeightLayout(layer: LayerSpec, expected: 'OIKW' | 'IOKW'): void {
  if (layer.weightLayout !== expected) {
    throw new Error(
      `[InferenceGraph] layer '${layer.name}' must use weightLayout='${expected}', got '${layer.weightLayout}'`,
    );
  }
}

function requireParameterlessLayer(layer: LayerSpec, channels: number): void {
  requireInputChannels(layer, channels);
  if (layer.params.outC !== channels || layer.weightLayout !== 'none') {
    throw new Error(
      `[InferenceGraph] layer '${layer.name}' must preserve ${channels} channels and use no weights`,
    );
  }
  if (
    layer.params.kH !== undefined || layer.params.kW !== undefined ||
    layer.params.stride !== undefined || layer.params.padding !== undefined ||
    layer.params.dilation !== undefined || layer.params.outputPadding !== undefined
  ) {
    throw new Error(`[InferenceGraph] layer '${layer.name}' has illegal convolution parameters`);
  }
}

function assertTensorDims(name: string, H: number, W: number, C: number): void {
  for (const [axis, value] of [['H', H], ['W', W], ['C', C]] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`[InferenceGraph] tensor '${name}' has invalid ${axis}=${value}`);
    }
  }
  const elements = H * W * C;
  if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(elements * 4)) {
    throw new RangeError(`[InferenceGraph] tensor '${name}' byte size is not a safe integer`);
  }
}

/**
 * Solve and validate the complete graph shape before GPU allocation begins.
 * This is intentionally separate from the allocator: an invalid resolution
 * must not leave even a placeholder buffer or compiled pipeline behind.
 */
export function preflightTensorDims(
  spec: UNetSpec,
  W: number,
  H: number,
): Map<string, TensorDims> {
  if (!Number.isSafeInteger(W) || !Number.isSafeInteger(H) || W <= 0 || H <= 0) {
    throw new RangeError(
      `[InferenceGraph] width and height must be positive safe integers; got ${W}x${H}`,
    );
  }
  const tensorDimsMap = computeTensorDims(spec, W, H);
  validateSkipShapes(spec, tensorDimsMap);
  return tensorDimsMap;
}

/**
 * Validate that every skipAdd layer's two operands have matching (H, W, C).
 * Throws if any mismatch is detected. Called once at initialize() time.
 */
export function validateSkipShapes(spec: UNetSpec, tensorDimsMap: Map<string, TensorDims>): void {
  for (const layer of spec.layers) {
    if (layer.kind !== 'skipAdd') continue;
    if (layer.inputs.length !== 2) {
      throw new Error(
        `[InferenceGraph] skipAdd layer '${layer.name}' must have exactly 2 inputs, ` +
        `got ${layer.inputs.length}`,
      );
    }
    const a = tensorDimsMap.get(layer.inputs[0]!);
    const b = tensorDimsMap.get(layer.inputs[1]!);
    if (!a || !b) {
      throw new Error(
        `[InferenceGraph] skipAdd layer '${layer.name}' — ` +
        `input tensor not found: '${!a ? layer.inputs[0] : layer.inputs[1]}'`,
      );
    }
    if (a.H !== b.H || a.W !== b.W || a.C !== b.C) {
      throw new Error(
        `[InferenceGraph] skipAdd layer '${layer.name}' shape mismatch: ` +
        `'${layer.inputs[0]}' = [${a.H}×${a.W}×${a.C}] vs ` +
        `'${layer.inputs[1]}' = [${b.H}×${b.W}×${b.C}]`,
      );
    }
  }
}

/**
 * Pack the uniform buffer data for a layer. Returns an ArrayBuffer (48 bytes =
 * 12 u32) that is written to the layer's uniform buffer. Pure: depends only on
 * the layer spec, the (already-computed) tensor-dim map, and the frame (H, W)
 * fallback. Byte-identical to the pre-extraction `_packUniform`.
 */
export function packLayerUniform(
  layer: LayerSpec,
  tensorDimsMap: Map<string, TensorDims>,
  H: number,
  W: number,
  maxComputeWorkgroupsPerDimension = 65_535,
): ArrayBuffer {
  const u32 = new Uint32Array(12); // 48 bytes = 12 u32
  const inDims = layer.inputs.length > 0 ? tensorDimsMap.get(layer.inputs[0]!) : undefined;

  switch (layer.kind) {
    case 'conv2d':
      u32[0] = inDims?.H ?? H;
      u32[1] = inDims?.W ?? W;
      u32[2] = layer.params.inC;
      u32[3] = layer.params.outC;
      u32[4] = layer.params.kH ?? 3;
      u32[5] = layer.params.kW ?? 3;
      u32[6] = layer.params.stride ?? 1;
      u32[7] = layer.params.padding ?? defaultConv2dPadding(u32[4], u32[5]);
      break;

    case 'transposedConv2d':
      u32[0] = inDims?.H ?? H;
      u32[1] = inDims?.W ?? W;
      u32[2] = layer.params.inC;
      u32[3] = layer.params.outC;
      u32[4] = layer.params.kH ?? 2;
      u32[5] = layer.params.kW ?? 2;
      u32[6] = layer.params.stride ?? 2;
      u32[7] = layer.params.padding ?? 0;
      u32[8] = layer.params.dilation ?? 1;
      u32[9] = layer.params.outputPadding ?? 0;
      break;

    case 'relu':
    case 'skipAdd': {
      const count = (inDims?.H ?? H) * (inDims?.W ?? W) * (inDims?.C ?? layer.params.inC);
      u32[0] = count;
      const groups = elementwiseDispatch(count, maxComputeWorkgroupsPerDimension);
      u32[1] = groups[0];
      break;
    }

    default:
      break;
  }

  return u32.buffer;
}

/**
 * Workgroup dispatch layout for a layer kind + its output dims. Must match the
 * @compute entry points in neural/wgsl/*. Pure. Byte-identical to the
 * pre-extraction `_dispatchWorkgroups`.
 */
export function dispatchWorkgroupsFor(
  kind: LayerKind,
  dims: TensorDims,
  maxComputeWorkgroupsPerDimension = 65_535,
): [number, number, number] {
  switch (kind) {
    case 'conv2d':
    case 'transposedConv2d':
      return [
        Math.ceil(dims.H / 8),
        Math.ceil(dims.W / 8),
        dims.C,
      ];
    case 'relu':
    case 'skipAdd':
      return elementwiseDispatch(
        dims.H * dims.W * dims.C,
        maxComputeWorkgroupsPerDimension,
      );
    default:
      return [1, 1, 1];
  }
}

function elementwiseDispatch(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number,
): [number, number, number] {
  if (!Number.isSafeInteger(maxComputeWorkgroupsPerDimension) || maxComputeWorkgroupsPerDimension <= 0) {
    throw new RangeError('[neural] maxComputeWorkgroupsPerDimension must be a positive safe integer');
  }
  const totalGroups = Math.max(1, Math.ceil(elementCount / 256));
  const groupsX = Math.min(totalGroups, maxComputeWorkgroupsPerDimension);
  return [groupsX, Math.ceil(totalGroups / groupsX), 1];
}
