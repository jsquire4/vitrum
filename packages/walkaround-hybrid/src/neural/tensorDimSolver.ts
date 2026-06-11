/**
 * tensorDimSolver.ts — PURE per-layer tensor-dimension solver for the U-Net
 * inference graph.
 *
 * Extracted from InferenceGraph (Task 4.5 Theme I) so the forward-shape
 * simulation is a stand-alone pure function: given a {@link UNetSpec} and the
 * frame (W, H) it returns the `(H, W, C)` of every named tensor in the graph.
 *
 * This is the ONLY place the graph's tensor shapes are derived. It has no GPU
 * dependency; InferenceGraph calls {@link computeTensorDims} ONCE at
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
export function computeTensorDims(spec: UNetSpec, W: number, H: number): Map<string, TensorDims> {
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
        const s = layer.params.stride ?? 2;
        // For kH=2, stride=2, padding=0: outH = inH * stride.
        outH = outH * s;
        outW = outW * s;
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

      case 'bilinearUpsample':
        outH = outH * 2;
        outW = outW * 2;
        break;
    }

    dims.set(layer.output, { H: outH, W: outW, C: outC });
  }

  return dims;
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
 * Pack the uniform buffer data for a layer. Returns an ArrayBuffer (32 bytes =
 * 8 u32) that is written to the layer's uniform buffer. Pure: depends only on
 * the layer spec, the (already-computed) tensor-dim map, and the frame (H, W)
 * fallback. Byte-identical to the pre-extraction `_packUniform`.
 */
export function packLayerUniform(
  layer: LayerSpec,
  tensorDimsMap: Map<string, TensorDims>,
  H: number,
  W: number,
): ArrayBuffer {
  const u32 = new Uint32Array(8); // 32 bytes = 8 u32
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
      break;

    case 'relu':
    case 'skipAdd':
    case 'bilinearUpsample': {
      const count = (inDims?.H ?? H) * (inDims?.W ?? W) * (inDims?.C ?? layer.params.inC);
      u32[0] = count;
      // remaining fields: 0 (padding)
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
export function dispatchWorkgroupsFor(kind: LayerKind, dims: TensorDims): [number, number, number] {
  switch (kind) {
    case 'conv2d':
    case 'transposedConv2d':
    case 'bilinearUpsample':
      return [
        Math.ceil(dims.H / 8),
        Math.ceil(dims.W / 8),
        dims.C,
      ];
    case 'relu':
    case 'skipAdd':
      return [Math.ceil((dims.H * dims.W * dims.C) / 256), 1, 1];
    default:
      return [1, 1, 1];
  }
}
