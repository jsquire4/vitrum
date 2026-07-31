import { WALKAROUND_UBO_SIZE_BYTES } from './constants.js';
import { RESERVOIR_DI_STRIDE_BYTES } from '../restir/reservoirDiLayout.js';
import { RESERVOIR_GI_STRIDE_BYTES } from '../gi/giLayout.js';

/**
 * Default steady-state ceiling for the resolution-dependent walkaround frame
 * graph. The renderer may choose a smaller internal resolution to stay below
 * this ceiling; presentation still targets the host's full swap-chain view.
 *
 * A transactional resize temporarily owns the old and candidate generations,
 * so its bounded peak is at most twice this value (768 MiB by default).
 */
export const DEFAULT_FRAME_RESOURCE_BUDGET_BYTES = 384 * 1024 * 1024;
/**
 * Deliberate quality ceiling for reservoir decimation. This is not inferred
 * from a WebGPU device limit: devices expose allocation limits, not an image-
 * quality threshold. Whole-frame resolution is reduced after scale 4.
 */
export const DEFAULT_MAX_RESTIR_RESERVOIR_SCALE = 4;

export type FrameResourceResolutionPolicy = 'auto' | 'native';

export interface FrameResourcePlanningOptions {
  readonly gtaoDownscale?: number;
  readonly gtaoEnabled?: boolean;
  readonly svgfEnabled?: boolean;
  readonly welfordPingPong?: boolean;
  readonly atrousVarianceEstimate?: boolean;
  readonly checkerboard?: boolean;
  /**
   * Integer reservoir-grid scale relative to the internal shading resolution.
   * DI uses floor(render/scale); GI uses floor(render/(2*scale)).
   */
  readonly reservoirScale?: number;
  /**
   * Optional host-owned ceiling for this frame graph's persistent logical GPU
   * bytes. WebGPU does not expose physical VRAM budgets, so a library cannot
   * manufacture a reliable universal default. When supplied, it is enforced
   * before the first allocation.
   */
  readonly maxPersistentBytes?: number;
}

export interface FrameResourceResolutionOptions extends FrameResourcePlanningOptions {
  /**
   * `auto` selects the largest aspect-preserving internal resolution supported
   * by the device and byte budget. `native` requires the exact requested
   * dimensions and rejects before allocation when they do not fit.
   */
  readonly resolutionPolicy?: FrameResourceResolutionPolicy;
  /**
   * Required dimension quantum for internal targets. Neural U-Net callers use
   * 8; the standard pipeline uses 1.
   */
  readonly dimensionAlignment?: number;
}

export interface FrameResourceAllocation {
  readonly label: string;
  readonly category: 'texture' | 'storage-buffer' | 'uniform-buffer';
  readonly bytes: number;
  readonly width?: number;
  readonly height?: number;
}

export interface FrameResourceFootprint {
  readonly width: number;
  readonly height: number;
  /** Exact logical bytes requested from WebGPU (driver padding is opaque). */
  readonly persistentBytes: number;
  readonly bytesPerFullResolutionPixel: number;
  readonly allocations: readonly FrameResourceAllocation[];
}

export interface ResolvedFrameResourcePlan {
  readonly requestedWidth: number;
  readonly requestedHeight: number;
  readonly effectiveWidth: number;
  readonly effectiveHeight: number;
  /** Requested/effective linear scale (1 means native). */
  readonly resolutionDownscale: number;
  /** ReSTIR-DI grid dimensions after the selected integer reservoir scale. */
  readonly restirDiWidth: number;
  readonly restirDiHeight: number;
  /** ReSTIR-GI grid dimensions after the 2x GI base stride and scale. */
  readonly restirGiWidth: number;
  readonly restirGiHeight: number;
  readonly restirReservoirScale: number;
  readonly policy: FrameResourceResolutionPolicy;
  readonly maxPersistentBytes: number;
  /** Steady-state bytes for the selected generation. */
  readonly persistentBytes: number;
  /**
   * Transactional replacement peak when the old generation remains live.
   * Callers supply the old generation's exact footprint.
   */
  readonly resizePeakBytes: number;
  readonly footprint: FrameResourceFootprint;
}

function checkedProduct(label: string, ...factors: number[]): number {
  let result = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(`[frame-resources] ${label} has invalid sizing factor ${factor}.`);
    }
    result *= factor;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`[frame-resources] ${label} exceeds safe-integer sizing.`);
    }
  }
  return result;
}

function assertDimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || width <= 0
      || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError(
      `[frame-resources] dimensions must be positive safe integers; got ${width}x${height}.`,
    );
  }
  const pixels = checkedProduct('pixel count', width, height);
  if (pixels > 0xffff_ffff) {
    throw new RangeError(
      `[frame-resources] ${width}x${height} has ${pixels} pixels, exceeding WGSL u32 indexing.`,
    );
  }
}

function assertBudget(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`[frame-resources] ${label} must be a positive safe integer.`);
  }
}

function assertAlignment(value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      '[frame-resources] dimensionAlignment must be a positive safe integer.',
    );
  }
}

export function assertFrameResourceReservoirScale(value: number): void {
  if (
    !Number.isSafeInteger(value)
    || value < 1
    || value > DEFAULT_MAX_RESTIR_RESERVOIR_SCALE
  ) {
    throw new RangeError(
      `[frame-resources] reservoirScale must be an integer in [1, ` +
      `${DEFAULT_MAX_RESTIR_RESERVOIR_SCALE}].`,
    );
  }
}

/**
 * Build the allocation ledger used by both diagnostics and the allocator.
 * Semantic aliases are listed only once, so this is the physical steady-state
 * footprint rather than a sum of public field names.
 */
export function planFrameResources(
  width: number,
  height: number,
  options: FrameResourcePlanningOptions = {},
): FrameResourceFootprint {
  assertDimensions(width, height);
  const pixels = checkedProduct('pixel count', width, height);
  const allocations: FrameResourceAllocation[] = [];
  const texture = (
    label: string,
    w: number,
    h: number,
    bytesPerTexel: number,
  ): void => {
    allocations.push({
      label,
      category: 'texture',
      bytes: checkedProduct(`${label} texture`, w, h, bytesPerTexel),
      width: w,
      height: h,
    });
  };
  const buffer = (
    label: string,
    bytes: number,
    category: 'storage-buffer' | 'uniform-buffer',
  ): void => {
    allocations.push({ label, category, bytes });
  };

  // Unique common rgba16float allocations. `resolved` aliases `combined`;
  // indirect à-trous aliases direct-pong + raw-indirect.
  for (const label of [
    'hdr-color',
    'hdr-indirect/indirect-atrous-pong',
    'combined/resolved',
    'transparent-composite',
    'hdr-total',
    'indirect-accum-a',
    'indirect-accum-b',
    'g-normal-depth',
    'direct-atrous-ping',
    'direct-atrous-pong/indirect-atrous-ping',
    'temporal-accum-a',
    'temporal-accum-b',
    'albedo',
  ]) texture(label, width, height, 8);
  texture('welford-a', width, height, 8);
  texture(
    'welford-b',
    options.welfordPingPong === false ? 1 : width,
    options.welfordPingPong === false ? 1 : height,
    8,
  );
  texture('motion-vectors', width, height, 8);
  texture(
    'checkerboard-snapshot',
    options.checkerboard === true ? width : 1,
    options.checkerboard === true ? height : 1,
    8,
  );
  texture('sample-tier', width, height, 4);
  const atrousVariance = options.atrousVarianceEstimate
    ?? options.welfordPingPong
    ?? true;
  texture(
    'atrous-variance-estimate',
    atrousVariance ? width : 1,
    atrousVariance ? height : 1,
    4,
  );
  buffer('walkaround-ubo', WALKAROUND_UBO_SIZE_BYTES, 'uniform-buffer');

  const reservoirScale = options.reservoirScale ?? 1;
  assertFrameResourceReservoirScale(reservoirScale);
  const diW = Math.max(1, Math.floor(width / reservoirScale));
  const diH = Math.max(1, Math.floor(height / reservoirScale));
  const diBytes = Math.max(
    256,
    checkedProduct('ReSTIR-DI reservoir', diW, diH, RESERVOIR_DI_STRIDE_BYTES),
  );
  for (const label of ['restir-di-current', 'restir-di-previous', 'restir-di-spatial']) {
    buffer(label, diBytes, 'storage-buffer');
  }
  const halfW = Math.max(1, Math.floor(width / (2 * reservoirScale)));
  const halfH = Math.max(1, Math.floor(height / (2 * reservoirScale)));
  const giBytes = Math.max(
    256,
    checkedProduct('ReSTIR-GI reservoir', halfW, halfH, RESERVOIR_GI_STRIDE_BYTES),
  );
  for (const label of ['restir-gi-current', 'restir-gi-previous', 'restir-gi-spatial']) {
    buffer(label, giBytes, 'storage-buffer');
  }

  const gtaoEnabled = options.gtaoEnabled ?? true;
  const downscale = Math.max(1, Math.floor(options.gtaoDownscale ?? 2));
  texture(
    'gtao-low',
    gtaoEnabled ? Math.max(1, Math.floor(width / downscale)) : 1,
    gtaoEnabled ? Math.max(1, Math.floor(height / downscale)) : 1,
    8,
  );
  texture('gtao-full', gtaoEnabled ? width : 1, gtaoEnabled ? height : 1, 8);
  buffer('gtao-ubo', 96, 'uniform-buffer');

  texture('ddgi-placeholder-irr', 1, 1, 8);
  texture('ddgi-placeholder-vis', 1, 1, 8);
  buffer('ddgi-ubo', 64, 'uniform-buffer');

  const svgfW = options.svgfEnabled === false ? 1 : width;
  const svgfH = options.svgfEnabled === false ? 1 : height;
  for (const label of [
    'svgf-current-object-id',
    'svgf-previous-object-id',
    'svgf-history-length-a',
    'svgf-history-length-b',
  ]) texture(label, svgfW, svgfH, 4);
  texture('svgf-prev-normal-depth', svgfW, svgfH, 8);
  texture('svgf-moments-a', svgfW, svgfH, 16);
  texture('svgf-moments-b', svgfW, svgfH, 16);
  texture('svgf-prev-radiance-a', svgfW, svgfH, 8);
  texture('svgf-prev-radiance-b', svgfW, svgfH, 8);
  texture('svgf-variance', svgfW, svgfH, 4);
  texture('svgf-variance-intermediate', svgfW, svgfH, 4);

  const persistentBytes = allocations.reduce((sum, item) => sum + item.bytes, 0);
  if (!Number.isSafeInteger(persistentBytes)) {
    throw new RangeError('[frame-resources] total persistent footprint exceeds safe-integer sizing.');
  }
  return Object.freeze({
    width,
    height,
    persistentBytes,
    bytesPerFullResolutionPixel: persistentBytes / pixels,
    allocations: Object.freeze(allocations),
  });
}

function reportedLimit(device: GPUDevice, name: string): number | undefined {
  const value = (device.limits as unknown as Record<string, unknown> | undefined)?.[name];
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

/** Validate every allocation and optional host budget before GPU mutation. */
export function assertFrameResourcePlanSupported(
  device: GPUDevice,
  plan: FrameResourceFootprint,
  options: FrameResourcePlanningOptions = {},
): void {
  const dimensionLimit = reportedLimit(device, 'maxTextureDimension2D');
  if (
    dimensionLimit !== undefined
    && (plan.width > dimensionLimit || plan.height > dimensionLimit)
  ) {
    throw new RangeError(
      `[frame-resources] ${plan.width}x${plan.height} exceeds ` +
      `device maxTextureDimension2D=${dimensionLimit}; no resources were allocated.`,
    );
  }
  const maxBufferSize = reportedLimit(device, 'maxBufferSize');
  const maxStorageBindingSize = reportedLimit(device, 'maxStorageBufferBindingSize');
  for (const allocation of plan.allocations) {
    if (allocation.category !== 'storage-buffer') continue;
    if (maxBufferSize !== undefined && allocation.bytes > maxBufferSize) {
      throw new RangeError(
        `[frame-resources] ${allocation.label} requires ${allocation.bytes} bytes, ` +
        `exceeding device maxBufferSize=${maxBufferSize}; no resources were allocated.`,
      );
    }
    if (
      maxStorageBindingSize !== undefined
      && allocation.bytes > maxStorageBindingSize
    ) {
      throw new RangeError(
        `[frame-resources] ${allocation.label} requires ${allocation.bytes} bytes, ` +
        `exceeding device maxStorageBufferBindingSize=${maxStorageBindingSize}; ` +
        `no resources were allocated.`,
      );
    }
  }
  const budget = options.maxPersistentBytes;
  if (budget !== undefined) {
    if (!Number.isSafeInteger(budget) || budget <= 0) {
      throw new RangeError('[frame-resources] maxPersistentBytes must be a positive safe integer.');
    }
    if (plan.persistentBytes > budget) {
      throw new RangeError(
        `[frame-resources] ${plan.width}x${plan.height} requires ` +
        `${plan.persistentBytes} persistent logical bytes ` +
        `(${plan.bytesPerFullResolutionPixel.toFixed(2)} B/pixel), exceeding ` +
        `host maxPersistentBytes=${budget}; no resources were allocated.`,
      );
    }
  }
}

function dimensionsAtMajorAxis(
  requestedWidth: number,
  requestedHeight: number,
  major: number,
  alignment: number,
): readonly [number, number] {
  const landscape = requestedWidth >= requestedHeight;
  const requestedMajor = landscape ? requestedWidth : requestedHeight;
  const requestedMinor = landscape ? requestedHeight : requestedWidth;
  const alignedMajor = Math.max(
    alignment,
    Math.min(requestedMajor, Math.floor(major / alignment) * alignment),
  );
  const proportionalMinor = Math.max(
    alignment,
    Math.floor((alignedMajor * requestedMinor) / requestedMajor / alignment) * alignment,
  );
  const alignedMinor = Math.min(requestedMinor, proportionalMinor);
  return landscape
    ? [alignedMajor, alignedMinor]
    : [alignedMinor, alignedMajor];
}

/**
 * Resolve a requested presentation-sized render target to a deterministic,
 * bounded internal frame graph.
 *
 * The search is over the major axis and is monotonic because every planned
 * allocation is non-decreasing with width/height. `auto` therefore returns the
 * largest aligned aspect-preserving candidate that satisfies both the device's
 * reported WebGPU limits and the persistent-byte ceiling. `native` never
 * degrades silently: it either returns the exact request or throws the exact
 * allocation/limit violation before the first GPU mutation.
 */
export function resolveFrameResourcePlan(
  device: GPUDevice,
  requestedWidth: number,
  requestedHeight: number,
  options: FrameResourceResolutionOptions = {},
  currentPersistentBytes = 0,
): ResolvedFrameResourcePlan {
  assertDimensions(requestedWidth, requestedHeight);
  if (!Number.isSafeInteger(currentPersistentBytes) || currentPersistentBytes < 0) {
    throw new RangeError(
      '[frame-resources] currentPersistentBytes must be a non-negative safe integer.',
    );
  }
  const maxPersistentBytes =
    options.maxPersistentBytes ?? DEFAULT_FRAME_RESOURCE_BUDGET_BYTES;
  assertBudget(maxPersistentBytes, 'maxPersistentBytes');
  const alignment = options.dimensionAlignment ?? 1;
  assertAlignment(alignment);
  if (requestedWidth < alignment || requestedHeight < alignment) {
    throw new RangeError(
      `[frame-resources] requested ${requestedWidth}x${requestedHeight} cannot satisfy ` +
      `dimensionAlignment=${alignment}.`,
    );
  }
  const policy = options.resolutionPolicy ?? 'auto';
  if (policy !== 'auto' && policy !== 'native') {
    throw new TypeError(
      `[frame-resources] resolutionPolicy must be "auto" or "native"; got ${String(policy)}.`,
    );
  }

  const explicitReservoirScale = options.reservoirScale;
  if (explicitReservoirScale !== undefined) {
    assertFrameResourceReservoirScale(explicitReservoirScale);
  }
  const checkedOptions: FrameResourcePlanningOptions = {
    ...options,
    maxPersistentBytes,
  };
  const verify = (
    width: number,
    height: number,
    reservoirScale: number,
  ): FrameResourceFootprint => {
    const scaledOptions = { ...checkedOptions, reservoirScale };
    const footprint = planFrameResources(width, height, scaledOptions);
    assertFrameResourcePlanSupported(device, footprint, scaledOptions);
    return footprint;
  };
  const resolveAtDimensions = (
    width: number,
    height: number,
  ): readonly [FrameResourceFootprint, number] => {
    if (explicitReservoirScale !== undefined) {
      return [verify(width, height, explicitReservoirScale), explicitReservoirScale];
    }
    let lastError: unknown;
    for (
      let reservoirScale = 1;
      reservoirScale <= DEFAULT_MAX_RESTIR_RESERVOIR_SCALE;
      reservoirScale += 1
    ) {
      try {
        return [verify(width, height, reservoirScale), reservoirScale];
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
  };

  let footprint: FrameResourceFootprint;
  let restirReservoirScale: number;
  if (policy === 'native') {
    if (requestedWidth % alignment !== 0 || requestedHeight % alignment !== 0) {
      throw new RangeError(
        `[frame-resources] native ${requestedWidth}x${requestedHeight} does not satisfy ` +
        `dimensionAlignment=${alignment}; no resources were allocated.`,
      );
    }
    [footprint, restirReservoirScale] = resolveAtDimensions(
      requestedWidth,
      requestedHeight,
    );
  } else {
    const requestedMajor = Math.max(requestedWidth, requestedHeight);
    const minimum = dimensionsAtMajorAxis(
      requestedWidth,
      requestedHeight,
      alignment,
      alignment,
    );
    // Produce the exact lower-bound error (budget or device limit) instead of
    // returning an unusable zero-sized graph.
    try {
      resolveAtDimensions(minimum[0], minimum[1]);
    } catch {
      // Re-run the most permissive exact candidate outside the search so the
      // caller receives its precise byte/device-limit error.
      verify(
        minimum[0],
        minimum[1],
        explicitReservoirScale ?? DEFAULT_MAX_RESTIR_RESERVOIR_SCALE,
      );
    }
    let bestWidth = minimum[0];
    let bestHeight = minimum[1];
    let low = alignment;
    let high = requestedMajor;
    while (low <= high) {
      const midpoint = Math.floor((low + high) / (2 * alignment)) * alignment;
      if (midpoint < alignment) break;
      const [candidateWidth, candidateHeight] = dimensionsAtMajorAxis(
        requestedWidth,
        requestedHeight,
        midpoint,
        alignment,
      );
      try {
        resolveAtDimensions(candidateWidth, candidateHeight);
        bestWidth = candidateWidth;
        bestHeight = candidateHeight;
        low = midpoint + alignment;
      } catch {
        high = midpoint - alignment;
      }
    }
    // Resolve once more at the selected dimensions so the footprint and scale
    // remain an inseparable pair even when adjacent major-axis probes collapse
    // to the same aligned width/height.
    [footprint, restirReservoirScale] = resolveAtDimensions(
      bestWidth,
      bestHeight,
    );
  }

  const resolutionDownscale = Math.max(
    requestedWidth / footprint.width,
    requestedHeight / footprint.height,
  );
  const resizePeakBytes = checkedProduct(
    'transactional resize peak',
    currentPersistentBytes + footprint.persistentBytes,
  );
  return Object.freeze({
    requestedWidth,
    requestedHeight,
    effectiveWidth: footprint.width,
    effectiveHeight: footprint.height,
    resolutionDownscale,
    restirDiWidth: Math.max(
      1,
      Math.floor(footprint.width / restirReservoirScale),
    ),
    restirDiHeight: Math.max(
      1,
      Math.floor(footprint.height / restirReservoirScale),
    ),
    restirGiWidth: Math.max(
      1,
      Math.floor(footprint.width / (2 * restirReservoirScale)),
    ),
    restirGiHeight: Math.max(
      1,
      Math.floor(footprint.height / (2 * restirReservoirScale)),
    ),
    restirReservoirScale,
    policy,
    maxPersistentBytes,
    persistentBytes: footprint.persistentBytes,
    resizePeakBytes,
    footprint,
  });
}
