/**
 * Browser-host helpers shared by the executable examples.
 *
 * The renderer contract uses physical pixels for the canvas backing store and
 * `FrameInput.viewport`, while CSS layout is expressed in logical pixels.
 * Keeping that conversion in one place prevents the examples from silently
 * diverging on high-DPI displays or after a browser-zoom / monitor-DPR change.
 */

export interface ExampleViewport {
  readonly width: number;
  readonly height: number;
  readonly devicePixelRatio: number;
  readonly resized: boolean;
}

export interface PerspectiveOptions {
  readonly fovY?: number;
  readonly near?: number;
  readonly far?: number;
}

export type ExampleVec3 = readonly [number, number, number];

export const CORNELL_CAMERA_POSITION: ExampleVec3 = [0, 1, 4];

function finitePositive(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function physicalDimension(
  clientDimension: number,
  backingDimension: number,
  devicePixelRatio: number,
): number {
  if (!(Number.isFinite(clientDimension) && clientDimension > 0)) {
    return Math.max(1, Math.floor(finitePositive(backingDimension, 1)));
  }
  const physical = Math.floor(clientDimension * devicePixelRatio);
  if (!Number.isSafeInteger(physical)) {
    throw new RangeError('example canvas physical dimension exceeds the safe-integer range');
  }
  return Math.max(1, physical);
}

/**
 * Synchronize a canvas backing store to its CSS size in physical pixels.
 *
 * Assigning `canvas.width` or `canvas.height` resets WebGL state even when the
 * assigned value is unchanged, so the properties are written only on a real
 * size transition. The returned `resized` bit is the host's signal to rebuild
 * size-dependent engine resources (and to reconfigure a WebGPU canvas).
 */
export function syncCanvasToDisplaySize(canvas: HTMLCanvasElement): ExampleViewport {
  const rawDpr = typeof window !== 'undefined' ? window.devicePixelRatio : 1;
  const devicePixelRatio = finitePositive(rawDpr, 1);
  const width = physicalDimension(canvas.clientWidth, canvas.width, devicePixelRatio);
  const height = physicalDimension(canvas.clientHeight, canvas.height, devicePixelRatio);
  const resized = canvas.width !== width || canvas.height !== height;
  if (resized) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, devicePixelRatio, resized };
}

/**
 * Write the OpenGL/WebGPU-compatible perspective convention used by the
 * examples into an existing column-major matrix.
 */
export function writePerspectiveProjection(
  out: Float32Array,
  width: number,
  height: number,
  options: PerspectiveOptions = {},
): Float32Array {
  if (out.length < 16) {
    throw new RangeError('perspective projection output must contain at least 16 elements');
  }
  const safeWidth = finitePositive(width, 1);
  const safeHeight = finitePositive(height, 1);
  const fovY = options.fovY ?? Math.PI / 3;
  const near = options.near ?? 0.1;
  const far = options.far ?? 100;
  if (!(Number.isFinite(fovY) && fovY > 0 && fovY < Math.PI)) {
    throw new RangeError('perspective fovY must be finite and between 0 and PI');
  }
  if (!(Number.isFinite(near) && near > 0 && Number.isFinite(far) && far > near)) {
    throw new RangeError('perspective clipping planes must satisfy 0 < near < far');
  }

  const f = 1 / Math.tan(fovY / 2);
  const aspect = safeWidth / safeHeight;
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

export function createPerspectiveProjection(
  width = 1,
  height = 1,
  options: PerspectiveOptions = {},
): Float32Array {
  return writePerspectiveProjection(new Float32Array(16), width, height, options);
}

/**
 * Create the view matrix for an axis-aligned camera looking down -Z.
 * Keeping the world-space position and inverse-world translation adjacent
 * avoids the sign disagreement that previously existed in two examples.
 */
export function createAxisAlignedView(cameraPosition: ExampleVec3): Float32Array {
  const tx = cameraPosition[0] === 0 ? 0 : -cameraPosition[0];
  const ty = cameraPosition[1] === 0 ? 0 : -cameraPosition[1];
  const tz = cameraPosition[2] === 0 ? 0 : -cameraPosition[2];
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1]);
}
