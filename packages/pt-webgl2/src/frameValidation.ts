import {
  resolveFrameCameraPosition,
  type FrameInput,
} from '@vitrum/core';
import { validateWebGl2FrameQuality } from './frameQuality.js';

const UINT32_MAX = 0xffff_ffff;

function assertPositiveSafeInteger(label: string, value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `${label} must be a positive safe integer (got ${String(value)})`,
    );
  }
}

export function validateWebGl2PixelSize(
  method: 'renderFrame' | 'setSize',
  width: unknown,
  height: unknown,
): void {
  assertPositiveSafeInteger(`${method}: width`, width);
  assertPositiveSafeInteger(`${method}: height`, height);
}

function assertFiniteArrayLike(
  label: string,
  value: ArrayLike<number> | null | undefined,
  expectedLength: number,
): void {
  if (value == null || typeof value.length !== 'number' || value.length !== expectedLength) {
    const actual = value == null || typeof value.length !== 'number'
      ? 'non-array-like'
      : String(value.length);
    throw new TypeError(
      `renderFrame: ${label} must contain exactly ${expectedLength} finite numbers (got length ${actual})`,
    );
  }
  for (let i = 0; i < expectedLength; i += 1) {
    if (!Number.isFinite(value[i])) {
      throw new TypeError(
        `renderFrame: ${label}[${i}] must be finite (got ${String(value[i])})`,
      );
    }
  }
}

/**
 * Validate all camera numerics before renderFrame allocates targets, compiles a
 * program, mutates accumulation state, or uploads uniforms. The nominal core
 * types are fixed-size, but JS hosts and deserialized payloads can bypass them.
 */
export function validateWebGl2FrameInput(input: FrameInput): void {
  const runtimeInput = input as FrameInput | null | undefined;
  if (runtimeInput == null || typeof runtimeInput !== 'object' || Array.isArray(runtimeInput)) {
    throw new TypeError('renderFrame: input must be a non-array object');
  }
  const viewport = runtimeInput.viewport as FrameInput['viewport'] | null | undefined;
  if (viewport == null || typeof viewport !== 'object' || Array.isArray(viewport)) {
    throw new TypeError('renderFrame: viewport must be a non-array object');
  }
  validateWebGl2PixelSize('renderFrame', viewport.width, viewport.height);
  if (
    typeof viewport.devicePixelRatio !== 'number' ||
    !Number.isFinite(viewport.devicePixelRatio) ||
    viewport.devicePixelRatio <= 0
  ) {
    throw new RangeError(
      `renderFrame: viewport.devicePixelRatio must be finite and > 0 ` +
        `(got ${String(viewport.devicePixelRatio)})`,
    );
  }
  if (
    !Number.isSafeInteger(runtimeInput.frameIndex) ||
    runtimeInput.frameIndex < 0 ||
    runtimeInput.frameIndex > UINT32_MAX
  ) {
    throw new RangeError(
      `renderFrame: frameIndex must be an unsigned 32-bit integer ` +
        `(got ${String(runtimeInput.frameIndex)})`,
    );
  }
  if (
    !Number.isSafeInteger(runtimeInput.frameSeed) ||
    runtimeInput.frameSeed < 0 ||
    runtimeInput.frameSeed > UINT32_MAX
  ) {
    throw new RangeError(
      `renderFrame: frameSeed must be an unsigned 32-bit integer ` +
        `(got ${String(runtimeInput.frameSeed)})`,
    );
  }
  assertFiniteArrayLike(
    'viewMatrix',
    runtimeInput.viewMatrix,
    16,
  );
  assertFiniteArrayLike(
    'projMatrix',
    runtimeInput.projMatrix,
    16,
  );
  if (runtimeInput.prevViewMatrix !== undefined) {
    assertFiniteArrayLike(
      'prevViewMatrix',
      runtimeInput.prevViewMatrix,
      16,
    );
  }
  if (runtimeInput.prevProjMatrix !== undefined) {
    assertFiniteArrayLike(
      'prevProjMatrix',
      runtimeInput.prevProjMatrix,
      16,
    );
  }
  if (runtimeInput.cameraPosition !== undefined) {
    assertFiniteArrayLike('cameraPosition', runtimeInput.cameraPosition, 3);
  }
  resolveFrameCameraPosition(runtimeInput, 'PTEngineWebGL2.renderFrame');
  validateWebGl2FrameQuality(runtimeInput.quality);
}
