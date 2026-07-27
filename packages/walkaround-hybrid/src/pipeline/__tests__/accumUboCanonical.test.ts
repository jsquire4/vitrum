import { describe, expect, it, vi } from 'vitest';
import {
  TEMPORAL_ACCUM_UBO_SIZE_BYTES,
  packTemporalAccumUniforms,
} from '@vitrum/shared-denoisers';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { writeAccumUbo, type UboRef } from '../bindGroupBuilders.js';

installWebGPUPolyfills();

describe('temporal accumulation UBO canonical packer', () => {
  it('allocates and uploads the shared-denoisers byte contract', () => {
    const buffer = {} as GPUBuffer;
    const createBuffer = vi.fn(() => buffer);
    const writeBuffer = vi.fn();
    const device = {
      createBuffer,
      queue: { writeBuffer },
    } as unknown as GPUDevice;
    const ref: UboRef = { buf: undefined };

    expect(writeAccumUbo(device, ref, 0.125)).toBe(buffer);
    expect(createBuffer).toHaveBeenCalledWith({
      size: TEMPORAL_ACCUM_UBO_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const expected = new ArrayBuffer(TEMPORAL_ACCUM_UBO_SIZE_BYTES);
    packTemporalAccumUniforms({ alpha: 0.125 }, expected);
    expect(writeBuffer).toHaveBeenCalledWith(buffer, 0, expected);
  });
});
