/**
 * nrcDispose.test.ts — host-owns-lifecycle pin for the NRC GPU-buffer cleanup.
 *
 * TASK 0.2 correctness fix: `FusedMlpTrainer` allocates ~18 GPU buffers in
 * build() and previously had NO dispose path — they leaked until device
 * teardown, and `NrcSubsystem.dispose()` carried a comment admitting it could
 * not release them. This test pins:
 *
 *   1. FusedMlpTrainer.build() allocates exactly the 18 buffers we enumerate,
 *      and dispose() calls .destroy() on every one EXACTLY ONCE.
 *   2. dispose() is IDEMPOTENT — a second call neither throws nor double-
 *      destroys.
 *   3. dispose() is safe to call before build() (never-built trainer).
 *   4. NrcSubsystem.dispose() forwards to trainer.dispose().
 */

import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import {
  FusedMlpTrainer,
  type FusedNetSpec,
  type FusedTrainerConfig,
} from '../src/neural/nrc/fusedMlpTrainer.js';
import { NrcSubsystem } from '../src/neural/nrc/nrcSubsystem.js';

installWebGPUPolyfills();

// A recording mock GPUDevice that hands back buffers whose `destroy` is a spy,
// and stubs out the shader-module / pipeline path build() also exercises. Every
// createBuffer() result is collected so the test can assert per-buffer destroy.
interface MockBuffer {
  size: number;
  usage: number;
  destroy: ReturnType<typeof vi.fn>;
}

function mockDevice(): { device: GPUDevice; buffers: MockBuffer[] } {
  const buffers: MockBuffer[] = [];
  const dev: Record<string, unknown> = {
    limits: { maxComputeWorkgroupStorageSize: 32768 },
    queue: { writeBuffer: () => {}, submit: () => {} },
    createBuffer: (desc: { size: number; usage: number }): MockBuffer => {
      const buf: MockBuffer = {
        size: desc.size,
        usage: desc.usage,
        destroy: vi.fn(),
      };
      buffers.push(buf);
      return buf;
    },
    createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: [] as GPUCompilationMessage[] }),
    }),
    createComputePipelineAsync: async () => ({
      getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({}),
  };
  return { device: dev as unknown as GPUDevice, buffers };
}

const SPEC: FusedNetSpec = { inW: 16, W: 16, outW: 3, hidden: 2 };
const CFG: FusedTrainerConfig = { useF16: false, tileB: 8 };
const NUM_SAMPLES = 32;

// The exact buffers build() allocates (see fusedMlpTrainer.ts build()). f32 path:
//   18 original compute storage buffers
// + 6 persistent UBOs now allocated in build() (_paramsUbo, _gradFinUboW/B/X,
//   _adamUboW/B) — previously created lazily in record*() per step.
// = 24 total. No downcast buffers (_downcastUboW/B) in the useF16=false path.
const EXPECTED_BUILD_BUFFER_COUNT = 24;

describe('FusedMlpTrainer.dispose()', () => {
  it('allocates 24 buffers in build() and destroys each exactly once', async () => {
    const { device, buffers } = mockDevice();
    const trainer = new FusedMlpTrainer(device, SPEC, CFG);
    await trainer.build(NUM_SAMPLES);

    expect(buffers.length).toBe(EXPECTED_BUILD_BUFFER_COUNT);

    trainer.dispose();
    for (const buf of buffers) {
      expect(buf.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('is idempotent — a second dispose() neither throws nor double-destroys', async () => {
    const { device, buffers } = mockDevice();
    const trainer = new FusedMlpTrainer(device, SPEC, CFG);
    await trainer.build(NUM_SAMPLES);

    trainer.dispose();
    expect(() => trainer.dispose()).not.toThrow();
    for (const buf of buffers) {
      expect(buf.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('is safe to call before build() (never-built trainer)', () => {
    const { device } = mockDevice();
    const trainer = new FusedMlpTrainer(device, SPEC, CFG);
    expect(() => trainer.dispose()).not.toThrow();
  });
});

describe('NrcSubsystem.dispose() forwards to the trainer', () => {
  it('calls trainer.dispose() so the trainer buffers are released', () => {
    const { device: _device } = mockDevice(); // available if NrcSubsystem constructor is tested directly
    // Build a subsystem without running the async initialize() (which would need
    // a far heavier device mock). Inject a trainer stub directly and assert the
    // dispose forwards. The subsystem's own buffer fields are undefined → the
    // `?.destroy()` guards make dispose() safe.
    const sub = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
    const disposeSpy = vi.fn();
    // _trainer is private; assign through an index cast for the test only.
    (sub as unknown as { _trainer: { dispose: () => void } })._trainer = {
      dispose: disposeSpy,
    };

    expect(() => sub.dispose()).not.toThrow();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
  });
});
