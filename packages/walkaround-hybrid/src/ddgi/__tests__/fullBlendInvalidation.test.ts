import { describe, expect, it, vi } from 'vitest';
import { SceneBvh } from '@vitrum/shared-bvh';
import { ProbeGrid } from '../probeGrid.js';
import {
  ProbeUpdatePass,
  submitProbeUpdateCommand,
} from '../probeUpdatePass.js';

type ProbeUpdatePassInternals = {
  _acknowledgeFullBlendStratum(
    generation: number,
    stride: number,
    offset: number,
    wasFullBlend: boolean,
  ): void;
};

function makePass(): ProbeUpdatePass {
  return new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
}

function acknowledge(
  pass: ProbeUpdatePass,
  generation: number,
  stride: number,
  offset: number,
): void {
  (pass as unknown as ProbeUpdatePassInternals)._acknowledgeFullBlendStratum(
    generation,
    stride,
    offset,
    true,
  );
}

function submitWith(
  finish: () => GPUCommandBuffer,
  submit: (commands: Iterable<GPUCommandBuffer>) => void,
  publish: () => void,
): void {
  submitProbeUpdateCommand(
    { finish },
    { submit } as Pick<GPUQueue, 'submit'>,
    publish,
  );
}

describe('DDGI full-blend invalidation generations', () => {
  it('arms and acknowledges every stratum for stride > 1', () => {
    const pass = makePass();
    pass.requestFullBlend(4);
    const generation = pass.fullBlendGeneration;
    const commandBuffer = {} as GPUCommandBuffer;

    expect(pass.captureFullBlendState()).toEqual({
      generation,
      stride: 4,
      pendingStrata: [0, 1, 2, 3],
    });

    for (let stratum = 0; stratum < 4; stratum++) {
      submitWith(
        () => commandBuffer,
        () => undefined,
        () => acknowledge(pass, generation, 4, stratum),
      );
      expect(pass.pendingFullBlendCount).toBe(3 - stratum);
    }

    expect(pass.pendingFullBlend).toBe(false);
    expect(pass.captureFullBlendState().pendingStrata).toEqual([]);
  });

  it('retains the exact pending set when encoder.finish throws', () => {
    const pass = makePass();
    pass.requestFullBlend(3);
    const before = pass.captureFullBlendState();
    const submit = vi.fn();
    const publish = vi.fn();

    expect(() => submitWith(
      () => { throw new Error('finish failed'); },
      submit,
      publish,
    )).toThrow('finish failed');

    expect(submit).not.toHaveBeenCalled();
    expect(publish).not.toHaveBeenCalled();
    expect(pass.captureFullBlendState()).toEqual(before);
  });

  it('retains the exact pending set when queue.submit throws', () => {
    const pass = makePass();
    pass.requestFullBlend(3);
    const before = pass.captureFullBlendState();
    const publish = vi.fn();

    expect(() => submitWith(
      () => ({} as GPUCommandBuffer),
      () => { throw new Error('submit failed'); },
      publish,
    )).toThrow('submit failed');

    expect(publish).not.toHaveBeenCalled();
    expect(pass.captureFullBlendState()).toEqual(before);
  });

  it('repeated invalidation supersedes the old generation and stale acks', () => {
    const pass = makePass();
    pass.requestFullBlend(4);
    const oldGeneration = pass.fullBlendGeneration;
    acknowledge(pass, oldGeneration, 4, 0);
    expect(pass.captureFullBlendState().pendingStrata).toEqual([1, 2, 3]);

    pass.requestFullBlend(3);
    const next = pass.captureFullBlendState();
    expect(next.generation).not.toBe(oldGeneration);
    expect(next).toEqual({
      generation: next.generation,
      stride: 3,
      pendingStrata: [0, 1, 2],
    });

    acknowledge(pass, oldGeneration, 4, 1);
    expect(pass.captureFullBlendState()).toEqual(next);
  });

  it('changing stride while invalidation is pending re-arms full coverage', () => {
    const pass = makePass();
    pass.setProbeUpdateDivisor(4);
    pass.requestFullBlend();
    const oldGeneration = pass.fullBlendGeneration;
    acknowledge(pass, oldGeneration, 4, 0);

    pass.setProbeUpdateDivisor(5);

    expect(pass.captureFullBlendState()).toEqual({
      generation: oldGeneration + 1,
      stride: 5,
      pendingStrata: [0, 1, 2, 3, 4],
    });
  });
});
