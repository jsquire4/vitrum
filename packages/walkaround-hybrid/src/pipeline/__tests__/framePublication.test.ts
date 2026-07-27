import { describe, expect, it, vi } from 'vitest';

import {
  FramePublicationTransaction,
  finishSubmitAndPublishFrame,
} from '../FramePublication.js';

describe('FramePublicationTransaction', () => {
  it('publishes every accepted owner before reporting callback failures', () => {
    const publication = new FramePublicationTransaction();
    const calls: string[] = [];
    const frameError = new Error('frame callback failed');
    const nrcError = new Error('nrc callback failed');
    const committed = { denoiser: 0, gi: 0 };
    publication.stage(() => {
      calls.push('frame');
      throw frameError;
    });
    publication.stage(() => {
      calls.push('denoiser');
      committed.denoiser += 1;
    });
    publication.stage(() => {
      calls.push('nrc');
      throw nrcError;
    });
    publication.stage(() => {
      calls.push('gi');
      committed.gi += 1;
    });

    let thrown: unknown;
    try {
      publication.accept();
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(AggregateError);
    expect((thrown as AggregateError).errors).toEqual([frameError, nrcError]);
    expect((thrown as AggregateError).message).toBe(
      '2 frame-publication callback(s) failed after GPU submission',
    );
    expect(calls).toEqual(['frame', 'denoiser', 'nrc', 'gi']);
    expect(committed).toEqual({ denoiser: 1, gi: 1 });
    expect(publication.state).toBe('accepted');

    publication.accept();
    publication.abort();
    expect(calls).toEqual(['frame', 'denoiser', 'nrc', 'gi']);
    expect(committed).toEqual({ denoiser: 1, gi: 1 });
  });

  it('runs every abort callback in reverse order and reports every failure', () => {
    const publication = new FramePublicationTransaction();
    const calls: number[] = [];
    const firstError = new Error('first abort failed');
    const thirdError = new Error('third abort failed');
    publication.stage(() => undefined, () => {
      calls.push(0);
      throw firstError;
    });
    publication.stage(() => undefined, () => {
      calls.push(1);
    });
    publication.stage(() => undefined, () => {
      calls.push(2);
      throw thirdError;
    });

    let thrown: unknown;
    try {
      publication.abort();
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(AggregateError);
    expect((thrown as AggregateError).errors).toEqual([thirdError, firstError]);
    expect(calls).toEqual([2, 1, 0]);
    expect(publication.state).toBe('aborted');
    publication.abort();
    expect(calls).toEqual([2, 1, 0]);
  });

  it('preserves submit failure plus every abort failure', () => {
    const publication = new FramePublicationTransaction();
    const calls: number[] = [];
    const primary = new Error('submit failed');
    const firstAbortError = new Error('first abort failed');
    const lastAbortError = new Error('last abort failed');
    publication.stage(() => undefined, () => {
      calls.push(0);
      throw firstAbortError;
    });
    publication.stage(() => undefined, () => {
      calls.push(1);
    });
    publication.stage(() => undefined, () => {
      calls.push(2);
      throw lastAbortError;
    });

    let thrown: unknown;
    try {
      finishSubmitAndPublishFrame(
        { finish: () => ({}) as GPUCommandBuffer },
        { submit: () => { throw primary; } },
        publication,
      );
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(AggregateError);
    expect((thrown as AggregateError).errors).toEqual([
      primary,
      lastAbortError,
      firstAbortError,
    ]);
    expect(calls).toEqual([2, 1, 0]);
    expect(publication.state).toBe('aborted');
  });

  it.each(['finish', 'submit'] as const)(
    'keeps the complete render publication snapshot on %s failure and retries identical work',
    (failure) => {
      const initial = {
        frameCount: 19,
        accumPingPong: 1,
        cameraHistory: [2, 3, 5] as [number, number, number],
        denoiserHistory: 0,
        giReservoirPublication: 7,
        nrcTicketSequence: 12,
      };
      const state = structuredClone(initial);
      const destroyedTickets: number[] = [];
      const copies: Array<readonly [string, string]> = [];
      const recordFrame = (
        publication: FramePublicationTransaction,
        ticket: number,
      ) => {
        // These are the same source/target identities selected while encoding
        // the render: DI/GI reservoir history and denoiser ping-pong.
        copies.push(['di-current', 'di-previous']);
        copies.push(['gi-current', 'gi-previous']);
        copies.push([
          state.denoiserHistory === 0 ? 'denoiser-a' : 'denoiser-b',
          state.denoiserHistory === 0 ? 'denoiser-b' : 'denoiser-a',
        ]);
        publication.stage(() => {
          state.frameCount += 1;
          state.accumPingPong = 1 - state.accumPingPong;
          state.cameraHistory = [11, 13, 17];
        });
        publication.stage(() => {
          state.denoiserHistory = 1 - state.denoiserHistory;
        });
        publication.stage(() => {
          state.giReservoirPublication += 1;
        });
        publication.stage(
          () => {
            state.nrcTicketSequence = ticket;
          },
          () => {
            destroyedTickets.push(ticket);
          },
        );
      };

      const first = new FramePublicationTransaction();
      recordFrame(first, 13);
      const commandBuffer = { label: 'frame-command-buffer' } as GPUCommandBuffer;
      const encoder = {
        finish: vi.fn(() => {
          if (failure === 'finish') throw new Error('injected finish failure');
          return commandBuffer;
        }),
      };
      const queue = {
        submit: vi.fn(() => {
          if (failure === 'submit') throw new Error('injected submit failure');
        }),
      };

      expect(() => finishSubmitAndPublishFrame(
        encoder as Pick<GPUCommandEncoder, 'finish'>,
        queue as Pick<GPUQueue, 'submit'>,
        first,
      )).toThrow(`injected ${failure} failure`);
      expect(state).toEqual(initial);
      expect(first.state).toBe('aborted');
      expect(destroyedTickets).toEqual([13]);
      const failedSelection = copies.slice();

      const retry = new FramePublicationTransaction();
      recordFrame(retry, 13);
      const retryEncoder = { finish: vi.fn(() => commandBuffer) };
        const retryQueue = { submit: vi.fn() };
        finishSubmitAndPublishFrame(
          retryEncoder,
          retryQueue,
          retry,
      );
      expect(copies.slice(failedSelection.length)).toEqual(failedSelection);
      expect(state).toEqual({
        frameCount: 20,
        accumPingPong: 0,
        cameraHistory: [11, 13, 17],
        denoiserHistory: 1,
        giReservoirPublication: 8,
        nrcTicketSequence: 13,
      });
      expect(retry.state).toBe('accepted');
      expect(retryQueue.submit).toHaveBeenCalledOnce();
      expect(retryQueue.submit).toHaveBeenCalledWith([commandBuffer]);

      retry.accept();
      retry.abort();
      expect(state.frameCount).toBe(20);
      expect(state.giReservoirPublication).toBe(8);
      expect(state.nrcTicketSequence).toBe(13);
      expect(destroyedTickets).toEqual([13]);
    },
  );
});
