/**
 * CPU-side frame state that becomes valid only when the frame command buffer is
 * accepted by GPUQueue.submit.
 *
 * Encoding is synchronous but fallible. Passes therefore register tiny,
 * non-throwing publication callbacks instead of advancing ping-pong/history
 * state inline. Abort callbacks are reserved for staging resources (for example
 * an NRC readback buffer) that must be released when encode/finish/submit fails.
 */
import {
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
} from '../SceneMutationTransaction.js';

export interface FramePublication {
  stage(onAccepted: () => void, onAborted?: () => void): void;
}

export class FramePublicationTransaction implements FramePublication {
  private _state: 'open' | 'accepted' | 'aborted' = 'open';
  private _entries: Array<{
    readonly onAccepted: () => void;
    readonly onAborted?: () => void;
  }> = [];

  get state(): 'open' | 'accepted' | 'aborted' {
    return this._state;
  }

  stage(onAccepted: () => void, onAborted?: () => void): void {
    if (this._state !== 'open') {
      throw new Error(`frame publication is already ${this._state}`);
    }
    this._entries.push({
      onAccepted,
      ...(onAborted === undefined ? {} : { onAborted }),
    });
  }

  /**
   * Publish after queue.submit returns. Callbacks registered here are restricted
   * to infallible assignments; GPU work is already irreversible at this point.
   */
  accept(): void {
    if (this._state !== 'open') return;
    this._state = 'accepted';
    const entries = this._entries;
    this._entries = [];
    const errors: unknown[] = [];
    for (const entry of entries) {
      try {
        entry.onAccepted();
      } catch (error) {
        // GPU submission is already irreversible. Continue publishing every
        // independent history owner so the CPU state cannot describe a partial
        // frame, then report the complete callback failure set to the host.
        errors.push(error);
      }
    }
    if (errors.length > 0) {
      throw new AggregateError(
        errors,
        `${errors.length} frame-publication callback(s) failed after GPU submission`,
      );
    }
  }

  /** Release staged resources and leave all publication state unchanged. */
  abort(): void {
    if (this._state !== 'open') return;
    this._state = 'aborted';
    const entries = this._entries;
    this._entries = [];
    runSceneMutationCleanups(
      [...entries]
        .reverse()
        .flatMap((entry) => entry.onAborted == null ? [] : [entry.onAborted]),
      'frame-publication abort failed',
    );
  }
}

/**
 * Standalone pass tests may omit a frame transaction. Production renderFrame
 * always supplies one; the fallback preserves the direct-dispatch test surface.
 */
export function publishFrameState(
  publication: FramePublication | undefined,
  onAccepted: () => void,
  onAborted?: () => void,
): void {
  if (publication == null) {
    onAccepted();
    return;
  }
  publication.stage(onAccepted, onAborted);
}

/**
 * The sole finish/submit/publication boundary for a pipeline frame.
 * A synchronous finish/submit exception aborts all staged resources and leaves
 * persistent history untouched. Once submit returns, every accepted callback is
 * run (even if a peer callback throws) because GPU execution is irreversible.
 */
export function finishSubmitAndPublishFrame(
  encoder: Pick<GPUCommandEncoder, 'finish'>,
  queue: Pick<GPUQueue, 'submit'>,
  publication: FramePublicationTransaction,
): GPUCommandBuffer {
  let commandBuffer: GPUCommandBuffer;
  try {
    commandBuffer = encoder.finish();
    queue.submit([commandBuffer]);
  } catch (error) {
    rethrowWithSceneMutationCleanup(
      error,
      [() => publication.abort()],
      'frame submission failed and publication abort also failed',
    );
  }
  publication.accept();
  return commandBuffer;
}
