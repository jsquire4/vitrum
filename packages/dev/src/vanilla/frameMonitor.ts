// frameMonitor.ts — intercepts renderFrame to observe FrameOutput.
// Used by GI signal diagnostics which need per-frame texture availability.

import type { FrameInput, FrameOutput } from '@vitrum/core';
import type { DebuggableEngine } from '../types.js';

export interface FrameMonitor {
  readonly supported: boolean;
  readonly message?: string;
  get(): FrameOutput | null;
  onFrame(cb: () => void): () => void;
  dispose(): void;
}

export function createFrameMonitor(engine: DebuggableEngine): FrameMonitor {
  // eslint-disable-next-line @typescript-eslint/unbound-method -- intentionally unbound; called below as originalRenderFrame.call(this, input)
  const originalRenderFrame = engine.renderFrame;
  let lastFrame: FrameOutput | null = null;
  let installed = false;
  let message = 'ready';
  const listeners = new Set<() => void>();

  const monitoredRenderFrame = function monitoredRenderFrame(
    this: DebuggableEngine,
    input: FrameInput,
  ): FrameOutput {
    const output = originalRenderFrame.call(this, input);
    lastFrame = output;
    for (const listener of listeners) listener();
    return output;
  };

  try {
    (engine as { renderFrame: DebuggableEngine['renderFrame'] }).renderFrame = monitoredRenderFrame;
    installed = true;
  } catch {
    message = 'renderFrame readonly';
  }

  return {
    get supported() { return installed; },
    get message() { return message; },
    get() { return lastFrame; },
    onFrame(cb: () => void): () => void {
      listeners.add(cb);
      return () => { listeners.delete(cb); };
    },
    dispose(): void {
      listeners.clear();
      if (!installed) return;
      const writable = engine as { renderFrame: DebuggableEngine['renderFrame'] };
      if (writable.renderFrame === monitoredRenderFrame) {
        writable.renderFrame = originalRenderFrame;
      }
      installed = false;
    },
  };
}
