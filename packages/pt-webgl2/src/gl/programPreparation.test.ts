import { describe, expect, it, vi } from 'vitest';
import { prepareProgramSequence, type PreparableProgram } from './programPreparation.js';

function controlledProgram(initialReady = false): {
  readonly program: PreparableProgram;
  readonly prepare: ReturnType<typeof vi.fn<[], boolean>>;
  setReady(ready: boolean): void;
} {
  let ready = initialReady;
  const prepare = vi.fn(() => ready);
  return {
    program: { prepare },
    prepare,
    setReady(nextReady: boolean): void {
      ready = nextReady;
    },
  };
}

describe('prepareProgramSequence', () => {
  it('never starts a later large link while the current one is pending', () => {
    const trace = controlledProgram();
    const candidate = controlledProgram();
    const resolve = controlledProgram();
    const programs = [trace.program, candidate.program, resolve.program] as const;

    expect(prepareProgramSequence(programs, 0)).toEqual({ nextIndex: 0, ready: false });
    expect(trace.prepare).toHaveBeenCalledTimes(1);
    expect(candidate.prepare).not.toHaveBeenCalled();
    expect(resolve.prepare).not.toHaveBeenCalled();

    trace.setReady(true);
    expect(prepareProgramSequence(programs, 0)).toEqual({ nextIndex: 1, ready: false });
    expect(trace.prepare).toHaveBeenCalledTimes(2);
    expect(candidate.prepare).toHaveBeenCalledTimes(1);
    expect(resolve.prepare).not.toHaveBeenCalled();

    candidate.setReady(true);
    expect(prepareProgramSequence(programs, 1)).toEqual({ nextIndex: 2, ready: false });
    expect(trace.prepare).toHaveBeenCalledTimes(2);
    expect(candidate.prepare).toHaveBeenCalledTimes(2);
    expect(resolve.prepare).toHaveBeenCalledTimes(1);

    resolve.setReady(true);
    expect(prepareProgramSequence(programs, 2)).toEqual({ nextIndex: 3, ready: true });
    expect(resolve.prepare).toHaveBeenCalledTimes(2);
  });

  it('rejects corrupt graph stages instead of silently skipping a program', () => {
    expect(() => prepareProgramSequence([], -1)).toThrow(/outside the sequence/);
    expect(() => prepareProgramSequence([], 1)).toThrow(/outside the sequence/);
    expect(() => prepareProgramSequence([], 0.5)).toThrow(/outside the sequence/);
  });
});
