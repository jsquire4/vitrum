export interface PreparableProgram {
  prepare(): boolean;
}

export interface ProgramSequencePreparation {
  /** Index of the first program that is not yet ready. */
  readonly nextIndex: number;
  readonly ready: boolean;
}

/**
 * Poll asynchronous shader links in strict order. Once one program reports
 * pending, no later program is touched until a future poll completes it.
 */
export function prepareProgramSequence(
  programs: readonly PreparableProgram[],
  startIndex: number,
): ProgramSequencePreparation {
  if (!Number.isSafeInteger(startIndex) || startIndex < 0 || startIndex > programs.length) {
    throw new RangeError('pt-webgl2: program preparation index is outside the sequence');
  }
  let nextIndex = startIndex;
  while (nextIndex < programs.length) {
    if (!programs[nextIndex]!.prepare()) return { nextIndex, ready: false };
    nextIndex += 1;
  }
  return { nextIndex, ready: true };
}
