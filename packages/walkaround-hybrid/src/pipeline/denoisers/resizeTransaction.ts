import {
  commitSceneMutations,
  type PreparedSceneMutation,
} from '../../SceneMutationTransaction.js';

export function noOpDenoiserResizeMutation(): PreparedSceneMutation {
  return {
    commit: () => undefined,
    rollback: () => undefined,
    finalize: () => undefined,
  };
}

export function commitPreparedDenoiserResize(
  mutation: PreparedSceneMutation,
): void {
  commitSceneMutations([mutation]);
}
