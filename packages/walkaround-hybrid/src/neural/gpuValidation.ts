export type NeuralGpuErrorFilter = 'validation' | 'out-of-memory' | 'internal';

/**
 * Await a complete WebGPU validation/OOM/internal error-scope set around async
 * candidate construction. If any scope reports an error, the candidate is
 * disposed before the promise rejects.
 */
export async function withNeuralGpuErrorScopes<T>(
  device: GPUDevice,
  label: string,
  construct: () => T | Promise<T>,
  disposeCandidate: (candidate: T) => void,
): Promise<T> {
  const canScope =
    typeof device.pushErrorScope === 'function' &&
    typeof device.popErrorScope === 'function';
  if (!canScope) return construct();

  const filters: readonly NeuralGpuErrorFilter[] = ['validation', 'out-of-memory', 'internal'];
  for (const filter of filters) device.pushErrorScope(filter);

  let candidate: T | undefined;
  let constructionError: unknown;
  try {
    candidate = await construct();
  } catch (error) {
    constructionError = error;
  }

  const scopeErrors: string[] = [];
  for (let i = filters.length - 1; i >= 0; i--) {
    try {
      const error = await device.popErrorScope();
      if (error != null) scopeErrors.push(filters[i]! + ': ' + error.message);
    } catch (error) {
      scopeErrors.push(filters[i]! + ' scope pop failed: ' + errorMessage(error));
    }
  }

  if (constructionError !== undefined || scopeErrors.length > 0) {
    if (candidate !== undefined) {
      try { disposeCandidate(candidate); } catch { /* preserve the primary error */ }
    }
    if (constructionError instanceof Error) throw constructionError;
    if (constructionError !== undefined) {
      throw new Error('[neural GPU validation] candidate construction failed', {
        cause: constructionError,
      });
    }
    throw new Error('[neural GPU validation] ' + label + ': ' + scopeErrors.join('; '));
  }
  if (candidate === undefined) {
    throw new Error('[neural GPU validation] ' + label + ' produced no candidate');
  }
  return candidate;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
