// @ts-check

export class DznStatusNumericValidationError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(`[behavioral-gate-dzn-numeric-validation] ${message}`);
    this.name = 'DznStatusNumericValidationError';
  }
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * @param {unknown} record
 * @param {string} key
 * @param {string} label
 * @param {{ integer?: boolean, minimum?: number, positive?: boolean }} [options]
 */
function requireFiniteField(record, key, label, options = {}) {
  if (!isRecord(record) || !Object.hasOwn(record, key)) {
    throw new DznStatusNumericValidationError(`${label}.${key} is missing`);
  }
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new DznStatusNumericValidationError(
      `${label}.${key} must be a number-typed finite value`,
    );
  }
  if (options.integer === true && !Number.isSafeInteger(value)) {
    throw new DznStatusNumericValidationError(`${label}.${key} must be a safe integer`);
  }
  if (options.positive === true && !(value > 0)) {
    throw new DznStatusNumericValidationError(`${label}.${key} must be positive`);
  }
  if (options.minimum != null && value < options.minimum) {
    throw new DznStatusNumericValidationError(`${label}.${key} must be >= ${options.minimum}`);
  }
  return value;
}

/**
 * Validate every numeric field that a committed PASS status uses as evidence.
 * Optional evidence groups become required when their expected contract selects
 * that group.
 *
 * @param {Record<string, any>} status
 * @param {{ path: string, configs: Array<Record<string, any>> }} expected
 */
export function validateDznStatusNumericFields(status, expected) {
  const root = expected.path;
  requireFiniteField(status, 'timeoutMs', root, { integer: true, positive: true });
  requireFiniteField(status, 'exitStatus', root, { integer: true, minimum: 0 });
  requireFiniteField(status.summary, 'totalConfigs', `${root}.summary`, {
    integer: true,
    minimum: 0,
  });
  requireFiniteField(status.summary, 'failures', `${root}.summary`, {
    integer: true,
    minimum: 0,
  });
  requireFiniteField(status.summary, 'knownResiduals', `${root}.summary`, {
    integer: true,
    minimum: 0,
  });

  if (!Array.isArray(status.configs)) {
    throw new DznStatusNumericValidationError(`${root}.configs must be an array`);
  }
  const byLabel = new Map(status.configs.map((config) => [config?.label, config]));
  for (const expectedConfig of expected.configs) {
    const label = `${root}:${expectedConfig.label}`;
    const config = byLabel.get(expectedConfig.label);
    if (!isRecord(config)) {
      throw new DznStatusNumericValidationError(`${label} is missing`);
    }
    requireFiniteField(config, 'luminance', label, { minimum: 0 });
    requireFiniteField(config, 'gpuErrors', label, { integer: true, minimum: 0 });

    const hasGoldenMetrics =
      expectedConfig.goldenStatus != null ||
      expectedConfig.minGoldenRmse != null ||
      expectedConfig.minGoldenMaxAbs != null;
    if (hasGoldenMetrics) {
      requireFiniteField(config, 'rmse', label, { minimum: 0 });
      requireFiniteField(config, 'meanAbs', label, { minimum: 0 });
      requireFiniteField(config, 'maxAbs', label, { minimum: 0 });
      for (const key of ['maxRmse', 'maxMeanAbs', 'maxAbs']) {
        requireFiniteField(config.thresholds, key, `${label}.thresholds`, {
          minimum: 0,
        });
      }
    }

    if (expectedConfig.mutationKind != null) {
      requireFiniteField(config, 'mutationMeanAbs', label, { minimum: 0 });
      requireFiniteField(config, 'mutationMaxAbs', label, { minimum: 0 });
    }

    if (expectedConfig.cwbvhParityKind != null) {
      for (const key of ['cwbvhParityRmse', 'cwbvhParityMeanAbs', 'cwbvhParityMaxAbs']) {
        requireFiniteField(config, key, label, { minimum: 0 });
      }
      for (const key of ['maxRmse', 'maxMeanAbs', 'maxAbs']) {
        requireFiniteField(config.cwbvhParityThresholds, key, `${label}.cwbvhParityThresholds`, {
          minimum: 0,
        });
      }
      for (const key of [
        'cwbvhBinaryRenderMs',
        'cwbvhRenderMs',
        'cwbvhRenderMsRatio',
        'cwbvhBinaryMemoryBytes',
        'cwbvhMemoryBytes',
        'cwbvhBinarySceneBytes',
        'cwbvhSceneBytes',
      ]) {
        requireFiniteField(config, key, label, { positive: true });
      }
      for (const key of ['cwbvhMemoryBytesDelta', 'cwbvhSceneBytesDelta']) {
        requireFiniteField(config, key, label);
      }
    }
  }
}

/**
 * Validate a newly parsed DZN status before the wrapper is allowed to report
 * success. Evidence groups are inferred from the parsed row itself.
 *
 * @param {Record<string, any>} status
 */
export function validateGeneratedDznStatusNumericFields(status) {
  const actualConfigs = Array.isArray(status.configs) ? status.configs : [];
  if (actualConfigs.length === 0) {
    throw new DznStatusNumericValidationError(
      'generated DZN status must contain at least one parsed config',
    );
  }
  const configs = actualConfigs.map((config) => {
    const label = typeof config?.label === 'string' ? config.label : '';
    const isMutation = config?.mutationKind != null || /\/mutation-/.test(label);
    const isCwbvhParity = config?.cwbvhParityKind != null || /^pt\/cwbvh-/.test(label);
    return {
      label,
      ...(config?.goldenStatus != null ? { goldenStatus: config.goldenStatus } : {}),
      ...(isMutation ? { mutationKind: config?.mutationKind ?? 'expected-from-label' } : {}),
      ...(isCwbvhParity
        ? { cwbvhParityKind: config?.cwbvhParityKind ?? 'expected-from-label' }
        : {}),
    };
  });
  validateDznStatusNumericFields(status, {
    path: 'generated DZN status',
    configs,
  });
  if (status.summary.totalConfigs !== actualConfigs.length) {
    throw new DznStatusNumericValidationError(
      'generated DZN status.summary.totalConfigs must equal the parsed config count',
    );
  }
  const labels = configs.map((config) => config.label);
  if (labels.some((label) => label.length === 0) || new Set(labels).size !== labels.length) {
    throw new DznStatusNumericValidationError(
      'generated DZN status config labels must be non-empty and unique',
    );
  }
}
