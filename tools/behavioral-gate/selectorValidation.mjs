// @ts-check

export class BehavioralGateSelectorError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(`[behavioral-gate-selector] ${message}`);
    this.name = 'BehavioralGateSelectorError';
  }
}

/**
 * Read an optional value-bearing CLI flag without allowing an omitted, empty,
 * or duplicate value to broaden a focused proof run into the full matrix.
 *
 * @param {string[]} args
 * @param {string} name
 */
export function readOptionalNonEmptyFlagValue(args, name) {
  /** @type {string[]} */
  const values = [];
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === name) {
      const value = args[index + 1];
      if (value == null || value.length === 0 || value.startsWith('--')) {
        throw new BehavioralGateSelectorError(`${name} requires a non-empty value`);
      }
      values.push(value);
      index++;
      continue;
    }
    if (arg.startsWith(`${name}=`)) {
      const value = arg.slice(name.length + 1);
      if (value.length === 0) {
        throw new BehavioralGateSelectorError(`${name} requires a non-empty value`);
      }
      values.push(value);
    }
  }
  if (values.length > 1) {
    throw new BehavioralGateSelectorError(`${name} may be provided at most once`);
  }
  return values[0] ?? '';
}

/**
 * Match the behavioral-gate's intentionally substring-based selectors while
 * preserving `default` as the two canonical default lanes. Without this
 * explicit rule, `default` also selects `pt/sobol-default`, contradicting the
 * checked DZN proof contract for that shard.
 *
 * @param {string} label
 * @param {string} filter
 * @param {boolean} [focused]
 */
export function configMatchesBehavioralFilter(label, filter, focused = false) {
  if (!filter) return true;
  if (filter === 'default') {
    return label === 'pt/default' || label === 'wh/default';
  }
  if (!label.includes(filter)) return false;
  // CWBVH promotion lanes may contain ordinary fixture names such as `gltf` or
  // `material-lobes`, but they require full-tier adapters and are selected
  // explicitly via `--filter cwbvh...`.
  if (focused && label.startsWith('pt/cwbvh-') && !filter.includes('cwbvh')) {
    return false;
  }
  return true;
}
