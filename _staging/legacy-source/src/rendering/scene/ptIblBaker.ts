import type { SkyParams } from './skyParams';
import { bakeSkyEquirect as bakeSkyEquirectFromPackage } from '@vitrum/pt-webgl';

/**
 * Staging adapter: preserve the historical local import path while delegating
 * to the canonical package implementation.
 */
export function bakeSkyEquirect(
  renderer: unknown,
  params: SkyParams,
) {
  return bakeSkyEquirectFromPackage(renderer as never, params as never);
}
