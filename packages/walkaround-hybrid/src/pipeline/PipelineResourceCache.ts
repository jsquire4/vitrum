/**
 * Small identity-keyed cache for stable pipeline GPU objects.
 *
 * The cache intentionally only memoizes descriptor-free texture views and bind
 * groups whose resource identities are supplied by the caller. Resize/resource
 * recreation naturally invalidates entries because the key objects change. A
 * small bounded variant list per id covers ping-pong groups without letting
 * scene-update churn accumulate forever.
 */

type CachedValue<T> = {
  readonly keys: readonly unknown[];
  readonly value: T;
};

function sameKeys(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

export class PipelineResourceCache {
  private static readonly MAX_BIND_GROUP_VARIANTS_PER_ID = 8;

  private _textureViews = new WeakMap<GPUTexture, GPUTextureView>();
  private readonly _bindGroups = new Map<string, CachedValue<unknown>[]>();

  textureView(texture: GPUTexture): GPUTextureView {
    const cached = this._textureViews.get(texture);
    if (cached) return cached;
    const view = texture.createView();
    this._textureViews.set(texture, view);
    return view;
  }

  bindGroup<T = GPUBindGroup>(
    id: string,
    keys: readonly unknown[],
    create: () => T,
  ): T {
    const cached = this._bindGroups.get(id);
    const hit = cached?.find((entry) => sameKeys(entry.keys, keys));
    if (hit) return hit.value as T;
    const value = create();
    const entry = { keys: [...keys], value };
    if (!cached) {
      this._bindGroups.set(id, [entry]);
    } else {
      cached.unshift(entry);
      cached.length = Math.min(cached.length, PipelineResourceCache.MAX_BIND_GROUP_VARIANTS_PER_ID);
    }
    return value;
  }

  clear(): void {
    this._textureViews = new WeakMap();
    this._bindGroups.clear();
  }
}

/**
 * Memoize a bind group through an OPTIONAL {@link PipelineResourceCache},
 * falling back to a fresh build when no cache is supplied. Collapses the
 * repeated `cache?.bindGroup(id, resources, build) ?? build()` idiom into a
 * single call so the `?? build()` fallback (and the double-listed resources ↔
 * cache-key coupling the callers warn about) live in ONE place.
 *
 * `resources` is BOTH the invalidation key AND the identity list the caller's
 * `build` closes over — passing them once here keeps the key and the built
 * group's contents from drifting (the stale-binding bug the comments guard).
 */
export function cachedBindGroup<T = GPUBindGroup>(
  cache: PipelineResourceCache | undefined,
  id: string,
  resources: readonly unknown[],
  build: () => T,
): T {
  return cache?.bindGroup(id, resources, build) ?? build();
}
