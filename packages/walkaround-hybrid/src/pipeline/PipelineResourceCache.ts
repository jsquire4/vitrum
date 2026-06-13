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
