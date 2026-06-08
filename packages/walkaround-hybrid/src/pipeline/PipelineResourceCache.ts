/**
 * Small identity-keyed cache for stable pipeline GPU objects.
 *
 * The cache intentionally only memoizes descriptor-free texture views and bind
 * groups whose resource identities are supplied by the caller. Resize/resource
 * recreation naturally invalidates entries because the key objects change.
 */

type CachedBindGroup = {
  readonly keys: readonly unknown[];
  readonly bindGroup: GPUBindGroup;
};

function sameKeys(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

export class PipelineResourceCache {
  private _textureViews = new WeakMap<GPUTexture, GPUTextureView>();
  private readonly _bindGroups = new Map<string, CachedBindGroup>();

  textureView(texture: GPUTexture): GPUTextureView {
    const cached = this._textureViews.get(texture);
    if (cached) return cached;
    const view = texture.createView();
    this._textureViews.set(texture, view);
    return view;
  }

  bindGroup(
    id: string,
    keys: readonly unknown[],
    create: () => GPUBindGroup,
  ): GPUBindGroup {
    const cached = this._bindGroups.get(id);
    if (cached && sameKeys(cached.keys, keys)) return cached.bindGroup;
    const bindGroup = create();
    this._bindGroups.set(id, { keys: [...keys], bindGroup });
    return bindGroup;
  }

  clear(): void {
    this._textureViews = new WeakMap();
    this._bindGroups.clear();
  }
}
