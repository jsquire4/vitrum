// errors.ts — structured glTF adapter failures.

export type GltfAssetResourceKind = 'asset' | 'buffer' | 'image';

export class GltfAdapterError extends Error {
  readonly code: string;

  constructor(code: string, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
    this.code = code;
  }
}

export interface GltfResourceErrorInit {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
  readonly message?: string;
}

export class GltfResourceNotFound extends GltfAdapterError {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;

  constructor(init: GltfResourceErrorInit) {
    super(
      'GLTF_RESOURCE_NOT_FOUND',
      init.message ?? `[vitrum/gltf-adapter] Could not resolve ${init.kind} resource "${init.url}".`,
    );
    this.url = init.url;
    this.kind = init.kind;
  }
}

export interface GltfFetchFailedInit extends GltfResourceErrorInit {
  readonly status?: number;
  readonly statusText?: string;
  readonly cause?: unknown;
}

export class GltfFetchFailed extends GltfAdapterError {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
  readonly status?: number;
  readonly statusText?: string;

  constructor(init: GltfFetchFailedInit) {
    const statusPart = init.status === undefined ? '' : ` HTTP ${init.status}`;
    const statusTextPart = init.statusText ? ` ${init.statusText}` : '';
    super(
      'GLTF_FETCH_FAILED',
      init.message ??
        `[vitrum/gltf-adapter] Failed to fetch ${init.kind} resource "${init.url}".${statusPart}${statusTextPart}`,
      init.cause === undefined ? undefined : { cause: init.cause },
    );
    this.url = init.url;
    this.kind = init.kind;
    if (init.status !== undefined) this.status = init.status;
    if (init.statusText !== undefined) this.statusText = init.statusText;
  }
}
