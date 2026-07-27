// errors.ts — structured glTF adapter failures.

export type GltfAssetResourceKind = 'asset' | 'buffer' | 'image';
export type GltfResourceDecodeFailureReason =
  | 'malformed-data-uri'
  | 'data-uri-atob-unavailable'
  | 'data-uri-decode-failed';
export type GltfParseFormat = 'gltf-json' | 'glb';
export type GltfParseFailureReason =
  | 'json-parse-failed'
  | 'glb-header-too-small'
  | 'glb-invalid-magic'
  | 'glb-unsupported-version'
  | 'glb-declared-length-exceeds-buffer'
  | 'glb-declared-length-mismatch'
  | 'glb-chunk-out-of-bounds'
  | 'glb-invalid-chunk-order'
  | 'glb-duplicate-chunk'
  | 'glb-invalid-chunk-alignment'
  | 'glb-trailing-bytes'
  | 'glb-json-parse-failed'
  | 'glb-json-missing';

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
  readonly sourcePath?: string;
  readonly message?: string;
}

export class GltfResourceNotFound extends GltfAdapterError {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
  readonly sourcePath?: string;

  constructor(init: GltfResourceErrorInit) {
    super(
      'GLTF_RESOURCE_NOT_FOUND',
      init.message ?? `[vitrum/gltf-adapter] Could not resolve ${init.kind} resource "${init.url}".`,
    );
    this.url = init.url;
    this.kind = init.kind;
    if (init.sourcePath !== undefined) this.sourcePath = init.sourcePath;
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
  readonly sourcePath?: string;
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
    if (init.sourcePath !== undefined) this.sourcePath = init.sourcePath;
    if (init.status !== undefined) this.status = init.status;
    if (init.statusText !== undefined) this.statusText = init.statusText;
  }
}

export interface GltfResourceDecodeFailedInit extends GltfResourceErrorInit {
  readonly reason: GltfResourceDecodeFailureReason;
  readonly cause?: unknown;
}

export class GltfResourceDecodeFailed extends GltfAdapterError {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
  readonly sourcePath?: string;
  readonly reason: GltfResourceDecodeFailureReason;

  constructor(init: GltfResourceDecodeFailedInit) {
    super(
      'GLTF_RESOURCE_DECODE_FAILED',
      init.message ??
        `[vitrum/gltf-adapter] Failed to decode ${init.kind} resource "${init.url}" (${init.reason}).`,
      init.cause === undefined ? undefined : { cause: init.cause },
    );
    this.url = init.url;
    this.kind = init.kind;
    if (init.sourcePath !== undefined) this.sourcePath = init.sourcePath;
    this.reason = init.reason;
  }
}

export interface GltfParseFailedInit {
  readonly format: GltfParseFormat;
  readonly reason: GltfParseFailureReason;
  readonly message: string;
  readonly cause?: unknown;
  readonly byteOffset?: number;
  readonly declaredLength?: number;
  readonly actualLength?: number;
  readonly chunkLength?: number;
  readonly version?: number;
}

export class GltfParseFailed extends GltfAdapterError {
  readonly format: GltfParseFormat;
  readonly reason: GltfParseFailureReason;
  readonly byteOffset?: number;
  readonly declaredLength?: number;
  readonly actualLength?: number;
  readonly chunkLength?: number;
  readonly version?: number;

  constructor(init: GltfParseFailedInit) {
    super(
      'GLTF_PARSE_FAILED',
      init.message,
      init.cause === undefined ? undefined : { cause: init.cause },
    );
    this.format = init.format;
    this.reason = init.reason;
    if (init.byteOffset !== undefined) this.byteOffset = init.byteOffset;
    if (init.declaredLength !== undefined) this.declaredLength = init.declaredLength;
    if (init.actualLength !== undefined) this.actualLength = init.actualLength;
    if (init.chunkLength !== undefined) this.chunkLength = init.chunkLength;
    if (init.version !== undefined) this.version = init.version;
  }
}

export type GltfCompatibilityErrorCode =
  | 'GLTF_COMPATIBILITY_REJECTED'
  | 'GLTF_COMPATIBILITY_PROFILE_MISSING'
  | 'GLTF_RUNTIME_PROFILE_MISMATCH';

export interface GltfCompatibilityErrorInit {
  readonly code: GltfCompatibilityErrorCode;
  readonly message: string;
  readonly backend?: string;
  readonly profileId?: string;
  readonly runtimeProfile?: string;
  readonly compatibilityMode?: string;
  readonly label?: string;
  readonly failures?: readonly string[];
  readonly failureDetails?: readonly GltfCompatibilityFailureDetail[];
  readonly cause?: unknown;
}

export interface GltfCompatibilityFailureDetail {
  readonly source: 'compatibility-issue' | 'import-diagnostic' | 'texture-readiness';
  readonly category?: string;
  readonly name?: string;
  readonly support?: string;
  readonly path?: string;
  readonly message?: string;
  readonly code?: string;
  readonly materialField?: string;
  readonly status?: string;
}

export class GltfCompatibilityError extends GltfAdapterError {
  readonly backend?: string;
  readonly profileId?: string;
  readonly runtimeProfile?: string;
  readonly compatibilityMode?: string;
  readonly label?: string;
  readonly failures: readonly string[];
  readonly failureDetails: readonly GltfCompatibilityFailureDetail[];

  constructor(init: GltfCompatibilityErrorInit) {
    super(
      init.code,
      init.message,
      init.cause === undefined ? undefined : { cause: init.cause },
    );
    if (init.backend !== undefined) this.backend = init.backend;
    if (init.profileId !== undefined) this.profileId = init.profileId;
    if (init.runtimeProfile !== undefined) this.runtimeProfile = init.runtimeProfile;
    if (init.compatibilityMode !== undefined) this.compatibilityMode = init.compatibilityMode;
    if (init.label !== undefined) this.label = init.label;
    this.failures = [...(init.failures ?? [])];
    this.failureDetails = [...(init.failureDetails ?? [])];
  }
}
