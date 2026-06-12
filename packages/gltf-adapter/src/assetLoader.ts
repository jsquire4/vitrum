// assetLoader.ts — turnkey glTF/GLB loading into Vitrum's core Scene contract.
//
// `gltfToScene()` remains the low-level converter for hosts that already own
// resource resolution. `loadGltfAsset()` is the predictable higher-level path:
// it accepts URL/string/ArrayBuffer/GltfJson, resolves external buffers/images,
// returns a structured feature report, and ranks the shipping backends against
// the asset's actual feature use.

import type { GltfJson } from './gltfTypes.js';
import { gltfToScene, type GltfToSceneOptions, type GltfToSceneResult } from './gltfToScene.js';
import { parseGlb } from './glbParser.js';
import type { GltfImageBytes } from './textures.js';
import {
  analyzeGltfAsset,
  rankGltfBackends,
  type GltfBackendCompatibility,
  type GltfBackendPolicy,
  type GltfFeatureReport,
} from './featureReport.js';

export type GltfAssetInput = string | URL | ArrayBuffer | GltfJson;

export interface GltfAssetFetchResponse {
  readonly ok?: boolean;
  readonly status?: number;
  readonly statusText?: string;
  readonly headers?: { get(name: string): string | null };
  arrayBuffer(): Promise<ArrayBuffer>;
}

export type GltfAssetFetch = (
  url: string,
  init?: { readonly signal?: AbortSignal },
) => Promise<GltfAssetFetchResponse>;

export interface LoadGltfAssetOptions extends GltfToSceneOptions {
  readonly baseUri?: string | URL;
  readonly fetch?: GltfAssetFetch;
  readonly signal?: AbortSignal;
  readonly backendPolicy?: GltfBackendPolicy;
}

export interface GltfAssetResult extends GltfToSceneResult {
  readonly gltf: GltfJson;
  readonly sceneIndex: number;
  readonly featureReport: GltfFeatureReport;
  readonly backendCompatibility: readonly GltfBackendCompatibility[];
  readonly recommendedBackend: GltfBackendCompatibility;
}

interface ParsedInput {
  readonly gltf: GltfJson;
  readonly baseUri?: string;
  readonly buffers: Map<number, ArrayBuffer>;
}

export async function loadGltfAsset(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions = {},
): Promise<GltfAssetResult> {
  const parsed = await parseInput(input, options);
  const buffers = normalizeBufferMap(options.buffers);
  for (const [index, buffer] of parsed.buffers) {
    if (!buffers.has(index)) buffers.set(index, buffer);
  }

  await resolveExternalBuffers(parsed.gltf, buffers, parsed.baseUri, options);
  const imageBytes = await resolveExternalImages(parsed.gltf, parsed.baseUri, options);

  const featureReport = analyzeGltfAsset(parsed.gltf);
  const backendCompatibility = rankGltfBackends(featureReport, options.backendPolicy ?? 'fidelity');
  const sceneIndex = options.sceneIndex ?? parsed.gltf.scene ?? 0;
  const sceneOptions: GltfToSceneOptions = {
    buffers,
    imageBytes,
    ...(options.decodeImage ? { decodeImage: options.decodeImage } : {}),
    sceneIndex,
    ...(options.dracoDecode ? { dracoDecode: options.dracoDecode } : {}),
    ...(options.meshoptDecode ? { meshoptDecode: options.meshoptDecode } : {}),
    ...(options.textureSourceExtensions ? { textureSourceExtensions: options.textureSourceExtensions } : {}),
    ...(options.materialVariant !== undefined ? { materialVariant: options.materialVariant } : {}),
  };
  const sceneResult = await gltfToScene(parsed.gltf, sceneOptions);

  return {
    ...sceneResult,
    gltf: parsed.gltf,
    sceneIndex,
    featureReport,
    backendCompatibility,
    recommendedBackend: backendCompatibility[0]!,
  };
}

async function parseInput(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions,
): Promise<ParsedInput> {
  if (typeof input === 'string' || input instanceof URL) {
    const url = resolveUri(String(input), options.baseUri);
    const bytes = await fetchArrayBuffer(url, options);
    return parseArrayBufferInput(bytes, directoryUrl(url));
  }
  if (input instanceof ArrayBuffer) {
    return parseArrayBufferInput(input, options.baseUri ? String(options.baseUri) : undefined);
  }
  return {
    gltf: input,
    ...(options.baseUri ? { baseUri: String(options.baseUri) } : {}),
    buffers: new Map(),
  };
}

function parseArrayBufferInput(buffer: ArrayBuffer, baseUri?: string): ParsedInput {
  if (isGlb(buffer)) {
    const glb = parseGlb(buffer);
    const buffers = new Map<number, ArrayBuffer>();
    if (glb.binChunk) buffers.set(0, glb.binChunk);
    return { gltf: glb.json, ...(baseUri ? { baseUri } : {}), buffers };
  }
  const text = new TextDecoder().decode(buffer);
  return { gltf: JSON.parse(text) as GltfJson, ...(baseUri ? { baseUri } : {}), buffers: new Map() };
}

async function resolveExternalBuffers(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
): Promise<void> {
  for (const [index, buffer] of (gltf.buffers ?? []).entries()) {
    if (buffers.has(index)) continue;
    if (buffer.uri == null) continue;
    if (buffer.uri.startsWith('data:')) {
      buffers.set(index, uint8ToArrayBuffer(decodeDataUri(buffer.uri, `buffer ${index}`)));
      continue;
    }
    const url = resolveUri(buffer.uri, baseUri);
    const data = await fetchArrayBuffer(url, options);
    buffers.set(index, data);
  }
}

async function resolveExternalImages(
  gltf: GltfJson,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
): Promise<Map<number, GltfImageBytes>> {
  const out = new Map<number, GltfImageBytes>();
  for (const [index, image] of (gltf.images ?? []).entries()) {
    if (image.bufferView !== undefined) continue;
    if (image.uri == null || image.uri.startsWith('data:')) continue;
    const url = resolveUri(image.uri, baseUri);
    const response = await fetchResource(url, options);
    const bytes = new Uint8Array(await response.arrayBuffer());
    out.set(index, {
      bytes,
      mimeType: image.mimeType ?? response.headers?.get('content-type') ?? inferMimeType(image.uri),
    });
  }
  return out;
}

async function fetchArrayBuffer(
  url: string,
  options: LoadGltfAssetOptions,
): Promise<ArrayBuffer> {
  const response = await fetchResource(url, options);
  return response.arrayBuffer();
}

async function fetchResource(
  url: string,
  options: LoadGltfAssetOptions,
): Promise<GltfAssetFetchResponse> {
  const fetchFn = (options.fetch ?? globalThis.fetch) as GltfAssetFetch | undefined;
  if (typeof fetchFn !== 'function') {
    throw new Error(
      '[vitrum/gltf-adapter] loadGltfAsset requires a fetch implementation for URL/external resources.',
    );
  }
  const init = options.signal ? { signal: options.signal } : undefined;
  const response = await fetchFn(url, init);
  if (response.ok === false) {
    throw new Error(
      `[vitrum/gltf-adapter] Failed to fetch "${url}": HTTP ${response.status ?? '?'} ` +
        `${response.statusText ?? ''}`.trim(),
    );
  }
  return response;
}

function normalizeBufferMap(
  buffers: GltfToSceneOptions['buffers'] | undefined,
): Map<number, ArrayBuffer> {
  const out = new Map<number, ArrayBuffer>();
  if (buffers == null) return out;
  if (buffers instanceof Map) {
    for (const [k, v] of buffers) out.set(k, v);
  } else {
    for (const [k, v] of Object.entries(buffers)) out.set(Number(k), v);
  }
  return out;
}

function isGlb(buffer: ArrayBuffer): boolean {
  return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === 0x46546c67;
}

function resolveUri(uri: string, baseUri: string | URL | undefined): string {
  if (uri.startsWith('data:')) return uri;
  try {
    return new URL(uri).toString();
  } catch {
    if (baseUri == null) {
      throw new Error(
        `[vitrum/gltf-adapter] Cannot resolve relative URI "${uri}" without a baseUri.`,
      );
    }
    return new URL(uri, baseUri).toString();
  }
}

function directoryUrl(url: string): string {
  return new URL('.', url).toString();
}

function decodeDataUri(uri: string, label: string): Uint8Array {
  const comma = uri.indexOf(',');
  if (comma < 0) {
    throw new Error(`[vitrum/gltf-adapter] ${label} has a malformed data: URI.`);
  }
  const meta = uri.slice(5, comma);
  const payload = uri.slice(comma + 1);
  const isBase64 = meta.split(';').some((p) => p.toLowerCase() === 'base64');
  if (isBase64) {
    if (typeof globalThis.atob !== 'function') {
      throw new Error(`[vitrum/gltf-adapter] ${label} uses base64 data URI but atob() is unavailable.`);
    }
    const bin = globalThis.atob(payload.replace(/\s+/g, ''));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  return new TextEncoder().encode(decodeURIComponent(payload));
}

function uint8ToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const out = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(out).set(bytes);
  return out;
}

function inferMimeType(uri: string): string {
  const lower = uri.toLowerCase().split('?')[0] ?? uri.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.ktx2')) return 'image/ktx2';
  if (lower.endsWith('.dds')) return 'image/vnd-ms.dds';
  return 'image/png';
}
