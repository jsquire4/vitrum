/**
 * Complete realtime-estimator persistence extension (v2 in GI snapshot v9).
 *
 * Kept separate from the historical DDGI/ReSTIR-GI/PPG container parser so
 * the legacy layouts remain auditable. The block always carries the live-input
 * compatibility key and optionally carries ReSTIR-DI and NRC state when those
 * subsystems are active. Extension v1 DI payloads are validated and reset to a
 * cold v2 reservoir cohort because their running-weight lane was linear.
 */

import {
  GI_STATE_COMPATIBILITY_WORDS,
  isValidGIStateCompatibility,
} from './giStateCompatibility.js';
import {
  assertRestirDISnapshot,
  coldMigrateLegacyRestirDISnapshot,
  type RestirDISnapshot,
} from './restir/restirDiStateSnapshot.js';
import { RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1 } from './restir/reservoirRepresentation.js';
import {
  assertNrcLearnedStateSnapshot,
  deserializeNrcLearnedState,
  nrcLearnedStateSerializedByteLength,
  serializeNrcLearnedState,
  type NrcLearnedStateSnapshot,
} from './neural/nrc/nrcStateSnapshot.js';

const EXTENSION_MAGIC = 0x47495853; // "GIXS"
const EXTENSION_VERSION = 2;
const LEGACY_EXTENSION_VERSION = 1;
const EXTENSION_HEADER_BYTES = 32;
const DI_SUBHEADER_BYTES = 20;
const FLAG_RESTIR_DI = 1 << 0;
const FLAG_NRC = 1 << 1;
const KNOWN_FLAGS = FLAG_RESTIR_DI | FLAG_NRC;
const U32_MAX = 0xffff_ffff;

export interface GIStateExtendedSections {
  readonly compatibility: Uint32Array;
  readonly restirDI?: RestirDISnapshot;
  readonly nrc?: NrcLearnedStateSnapshot;
}

export function assertGIStateExtendedSections(
  value: unknown,
): asserts value is GIStateExtendedSections {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('GI-state extended sections must be an object.');
  }
  const sections = value as GIStateExtendedSections;
  if (!isValidGIStateCompatibility(sections.compatibility)) {
    throw new RangeError(
      `GI-state compatibility must contain ${GI_STATE_COMPATIBILITY_WORDS} schema-valid u32 words.`,
    );
  }
  if (sections.restirDI != null) {
    assertRestirDISnapshot(sections.restirDI);
  }
  if (sections.nrc != null) {
    assertNrcLearnedStateSnapshot(sections.nrc);
  }
}

export function serializeGIStateExtendedSections(
  sections: GIStateExtendedSections,
): ArrayBuffer {
  const totalBytes = giStateExtendedSectionsByteLength(sections);
  const diBytes = sections.restirDI == null
    ? 0
    : checkedSum(
        DI_SUBHEADER_BYTES,
        sections.restirDI.current.byteLength,
        sections.restirDI.previous.byteLength,
        sections.restirDI.spatial.byteLength,
        'GI-state ReSTIR-DI section bytes',
      );
  const nrcBuffer = sections.nrc == null
    ? null
    : serializeNrcLearnedState(sections.nrc);
  const nrcBytes = nrcBuffer?.byteLength ?? 0;
  const compatibilityBytes = sections.compatibility.byteLength;

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  let offset = 0;
  view.setUint32(offset, EXTENSION_MAGIC, true); offset += 4;
  view.setUint32(offset, EXTENSION_VERSION, true); offset += 4;
  view.setUint32(offset, totalBytes, true); offset += 4;
  const flags =
    (sections.restirDI != null ? FLAG_RESTIR_DI : 0) |
    (sections.nrc != null ? FLAG_NRC : 0);
  view.setUint32(offset, flags, true); offset += 4;
  view.setUint32(offset, GI_STATE_COMPATIBILITY_WORDS, true); offset += 4;
  view.setUint32(offset, diBytes, true); offset += 4;
  view.setUint32(offset, nrcBytes, true); offset += 4;
  view.setUint32(offset, 0, true); offset += 4;
  copyBytes(buffer, offset, sections.compatibility);
  offset += compatibilityBytes;

  if (sections.restirDI != null) {
    const di = sections.restirDI;
    view.setUint32(offset, di.width, true); offset += 4;
    view.setUint32(offset, di.height, true); offset += 4;
    view.setUint32(offset, di.strideU32, true); offset += 4;
    view.setUint32(offset, di.current.length, true); offset += 4;
    view.setUint32(offset, di.representationVersion, true); offset += 4;
    copyBytes(buffer, offset, di.current); offset += di.current.byteLength;
    copyBytes(buffer, offset, di.previous); offset += di.previous.byteLength;
    copyBytes(buffer, offset, di.spatial); offset += di.spatial.byteLength;
  }
  if (nrcBuffer != null) {
    new Uint8Array(buffer, offset, nrcBuffer.byteLength).set(
      new Uint8Array(nrcBuffer),
    );
    offset += nrcBuffer.byteLength;
  }
  if (offset !== totalBytes) {
    throw new Error('GI-state extension encoder did not fill its buffer.');
  }
  return buffer;
}

export function giStateExtendedSectionsByteLength(
  sections: GIStateExtendedSections,
): number {
  assertGIStateExtendedSections(sections);
  const diBytes = sections.restirDI == null
    ? 0
    : checkedSum(
        DI_SUBHEADER_BYTES,
        sections.restirDI.current.byteLength,
        sections.restirDI.previous.byteLength,
        sections.restirDI.spatial.byteLength,
        'GI-state ReSTIR-DI section bytes',
      );
  const nrcBytes = sections.nrc == null
    ? 0
    : nrcLearnedStateSerializedByteLength(sections.nrc);
  const totalBytes = checkedSum(
    EXTENSION_HEADER_BYTES,
    sections.compatibility.byteLength,
    diBytes,
    nrcBytes,
    'GI-state extension byte length',
  );
  if (totalBytes > U32_MAX) {
    throw new RangeError('GI-state extension exceeds the uint32 byte domain.');
  }
  return totalBytes;
}

export function deserializeGIStateExtendedSections(
  buffer: ArrayBuffer,
): GIStateExtendedSections {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new TypeError('GI-state extension input must be an ArrayBuffer.');
  }
  if (buffer.byteLength < EXTENSION_HEADER_BYTES) {
    throw new RangeError('GI-state extension is smaller than its header.');
  }
  const view = new DataView(buffer);
  let offset = 0;
  const magic = view.getUint32(offset, true); offset += 4;
  if (magic !== EXTENSION_MAGIC) {
    throw new Error('GI-state extension has an invalid magic value.');
  }
  const version = view.getUint32(offset, true); offset += 4;
  if (
    version !== EXTENSION_VERSION &&
    version !== LEGACY_EXTENSION_VERSION
  ) {
    throw new Error(
      `GI-state extension version ${version} is unsupported (expected ${EXTENSION_VERSION} or ${LEGACY_EXTENSION_VERSION}).`,
    );
  }
  const totalBytes = view.getUint32(offset, true); offset += 4;
  const flags = view.getUint32(offset, true); offset += 4;
  const compatibilityWords = view.getUint32(offset, true); offset += 4;
  const diBytes = view.getUint32(offset, true); offset += 4;
  const nrcBytes = view.getUint32(offset, true); offset += 4;
  const reserved = view.getUint32(offset, true); offset += 4;
  if (
    totalBytes !== buffer.byteLength ||
    (flags & ~KNOWN_FLAGS) !== 0 ||
    compatibilityWords !== GI_STATE_COMPATIBILITY_WORDS ||
    reserved !== 0
  ) {
    throw new RangeError('GI-state extension header is inconsistent.');
  }
  if (
    ((flags & FLAG_RESTIR_DI) === 0) !== (diBytes === 0) ||
    ((flags & FLAG_NRC) === 0) !== (nrcBytes === 0)
  ) {
    throw new RangeError(
      'GI-state extension flags do not match their section lengths.',
    );
  }
  const expectedTotal = checkedSum(
    EXTENSION_HEADER_BYTES,
    checkedProduct(
      compatibilityWords,
      Uint32Array.BYTES_PER_ELEMENT,
      'GI-state compatibility bytes',
    ),
    diBytes,
    nrcBytes,
    'GI-state extension declared bytes',
  );
  if (expectedTotal !== buffer.byteLength) {
    throw new RangeError(
      'GI-state extension section lengths do not match the container.',
    );
  }

  const compatibilityByteLength =
    compatibilityWords * Uint32Array.BYTES_PER_ELEMENT;
  const compatibility = new Uint32Array(
    buffer.slice(offset, offset + compatibilityByteLength),
  );
  offset += compatibilityByteLength;
  if (!isValidGIStateCompatibility(compatibility)) {
    throw new RangeError('GI-state extension compatibility key is invalid.');
  }

  let restirDI: RestirDISnapshot | undefined;
  if ((flags & FLAG_RESTIR_DI) !== 0) {
    const diStart = offset;
    if (diBytes < DI_SUBHEADER_BYTES) {
      throw new RangeError('GI-state ReSTIR-DI section is too small.');
    }
    const width = view.getUint32(offset, true); offset += 4;
    const height = view.getUint32(offset, true); offset += 4;
    const strideU32 = view.getUint32(offset, true); offset += 4;
    const bufferLengthU32 = view.getUint32(offset, true); offset += 4;
    const representationVersion = view.getUint32(offset, true); offset += 4;
    if (
      (version === LEGACY_EXTENSION_VERSION && representationVersion !== 0) ||
      (version === EXTENSION_VERSION &&
        representationVersion !==
          RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1)
    ) {
      throw new RangeError(
        'GI-state ReSTIR-DI representation marker is incompatible with the extension version.',
      );
    }
    const oneBufferBytes = checkedProduct(
      bufferLengthU32,
      Uint32Array.BYTES_PER_ELEMENT,
      'GI-state ReSTIR-DI buffer bytes',
    );
    const expectedDiBytes = checkedSum(
      DI_SUBHEADER_BYTES,
      oneBufferBytes,
      oneBufferBytes,
      oneBufferBytes,
      'GI-state ReSTIR-DI declared bytes',
    );
    if (expectedDiBytes !== diBytes) {
      throw new RangeError(
        'GI-state ReSTIR-DI section length is inconsistent.',
      );
    }
    const take = (): Uint32Array => {
      const end = checkedSum(
        offset,
        oneBufferBytes,
        'GI-state ReSTIR-DI buffer end',
      );
      if (end > buffer.byteLength) {
        throw new RangeError('GI-state ReSTIR-DI buffer exceeds its section.');
      }
      const result = new Uint32Array(buffer.slice(offset, end));
      offset = end;
      return result;
    };
    const current = take();
    const previous = take();
    const spatial = take();
    if (version === LEGACY_EXTENSION_VERSION) {
      restirDI = coldMigrateLegacyRestirDISnapshot({
        width,
        height,
        strideU32,
        current,
        previous,
        spatial,
      });
    } else {
      restirDI = {
        representationVersion:
          RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1,
        width,
        height,
        strideU32,
        current,
        previous,
        spatial,
      };
      assertRestirDISnapshot(restirDI);
    }
    if (offset !== diStart + diBytes) {
      throw new Error('GI-state ReSTIR-DI decoder left trailing bytes.');
    }
  }

  let nrc: NrcLearnedStateSnapshot | undefined;
  if ((flags & FLAG_NRC) !== 0) {
    const end = checkedSum(offset, nrcBytes, 'GI-state NRC section end');
    if (end > buffer.byteLength) {
      throw new RangeError('GI-state NRC section exceeds the extension.');
    }
    nrc = deserializeNrcLearnedState(buffer.slice(offset, end));
    offset = end;
  }
  if (offset !== buffer.byteLength) {
    throw new Error('GI-state extension decoder left trailing bytes.');
  }
  return {
    compatibility,
    ...(restirDI != null ? { restirDI } : {}),
    ...(nrc != null ? { nrc } : {}),
  };
}

function copyBytes(
  destination: ArrayBuffer,
  byteOffset: number,
  source: ArrayBufferView,
): void {
  new Uint8Array(destination, byteOffset, source.byteLength).set(
    new Uint8Array(source.buffer, source.byteOffset, source.byteLength),
  );
}

function checkedProduct(a: number, b: number, label: string): number {
  if (
    !Number.isSafeInteger(a) ||
    !Number.isSafeInteger(b) ||
    a < 0 ||
    b < 0 ||
    (a !== 0 && b > Number.MAX_SAFE_INTEGER / a)
  ) {
    throw new RangeError(`${label} exceeds safe-integer arithmetic.`);
  }
  return a * b;
}

function checkedSum(...args: [...number[], string]): number {
  const label = args.pop() as string;
  let sum = 0;
  for (const term of args as number[]) {
    if (
      !Number.isSafeInteger(term) ||
      term < 0 ||
      sum > Number.MAX_SAFE_INTEGER - term
    ) {
      throw new RangeError(`${label} exceeds safe-integer arithmetic.`);
    }
    sum += term;
  }
  return sum;
}
