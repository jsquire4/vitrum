declare module 'jpeg-js' {
  export interface JpegRawImageData {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array;
  }

  export interface JpegEncodeResult {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array;
  }

  export function decode(
    data: unknown,
    options?: { readonly useTArray?: boolean },
  ): JpegRawImageData;

  export function encode(
    rawImageData: JpegRawImageData,
    quality?: number,
  ): JpegEncodeResult;
}
