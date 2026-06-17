declare module '*.wasm?url' {
  const url: string;
  export default url;
}

declare module 'draco3d/draco_decoder_nodejs.js' {
  interface DracoDecoderBuffer {
    Init(data: Int8Array, byteLength: number): void;
  }

  interface DracoMesh {
    num_faces(): number;
  }

  interface DracoStatus {
    ok(): boolean;
    error_msg(): string;
  }

  interface DracoAttribute {
    readonly ptr: number;
  }

  interface DracoDecoder {
    GetEncodedGeometryType(buffer: DracoDecoderBuffer): number;
    DecodeBufferToMesh(buffer: DracoDecoderBuffer, mesh: DracoMesh): DracoStatus;
    GetAttributeByUniqueId(mesh: DracoMesh, uniqueId: number): DracoAttribute | null;
    GetAttributeFloatForAllPoints(mesh: DracoMesh, attribute: DracoAttribute, out: DracoFloat32Array): boolean;
    GetFaceFromMesh(mesh: DracoMesh, faceIndex: number, out: DracoInt32Array): boolean;
  }

  interface DracoFloat32Array {
    size(): number;
    GetValue(index: number): number;
  }

  interface DracoInt32Array {
    GetValue(index: number): number;
  }

  interface DracoModule {
    readonly TRIANGULAR_MESH: number;
    readonly Decoder: new () => DracoDecoder;
    readonly DecoderBuffer: new () => DracoDecoderBuffer;
    readonly Mesh: new () => DracoMesh;
    readonly DracoFloat32Array: new () => DracoFloat32Array;
    readonly DracoInt32Array: new () => DracoInt32Array;
    destroy(value: unknown): void;
  }

  export default function DracoDecoderModule(options?: { readonly wasmBinary?: ArrayBuffer }): Promise<DracoModule>;
}
