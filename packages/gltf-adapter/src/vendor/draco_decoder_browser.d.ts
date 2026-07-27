interface DracoDecoderModuleOptions {
  readonly wasmBinary?: ArrayBuffer | Uint8Array;
  readonly locateFile?: (path: string, prefix: string) => string;
}

export default function createDecoderModule(
  options?: DracoDecoderModuleOptions,
): Promise<unknown>;
