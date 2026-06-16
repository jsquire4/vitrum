declare module 'draco3d' {
  const draco3d: {
    createDecoderModule(options?: Record<string, unknown>): Promise<unknown>;
    createEncoderModule(options?: Record<string, unknown>): Promise<unknown>;
  };
  export default draco3d;
}
