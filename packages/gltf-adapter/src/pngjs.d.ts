declare module 'pngjs' {
  export const PNG: {
    sync: {
      read(data: unknown): {
        width: number;
        height: number;
        data: Uint8Array;
      };
      write(input: {
        width: number;
        height: number;
        data: Uint8Array;
      }): Uint8Array;
    };
  };
}

declare module 'pngjs/browser.js' {
  export const PNG: {
    sync: {
      read(data: unknown): {
        width: number;
        height: number;
        data: Uint8Array;
      };
      write(input: {
        width: number;
        height: number;
        data: Uint8Array;
      }): Uint8Array;
    };
  };
}
