interface UniformRef<T> {
  value: T;
}

export interface PathTracerMaterialLike {
  uniforms?: Record<string, UniformRef<unknown>>;
  setDefine?(name: string, value: number): void;
}

interface ForkTracerLike {
  _pathTracer?: {
    material?: PathTracerMaterialLike;
    target?: { texture?: unknown };
  };
  target?: { texture?: unknown };
}

/** Centralized compatibility access for private fork fields. */
export class ForkAccess {
  static getMaterial(pathTracer: unknown): PathTracerMaterialLike | null {
    const tracer = pathTracer as ForkTracerLike;
    return tracer._pathTracer?.material ?? null;
  }

  static getRenderTexture(pathTracer: unknown): unknown | null {
    const tracer = pathTracer as ForkTracerLike;
    return tracer.target?.texture ?? tracer._pathTracer?.target?.texture ?? null;
  }
}
