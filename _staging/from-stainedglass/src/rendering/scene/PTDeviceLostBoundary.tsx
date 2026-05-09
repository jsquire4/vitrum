// Slice 3.4 — Path-tracer-specific error boundary.
//
// THREE.WebGPURenderer / WebGLPathTracer can fire device-lost errors
// (DXGI_ERROR_DEVICE_HUNG on Windows, GL_OUT_OF_MEMORY on Linux/Intel)
// when the host GPU is starved by rapid setScene rebuilds. Without a
// PT-specific boundary the error bubbles to CanvasErrorBoundary, which
// blanks the entire canvas — the user has no way to keep editing in
// raster mode without reloading.
//
// This boundary catches errors thrown inside <Pathtracer>'s subtree,
// invokes onError(error) so the parent can flip pathTracerEnabled off
// and surface a recoverable banner, then renders nothing for the rest
// of the unmount cycle. Once setPathTracerEnabled(false) propagates,
// the parent <PathTracingLayer> stops mounting <PathTracingInner>
// entirely, taking the boundary with it.

import React, { Component } from 'react';

interface Props {
  children: React.ReactNode;
  /** Called once with the caught error. Parent should dispatch
   *  setPathTracerEnabled(false) + setPtDeviceLost(true). */
  onError: (error: Error) => void;
}

interface State {
  hasError: boolean;
}

export class PTDeviceLostBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error): void {
    console.error(
      '[PTDeviceLostBoundary] PT subtree threw — flipping PT off.',
      error,
    );
    this.props.onError(error);
  }

  render() {
    // Fallback renders nothing. The user-visible banner is driven by
    // viewport.ptDeviceLost (set in onError) so it lives outside the
    // boundary's subtree and stays visible after this component unmounts.
    if (this.state.hasError) return null;
    return this.props.children;
  }
}
