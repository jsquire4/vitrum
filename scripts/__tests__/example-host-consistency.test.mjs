import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const sourceUrl = new URL('../../examples/shared/exampleHost.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  reportDiagnostics: true,
});
assert.deepEqual(
  transpiled.diagnostics?.filter(
    (diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error,
  ),
  [],
);
const helpers = await import(
  `data:text/javascript;base64,${Buffer.from(transpiled.outputText).toString('base64')}`
);

function withDevicePixelRatio(devicePixelRatio, callback) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { devicePixelRatio },
  });
  try {
    callback();
  } finally {
    if (previous == null) delete globalThis.window;
    else Object.defineProperty(globalThis, 'window', previous);
  }
}

function trackedCanvas(clientWidth, clientHeight, width = 300, height = 150) {
  let backingWidth = width;
  let backingHeight = height;
  let writes = 0;
  return {
    canvas: {
      clientWidth,
      clientHeight,
      get width() {
        return backingWidth;
      },
      set width(value) {
        writes += 1;
        backingWidth = value;
      },
      get height() {
        return backingHeight;
      },
      set height(value) {
        writes += 1;
        backingHeight = value;
      },
    },
    writes: () => writes,
  };
}

test('example host uses physical pixels and does not rewrite an unchanged backing store', () => {
  withDevicePixelRatio(2, () => {
    const tracked = trackedCanvas(320, 180);
    assert.deepEqual(helpers.syncCanvasToDisplaySize(tracked.canvas), {
      width: 640,
      height: 360,
      devicePixelRatio: 2,
      resized: true,
    });
    assert.equal(tracked.writes(), 2);

    assert.equal(helpers.syncCanvasToDisplaySize(tracked.canvas).resized, false);
    assert.equal(tracked.writes(), 2);
  });
});

test('example host preserves backing size without layout and rejects unsafe dimensions', () => {
  withDevicePixelRatio(Number.NaN, () => {
    const tracked = trackedCanvas(0, 0, 512, 256);
    assert.deepEqual(helpers.syncCanvasToDisplaySize(tracked.canvas), {
      width: 512,
      height: 256,
      devicePixelRatio: 1,
      resized: false,
    });
  });

  withDevicePixelRatio(Number.MAX_VALUE, () => {
    const tracked = trackedCanvas(2, 2);
    assert.throws(() => helpers.syncCanvasToDisplaySize(tracked.canvas), /safe-integer range/);
  });
});

test('camera helpers keep view position and projection aspect mathematically consistent', () => {
  const view = helpers.createAxisAlignedView(helpers.CORNELL_CAMERA_POSITION);
  assert.deepEqual(Array.from(view.slice(12, 16)), [0, -1, -4, 1]);

  const square = helpers.createPerspectiveProjection(100, 100);
  const wide = helpers.createPerspectiveProjection(200, 100);
  assert.ok(Math.abs(wide[0] * 2 - square[0]) < 1e-6);
  assert.equal(wide[5], square[5]);
  assert.equal(wide[11], -1);
});
