# Exporting Trained Weights for InferenceGraph

After training the PyTorch model (see `train.py.md`), export the weights to the
binary format consumed by `@vitrum/walkaround-hybrid`'s `ModelWeights` interface.

---

## ModelWeights format (TypeScript)

```typescript
export interface ModelWeights {
  readonly weights: ReadonlyMap<string, Float32Array>;  // layer name → weight tensor
  readonly biases:  ReadonlyMap<string, Float32Array>;  // layer name → bias tensor
}
```

Keys in the maps are the `output` field of each `InferenceLayer` in
`WALKAROUND_DENOISER_UNET_SPEC`. The weight array must match the WGSL buffer
layout expected by each kernel.

---

## Weight tensor layouts

| Layer kind | WGSL kernel | Weight layout | Shape |
|---|---|---|---|
| `conv2d` | `conv2dKernel` | OIKK (output, input, kH, kW) | `[outC, inC, kH, kW]` |
| `transposed_conv2d` | `transposedConv2dKernel` | IOKK (input, output, kH, kW) | `[inC, outC, kH, kW]` |
| `relu` | `reluKernel` | no weights | — |
| `skip` | `skipConnectionKernel` | no weights | — |

Bias shape is always `[outC]` for layers that have biases (`conv2d`, `transposed_conv2d`).

---

## PyTorch weight name mapping

PyTorch stores weights as `[outC, inC, kH, kW]` for `nn.Conv2d` and
`[inC, outC, kH, kW]` for `nn.ConvTranspose2d`. These match the WGSL layouts above
exactly — no transposition needed.

```python
import torch
import numpy as np
import struct

model = VitDenoiserUNet()
model.load_state_dict(torch.load('vitrum-denoiser.pth', map_location='cpu'))
model.eval()

# Map from PyTorch parameter name → UNET_SPEC layer output name.
# Each tuple: (weight_param, bias_param, output_tensor_name).
LAYER_MAP = [
    ('enc1.0.weight', 'enc1.0.bias',   'enc1_conv'),
    ('enc2.0.weight', 'enc2.0.bias',   'enc2_conv'),
    ('enc3.0.weight', 'enc3.0.bias',   'enc3_conv'),
    ('btn.0.weight',  'btn.0.bias',    'btn_conv'),
    ('dec3_up.weight','dec3_up.bias',  'dec3_up'),
    ('dec3_conv.0.weight','dec3_conv.0.bias', 'dec3_conv'),
    ('dec2_up.weight','dec2_up.bias',  'dec2_up'),
    ('dec2_conv.0.weight','dec2_conv.0.bias', 'dec2_conv'),
    ('dec1_up.weight','dec1_up.bias',  'dec1_up'),
    ('dec1_conv.0.weight','dec1_conv.0.bias', 'dec1_conv'),
    ('proj.weight',   'proj.bias',     'denoisedColor'),
]

def export_weights(model, layer_map, out_path):
    weights = {}
    biases  = {}
    state   = model.state_dict()

    for w_name, b_name, layer_output in layer_map:
        w = state[w_name].numpy().astype(np.float32)
        b = state[b_name].numpy().astype(np.float32)
        weights[layer_output] = w.flatten()   # OIKK or IOKK, row-major
        biases[layer_output]  = b.flatten()

    # Write as a simple binary file: header + data sections.
    # Format: see "Binary file format" section below.
    with open(out_path, 'wb') as f:
        # Magic + version header (8 bytes)
        f.write(b'VITRUMW1')
        # Number of entries (u32 LE)
        f.write(struct.pack('<I', len(weights)))
        for name, arr in weights.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<Q', len(arr)))       # element count (u64 LE)
            f.write(arr.tobytes())
        for name, arr in biases.items():
            # Biases stored in a second block, same format but tag 'B'
            name_bytes = (name + ':bias').encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<Q', len(arr)))
            f.write(arr.tobytes())

export_weights(model, LAYER_MAP, 'vitrum-denoiser.vitrum-weights')
print('Exported vitrum-denoiser.vitrum-weights')
```

---

## Binary file format

```
Offset  Length  Content
──────────────────────────────────────────────────────
0       8       Magic: "VITRUMW1" (ASCII)
8       4       entryCount (u32 LE) — number of weight tensors
                (NOT including bias; biases follow with ":bias" suffix)
12+     variable  Entry records (repeated entryCount × 2 times):
                  nameLen  (u32 LE)
                  name     (UTF-8 string, nameLen bytes)
                  elemCount (u64 LE) — number of f32 elements
                  data      (float32 LE, elemCount × 4 bytes)
```

Bias entries use the same entry format but with the name suffix `:bias`
(e.g. `enc1_conv:bias`).

---

## JavaScript loader (host-side)

```typescript
async function loadWeightsFromFile(url: string): Promise<ModelWeights> {
  const buf = await (await fetch(url)).arrayBuffer();
  const view = new DataView(buf);

  // Verify magic
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== 'VITRUMW1') throw new Error('Invalid vitrum-weights file');

  const entryCount = view.getUint32(8, true);  // LE
  const weights = new Map<string, Float32Array>();
  const biases  = new Map<string, Float32Array>();
  let offset = 12;

  const readEntry = () => {
    const nameLen   = view.getUint32(offset, true); offset += 4;
    const name      = new TextDecoder().decode(new Uint8Array(buf, offset, nameLen));
    offset += nameLen;
    const elemCount = Number(view.getBigUint64(offset, true)); offset += 8;
    const data      = new Float32Array(buf, offset, elemCount);
    offset += elemCount * 4;
    return { name, data };
  };

  // Read weight entries × 2 (weights + biases, interleaved)
  for (let i = 0; i < entryCount * 2; i++) {
    const { name, data } = readEntry();
    if (name.endsWith(':bias')) {
      biases.set(name.slice(0, -5), data);
    } else {
      weights.set(name, data);
    }
  }

  return { weights, biases };
}
```

---

## Validation after export

Verify the exported file is self-consistent:

```python
# Reload and compare first layer's weight sum to PyTorch reference.
with open('vitrum-denoiser.vitrum-weights', 'rb') as f:
    raw = f.read()
# Parse and check enc1_conv weights sum against model.enc1[0].weight.sum()
```

And in TypeScript (Node or browser), check:

```typescript
const weights = await loadWeightsFromFile('/models/vitrum-denoiser.vitrum-weights');
console.assert(weights.weights.has('enc1_conv'), 'enc1_conv weights present');
console.assert(weights.biases.has('enc1_conv'),  'enc1_conv biases present');
// Expected element counts from unetArchitecture.ts:
// enc1_conv weights: 9 × 24 × 3 × 3 = 1,944 elements
// enc1_conv biases:  24 elements
console.assert(weights.weights.get('enc1_conv')!.length === 1944);
console.assert(weights.biases.get('enc1_conv')!.length  === 24);
```
