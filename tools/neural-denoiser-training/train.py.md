# Training Loop Specification (PyTorch Pseudocode)

This document specifies the PyTorch architecture and training loop that produces
a model matching the `WALKAROUND_DENOISER_UNET_SPEC` in
`packages/walkaround-hybrid/src/neural/unetArchitecture.ts`.

This is a **Markdown spec, not executable Python**. The training pipeline is
researcher/host concern; vitrum does not ship Python code. Implement this spec
in PyTorch to produce the model weights consumed by `InferenceGraph`.

---

## Architecture

```python
# Vitrum Walkaround Neural Denoiser — UNet
# Matches WALKAROUND_DENOISER_UNET_SPEC in unetArchitecture.ts exactly.
# Total parameters: 426,075.  Storage: ~1.63 MB (f32).

class VitDenoiserUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder — stride-2 conv downsample at each level
        self.enc1 = nn.Sequential(nn.Conv2d( 9, 24, 3, stride=2, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.ReLU())

        # Bottleneck — stride-1
        self.btn  = nn.Sequential(nn.Conv2d(96, 192, 3, stride=1, padding=1), nn.ReLU())

        # Decoder — transposed conv upsample + skip-add + conv
        self.dec3_up   = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec3_conv = nn.Sequential(nn.Conv2d(96, 96, 3, stride=1, padding=1), nn.ReLU())

        self.dec2_up   = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec2_conv = nn.Sequential(nn.Conv2d(48, 48, 3, stride=1, padding=1), nn.ReLU())

        self.dec1_up   = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec1_conv = nn.Sequential(nn.Conv2d(24, 24, 3, stride=1, padding=1), nn.ReLU())

        # Output projection — 1×1 conv to RGB
        self.proj = nn.Conv2d(24, 3, 1)

    def forward(self, x):
        # x: (B, 9, H, W) — noisy(3) + albedo(3) + normals(3), all in [0,1]

        # Encode
        e1 = self.enc1(x)    # (B, 24, H/2, W/2)
        e2 = self.enc2(e1)   # (B, 48, H/4, W/4)
        e3 = self.enc3(e2)   # (B, 96, H/8, W/8)

        # Bottleneck
        b  = self.btn(e3)    # (B, 192, H/8, W/8)

        # Decode with skip-add connections
        d3 = F.relu(self.dec3_up(b) + e3)   # (B, 96, H/4, W/4)
        d3 = self.dec3_conv(d3)

        d2 = F.relu(self.dec2_up(d3) + e2)  # (B, 48, H/2, W/2)
        d2 = self.dec2_conv(d2)

        d1 = F.relu(self.dec1_up(d2) + e1)  # (B, 24, H, W)
        d1 = self.dec1_conv(d1)

        return self.proj(d1)                 # (B, 3, H, W)  denoised RGB
```

**Important:** Skip connections use **addition** (not concatenation). This matches
the `skip` layer kind in `InferenceGraph` (`skipConnectionKernel`: output = A + B).
If you change to concatenation in PyTorch, the exported weight shapes will not match.

---

## Dataset loader

```python
class DenoisePairDataset(Dataset):
    def __init__(self, pair_dir, patch_size=128):
        self.files = sorted(Path(pair_dir).glob('*.npz'))
        self.patch = patch_size

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        noisy   = data['noisy']    # (H, W, 3)
        albedo  = data['albedo']   # (H, W, 3)
        normals = (data['normals'] + 1) / 2  # [-1,1] → [0,1]
        clean   = data['clean']    # (H, W, 3)

        # Random 128×128 crop (applied identically to all tensors)
        h, w = noisy.shape[:2]
        y = random.randint(0, h - self.patch)
        x = random.randint(0, w - self.patch)
        crop = slice(y, y+self.patch), slice(x, x+self.patch)
        noisy, albedo, normals, clean = [t[crop] for t in [noisy, albedo, normals, clean]]

        # Random horizontal flip
        if random.random() < 0.5:
            noisy, albedo, normals, clean = [np.fliplr(t) for t in [noisy, albedo, normals, clean]]

        # Reinhard tone-mapping for noisy + clean
        noisy = noisy / (1 + noisy)
        clean = clean / (1 + clean)

        # Stack input: (9, H, W)
        inp = np.concatenate([noisy, albedo, normals], axis=-1)
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float()
        tgt = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        return inp, tgt
```

---

## Training loop

```python
model = VitDenoiserUNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

dataset = DenoisePairDataset('tools/neural-denoiser-training/data/pairs/')
loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

for epoch in range(100):
    for inp, tgt in loader:
        inp, tgt = inp.cuda(), tgt.cuda()
        pred = model(inp)
        # Combine L1 + SSIM loss (common for denoising)
        loss = F.l1_loss(pred, tgt) + (1 - ssim(pred, tgt))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

torch.save(model.state_dict(), 'vitrum-denoiser.pth')
```

Recommended training budget: **100 epochs**, batch size 16, ~24 hours on a single A100.
For consumer GPUs: reduce batch size to 4, expect ~48 hours on an RTX 3090.

---

## Validation

After each epoch, evaluate on a held-out validation set (10% of pairs):

- **PSNR** (peak signal-to-noise ratio) — target: ≥35 dB at 8 SPP walkaround input.
- **SSIM** (structural similarity) — target: ≥0.95.
- **Visual inspection**: render a reference stained-glass scene at 4 SPP, run inference,
  compare to the 192 SPP PT_FINAL reference.

Checkpoint the model at the epoch with the best validation PSNR.

---

## SSIM note

PyTorch does not ship a built-in SSIM loss. Use `pytorch-msssim` or implement:

```python
from pytorch_msssim import ssim
# pip install pytorch-msssim
```

Or replace with purely L1 loss if SSIM dependency is inconvenient; quality will be
slightly softer on edges.

---

## Architecture validation (before training)

Verify parameter count matches the vitrum spec before committing to a long training run:

```python
model = VitDenoiserUNet()
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 426_075, f"Parameter count mismatch: {n_params}"
```
