# ─────────────────────────────────────────────────────────────────────────────
# Imports & Class Definitions only at the top level.
# Everything else is inside if __name__ == '__main__' to prevent
# DataLoader worker processes from re-running setup code on Windows.
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys, os, time


class InnerResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, is_forth=False):
        super().__init__()
        if is_forth:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, (3,1), padding='same'), nn.ReLU(),
                nn.Conv2d(out_c, out_c, (1,3), padding='same'), nn.ReLU())
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding='same'), nn.ReLU())
        self.bn       = nn.BatchNorm2d(out_c)
        self.pool     = nn.MaxPool2d(2, stride=1, padding=1)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.conv(x)
        res = self.pool(res)[:, :, :x.shape[2], :x.shape[3]]
        return self.bn(res) + self.shortcut(x)


class SauberModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'), nn.ReLU())
        self.b1 = nn.ModuleList([InnerResidualBlock(64 if i==0 else 96,  96,  3) for i in range(3)])
        self.b2 = nn.ModuleList([InnerResidualBlock(64 if i==0 else 32,  32,  5) for i in range(3)])
        self.b3 = nn.ModuleList([InnerResidualBlock(64 if i==0 else 128, 128, 3) for i in range(3)])
        self.b4 = nn.ModuleList([InnerResidualBlock(64 if i==0 else 64,  64,  is_forth=True) for i in range(3)])
        self.concat_conv = nn.Conv2d(96+32+128+64, 64, 1)
        self.dropout     = nn.Dropout(0.3)
        self.gap         = nn.AdaptiveAvgPool2d(1)           # 64×64 → 1×1
        self.classifier  = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))  # no Sigmoid — BCEWithLogitsLoss handles it

    def forward(self, x):
        x        = self.stem(x)
        id_outer = x
        out1, out2, out3, out4 = x, x, x, x
        for i in range(3):
            out1, out2, out3, out4 = (self.b1[i](out1), self.b2[i](out2),
                                      self.b3[i](out3), self.b4[i](out4))
        m = torch.cat([out1, out2, out3, out4], dim=1)
        m = self.dropout(self.concat_conv(m))
        m = self.gap(m + id_outer)
        return self.classifier(m)


# ─────────────────────────────────────────────────────────────────────────────
# Everything below only runs in the main process
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── CUDA check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        print(f"       torch version: {torch.__version__}")
        print("       Run with:  .venv\\Scripts\\python pneumonia.py")
        sys.exit(1)

    torch.backends.cudnn.benchmark = True   # auto-select fastest cuDNN kernels
    device = torch.device('cuda')
    print(f'Using Device: {device}  ({torch.cuda.get_device_name(0)})', flush=True)
    print(f'Torch version: {torch.__version__}', flush=True)

    # ── Dataset setup ─────────────────────────────────────────────────────────
    import zipfile, subprocess

    os.environ['KAGGLE_USERNAME'] = "amrmsauber"
    os.environ['KAGGLE_KEY']      = "KGAT_ceeffe620f713b8e6130fd931de9cb9c"
    import kaggle

    base_dir = "chest_xray"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        dataset_slug = "paultimothymooney/chest-xray-pneumonia"
        zip_file = "chest-xray-pneumonia.zip"
        if not os.path.exists(zip_file):
            print("Starting download from Kaggle...", flush=True)
            subprocess.run(f"kaggle datasets download -d {dataset_slug}", shell=True)
        else:
            print("Dataset zip already exists. Skipping download.", flush=True)

    if not os.path.exists(os.path.join(base_dir, "train")):
        print("Extracting dataset...", flush=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        print("Extraction complete.", flush=True)
    else:
        print("Dataset already extracted.", flush=True)

    train_path = os.path.join(base_dir, 'train')
    test_path  = os.path.join(base_dir, 'test')
    if os.path.exists(base_dir):
        print(f"Data organized into: {os.listdir(base_dir)}", flush=True)
        def count_files(d): return sum(len(f) for _, _, f in os.walk(d))
        if os.path.exists(train_path): print(f"Train images: {count_files(train_path)}", flush=True)
        if os.path.exists(test_path):  print(f"Test  images: {count_files(test_path)}",  flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # num_workers=8  → 8 of your 72 Xeon cores prefetch images in parallel
    # persistent_workers → workers stay alive between epochs (no respawn cost)
    # prefetch_factor=4  → each worker queues 4 batches ahead
    N_WORKERS = 8
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(
        datasets.ImageFolder(train_path, transform=transform),
        batch_size=512, shuffle=True,
        num_workers=N_WORKERS, persistent_workers=True, prefetch_factor=4,
    )
    test_loader = DataLoader(
        datasets.ImageFolder(test_path, transform=transform),
        batch_size=512, shuffle=False,
        num_workers=N_WORKERS, persistent_workers=True, prefetch_factor=4,
    )
    n_batches = len(train_loader)
    print(f'DataLoaders ready — {N_WORKERS} workers, batch 512, '
          f'{n_batches} train batches / epoch', flush=True)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    # NOTE: torch.compile is disabled — requires Triton which is Linux-only.
    #       All other optimisations (AMP, cuDNN benchmark, workers) still apply.
    model     = SauberModel3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()  # AMP-safe: fuses Sigmoid + BCE in float32

    # AMP: float16 on RTX Tensor Cores → ~2× faster, halves activation memory
    scaler = torch.amp.GradScaler('cuda')

    epoch_losses, epoch_accs, epoch_recalls, epoch_times = [], [], [], []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(50):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        correct          = torch.tensor(0.0, device=device)
        total            = torch.tensor(0.0, device=device)
        true_positives   = torch.tensor(0.0, device=device)
        actual_positives = torch.tensor(0.0, device=device)

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            print(f'  Epoch {epoch+1}/50 | Batch {batch_idx+1}/{n_batches}',
                  end='\r', flush=True)

            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss     += loss.item()
            predictions       = torch.round(torch.sigmoid(outputs.detach()))
            correct          += (predictions == labels).sum()
            total            += labels.size(0)
            true_positives   += ((predictions == 1) & (labels == 1)).sum()
            actual_positives += (labels == 1).sum()

        avg_loss     = running_loss / n_batches
        epoch_acc    = (correct / total).item()
        epoch_recall = (true_positives / (actual_positives + 1e-8)).item()
        epoch_time   = (time.time() - start_time) / 60

        epoch_losses.append(avg_loss)
        epoch_accs.append(epoch_acc)
        epoch_recalls.append(epoch_recall)
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch+1}/50 | Loss: {avg_loss:.4f} | '
              f'Acc: {epoch_acc:.4f} | Recall: {epoch_recall:.4f} | '
              f'Time: {epoch_time:.2f} mins', flush=True)

    # ── Visualise ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(15, 5))
    for idx, (data, title, color) in enumerate([
        (epoch_losses,  'Training Loss',     'b'),
        (epoch_accs,    'Training Accuracy', 'g'),
        (epoch_recalls, 'Training Recall',   'r'),
    ]):
        plt.subplot(1, 3, idx+1)
        plt.plot(range(1, 51), data, marker='o', linestyle='-', color=color)
        plt.title(title + ' per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(title.split()[-1])
        plt.grid(True)
    plt.tight_layout()
    plt.show()
