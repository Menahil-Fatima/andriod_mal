import random
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from SETTINGS import ensure_folders, GRAY_DATASET_DIR, TRAINED_MODELS_DIR
from Models.Exp1_CNN4 import Exp1_CNN4

class PairDataset(Dataset):
    def __init__(self, root: Path, train_families, pairs_per_epoch=3000, seed=7):
        self.root = root
        self.train_families = train_families
        self.pairs_per_epoch = pairs_per_epoch
        random.seed(seed)
        self.paths = {fam: list((root / fam).glob("*.png")) for fam in train_families}
        for fam in train_families:
            if len(self.paths[fam]) < 2:
                raise ValueError(f"Need at least 2 images in {fam} for training pairs.")

    def __len__(self):
        return self.pairs_per_epoch

    def _load(self, p: Path):
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

    def __getitem__(self, idx):
        same = random.random() < 0.5
        if same:
            fam = random.choice(self.train_families)
            a, b = random.sample(self.paths[fam], 2)
            y = torch.tensor(1.0)
        else:
            fam1, fam2 = random.sample(self.train_families, 2)
            a = random.choice(self.paths[fam1])
            b = random.choice(self.paths[fam2])
            y = torch.tensor(0.0)
        return self._load(a), self._load(b), y

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, y):
        d = torch.norm(e1 - e2, p=2, dim=1)
        loss_same = y * (d ** 2)
        loss_diff = (1 - y) * (torch.clamp(self.margin - d, min=0.0) ** 2)
        return (loss_same + loss_diff).mean()

def train_exp1(train_families, epochs=3, batch=32, lr=1e-3, device="cpu"):
    ensure_folders()
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ds = PairDataset(GRAY_DATASET_DIR, train_families, pairs_per_epoch=4000)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    model = Exp1_CNN4(out_dim=128).to(device)
    loss_fn = ContrastiveLoss(margin=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for x1, x2, y in tqdm(dl, desc=f"Exp1 Epoch {ep}/{epochs}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            e1, e2 = model(x1), model(x2)
            loss = loss_fn(e1, e2, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"Epoch {ep}: loss={total/len(dl):.4f}")

    out = TRAINED_MODELS_DIR / "exp1_cnn4.pt"
    torch.save(model.state_dict(), out)
    print("Saved:", out)

if __name__ == "__main__":
    # ✅ Train only 3 families (example)
    train_exp1(["Adware", "Banking", "Riskware"], epochs=3, device="cpu")
