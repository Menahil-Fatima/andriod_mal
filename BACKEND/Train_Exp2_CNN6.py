import torch
from SETTINGS import ensure_folders, GRAY_DATASET_DIR, TRAINED_MODELS_DIR
from Models.Exp2_CNN6 import Exp2_CNN6
from Train_Exp1_CNN4 import PairDataset, ContrastiveLoss

def train_exp2(train_families, epochs=3, batch=24, lr=1e-3, device="cpu"):
    ensure_folders()
    ds = PairDataset(GRAY_DATASET_DIR, train_families, pairs_per_epoch=4000)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    model = Exp2_CNN6(out_dim=128, dropout=0.5).to(device)
    loss_fn = ContrastiveLoss(margin=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for x1, x2, y in dl:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            e1, e2 = model(x1), model(x2)
            loss = loss_fn(e1, e2, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"Exp2 Epoch {ep}: loss={total/len(dl):.4f}")

    out = TRAINED_MODELS_DIR / "exp2_cnn6.pt"
    torch.save(model.state_dict(), out)
    print("Saved:", out)

if __name__ == "__main__":
    train_exp2(["Adware", "Banking", "Riskware"], epochs=3, device="cpu")
