import torch
from SETTINGS import ensure_folders, GRAY_DATASET_DIR, TRAINED_MODELS_DIR
from Models.Exp3_ResNet34 import Exp3_ResNet34
from Train_Exp1_CNN4 import PairDataset, ContrastiveLoss

def train_exp3(train_families, epochs=2, batch=8, lr=1e-4, device="cpu"):
    ensure_folders()
    ds = PairDataset(GRAY_DATASET_DIR, train_families, pairs_per_epoch=2000)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    model = Exp3_ResNet34(out_dim=128).to(device)
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
        print(f"Exp3 Epoch {ep}: loss={total/len(dl):.4f}")

    out = TRAINED_MODELS_DIR / "exp3_resnet34.pt"
    torch.save(model.state_dict(), out)
    print("Saved:", out)

if __name__ == "__main__":
    train_exp3(["Adware", "Banking", "Riskware"], epochs=2, device="cpu")
________________________________________
