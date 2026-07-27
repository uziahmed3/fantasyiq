"""PyTorch MLP.

Included because the original project was PyTorch, and because it is a fair test of
whether a neural net earns its keep on ~10 tabular features (usually: it does not, and
saying so is a better interview answer than pretending otherwise).
"""

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.features import FEATURE_ORDER
from app.torch_model import FantasyMLP
from train.common import metrics, report, write_metadata
from train.dataset import ARTIFACT_DIR, load_dataset, time_split

VERSION = "torch_v1"
HIDDEN = 64
EPOCHS = 120
PATIENCE = 15
BATCH = 256
LR = 1e-3


def main() -> dict:
    torch.manual_seed(7)
    np.random.seed(7)

    df, source = load_dataset()
    x_tr, y_tr, x_va, y_va, cutoff = time_split(df)

    # Scaler is fit on training rows only, then persisted alongside the weights - fitting
    # it on the full dataset is a subtle and very common leak.
    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_va_s = scaler.transform(x_tr), scaler.transform(x_va)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_tr_s.astype("float32")), torch.from_numpy(y_tr.reshape(-1, 1))
        ),
        batch_size=BATCH,
        shuffle=True,
    )
    model = FantasyMLP(n_features=len(FEATURE_ORDER), hidden=HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.HuberLoss(delta=6.0)  # fantasy points are heavy-tailed; Huber > MSE here

    x_va_t = torch.from_numpy(x_va_s.astype("float32"))
    y_va_t = torch.from_numpy(y_va.reshape(-1, 1))

    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va_t), y_va_t))
        sched.step(val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(x_va_t).numpy().ravel()
    scores = metrics(y_va, preds)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / f"{VERSION}.pt")
    joblib.dump(scaler, ARTIFACT_DIR / f"{VERSION}.scaler.joblib")
    write_metadata(
        VERSION,
        "pytorch",
        scores,
        {"hidden": HIDDEN, "epochs_max": EPOCHS, "lr": LR, "loss": "HuberLoss(delta=6.0)"},
    )
    report(VERSION, scores, source, cutoff)
    return scores


if __name__ == "__main__":
    main()
