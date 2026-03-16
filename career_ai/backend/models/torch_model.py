"""
PyTorch CareerNet and TorchCareerClassifier — shared module.
Imported by train_model.py (training) and predictor.py / research_report.py (inference).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CareerNet(nn.Module):
    """4-block MLP with BatchNorm + Dropout: 512 → 256 → 128 → 64 → n_classes."""

    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),    # lighter deeper in
            # Block 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            # Output
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TorchCareerClassifier:
    """
    Sklearn-compatible wrapper around CareerNet.
    Exposes: fit, predict, predict_proba, loss_curve_, validation_scores_, n_iter_, classes_
    """

    def __init__(self):
        self.model_              = None
        self.classes_            = None
        self.n_iter_             = 0
        self.loss_curve_         = []
        self.validation_scores_  = []

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_tensor(X, dtype=torch.float32):
        return torch.tensor(np.array(X), dtype=dtype)

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=300, lr=3e-3, batch_size=128, patience=25):

        n_features = X_train.shape[1]
        n_classes  = len(np.unique(y_train))
        device     = torch.device("cpu")

        net = CareerNet(n_features, n_classes, dropout=0.2).to(device)

        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        # Cosine annealing: smooth LR decay, no aggressive spike → lower loss
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()  # no label smoothing — keeps loss honest

        Xt = self._to_tensor(X_train)
        yt = torch.tensor(y_train, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        best_val_acc = 0.0
        best_state   = None
        no_improve   = 0
        self.loss_curve_        = []
        self.validation_scores_ = []

        print(f"      Training on {len(X_train)} samples | {epochs} max epochs | patience={patience}")

        for epoch in range(1, epochs + 1):
            net.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(net(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            scheduler.step()   # CosineAnnealingLR steps once per epoch
            avg_loss = epoch_loss / len(X_train)
            self.loss_curve_.append(avg_loss)

            if X_val is not None:
                val_acc = self._score(net, X_val, y_val)
                self.validation_scores_.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = {k: v.clone() for k, v in net.state_dict().items()}
                    no_improve   = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"      Early stop at epoch {epoch}  |  best val acc: {best_val_acc*100:.2f}%")
                        break

            if epoch % 25 == 0:
                vs = f"  val_acc={self.validation_scores_[-1]*100:.2f}%" if X_val is not None else ""
                print(f"      Epoch {epoch:3d}  loss={avg_loss:.4f}{vs}")

        if best_state is not None:
            net.load_state_dict(best_state)

        self.model_  = net
        self.n_iter_ = epoch
        return self

    def _score(self, net, X, y):
        net.eval()
        with torch.no_grad():
            logits = net(self._to_tensor(X))
            preds  = logits.argmax(dim=1).numpy()
        return float((preds == np.array(y)).mean())

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._to_tensor(X))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._to_tensor(X))
            return torch.softmax(logits, dim=1).numpy()
