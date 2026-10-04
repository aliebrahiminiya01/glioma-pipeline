"""
Step 2 - Train the U-Net for one fold.

Fold k is held out completely: the model never sees those patients, so the masks it
predicts for them later are as honest as masks for new patients. The other two folds are
split 90/10 into training and validation (validation picks the best epoch).

Model, loss and optimizer are the same as in the U-Net project (../C1).
10 epochs instead of 30: there, validation Dice was already 0.86 after epoch 1 and
improved by only ~0.02 afterwards.

Usage:
    python src/train.py --fold 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader

from dataset import BrainSliceDataset


def make_model():
    return UNet(spatial_dims=2, in_channels=4, out_channels=4,
                channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)


def dice_score(pred, target, min_class):
    p, t = pred >= min_class, target >= min_class
    total = p.sum().item() + t.sum().item()
    return 1.0 if total == 0 else 2 * (p & t).sum().item() / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).to(torch.uint8).cpu())
        targets.append(y.to(torch.uint8))
    preds, targets = torch.cat(preds), torch.cat(targets)
    return {"WT": dice_score(preds, targets, 1),
            "TC": dice_score(preds, targets, 2),
            "ET": dice_score(preds, targets, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--slices-dir", default="data/slices")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    folds = json.loads(Path("data/folds.json").read_text())
    rest = sorted(c for c, k in folds.items() if k != args.fold)
    random.Random(args.seed).shuffle(rest)
    n_val = len(rest) // 10
    val_cases, train_cases = sorted(rest[:n_val]), sorted(rest[n_val:])

    train_dl = DataLoader(BrainSliceDataset(args.slices_dir, train_cases, augment=True), batch_size=16, shuffle=True)
    val_dl = DataLoader(BrainSliceDataset(args.slices_dir, val_cases), batch_size=16)
    print(f"fold {args.fold}: {len(train_cases)} train / {len(val_cases)} val patients, "
          f"{len(train_dl.dataset)} training slices | device {device}")

    model = make_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(f"runs/fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        dice = evaluate(model, val_dl, device)
        mean = sum(dice.values()) / 3
        history.append({"epoch": epoch, "loss": total_loss / len(train_dl), **dice, "mean": mean})

        saved = ""
        if mean > best:
            best = mean
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            saved = "  <- saved"
        print(f"epoch {epoch:2d} | loss {history[-1]['loss']:.4f} | WT {dice['WT']:.3f}  TC {dice['TC']:.3f}  "
              f"ET {dice['ET']:.3f} | mean {mean:.3f} | {time.time() - t0:.0f}s{saved}", flush=True)

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "split.json").write_text(json.dumps({"train": train_cases, "val": val_cases}, indent=1))
    print(f"best mean val Dice: {best:.3f}")


if __name__ == "__main__":
    main()
