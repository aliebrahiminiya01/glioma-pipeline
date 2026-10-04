"""
Step 3 - Predict a full 3D tumor mask for every patient with the model of the fold
that held this patient out, and save it in BraTS format (labels 1, 2, 4).

Every axial slice goes through the same preprocessing as in training (z-score, crop 240 -> 192,
resize to 128). The class probabilities are resized back to 192x192 before taking the
argmax, and the crop is padded back to 240x240.

Usage:
    python src/predict.py --brats-dir ../Glioma_Grade_Radiomics/data/brats2020
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm

from prepare_slices import load_case
from train import make_model

UNET_TO_BRATS = np.array([0, 2, 1, 4], dtype=np.uint8)  # index = U-Net class


@torch.no_grad()
def predict_volume(model, image, device, crop=192, size=128, batch=32):
    h, w, depth, _ = image.shape
    s = (h - crop) // 2
    x = image[s:s + crop, s:s + crop]                       # (192, 192, D, 4)
    x = zoom(x, (size / crop, size / crop, 1, 1), order=1)  # (128, 128, D, 4)
    x = torch.from_numpy(x.transpose(2, 3, 0, 1).copy())    # (D, 4, 128, 128)

    labels = []
    for i in range(0, depth, batch):
        probs = model(x[i:i + batch].to(device)).softmax(dim=1)
        probs = F.interpolate(probs, size=(crop, crop), mode="bilinear", align_corners=False)
        labels.append(probs.argmax(dim=1).to(torch.uint8).cpu())
    labels = torch.cat(labels).numpy().transpose(1, 2, 0)  # (192, 192, D)

    mask = np.zeros((h, w, depth), dtype=np.uint8)
    mask[s:s + crop, s:s + crop] = labels
    mask[~image.any(axis=-1)] = 0  # nothing outside the brain
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-dir", default="../Glioma_Grade_Radiomics/data/brats2020")
    p.add_argument("--out-dir", default="data/auto_masks")
    args = p.parse_args()

    brats_dir, out_dir = Path(args.brats_dir).expanduser(), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    folds = json.loads(Path("data/folds.json").read_text())

    for k in sorted(set(folds.values())):
        model = make_model().to(device)
        model.load_state_dict(torch.load(f"runs/fold{k}/best_model.pt", map_location=device))
        model.eval()
        cases = [c for c, f in folds.items() if f == k]
        for case in tqdm(cases, desc=f"fold {k}"):
            target = out_dir / f"{case}_seg.nii.gz"
            if target.exists():
                continue
            image, _ = load_case(brats_dir / case)
            mask = UNET_TO_BRATS[predict_volume(model, image, device)]
            reference = nib.load(str(brats_dir / case / f"{case}_seg.nii.gz"))  # same grid and header
            nib.save(nib.Nifti1Image(mask, reference.affine, reference.header), str(target))


if __name__ == "__main__":
    main()
