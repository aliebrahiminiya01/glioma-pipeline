"""
Step 1 - Split BraTS 2020 into 3 folds and convert the volumes into 2D training slices.

Same preprocessing as the U-Net project (../C1): z-score per modality, all tumor slices
+ 10 empty brain slices per patient, center crop 240 -> 192, resize to 128x128.

Labels are renumbered to the convention the U-Net code uses:
    BraTS 2020:  1 = necrotic/non-enhancing, 2 = edema, 4 = enhancing
    here:        1 = edema, 2 = non-enhancing, 3 = enhancing
so that ">= 1" is the whole tumor, ">= 2" the tumor core and "== 3" the enhancing part.

Usage:
    python src/prepare_slices.py --brats-dir ../Glioma_Grade_Radiomics/data/brats2020
"""
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

MODALITIES = ["flair", "t1", "t1ce", "t2"]  # same channel order as the MSD data in C1
BRATS_TO_UNET = {0: 0, 1: 2, 2: 1, 4: 3}


def normalize(volume):
    """Z-score each modality over brain voxels only; the background stays 0."""
    out = np.zeros_like(volume, dtype=np.float32)
    for m in range(volume.shape[-1]):
        v = volume[..., m]
        brain = v > 0
        if brain.any():
            out[..., m][brain] = (v[brain] - v[brain].mean()) / (v[brain].std() + 1e-8)
    return out


def load_case(case_dir: Path):
    case = case_dir.name
    image = np.stack([nib.load(str(case_dir / f"{case}_{m}.nii.gz")).get_fdata(dtype=np.float32)
                      for m in MODALITIES], axis=-1)  # (240, 240, 155, 4)
    seg = np.asarray(nib.load(str(case_dir / f"{case}_seg.nii.gz")).dataobj).astype(np.uint8)
    label = np.zeros_like(seg)
    for brats, unet in BRATS_TO_UNET.items():
        label[seg == brats] = unet
    return normalize(image), label


def select_slices(label, brain, n_empty, rng):
    tumor = [z for z in range(label.shape[2]) if label[:, :, z].any()]
    empty = [z for z in range(label.shape[2]) if not label[:, :, z].any() and brain[:, :, z].any()]
    return sorted(tumor + rng.sample(empty, min(n_empty, len(empty))))


def to_slices(image, label, zs, crop, size):
    s = (image.shape[0] - crop) // 2
    image = image[s:s + crop, s:s + crop, zs, :]
    label = label[s:s + crop, s:s + crop, zs]
    f = size / crop
    image = zoom(image, (f, f, 1, 1), order=1)
    label = zoom(label, (f, f, 1), order=0)  # nearest neighbor: labels must stay whole numbers
    return image.transpose(2, 3, 0, 1).astype(np.float16), label.transpose(2, 0, 1).astype(np.uint8)


def make_folds(brats_dir: Path, n_folds: int, seed: int):
    """Stratify on grade and on whether the patient has survival data, so every fold gets its share of both."""
    names = pd.read_csv(brats_dir / "name_mapping.csv")
    survival = pd.read_csv(brats_dir / "survival_info.csv")
    cases = names["BraTS_2020_subject_ID"]
    group = names["Grade"] + cases.isin(survival["Brats20ID"]).map({True: "_surv", False: ""})

    folds = {}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(skf.split(cases, group)):
        for case in cases.iloc[test]:
            folds[case] = k
    return dict(sorted(folds.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-dir", default="../Glioma_Grade_Radiomics/data/brats2020")
    p.add_argument("--out-dir", default="data/slices")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--empty-slices", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    brats_dir, out_dir = Path(args.brats_dir).expanduser(), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    folds = make_folds(brats_dir, args.folds, args.seed)
    (out_dir.parent / "folds.json").write_text(json.dumps(folds, indent=1))

    total = 0
    for case in tqdm(folds, desc="cases"):
        if (out_dir / f"{case}_lbl.npy").exists():
            continue
        image, label = load_case(brats_dir / case)
        zs = select_slices(label, image.any(axis=-1), args.empty_slices, rng)
        img, lbl = to_slices(image, label, zs, args.crop, args.size)
        np.save(out_dir / f"{case}_img.npy", img)
        np.save(out_dir / f"{case}_lbl.npy", lbl)
        total += len(lbl)

    sizes = pd.Series(folds).value_counts().sort_index().to_dict()
    print(f"saved {total} new slices to {out_dir} | cases per fold: {sizes}")


if __name__ == "__main__":
    main()
