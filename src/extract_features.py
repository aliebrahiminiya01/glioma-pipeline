"""Extract radiomics features for every BraTS 2020 case with PyRadiomics.

Copied from the radiomics project (../Glioma_Grade_Radiomics). The only change: the tumor
mask can come from any folder, so the same features can be computed from the expert masks
and from the U-Net masks.

For each case:
  * shape features (volume, surface, sphericity, ...) of the whole tumour (WT), tumour core (TC)
    and enhancing tumour (ET);
  * first-order and texture features (GLCM, GLRLM, GLSZM) of each MRI sequence inside WT and TC.

BraTS labels: 1 = necrotic / non-enhancing core, 2 = edema, 4 = enhancing tumour.
WT = {1, 2, 4}, TC = {1, 4}, ET = {4}.

MRI intensities have no absolute scale, so every sequence is z-scored over brain voxels before extraction.
Texture features of ET are not used: many low-grade tumours have no enhancing part at all, and the ET volume
(0 for those cases) already carries that information.

Usage:
    python src/extract_features.py --masks expert --out data/features_expert.csv
    python src/extract_features.py --masks data/auto_masks --out data/features_auto.csv
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Glioma_Grade_Radiomics" / "data" / "brats2020"
SEQUENCES = ("flair", "t1", "t1ce", "t2")
REGIONS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}
TEXTURE_REGIONS = ("WT", "TC")
MIN_VOXELS = 30  # smaller regions give unstable texture features

TEXTURE_SETTINGS = {
    "binWidth": 0.25,          # on z-scored intensities: roughly 30-40 grey levels inside a tumour
    "label": 1,
    "minimumROISize": MIN_VOXELS,
    "additionalInfo": False,
}


def zscore_in_brain(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    brain = array > 0
    mean, std = array[brain].mean(), array[brain].std()
    normalized = np.where(brain, (array - mean) / (std + 1e-8), 0).astype(np.float32)
    out = sitk.GetImageFromArray(normalized)
    out.CopyInformation(image)
    return out


def region_mask(seg: np.ndarray, labels: tuple[int, ...], reference: sitk.Image) -> tuple[sitk.Image, int]:
    mask = np.isin(seg, labels).astype(np.uint8)
    image = sitk.GetImageFromArray(mask)
    image.CopyInformation(reference)
    return image, int(mask.sum())


def make_extractor(feature_classes: dict) -> featureextractor.RadiomicsFeatureExtractor:
    extractor = featureextractor.RadiomicsFeatureExtractor(**TEXTURE_SETTINGS)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**feature_classes)
    return extractor


def extract_case(case_dir: Path, mask_dir: Path | None = None) -> dict:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    case = case_dir.name
    seg_path = (mask_dir or case_dir) / f"{case}_seg.nii.gz"
    seg_image = sitk.ReadImage(str(seg_path))
    seg = sitk.GetArrayFromImage(seg_image)
    row: dict = {"case": case}

    shape_extractor = make_extractor({"shape": []})
    texture_extractor = make_extractor({"firstorder": [], "glcm": [], "glrlm": [], "glszm": []})

    masks = {}
    for region, labels in REGIONS.items():
        mask, voxels = region_mask(seg, labels, seg_image)
        masks[region] = (mask, voxels)
        row[f"{region}_voxels"] = voxels
        if voxels >= MIN_VOXELS:
            for name, value in shape_extractor.execute(seg_image, mask).items():
                row[f"{region}_{name.replace('original_', '')}"] = float(value)

    for sequence in SEQUENCES:
        image = zscore_in_brain(sitk.ReadImage(str(case_dir / f"{case}_{sequence}.nii.gz")))
        for region in TEXTURE_REGIONS:
            mask, voxels = masks[region]
            if voxels < MIN_VOXELS:
                continue
            for name, value in texture_extractor.execute(image, mask).items():
                row[f"{sequence}_{region}_{name.replace('original_', '')}"] = float(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", default="expert", help='"expert" or a folder with <case>_seg.nii.gz files')
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mask_dir = None if args.masks == "expert" else Path(args.masks).resolve()
    cases = sorted(p for p in DATA.iterdir() if p.is_dir())
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_case, case, mask_dir): case.name for case in cases}
        for future in tqdm(as_completed(futures), total=len(futures), desc="radiomics"):
            try:
                rows.append(future.result())
            except Exception as error:  # report and continue; a single bad case should not stop the run
                failed.append((futures[future], repr(error)))

    features = pd.DataFrame(rows).sort_values("case")
    labels = pd.read_csv(DATA / "name_mapping.csv")[["BraTS_2020_subject_ID", "Grade"]]
    labels.columns = ["case", "grade"]
    features = labels.merge(features, on="case", how="inner")
    features.to_csv(args.out, index=False)
    print(f"{len(features)} cases, {features.shape[1] - 2} features -> {args.out}")
    for case, error in failed:
        print("FAILED", case, error)


if __name__ == "__main__":
    main()
