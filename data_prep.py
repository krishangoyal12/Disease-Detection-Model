import json
import os
import random
import shutil
from pathlib import Path

# Builds a combined dataset with train/valid splits and consistent class naming.
# Usage: python data_prep.py
# Optional env vars:
#   DATA_ROOT   (default: "disease data")
#   OUT_DIR     (default: "disease data/combined")
#   LEGACY_DATASET (path to an existing dataset with train/valid folders)
#   SPLIT_RATIO (default: 0.2)
#   SEED        (default: 42)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "disease data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_ROOT / "combined"))
LEGACY_DATASET = os.environ.get("LEGACY_DATASET")
SPLIT_RATIO = float(os.environ.get("SPLIT_RATIO", "0.2"))
SEED = int(os.environ.get("SEED", "42"))

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CROP_MAP = {
    "corn or maize disease": "Corn",
    "Cotton Disease": "Cotton",
    "rice-leaf-disease-image": "Rice",
    "sugarcane-leaf-disease-dataset": "Sugarcane",
    "wheat data": "Wheat",
}


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_")


def _copy_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix.lower() in VALID_EXTS:
            shutil.copy2(item, dst_dir / item.name)
            count += 1
    return count


def _split_and_copy(class_dir: Path, crop: str) -> None:
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not images:
        return
    random.shuffle(images)
    split_idx = max(1, int(len(images) * (1.0 - SPLIT_RATIO)))
    train_imgs = images[:split_idx]
    valid_imgs = images[split_idx:]

    class_name = _sanitize(f"{crop}___{class_dir.name}")
    train_out = OUT_DIR / "train" / class_name
    valid_out = OUT_DIR / "valid" / class_name
    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(img, train_out / img.name)
    for img in valid_imgs:
        shutil.copy2(img, valid_out / img.name)


def _copy_split_dataset(root: Path, crop: str) -> None:
    train_dir = root / "train"
    valid_dir = root / "valid"
    if not valid_dir.exists():
        valid_dir = root / "val"

    for split_name, split_dir in ("train", train_dir), ("valid", valid_dir):
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = _sanitize(f"{crop}___{class_dir.name}")
            out_dir = OUT_DIR / split_name / class_name
            _copy_images(class_dir, out_dir)


def _copy_legacy_dataset(root: Path) -> None:
    for split_name in ("train", "valid"):
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            out_dir = OUT_DIR / split_name / _sanitize(class_dir.name)
            _copy_images(class_dir, out_dir)


def _ensure_valid_samples() -> None:
    train_root = OUT_DIR / "train"
    valid_root = OUT_DIR / "valid"
    if not train_root.exists() or not valid_root.exists():
        return

    for class_dir in train_root.iterdir():
        if not class_dir.is_dir():
            continue
        valid_dir = valid_root / class_dir.name
        valid_dir.mkdir(parents=True, exist_ok=True)

        valid_count = sum(1 for p in valid_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
        if valid_count > 0:
            continue

        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        if len(images) < 2:
            continue

        random.shuffle(images)
        move_count = max(1, int(len(images) * SPLIT_RATIO))
        for img in images[:move_count]:
            shutil.move(str(img), valid_dir / img.name)


def _sync_valid_with_train() -> None:
    train_root = OUT_DIR / "train"
    valid_root = OUT_DIR / "valid"
    if not train_root.exists() or not valid_root.exists():
        return

    train_classes = {p.name for p in train_root.iterdir() if p.is_dir()}
    valid_classes = {p.name for p in valid_root.iterdir() if p.is_dir()}

    # Remove any valid-only classes so train/valid class lists match.
    for extra in valid_classes - train_classes:
        shutil.rmtree(valid_root / extra, ignore_errors=True)

    # Ensure missing valid classes get a small split from train.
    for missing in train_classes - valid_classes:
        valid_dir = valid_root / missing
        valid_dir.mkdir(parents=True, exist_ok=True)

        train_dir = train_root / missing
        images = [p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        if len(images) < 2:
            continue
        random.shuffle(images)
        move_count = max(1, int(len(images) * SPLIT_RATIO))
        for img in images[:move_count]:
            shutil.move(str(img), valid_dir / img.name)


def main() -> None:
    random.seed(SEED)

    if OUT_DIR.exists():
        raise SystemExit(f"Output directory already exists: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for folder_name, crop in CROP_MAP.items():
        src = DATA_ROOT / folder_name
        if not src.exists():
            continue

        has_train = (src / "train").exists() or (src / "val").exists() or (src / "valid").exists()
        if has_train:
            _copy_split_dataset(src, crop)
        else:
            for class_dir in src.iterdir():
                if class_dir.is_dir():
                    _split_and_copy(class_dir, crop)

    if LEGACY_DATASET:
        legacy_root = Path(LEGACY_DATASET)
        if legacy_root.exists():
            _copy_legacy_dataset(legacy_root)

    _sync_valid_with_train()
    _ensure_valid_samples()

    # Write class list in training order (alphabetical by folder name)
    class_dirs = sorted((OUT_DIR / "train").iterdir())
    class_names = [d.name for d in class_dirs if d.is_dir()]
    with open("class_names.json", "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    print("Combined dataset ready at:", OUT_DIR)
    print("Class names written to class_names.json")


if __name__ == "__main__":
    main()
