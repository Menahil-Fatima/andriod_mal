from SETTINGS import (
    ensure_folders, FAMILIES,
    RAW_APK_DIR, SUPPORT_SET_APK_DIR,
    GRAY_DATASET_DIR, GRAY_SUPPORT_DIR,
    EXP_IMAGE_SIZE
)

from ApkToGrayImage import apk_bytes_to_gray_image
from ResizeImage import resize_gray
from pathlib import Path


def convert_one_root(apk_root, out_root, exp: str):

    size = EXP_IMAGE_SIZE[exp]

    for fam in FAMILIES:

        in_dir = apk_root / fam
        out_dir = out_root / fam
        out_dir.mkdir(parents=True, exist_ok=True)

        # iterate over APK files
        for apk_path in in_dir.iterdir():
            if not apk_path.is_file():
                continue

            try:
                # Windows will raise Errno 22 for too‑long paths; prefixing
                # with \\?\ disables the MAX_PATH check.  Catching the
                # exception lets us skip a problematic file.
                p = str(apk_path)
                if len(p) > 260:
                    p = "\\\\?\\" + p
                with open(p, "rb") as f:
                    data = f.read()
            except OSError as err:
                print(f"skipping {apk_path!r}: {err}")
                continue

            img = apk_bytes_to_gray_image(data)
            img = resize_gray(img, size)
            img.save(out_dir / f"{apk_path.stem}.png")


def convert_all(exp="exp1"):

    ensure_folders()

    convert_one_root(RAW_APK_DIR, GRAY_DATASET_DIR, exp)

    convert_one_root(SUPPORT_SET_APK_DIR, GRAY_SUPPORT_DIR, exp)


if __name__ == "__main__":

    # change exp1 / exp2 / exp3
    convert_all("exp1")

    print("Converted RAW_APK and SUPPORT_SET_APK to grayscale images.")