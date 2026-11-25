import argparse
from pathlib import Path

import cv2
import numpy as np


def ensure_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

def save(path, image):
    cv2.imwrite(str(path), image)

def load_or_create(path: Path, size=(300, 400)):
    image = cv2.imread(str(path)) if path else None
    if image is not None:
        return image
    h, w = size
    y, x = np.indices((h, w))
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = (x * 255 // w).astype(np.uint8)
    image[..., 1] = ((x + y) * 255 // (w + h)).astype(np.uint8)
    image[..., 2] = (y * 255 // h).astype(np.uint8)
    cv2.circle(image, (w // 2, h // 2), 75, (255, 255, 255), -1)
    cv2.rectangle(image, (40, 40), (150, 130), (0, 0, 255), 5)
    return image

def practical_06(image, out):
    gray = ensure_gray(image)
    h, w = gray.shape
    translation_matrix = np.float32([[1, 0, 40], [0, 1, 25]])
    save(out / "06_translation.png", cv2.warpAffine(gray, translation_matrix, (w, h)))

    save(out / "06_scaling.png", cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1)
    save(out / "06_rotation.png", cv2.warpAffine(gray, rotation_matrix, (w, h)))

    save(out / "06_shrinking.png", cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    save(out / "06_zooming.png", cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 06: Geometric Transformations")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_06(image, args.output)
    print(f"Practical 06 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
