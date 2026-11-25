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

def practical_03(image, out):
    gray = ensure_gray(image)
    second = cv2.flip(gray, 1)
    save(out / "03_addition.png", cv2.add(gray, second))
    save(out / "03_subtraction.png", cv2.subtract(gray, second))
    save(out / "03_mean.png", np.full_like(gray, int(np.mean(gray))))

    brighter = np.clip(gray.astype(np.float32) + 40, 0, 255).astype(np.uint8)
    darker = np.clip(gray.astype(np.float32) - 40, 0, 255).astype(np.uint8)
    save(out / "03_brighter_mean.png", brighter)
    save(out / "03_darker_mean.png", darker)



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 03: Image Arithmetic")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_03(image, args.output)
    print(f"Practical 03 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
