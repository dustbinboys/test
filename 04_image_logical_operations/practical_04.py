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

def practical_04(image, out):
    gray = ensure_gray(image)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (gray.shape[1] // 2, gray.shape[0] // 2), 60, 255, -1)
    save(out / "04_and.png", cv2.bitwise_and(gray, mask))
    save(out / "04_or.png", cv2.bitwise_or(gray, mask))
    save(out / "04_intersection.png", cv2.bitwise_and(gray, mask))
    save(out / "04_not_negative.png", cv2.bitwise_not(gray))

    watermark = cv2.resize(cv2.flip(gray, 1), (gray.shape[1], gray.shape[0]))
    save(out / "04_xor_watermark.png", cv2.bitwise_xor(gray, watermark))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 04: Image Logical Operations")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_04(image, args.output)
    print(f"Practical 04 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
