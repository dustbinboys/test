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

def practical_11(image, out):
    gray = ensure_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    save(out / "11_binary_input.png", binary)
    save(out / "11_erosion.png", cv2.erode(binary, kernel, iterations=1))
    save(out / "11_dilation.png", cv2.dilate(binary, kernel, iterations=1))
    save(out / "11_opening.png", cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))
    save(out / "11_closing.png", cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 11: Morphological Operations")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_11(image, args.output)
    print(f"Practical 11 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
