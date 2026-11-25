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

def manual_histogram(gray):
    histogram = np.zeros(256, dtype=np.int64)
    for value in gray.ravel():
        histogram[int(value)] += 1
    return histogram

def practical_05(image, out):
    gray = ensure_gray(image)
    built_in = cv2.equalizeHist(gray)
    save(out / "05_histogram_equalized_builtin.png", built_in)

    histogram = manual_histogram(gray)
    cdf = np.cumsum(histogram)
    cdf_masked = cdf - cdf[np.nonzero(cdf)[0][0]]
    cdf_max = cdf_masked[-1]
    lookup = np.where(cdf_max > 0, np.rint(cdf_masked * 255 / cdf_max).astype(np.uint8), 0)
    manual = lookup[gray]
    save(out / "05_histogram_equalized_manual.png", manual)



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 05: Histogram")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_05(image, args.output)
    print(f"Practical 05 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
