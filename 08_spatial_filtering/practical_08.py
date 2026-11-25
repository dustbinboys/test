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

def convolve2d_manual(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="symmetric")
    result = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y, x] = np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)
    return np.clip(result, 0, 255).astype(np.uint8)

def practical_08(image, out):
    gray = ensure_gray(image)
    vector = np.arange(1, 6, dtype=np.float32)
    kernel_1d = np.array([1, 2, 1], dtype=np.float32) / 4
    convolved_1d = np.convolve(vector, kernel_1d, mode="same")
    np.savetxt(out / "08_1d_convolution.txt", convolved_1d, fmt="%.4f")

    low_pass = np.ones((3, 3), dtype=np.float32) / 9
    high_pass = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    save(out / "08_low_pass_manual.png", convolve2d_manual(gray, low_pass))
    save(out / "08_high_pass_manual.png", convolve2d_manual(gray, high_pass))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 08: Spatial Filtering")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_08(image, args.output)
    print(f"Practical 08 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
