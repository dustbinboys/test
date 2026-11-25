import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, signal


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

def practical_07(image, out):
    gray = ensure_gray(image)
    gaussian_noisy = gray.astype(np.float32) + np.random.default_rng(42).normal(0, 20, gray.shape)
    gaussian_noisy = np.clip(gaussian_noisy, 0, 255).astype(np.uint8)
    save(out / "07_gaussian_noise.png", gaussian_noisy)

    rng = np.random.default_rng(42)
    salt_pepper = gray.copy()
    probability = 0.05
    mask = rng.random(gray.shape)
    salt_pepper[mask < probability / 2] = 0
    salt_pepper[mask > 1 - probability / 2] = 255
    save(out / "07_salt_pepper_noise.png", salt_pepper)

    save(out / "07_salt_pepper_removed_median.png", cv2.medianBlur(salt_pepper, 3))
    save(out / "07_gaussian_removed_blur.png", cv2.GaussianBlur(gaussian_noisy, (5, 5), 1))
    save(out / "07_median_filter.png", cv2.medianBlur(gaussian_noisy, 3))

    wiener = signal.wiener(gray.astype(np.float32), (5, 5), noise=20)
    save(out / "07_wiener_filter.png", np.clip(wiener, 0, 255).astype(np.uint8))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 07: Noise And Restoration")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_07(image, args.output)
    print(f"Practical 07 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
