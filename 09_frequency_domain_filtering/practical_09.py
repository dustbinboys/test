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

def ideal_low_pass(shape, radius):
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).astype(np.float32)

def butterworth_low_pass(shape, radius, order=2):
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return 1 / (1 + (distance / radius) ** (2 * order))

def practical_09(image, out):
    gray = ensure_gray(image)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = 20 * np.log1p(np.abs(f_shift))
    save(out / "09_fft_magnitude.png", cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    ideal = ideal_low_pass(gray.shape, 40)
    butter = butterworth_low_pass(gray.shape, 40)

    ideal_low = np.fft.ifft2(np.fft.ifftshift(f_shift * ideal)).real
    butter_low = np.fft.ifft2(np.fft.ifftshift(f_shift * butter)).real

    ideal_high = np.fft.ifft2(np.fft.ifftshift(f_shift * (1 - ideal))).real
    butter_high = np.fft.ifft2(np.fft.ifftshift(f_shift * (1 - butter))).real

    for name, result in [("09_ideal_low_pass.png", ideal_low), ("09_butterworth_low_pass.png", butter_low),
                         ("09_ideal_high_pass.png", ideal_high), ("09_butterworth_high_pass.png", butter_high)]:
        save(out / name, np.clip(result, 0, 255).astype(np.uint8))



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 09: Frequency Domain Filtering")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_09(image, args.output)
    print(f"Practical 09 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
