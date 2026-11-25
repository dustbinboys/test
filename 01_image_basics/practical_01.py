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

def practical_01(image, out):
    gray = ensure_gray(image)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(image, (200, 150), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(image)
    reconstructed = cv2.merge((b, g, r))
    profile = gray[gray.shape[0] // 2, :]

    save(out / "01_read_original.png", image)
    save(out / "01_resized.png", resized)
    save(out / "01_gray.png", gray)
    save(out / "01_black_and_white.png", bw)
    save(out / "01_blue_plane.png", cv2.merge((b, np.zeros_like(b), np.zeros_like(b))))
    save(out / "01_green_plane.png", cv2.merge((np.zeros_like(g), g, np.zeros_like(g))))
    save(out / "01_red_plane.png", cv2.merge((np.zeros_like(r), np.zeros_like(r), r)))
    save(out / "01_reconstructed.png", reconstructed)
    np.savetxt(out / "01_2d_gray_data.txt", gray, fmt="%d")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(profile)
    plt.title("Image Profile (Middle Row)")
    plt.xlabel("Column")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.savefig(out / "01_image_profile.png")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="DIP Practical 01: Image Basics")
    parser.add_argument("--input", type=Path, default=None, help="Optional input image path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_or_create(args.input)
    practical_01(image, args.output)
    print(f"Practical 01 completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
