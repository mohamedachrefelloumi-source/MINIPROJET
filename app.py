

import numpy as np
import cv2
from scipy.fft import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


DELTA      = 30
SECRET_KEY = 42
N_BITS     = 64



def choose_image():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return path


def get_coords(shape):
    rng = np.random.default_rng(SECRET_KEY)
    r = rng.integers(shape[0] // 4, 3 * shape[0] // 4, size=N_BITS)
    c = rng.integers(shape[1] // 4, 3 * shape[1] // 4, size=N_BITS)
    return list(zip(r, c))


def to_dct(img):
    return dct(dct(img.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


def from_dct(d):
    return np.clip(idct(idct(d, axis=1, norm='ortho'), axis=0, norm='ortho'), 0, 255).astype(np.uint8)



def generate_watermark():
    return np.random.default_rng(SECRET_KEY).integers(0, 2, size=N_BITS).astype(int)


def embed(img, wm):
    d = to_dct(img)
    for i, (r, c) in enumerate(get_coords(d.shape)):
        if wm[i] == 0:
            d[r, c] = np.round(d[r, c] / DELTA) * DELTA
        else:
            d[r, c] = (np.round(d[r, c] / DELTA - 0.5) + 0.5) * DELTA
    return from_dct(d)


def extract(img):
    d = to_dct(img)
    bits = []
    for (r, c) in get_coords(d.shape):
        q0 = np.round(d[r, c] / DELTA) * DELTA
        q1 = (np.round(d[r, c] / DELTA - 0.5) + 0.5) * DELTA
        bits.append(0 if abs(d[r, c] - q0) < abs(d[r, c] - q1) else 1)
    return np.array(bits)


def ber(original, extracted):
    return np.sum(original != extracted) / len(original)




def attack_noise(img, sigma=10):
    return np.clip(img.astype(float) + np.random.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def attack_jpeg(img, quality=50):
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)



if __name__ == "__main__":

    path = choose_image()
    if path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("Image chargée : {path} {img.shape}")
    else:
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        print("⚠Aucune image choisie — image de test générée (256×256).")


    wm = generate_watermark()
    wm_img = embed(img, wm)

    score = psnr(img, wm_img)
    print("PSNR : {score:.2f} dB")
    print("BER sans attaque  : {ber(wm, extract(wm_img)):.4f}")
    print("BER bruit (σ=10)  : {ber(wm, extract(attack_noise(wm_img))):.4f}")
    print("BER JPEG (q=50)   : {ber(wm, extract(attack_jpeg(wm_img))):.4f}")

    images = [img, wm_img, attack_noise(wm_img), attack_jpeg(wm_img)]
    titles = ["Original", "Tatouée", "Après bruit", "Après JPEG"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap='gray'); ax.set_title(t); ax.axis('off')
    plt.suptitle(f"PSNR={score:.1f} dB", fontsize=12)
    plt.tight_layout()
    plt.savefig("resultats.png", dpi=150)
    plt.show()
    print(" Sauvegardé dans resultats.png")
