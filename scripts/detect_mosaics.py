import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)
    _, img_otsu = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_otsu

def generate_kernel_cross(size=501):
    if size % 2 == 0:
        raise ValueError("El tamaño debe ser impar.")
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    kernel[:, center] = 1
    kernel[center, :] = 1
    return kernel

def get_gradient(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def get_standard_deviation(grad_x, grad_y):
    grad_mag = cv2.magnitude(grad_x, grad_y)
    std_dev = np.std(grad_mag)
    return std_dev, grad_mag

def is_mosaic(gray, threshold=0.3, show=True):
    kernel = generate_kernel_cross()
    grad_x, grad_y = get_gradient(gray)
    std_dev, grad_mag = get_standard_deviation(grad_x, grad_y)
    umbral = threshold * std_dev
    _, grad_bin = cv2.threshold(grad_mag, umbral, 255, cv2.THRESH_BINARY)
    res = cv2.matchTemplate(grad_bin.astype(np.uint8), kernel, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    es_mosaic = max_val > threshold

    if show:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        top_left = max_loc
        h, w = kernel.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(vis, top_left, bottom_right, (0, 0, 255), 10)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Match: {max_val:.2f} {"" if es_mosaic else "NO "}es mosaico")
        plt.axis("off")
        plt.show()

    return es_mosaic

def main():
    mosaic = "augmentated_images"
    no_mosaic = "not_augmentated_images"
    os.makedirs(mosaic, exist_ok=True)
    os.makedirs(no_mosaic, exist_ok=True)

    c = 0
    for file in ["train"]:
        img_path = os.path.join("yolov11", "data_v11", file, "images")
        for img in os.listdir(img_path):
            path = os.path.join(img_path, img)
            imagen = cv2.imread(path)
            image_name = os.path.basename(path)
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = apply_clahe(gray)
            if imagen is not None:
                if is_mosaic(imagen):
                    print(f"{file} -> POSIBLE MOSAIC")
                    shutil.copy(path, os.path.join(mosaic, image_name)) 
                else:
                    print(f"{file} -> ORIGINAL")
                    shutil.copy(path, os.path.join(no_mosaic, image_name)) 
            c += 1
            if c == 100:
                break


if __name__ == "__main__":
    main()
