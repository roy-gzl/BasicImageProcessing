import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")
    return Image.open(path).convert("RGB")

def zoom_and_crop_center(img: Image.Image, scale: float = 1.02) -> Image.Image:

    w, h = img.size

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    right = left + w
    bottom = top + h

    cropped = resized.crop((left, top, right, bottom))

    return cropped

def save_image(img, title):
    os.makedirs("image", exist_ok=True)

    filename = title.replace(" ", "_") + ".jpg"
    save_dir = os.path.join("image", "03")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)

    if isinstance(img, Image.Image):
        img.convert("RGB").save(save_path, quality=95)
    else:
        arr = np.asarray(img)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            Image.fromarray(arr, mode="L").save(save_path, quality=95)
        else:
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            Image.fromarray(arr, mode="RGB").save(save_path, quality=95)

def main():
    img1 = load_image("0.jpg")
    img2 = img1.rotate(2, expand=False) 
    img3 = zoom_and_crop_center(img1, scale=1.02)

    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)
    arr3 = np.array(img3, dtype=np.int16)

    diff1 = np.abs(arr1 - arr2)
    diff2 = np.abs(arr1 - arr3)

    diff1 = np.clip(diff1, 0, 255).astype(np.uint8)
    diff2 = np.clip(diff2, 0, 255).astype(np.uint8)

    plt.figure()
    title = "Image 1"
    plt.title(title)
    plt.imshow(img1)
    plt.axis("off")
    save_image(img1, title)

    plt.figure()
    title = "Image 2 (Rotated)"
    plt.title(title)
    plt.imshow(img2)
    plt.axis("off")
    save_image(img2, title)
    
    plt.figure()
    title = "Image 3 (Expanded)"
    plt.title(title)
    plt.imshow(img3)
    plt.axis("off")
    save_image(img3, title)

    plt.figure()
    title = "Difference Image (Rotated)"
    plt.title(title)
    plt.imshow(diff1)
    plt.axis("off")
    save_image(diff1, title)
    
    plt.figure()
    title = "Difference Image (Expanded)"
    plt.title(title)
    plt.imshow(diff2)
    plt.axis("off")
    save_image(diff2, title)

    plt.show()

if __name__ == "__main__":
    main()