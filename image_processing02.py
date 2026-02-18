import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def load_or_make_sample(image_path: str) -> Image.Image:
    if not image_path:
        raise ValueError("画像パスが指定されていません．")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return img


def to_grayscale_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def binarize(gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def annotate_size(img: Image.Image, label: str) -> Image.Image:
    """画像にラベルとピクセル寸法を焼き込む（第三者向け証拠）"""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    w, h = out.size
    text = f"{label}  |  {w} x {h} px"

    # フォント（環境により無いので安全にフォールバック）
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    # 文字の背景（半透明風に黒の矩形）
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y = 10, 10
    draw.rectangle([x-pad, y-pad, x+tw+pad, y+th+pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return out


def save_image(img, title):
    os.makedirs("image", exist_ok=True)

    filename = title.replace(" ", "_") + ".jpg"
    save_dir = os.path.join("image", "02")
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
    image_path = "0.jpg"

    img = load_or_make_sample(image_path)

    #表示
    plt.figure()
    title = "Original"
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    save_image(img, title)

    #縮小拡大
    scale_up = img.resize((img.width * 2, img.height * 2), resample=Image.BICUBIC)
    scale_down = img.resize((img.width // 2, img.height // 2), resample=Image.BICUBIC)

    plt.figure()
    title = "Scaled Up (x2)"
    plt.title(title)
    plt.imshow(scale_up)
    plt.axis("off")
    save_image(scale_up, title)

    plt.figure()
    title = "Scaled Down (half)"
    plt.title(title)
    plt.imshow(scale_down)
    plt.axis("off")
    save_image(scale_down, title)

    #回転
    rotated_30 = img.rotate(30, resample=Image.BICUBIC, expand=True)

    plt.figure()
    title = "Rotated 30 degrees"
    plt.title(title)
    plt.imshow(rotated_30)
    plt.axis("off")
    save_image(rotated_30, title)

    #二値化
    gray = to_grayscale_np(img)
    binary = binarize(gray, threshold=128)

    plt.figure()
    title = "Binary (threshold=128)"
    plt.title(title)
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    save_image(binary, title)

    plt.show()

if __name__ == "__main__":
    main()