import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from PIL import Image  # 追加（save_imageで使っているため）


def load_bgr(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {path}")

    return img


def color_hist_bgr(img_bgr: np.ndarray):
    hists = []
    for ch in range(3):
        hist = cv2.calcHist([img_bgr], [ch], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-12)
        hists.append(hist)
    return hists


def gray_hist(img_gray: np.ndarray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    return hist


def plot_color_hist(hists, title):
    plt.figure()
    plt.title(title)

    colors = ["blue", "green", "red"]
    labels = ["B", "G", "R"]

    for hist, color, lab in zip(hists, colors, labels):
        plt.plot(hist, color=color, label=lab)

    plt.xlim([0, 255])
    plt.legend()


def plot_gray_hist(hist, title):
    plt.figure()
    plt.title(title)
    plt.plot(hist)
    plt.xlim([0, 255])


def create_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT_create がありません（opencv-python を更新してください）")
    return cv2.SIFT_create()


def sift_match(img1_bgr, img2_bgr):
    sift = create_sift()

    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("SIFT特徴点が検出できませんでした")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:50]

    img1_kp = cv2.drawKeypoints(img1_bgr, kp1, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2_bgr, kp2, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    match_vis = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img1_kp, img2_kp, match_vis, len(kp1), len(kp2), len(matches)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) /
                 ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def save_image(img, title):
    os.makedirs("image", exist_ok=True)

    filename = title.replace(" ", "_") + ".jpg"
    save_dir = os.path.join("image", "04")
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
    path1 = "0.jpg"
    path2 = "1.jpg"

    img1 = load_bgr(path1)
    img2 = load_bgr(path2)

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 画像表示
    plt.figure()
    title = "Image 1"
    plt.title(title)
    plt.imshow(img1_rgb)
    plt.axis("off")
    save_image(img1_rgb, title)

    plt.figure()
    title = "Image 2"
    plt.title(title)
    plt.imshow(img2_rgb)
    plt.axis("off")
    save_image(img2_rgb, title)

    # ヒストグラム
    h1 = color_hist_bgr(img1)
    h2 = color_hist_bgr(img2)
    title1 = "Color Histogram (Image 1)"
    title2 = "Color Histogram (Image 2)"
    plot_color_hist(h1, title1)
    plot_color_hist(h2, title2)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gh1 = gray_hist(g1)
    gh2 = gray_hist(g2)
    title1 = "Gray Histogram (Image 1)"
    title2 = "Gray Histogram (Image 2)"
    plot_gray_hist(gh1, title1)
    plot_gray_hist(gh2, title2)

    # SIFT
    try:
        img1_kp, img2_kp, match_vis, n1, n2, nm = sift_match(img1, img2)

        plt.figure()
        title = f"SIFT Keypoints Image 1 (n={n1})"
        plt.title(title)
        vis = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
        plt.imshow(vis)
        plt.axis("off")
        save_image(vis, title)

        plt.figure()
        title = f"SIFT Keypoints Image 2 (n={n2})"
        plt.title(title)
        vis = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)
        plt.imshow(vis)
        plt.axis("off")
        save_image(vis, title)

        plt.figure()
        title = f"SIFT Matches (top {nm})"
        plt.title(title)
        vis = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)
        plt.imshow(vis)
        plt.axis("off")
        save_image(vis, title)

        print(f"SIFT keypoints: img1={n1}, img2={n2}, matches={nm}")

    except Exception as e:
        print("SIFT 実行エラー:", e)

    plt.show()


if __name__ == "__main__":
    main()