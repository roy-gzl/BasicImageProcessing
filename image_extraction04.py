import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

    os.makedirs("image/04", exist_ok=True)
    filename = title.replace(" ", "_") + ".jpg"
    plt.savefig(os.path.join("image/04", filename), dpi=300)


def plot_gray_hist(hist, title):
    plt.figure()
    plt.title(title)
    plt.plot(hist, color="black")
    plt.xlim([0, 255])

    os.makedirs("image/04", exist_ok=True)
    filename = title.replace(" ", "_") + ".jpg"
    plt.savefig(os.path.join("image/04", filename), dpi=300)


def create_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT_create がありません（opencv-python を更新してください）")
    return cv2.SIFT_create()


def sift_match(img1_bgr, img2_bgr, ratio: float = 0.75):
    sift = create_sift()

    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("SIFT特徴点が検出できませんでした")

    # ratio test 用：crossCheck は使わない
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    total_good = len(good)

    # 表示は上位50
    good = sorted(good, key=lambda m: m.distance)
    shown = good[:50]
    shown_m = len(shown)

    img1_kp = cv2.drawKeypoints(img1_bgr, kp1, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2_bgr, kp2, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    match_vis = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, shown, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img1_kp, img2_kp, match_vis, len(kp1), len(kp2), total_good, shown_m


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


def rotate_scale_center_crop(bgr: np.ndarray, angle_deg: float = 15.0, scale: float = 0.5) -> np.ndarray:
    h, w = bgr.shape[:2]

    pad = int(max(h, w) * 0.6)
    H = h + 2 * pad
    W = w + 2 * pad

    canvas = np.zeros((H, W, 3), dtype=bgr.dtype)

    top0 = 0
    left0 = W - w
    canvas[top0:top0 + h, left0:left0 + w] = bgr

    cx = left0 + w / 2
    cy = top0 + h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    warped = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    top = 0
    left = W - w
    out = warped[top:top + h, left:left + w].copy()

    return out


def run_sift_case(img1_bgr, img2_bgr, case_name: str):
    img1_kp, img2_kp, match_vis, n1, n2, total_good, shown_m = sift_match(img1_bgr, img2_bgr)

    plt.figure()
    title = f"{case_name} - SIFT Keypoints Image 1 (n={n1})"
    plt.title(title)
    vis = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
    plt.imshow(vis)
    plt.axis("off")
    save_image(vis, title)

    plt.figure()
    title = f"{case_name} - SIFT Keypoints Image 2 (n={n2})"
    plt.title(title)
    vis = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)
    plt.imshow(vis)
    plt.axis("off")
    save_image(vis, title)

    plt.figure()
    title = f"{case_name} - SIFT Matches (Good Matches: {total_good}, Displayed: {shown_m})"
    plt.title(title)
    vis = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)
    plt.imshow(vis)
    plt.axis("off")
    save_image(vis, title)

    print(f"[{case_name}] SIFT keypoints: img1={n1}, img2={n2}, good_matches={total_good}, shown={shown_m}")


def main():
    path1 = "0.jpg"
    path2 = "1.jpg"

    img0 = load_bgr(path1)
    img1 = load_bgr(path2)

    # --- 画像表示＆保存（元の2枚だけ） ---
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    plt.figure()
    title = "Image 1"
    plt.title(title)
    plt.imshow(img0_rgb)
    plt.axis("off")
    save_image(img0_rgb, title)

    plt.figure()
    title = "Image 2"
    plt.title(title)
    plt.imshow(img1_rgb)
    plt.axis("off")
    save_image(img1_rgb, title)

    # --- ヒストグラム（元の2枚だけ） ---
    plot_color_hist(color_hist_bgr(img0), "Color Histogram (Image 1)")
    plot_color_hist(color_hist_bgr(img1), "Color Histogram (Image 2)")

    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    plot_gray_hist(gray_hist(g0), "Gray Histogram (Image 1)")
    plot_gray_hist(gray_hist(g1), "Gray Histogram (Image 2)")

    # --- SIFTだけ自動で2ケース ---
    try:
        run_sift_case(img0, img1, "CaseA_DifferentImages")
    except Exception as e:
        print("CaseA エラー:", e)

    try:
        img0_trans = rotate_scale_center_crop(img0, angle_deg=15.0, scale=0.5)

        # ★ 追加：変換後画像も保存
        img0_trans_rgb = cv2.cvtColor(img0_trans, cv2.COLOR_BGR2RGB)
        plt.figure()
        title = "Image 1 (Rotate+Scale+Crop)"
        plt.title(title)
        plt.imshow(img0_trans_rgb)
        plt.axis("off")
        save_image(img0_trans_rgb, title)

        run_sift_case(img0, img0_trans, "CaseB_RotateScaleCrop")
    except Exception as e:
        print("CaseB エラー:", e)

    plt.show()


if __name__ == "__main__":
    main()