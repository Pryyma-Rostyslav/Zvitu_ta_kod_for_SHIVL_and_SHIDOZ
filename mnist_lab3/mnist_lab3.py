import os
import glob
import cv2
import numpy as np
import urllib.request


DATASET_FOLDER = "dataset"
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "result.png")


def ensure_folders():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def download_small_dataset():
    files = [
        (
            os.path.join(DATASET_FOLDER, "box.png"),
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png"
        ),
        (
            os.path.join(DATASET_FOLDER, "box_in_scene.png"),
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box_in_scene.png"
        )
    ]

    for path, url in files:
        if not os.path.exists(path):
            print(f"Downloading dataset file: {os.path.basename(path)}")
            urllib.request.urlretrieve(url, path)

    print("Dataset is ready.")


def load_dataset_pairs():
    img1_path = os.path.join(DATASET_FOLDER, "box.png")
    img2_path = os.path.join(DATASET_FOLDER, "box_in_scene.png")

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Could not load dataset images.")

    return [(img1, img2)]


def load_user_images():
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []

    for pattern in patterns:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, pattern)))

    files.sort()

    if len(files) < 2:
        raise ValueError(
            "Put at least 2 images into the 'input' folder.\n"
            "Example: left.png and right.png"
        )

    img1 = cv2.imread(files[0], cv2.IMREAD_COLOR)
    img2 = cv2.imread(files[1], cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Could not load input images.")

    print("Using user images:")
    print(" -", files[0])
    print(" -", files[1])

    return img1, img2


def detect_and_describe(image, nfeatures=5000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio=0.78):
    if desc1 is None or desc2 is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches


def estimate_shift_from_matches(kp1, kp2, matches):
    if len(matches) < 4:
        return None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    _, inlier_mask = cv2.estimateAffinePartial2D(
        pts2,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=5000,
        confidence=0.99,
        refineIters=10
    )

    if inlier_mask is None:
        return None, None, None

    inlier_mask = inlier_mask.ravel().astype(bool)

    if inlier_mask.sum() < 4:
        return None, None, None

    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    diffs = inlier_pts1 - inlier_pts2

    dx = int(round(np.median(diffs[:, 0])))
    dy = int(round(np.median(diffs[:, 1])))

    return dx, dy, inlier_mask


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=40):
    return cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


def merge_by_shift(img1, img2, dx, dy):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    x1_min, y1_min = 0, 0
    x1_max, y1_max = w1, h1

    x2_min, y2_min = dx, dy
    x2_max, y2_max = dx + w2, dy + h2

    xmin = min(x1_min, x2_min)
    ymin = min(y1_min, y2_min)
    xmax = max(x1_max, x2_max)
    ymax = max(y1_max, y2_max)

    tx = -xmin
    ty = -ymin

    result_w = xmax - xmin
    result_h = ymax - ymin

    canvas1 = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    canvas2 = np.zeros((result_h, result_w, 3), dtype=np.uint8)

    canvas1[ty:ty + h1, tx:tx + w1] = img1
    canvas2[ty + dy:ty + dy + h2, tx + dx:tx + dx + w2] = img2

    mask1 = np.zeros((result_h, result_w), dtype=np.uint8)
    mask2 = np.zeros((result_h, result_w), dtype=np.uint8)

    mask1[ty:ty + h1, tx:tx + w1] = 255
    mask2[ty + dy:ty + dy + h2, tx + dx:tx + dx + w2] = 255

    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask2 > 0) & (mask1 == 0)
    overlap = (mask1 > 0) & (mask2 > 0)

    result = np.zeros_like(canvas1)
    result[only1] = canvas1[only1]
    result[only2] = canvas2[only2]

    if np.any(overlap):
        ys, xs = np.where(overlap)
        x_min = xs.min()
        x_max = xs.max()

        seam_x = (x_min + x_max) // 2

        col_indices = np.arange(result_w)[None, :]
        overlap_left = overlap & (col_indices <= seam_x)
        overlap_right = overlap & (col_indices > seam_x)

        result[overlap_left] = canvas1[overlap_left]
        result[overlap_right] = canvas2[overlap_right]

    return result


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y + h, x:x + w]


def resize_for_screen(image, max_width=1400, max_height=850):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def stitch_two_images(img1, img2, save_debug=True, debug_prefix="user"):
    kp1, desc1 = detect_and_describe(img1)
    kp2, desc2 = detect_and_describe(img2)

    matches = match_features(desc1, desc2)

    print(f"{debug_prefix}: keypoints 1 = {len(kp1)}, keypoints 2 = {len(kp2)}")
    print(f"{debug_prefix}: good matches = {len(matches)}")

    if len(matches) < 4:
        raise ValueError("Not enough good matches to estimate shift.")

    dx, dy, inlier_mask = estimate_shift_from_matches(kp1, kp2, matches)

    if dx is None:
        raise ValueError("Could not estimate shift from matches.")

    inliers = int(inlier_mask.sum())
    print(f"{debug_prefix}: RANSAC inliers = {inliers}")
    print(f"{debug_prefix}: estimated shift dx = {dx}, dy = {dy}")

    if save_debug:
        matches_vis = draw_matches(img1, kp1, img2, kp2, matches)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{debug_prefix}_matches.png"), matches_vis)

    result = merge_by_shift(img1, img2, dx, dy)
    result = crop_black_borders(result)

    return result


def evaluate_on_dataset():
    print("\n=== DATASET EVALUATION ===")
    pairs = load_dataset_pairs()

    for i, (img1, img2) in enumerate(pairs, start=1):
        try:
            result = stitch_two_images(
                img1,
                img2,
                save_debug=True,
                debug_prefix=f"dataset_pair_{i}"
            )
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"dataset_result_{i}.png"), result)
            print(f"Dataset pair {i}: success")
        except Exception as e:
            print(f"Dataset pair {i}: failed -> {e}")


def process_user_images():
    print("\n=== USER IMAGE PROCESSING ===")
    img1, img2 = load_user_images()
    result = stitch_two_images(img1, img2, save_debug=True, debug_prefix="user")

    cv2.imwrite(OUTPUT_FILE, result)
    print(f"Saved final result to: {OUTPUT_FILE}")

    show = resize_for_screen(result)
    cv2.imshow("Final Result", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ensure_folders()
    download_small_dataset()
    evaluate_on_dataset()
    process_user_images()


if __name__ == "__main__":
    main()