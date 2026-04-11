import os
import json
from glob import glob

import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# -------------------- config --------------------
DATA_DIR = "mnist_data"
IMG_DIR = "images"
OUT_DIR = "results"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

TRY_ANGLES = [-15, -10, -5, 0, 5, 10, 15]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- data --------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x + 0.05 * torch.randn_like(x), 0, 1)),
])

test_transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------- model --------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -------------------- utils --------------------
def save_loss_plot(losses, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.show()
    plt.close()


@torch.no_grad()
def test_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def center_of_mass_shift(img01):
    """
    img01: float32 (H,W) in [0,1], digit bright on dark background (MNIST style)
    shifts the digit so its center-of-mass is near the image center.
    """
    h, w = img01.shape
    ys, xs = np.indices((h, w))
    mass = img01.sum()

    if mass <= 1e-6:
        return img01

    cy = (ys * img01).sum() / mass
    cx = (xs * img01).sum() / mass

    shift_y = int(round(h / 2.0 - cy))
    shift_x = int(round(w / 2.0 - cx))

    shifted = np.zeros_like(img01)
    y0_src = max(0, -shift_y)
    y1_src = min(h, h - shift_y)
    x0_src = max(0, -shift_x)
    x1_src = min(w, w - shift_x)

    y0_dst = max(0, shift_y)
    y1_dst = min(h, h + shift_y)
    x0_dst = max(0, shift_x)
    x1_dst = min(w, w + shift_x)

    shifted[y0_dst:y1_dst, x0_dst:x1_dst] = img01[y0_src:y1_src, x0_src:x1_src]
    return shifted


def preprocess_external_image(path):
    """
    Makes an external image look more like MNIST:
    - grayscale
    - autocontrast + slight blur (reduces broken dots / compression noise)
    - decide inversion automatically
    - crop tight bbox by intensity
    - resize into 20x20 area, pad to 28x28
    - center-of-mass shift
    Returns float32 (28,28) in [0,1] with digit bright on dark background.
    """
    img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    img = img.convert("L")  # grayscale

    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    arr = np.asarray(img).astype(np.float32) / 255.0

    # Auto inversion:
    # If background is bright (mean high), invert so digit becomes bright on dark.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Soft threshold for bbox only (not hard binarization)
    # This avoids destroying the digit 
    mask = arr > 0.15
    ys, xs = np.where(mask)
    if ys.size > 0 and xs.size > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        arr = arr[y0:y1 + 1, x0:x1 + 1]

    # Resize to fit 20x20 keeping aspect ratio
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.float32)

    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil_digit = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil_digit = pil_digit.resize((new_w, new_h), resample=Image.BILINEAR)
    small = np.asarray(pil_digit).astype(np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    canvas = center_of_mass_shift(canvas)
    canvas = np.clip(canvas, 0.0, 1.0)
    return canvas


def rotate_image_pil(img28, angle_deg):
    pil = Image.fromarray((img28 * 255).astype(np.uint8), mode="L")
    rot = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return np.asarray(rot).astype(np.float32) / 255.0


@torch.no_grad()
def recognize_digit(path, model):
    base = preprocess_external_image(path)
    best_digit = None
    best_prob = 0.0

    for a in TRY_ANGLES:
        img_rot = rotate_image_pil(base, a)
        x = torch.from_numpy(img_rot).float().unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        p, cls = probs.max(dim=0)

        p = float(p.item())
        cls = int(cls.item())
        if p > best_prob:
            best_prob = p
            best_digit = cls

    return best_digit, best_prob


# -------------------- main --------------------
def main():
    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for ep in range(EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion)
        losses.append(avg_loss)
        print(f"Epoch {ep + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_loss_plot(losses, os.path.join(OUT_DIR, "training_loss.png"))

    acc = test_accuracy(model, test_loader)
    print(f"Test Accuracy: {acc:.4f}")

    image_files = glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")) + glob(os.path.join(IMG_DIR, "*.jpeg"))
    if not image_files:
        print("No images found in images/ (put .png/.jpg/.jpeg there)")
        return

    results = {}
    for p in image_files:
        d, prob = recognize_digit(p, model)
        results[os.path.basename(p)] = {"digit": int(d), "probability": float(prob)}
        print(f"{p}: Digit={d}, Probability={prob:.3f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()