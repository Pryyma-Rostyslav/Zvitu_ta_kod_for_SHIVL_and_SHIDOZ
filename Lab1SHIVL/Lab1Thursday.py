import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# -----------------------------
# 0) Download dataset (no Kaggle API)
# -----------------------------
ZIP_URL = "https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip"
DATA_ROOT = Path("data")
ZIP_PATH = DATA_ROOT / "captcha_images_v2.zip"
EXTRACT_DIR = DATA_ROOT / "captcha_images_v2"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print("Downloading dataset zip...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH.as_posix())
    print("Downloaded:", ZIP_PATH)

marker = EXTRACT_DIR / ".extracted"
if not marker.exists():
    print("Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    marker.write_text("ok", encoding="utf-8")
    print("Extracted to:", EXTRACT_DIR)

valid_ext = (".png", ".jpg", ".jpeg")


def find_image_folder(root: Path) -> Path:
    best_path = None
    best_count = 0
    for p in root.rglob("*"):
        if p.is_dir():
            count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_ext)
            if count > best_count:
                best_count = count
                best_path = p
    if best_path is None or best_count == 0:
        raise RuntimeError("No images found after extracting dataset.")
    return best_path


IMG_DIR = find_image_folder(EXTRACT_DIR)
image_files = sorted([f.name for f in IMG_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_ext])
labels = [os.path.splitext(f)[0] for f in image_files]

if len(image_files) < 50:
    raise RuntimeError(f"Too few images found ({len(image_files)}). Dataset extraction might be wrong.")

print("Image folder:", IMG_DIR)
print("Image count:", len(image_files))


# -----------------------------
# 1) Captcha length + vocabulary
# -----------------------------
lengths = [len(t) for t in labels]
captcha_len = max(set(lengths), key=lengths.count)

# Keep only labels that match the common length (avoid odd files)
pairs = [(f, t) for f, t in zip(image_files, labels) if len(t) == captcha_len]
image_files = [p[0] for p in pairs]
labels = [p[1] for p in pairs]

all_text = "".join(labels)
vocab = sorted(list(set(all_text)))
num_classes = len(vocab)

char_to_num = layers.StringLookup(vocabulary=vocab, mask_token=None, num_oov_indices=0)
num_to_char = layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True, num_oov_indices=0)

print("Captcha length:", captcha_len)
print("Vocab:", "".join(vocab))
print("Num classes:", num_classes)


# -----------------------------
# 2) Safe conversion helpers (NO MORE bytes/str issues)
# -----------------------------
def to_py_str(x) -> str:
    """
    Convert bytes/np.bytes_/tf.Tensor/np scalar -> Python str safely.
    """
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.bytes_):
        return x.decode("utf-8", errors="ignore")
    # numpy scalar like array(['a'], dtype='<U1') element is already str
    if isinstance(x, str):
        return x
    # tf.Tensor or numpy array element
    try:
        # if it's a 0-d or 1-element tensor/array, .item() works
        v = x.item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return str(v)
    except Exception:
        return str(x)


def idx_to_char(idx: int) -> str:
    """
    Always returns Python str (never bytes).
    """
    out = num_to_char([idx]).numpy()
    # out is typically shape (1,)
    return to_py_str(out[0])


# -----------------------------
# 3) Train/Val/Test split
# -----------------------------
rng = np.random.default_rng(42)
idx = np.arange(len(image_files))
rng.shuffle(idx)

n = len(idx)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]


def subset(indices):
    files = [str(IMG_DIR / image_files[i]) for i in indices]
    texts = [labels[i] for i in indices]
    return files, texts


train_files, train_texts = subset(train_idx)
val_files, val_texts = subset(val_idx)
test_files, test_texts = subset(test_idx)

print("Train/Val/Test:", len(train_files), len(val_files), len(test_files))


# -----------------------------
# 4) Preprocessing + datasets
# -----------------------------
IMG_HEIGHT = 50
IMG_WIDTH = 200
BATCH_SIZE = 64
EPOCHS = 30


def preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    # dataset is png; this is stable
    img = tf.io.decode_png(img_bytes, channels=1)
    img.set_shape([None, None, 1])
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def vectorize_label(text):
    chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
    ids = char_to_num(chars)
    return ids  # shape (captcha_len,)


def encode_sample(img_path, text):
    img = preprocess_image(img_path)
    y = vectorize_label(text)
    out = {f"c{i}": y[i] for i in range(captcha_len)}
    return img, out


def make_dataset(files, texts, training=True):
    ds = tf.data.Dataset.from_tensor_slices((files, texts))
    ds = ds.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(4096)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_files, train_texts, training=True)
val_ds = make_dataset(val_files, val_texts, training=False)
test_ds = make_dataset(test_files, test_texts, training=False)


# -----------------------------
# 5) Model: CNN + heads (no CTC)
# -----------------------------
def build_model():
    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outs = {f"c{i}": layers.Dense(num_classes, activation="softmax", name=f"c{i}")(x)
            for i in range(captcha_len)}

    model = keras.Model(inputs=inp, outputs=outs)

    losses = {f"c{i}": "sparse_categorical_crossentropy" for i in range(captcha_len)}
    metrics = {f"c{i}": ["accuracy"] for i in range(captcha_len)}

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, metrics=metrics)
    return model


model = build_model()

# -----------------------------
# 6) Train (silent)
# -----------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)

# -----------------------------
# 7) Plot loss
# -----------------------------
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training Loss")
plt.legend()
plt.show()


# -----------------------------
# 8) Evaluate full captcha accuracy on test set
# -----------------------------
correct = 0
total = 0

for batch_imgs, batch_labels in test_ds:
    pred = model.predict(batch_imgs, verbose=0)
    B = int(batch_imgs.shape[0])

    for b in range(B):
        pred_text = ""
        gt_text = ""

        for i in range(captcha_len):
            pred_idx = int(np.argmax(pred[f"c{i}"][b]))
            gt_idx = int(batch_labels[f"c{i}"][b].numpy())

            pred_text += idx_to_char(pred_idx)
            gt_text += idx_to_char(gt_idx)

        if pred_text == gt_text:
            correct += 1
        total += 1

test_exact = correct / total if total else 0.0
print("Exact match accuracy (dataset test):", test_exact)


# -----------------------------
# 9) Predict a specific uploaded image from data/image_to_test/
# -----------------------------
TEST_IMAGE_DIR = DATA_ROOT / "image_to_test"
TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

test_candidates = [p for p in TEST_IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
if len(test_candidates) == 0:
    print("No test image found. Put one captcha image into: data/image_to_test/")
else:
    test_path = str(test_candidates[0])
    img = preprocess_image(test_path)
    img_b = tf.expand_dims(img, axis=0)

    pred = model.predict(img_b, verbose=0)

    pred_text = ""
    for i in range(captcha_len):
        pred_idx = int(np.argmax(pred[f"c{i}"][0]))
        pred_text += idx_to_char(pred_idx)

    plt.figure()
    plt.imshow(tf.squeeze(img).numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"Predicted: {pred_text}")
    plt.show()

    print("Test image file:", os.path.basename(test_path))
    print("Predicted text:", pred_text)