import os
import random
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyttsx3
from glob import glob
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
N_MFCC = 40
EPOCHS = 15
BATCH_SIZE = 32

DATASET_DIR = "dataset"
MODEL_FILE = "speech_nato.pth"


NATO = [
"Alpha","Bravo","Charlie","Delta","Echo","Foxtrot","Golf","Hotel",
"India","Juliett","Kilo","Lima","Mike","November","Oscar","Papa",
"Quebec","Romeo","Sierra","Tango","Uniform","Victor","Whiskey",
"Xray","Yankee","Zulu"
]

DIGITS = [
"Zero","One","Two","Three","Four",
"Five","Six","Seven","Eight","Nine"
]

CLASSES = NATO + DIGITS

CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
IDX_TO_CLASS = {i:c for c,i in CLASS_TO_IDX.items()}


def word_to_char(word):

    if word in NATO:
        return word[0]

    if word in DIGITS:
        return str(DIGITS.index(word))

    return ""


# =========================
# AUDIO AUGMENTATION
# =========================

def add_noise(audio):
    noise = np.random.randn(len(audio)) * 0.003
    return audio + noise


def change_speed(audio):
    rate = random.uniform(0.85,1.15)
    return librosa.effects.time_stretch(audio, rate=rate)


def change_pitch(audio):
    steps = random.uniform(-1.5,1.5)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)


def augment(audio):

    if random.random() < 0.5:
        audio = change_speed(audio)

    if random.random() < 0.5:
        audio = change_pitch(audio)

    if random.random() < 0.7:
        audio = add_noise(audio)

    return audio


# =========================
# DATASET CREATION
# =========================

def create_dataset():

    print("Generating NATO dataset")

    os.makedirs(DATASET_DIR, exist_ok=True)

    engine = pyttsx3.init()

    base_files = []

    # Step 1 — generate base wavs
    for word in CLASSES:

        folder = os.path.join(DATASET_DIR, word)
        os.makedirs(folder, exist_ok=True)

        base_file = os.path.join(folder, "base.wav")

        engine.save_to_file(word, base_file)

        base_files.append((word, base_file))

    print("Generating base speech files...")
    engine.runAndWait()

    # Step 2 — augmentation
    for word, base_file in base_files:

        audio, sr = librosa.load(base_file, sr=SAMPLE_RATE)

        folder = os.path.join(DATASET_DIR, word)

        for i in range(60):

            aug = augment(audio)

            out_file = os.path.join(folder, f"{word}_{i}.wav")

            sf.write(out_file, aug, SAMPLE_RATE)

        os.remove(base_file)

        print("Generated", word)

    print("Dataset generation complete")


# =========================
# DATASET CLASS
# =========================

class SpeechDataset(Dataset):

    def __init__(self):

        self.files = []

        for cls in CLASSES:

            folder = os.path.join(DATASET_DIR,cls)

            for f in glob(os.path.join(folder,"*.wav")):

                self.files.append((f,CLASS_TO_IDX[cls]))

        print("Training samples:",len(self.files))


    def __len__(self):
        return len(self.files)


    def __getitem__(self,idx):

        path,label = self.files[idx]

        y,sr = librosa.load(path,sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)

        mfcc = torch.tensor(mfcc,dtype=torch.float32)

        return mfcc,torch.tensor(label)


def collate_fn(batch):

    features = [b[0].T for b in batch]
    labels = torch.stack([b[1] for b in batch])

    max_len = max([f.shape[0] for f in features])

    padded=[]

    for f in features:

        pad = max_len - f.shape[0]

        if pad>0:
            f = torch.nn.functional.pad(f,(0,0,0,pad))

        padded.append(f)

    x = torch.stack(padded)

    return x.to(DEVICE),labels.to(DEVICE)


# =========================
# MODEL
# =========================

class SpeechModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(N_MFCC,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(128,128,batch_first=True)

        self.fc = nn.Linear(128,len(CLASSES))


    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.cnn(x)

        x = x.permute(0,2,1)

        _,(h,_) = self.lstm(x)

        return self.fc(h[-1])


# =========================
# TRAIN
# =========================

def train():

    dataset = SpeechDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = SpeechModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):

        correct = 0
        total = 0
        loss_sum = 0

        for x, y in loader:

            optimizer.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item()

            preds = out.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = loss_sum / len(loader)
        epoch_acc = correct / total

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print("Epoch", epoch + 1,
              "loss", epoch_loss,
              "acc", epoch_acc)

    torch.save(model.state_dict(), MODEL_FILE)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, marker="o")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    return model

# =========================
# PREDICTION
# =========================

def predict_word(model,file):

    y,sr = librosa.load(file,sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)

    x = torch.tensor(mfcc,dtype=torch.float32).T.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        out = model(x)

        pred = out.argmax(1).item()

    return IDX_TO_CLASS[pred]


def recognize(model,files):

    result=""

    for f in files:

        word = predict_word(model,f)

        char = word_to_char(word)

        print(os.path.basename(f),"->",word,"->",char)

        result+=char

    return result


# =========================
# MAIN
# =========================

def main():

    print("Device:",DEVICE)

    create_dataset()

    if os.path.exists(MODEL_FILE):

        print("Loading model")

        model = SpeechModel().to(DEVICE)

        model.load_state_dict(torch.load(MODEL_FILE,map_location=DEVICE))

        model.eval()

    else:

        print("Training model")

        model = train()

    files = [
        "audio/One.wav",
        "audio/November.wav",
        "audio/Eight.wav",
        "audio/Juliett.wav"
    ]

    print("Recognizing")

    flight = recognize(model,files)

    print("Flight number:",flight)


if __name__ == "__main__":
    main()