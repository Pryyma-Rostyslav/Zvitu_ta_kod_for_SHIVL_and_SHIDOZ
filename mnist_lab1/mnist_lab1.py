import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    # 1) Завантаження MNIST (вбудовано в Keras)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2) Передобробка даних:
    #    MNIST має картинки 28x28 (0..255). Нормалізуємо до 0..1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3) Побудова нейронної мережі (простий MLP):
    #    Flatten -> Dense(128, ReLU) -> Dropout -> Dense(10, Softmax)
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
    ])

    # 4) Компіляція моделі
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5) Навчання
    # validation_split=0.1 означає: 10% train-даних піде на валідацію під час навчання
    history = model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # 6) Оцінка якості на тестовій вибірці
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\n--- Результати на тестовій вибірці ---")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # 7) Графіки навчання
    plot_history(history)

    # 8) Приклад прогнозів для кількох тестових зображень
    show_predictions(model, x_test, y_test, n=12)

def plot_history(history):
    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy під час навчання")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss під час навчання")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def show_predictions(model, x_test, y_test, n=12):
    # Прогнозуємо ймовірності для тестових даних
    probs = model.predict(x_test[:n], verbose=0)
    preds = np.argmax(probs, axis=1)

    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(x_test[i], cmap="gray")
        plt.axis("off")
        plt.title(f"True: {y_test[i]}\nPred: {preds[i]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

