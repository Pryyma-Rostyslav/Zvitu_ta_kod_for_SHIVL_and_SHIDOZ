import os
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, filedialog

from gtts import gTTS
import pygame


class TextToSpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Генерування мовлення")
        self.root.geometry("700x450")
        self.root.resizable(False, False)

        self.is_playing = False
        self.current_temp_file = None

        pygame.mixer.init()

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Лабораторна робота №4\nГенерування мовлення",
            font=("Arial", 16, "bold"),
            justify="center"
        )
        title_label.pack(pady=15)

        instruction_label = tk.Label(
            self.root,
            text="Введіть текст, який потрібно перетворити на мовлення:",
            font=("Arial", 11)
        )
        instruction_label.pack()

        self.text_box = tk.Text(
            self.root,
            width=75,
            height=12,
            font=("Arial", 11),
            wrap="word"
        )
        self.text_box.pack(pady=10)

        lang_frame = tk.Frame(self.root)
        lang_frame.pack(pady=5)

        lang_label = tk.Label(lang_frame, text="Мова:", font=("Arial", 11))
        lang_label.pack(side=tk.LEFT, padx=5)

        self.language_var = tk.StringVar(value="uk")
        lang_menu = tk.OptionMenu(lang_frame, self.language_var, "uk", "en")
        lang_menu.config(font=("Arial", 10), width=10)
        lang_menu.pack(side=tk.LEFT)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)

        self.speak_button = tk.Button(
            button_frame,
            text="Озвучити текст",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.start_speaking
        )
        self.speak_button.grid(row=0, column=0, padx=8)

        self.save_button = tk.Button(
            button_frame,
            text="Зберегти у MP3",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.save_to_mp3
        )
        self.save_button.grid(row=0, column=1, padx=8)

        self.stop_button = tk.Button(
            button_frame,
            text="Зупинити",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.stop_audio
        )
        self.stop_button.grid(row=0, column=2, padx=8)

        self.status_label = tk.Label(
            self.root,
            text="Статус: очікування вводу тексту",
            font=("Arial", 10),
            fg="blue"
        )
        self.status_label.pack(pady=10)

        info_label = tk.Label(
            self.root,
            text="Підтримувані мови: uk — українська, en — англійська",
            font=("Arial", 9),
            fg="gray"
        )
        info_label.pack()

    def get_input_text(self):
        return self.text_box.get("1.0", tk.END).strip()

    def update_status(self, message):
        self.status_label.config(text=f"Статус: {message}")
        self.root.update_idletasks()

    def start_speaking(self):
        text = self.get_input_text()

        if not text:
            messagebox.showwarning("Попередження", "Будь ласка, введіть текст.")
            return

        thread = threading.Thread(target=self.speak_text, daemon=True)
        thread.start()

    def speak_text(self):
        try:
            text = self.get_input_text()
            lang = self.language_var.get()

            self.update_status("генерується аудіо...")
            self.is_playing = False

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.close()
            self.current_temp_file = temp_file.name

            tts = gTTS(text=text, lang=lang)
            tts.save(self.current_temp_file)

            self.update_status("відтворення...")
            pygame.mixer.music.load(self.current_temp_file)
            pygame.mixer.music.play()
            self.is_playing = True

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            self.is_playing = False
            self.update_status("відтворення завершено")

        except Exception as e:
            self.update_status("помилка")
            messagebox.showerror("Помилка", f"Не вдалося озвучити текст.\n\n{e}")

    def save_to_mp3(self):
        text = self.get_input_text()

        if not text:
            messagebox.showwarning("Попередження", "Будь ласка, введіть текст.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3")],
            title="Зберегти аудіо як"
        )

        if not file_path:
            return

        try:
            lang = self.language_var.get()
            self.update_status("збереження у mp3...")

            tts = gTTS(text=text, lang=lang)
            tts.save(file_path)

            self.update_status("файл успішно збережено")
            messagebox.showinfo("Успіх", f"Файл збережено:\n{file_path}")

        except Exception as e:
            self.update_status("помилка")
            messagebox.showerror("Помилка", f"Не вдалося зберегти файл.\n\n{e}")

    def stop_audio(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                self.is_playing = False
                self.update_status("відтворення зупинено")
            else:
                self.update_status("немає активного відтворення")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося зупинити аудіо.\n\n{e}")


def main():
    root = tk.Tk()
    app = TextToSpeechApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()