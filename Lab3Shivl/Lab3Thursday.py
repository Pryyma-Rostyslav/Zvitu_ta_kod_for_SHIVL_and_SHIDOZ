import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторна робота №3 - Генерування тексту")
        self.root.geometry("900x650")

        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.history = []
        self.model_loaded = False
        self.is_generating = False

        self.create_ui()
        self.start_model_loading()

    def create_ui(self):
        title = tk.Label(
            self.root,
            text="Чат-бот для генерування тексту",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=100,
            height=28,
            font=("Consolas", 11)
        )
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.config(state=tk.DISABLED)

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.input_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            input_frame,
            text="Надіслати",
            font=("Arial", 12),
            command=self.send_message,
            state=tk.DISABLED
        )
        self.send_button.pack(side=tk.RIGHT)

        self.clear_button = tk.Button(
            self.root,
            text="Очистити чат",
            font=("Arial", 11),
            command=self.clear_chat
        )
        self.clear_button.pack(pady=5)

        self.status_label = tk.Label(
            self.root,
            text="Підготовка до завантаження моделі...",
            font=("Arial", 10),
            fg="blue"
        )
        self.status_label.pack(pady=5)

        self.add_message("Система", "Запуск програми...")
        self.add_message("Система", "Модель завантажується у фоновому режимі. Зачекайте.")

    def add_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.yview(tk.END)

    def set_status(self, text, color="blue"):
        self.status_label.config(text=text, fg=color)

    def start_model_loading(self):
        thread = threading.Thread(target=self.load_model, daemon=True)
        thread.start()

    def load_model(self):
        try:
            self.root.after(0, lambda: self.set_status("Завантаження FLAN-T5...", "blue"))

            model_name = "google/flan-t5-small"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)

            self.tokenizer = tokenizer
            self.model = model
            self.model_loaded = True

            self.root.after(0, self.on_model_loaded)

        except Exception as e:
            self.root.after(0, lambda: self.on_model_error(str(e)))

    def on_model_loaded(self):
        self.set_status(f"Модель успішно завантажена ({self.device})", "green")
        self.input_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.add_message("Бот", "Вітаю! Я готовий відповідати на ваші запитання.")

    def on_model_error(self, error_text):
        self.set_status("Помилка завантаження моделі", "red")
        messagebox.showerror("Помилка", f"Не вдалося завантажити модель:\n{error_text}")
        self.add_message("Система", "Не вдалося завантажити модель.")

    def solve_simple_math(self, text):
        text = text.lower()

        match = re.search(r"-?\d+(?:\.\d+)?\s*[\+\-\*/]\s*-?\d+(?:\.\d+)?", text)
        if not match:
            return None

        expr = match.group()

        try:
            result = eval(expr, {"__builtins__": {}}, {})

            if isinstance(result, float) and result.is_integer():
                result = int(result)

            return str(result)
        except Exception:
            return None

    def build_prompt(self, user_text):
        recent_history = self.history[-3:]

        history_text = ""
        for user_msg, bot_msg in recent_history:
            history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        prompt = (
            "You are a helpful chatbot. "
            "Answer clearly, correctly and briefly. "
            "If the question is simple, answer in one short sentence.\n\n"
            f"{history_text}"
            f"User: {user_text}\n"
            "Assistant:"
        )

        return prompt

    def generate_response(self, user_text):
        math_answer = self.solve_simple_math(user_text)
        if math_answer is not None:
            return math_answer

        prompt = self.build_prompt(user_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if "Assistant:" in answer:
            answer = answer.split("Assistant:")[-1].strip()

        if not answer:
            answer = "I could not generate an answer."

        return answer

    def send_message(self, event=None):
        if not self.model_loaded:
            messagebox.showwarning("Увага", "Модель ще завантажується. Зачекайте.")
            return

        if self.is_generating:
            return

        user_text = self.input_entry.get().strip()
        if not user_text:
            return

        self.add_message("Ви", user_text)
        self.input_entry.delete(0, tk.END)

        self.is_generating = True
        self.input_entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.set_status("Генерація відповіді...", "blue")

        thread = threading.Thread(
            target=self.generate_response_thread,
            args=(user_text,),
            daemon=True
        )
        thread.start()

    def generate_response_thread(self, user_text):
        try:
            response = self.generate_response(user_text)
            self.root.after(0, lambda: self.on_response_ready(user_text, response))
        except Exception as e:
            self.root.after(0, lambda: self.on_response_ready(user_text, f"Помилка: {e}"))

    def on_response_ready(self, user_text, response):
        self.add_message("Бот", response)
        self.history.append((user_text, response))

        self.is_generating = False
        self.input_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.set_status("Готово", "green")
        self.input_entry.focus_set()

    def clear_chat(self):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state=tk.DISABLED)

        self.history.clear()
        self.add_message("Система", "Чат очищено.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextChatBot(root)
    root.mainloop()