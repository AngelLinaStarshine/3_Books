import requests
import os
import re
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

logging.getLogger("transformers").setLevel(logging.ERROR)

class TextGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def load_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def clean_text(self, text: str) -> str:
        cleaned_text = re.sub(r'https?://\S+', '', text)
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'[^\w\s,.!?-]', '', cleaned_text)
        cleaned_text = re.sub(r'\b\w{1,2}\b', '', cleaned_text) 
        cleaned_text = re.sub(
            r'\b(?:DONT|GET|FUCKED|E|R|M|D|O|Z|RAGE|AGAINST|EVE|ESL|DLC|FALLENOVA|ZEACKE|OXYGENLEAGUE|BUBBLES|VIZINAGA|S_FANTASY|G_L)\b',
            '', cleaned_text, flags=re.IGNORECASE)
        return cleaned_text.strip()

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attention_mask = (inputs != self.tokenizer.pad_token_id).type(torch.int)

        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


gutenberg_texts = {
    "Mark Twain": "http://www.gutenberg.org/files/74/74-0.txt",
    "Fyodor Dostoevsky": "http://www.gutenberg.org/files/2554/2554-0.txt",
    "Alexandre Dumas": "http://www.gutenberg.org/files/1184/1184-0.txt"
}

save_directory = "books"
os.makedirs(save_directory, exist_ok=True)

for author, url in gutenberg_texts.items():
    try:
        response = requests.get(url)
        response.raise_for_status()

        file_path = os.path.join(save_directory, f"{author}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)

        gutenberg_texts[author] = file_path
        print(f"Downloaded and saved {author}'s book successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {author}'s book. Error: {e}")

text_generator = TextGenerator(model_name='gpt2')

for author, file_path in gutenberg_texts.items():
    try:
        text = text_generator.load_text(file_path)
        if not text.strip():
            print(f"{author}'s book has no text available for generation.\n")
            continue

        prompt = f"{author}: "
        generated_text = text_generator.generate_text(prompt)
        cleaned_text = text_generator.clean_text(generated_text)
        print(f"{author} generated text:\n{cleaned_text}\n")

    except FileNotFoundError:
        print(f"File not found for {author}. Please check the download step.")
