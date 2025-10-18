from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration
import torch

# Supported filetypes: CSV, JSON, Parquet, txt, PNG, JPEG, WAV, MP3
dataset_dir = "data/"
dataset_path = dataset_dir + ""
dataset = load_dataset(dataset_path)

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)