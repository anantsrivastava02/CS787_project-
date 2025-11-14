#save_embedding.py
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from tqdm import tqdm
import torch.nn.functional as F

# ==============================
# Helper: Last-token pooling
# ==============================
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# ==============================
# Configuration
# ==============================
max_length = 512

# ✅ Use the 1.5B GTE-Qwen model from Hugging Face
pretrained_model_name_or_path = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
which_embedding = "gte-qwen1.5-1.5B_all_embedding"

# ==============================
# Load tokenizer & model
# ==============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a few minutes on first run)...")
model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, device_map="auto")
model.eval()

# ==============================
# Embedding function
# ==============================
# def get_all_embedding(model, input_texts):
#     batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**batch_dict, output_hidden_states=True)
#     all_embed = [last_token_pool(outputs.hidden_states[i], batch_dict['attention_mask']) for i in range(len(outputs.hidden_states))]
#     all_embed = torch.cat(all_embed, dim=1).cpu()
#     return all_embed
def get_all_embedding(model, input_texts):
    batch_dict = tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **batch_dict,
            output_hidden_states=True,
            use_cache=False   # <-- FIX HERE
        )

    all_embed = [
        last_token_pool(outputs.hidden_states[i], batch_dict['attention_mask'])
        for i in range(len(outputs.hidden_states))
    ]
    all_embed = torch.cat(all_embed, dim=1).cpu()
    return all_embed

# ==============================
# Directory setup
# ==============================
data_dir = "/content/drive/MyDrive/Text-Fluoroscopy/dataset/processed_data"
save_dir = f"save/{which_embedding}/save_embedding/"
os.makedirs(save_dir, exist_ok=True)

# ==============================
# Processing and saving embeddings
# ==============================
for file_name in os.listdir(data_dir):
    save_path = os.path.join(save_dir, file_name.split('.')[0] + '.pt')
    if os.path.exists(save_path):
        continue

    print(f"\nProcessing: {file_name}")
    with open(os.path.join(data_dir, file_name), "r") as f:
        data = json.load(f)

    embeddings = []
    for text_info in tqdm(data, desc=f"Embedding {file_name}"):
        text = text_info["text"]
        embedding = get_all_embedding(model, [text])
        embeddings.append(embedding)
        if len(embeddings) >= 300:  # limit for testing
            break

    embeddings = torch.cat(embeddings, dim=0)
    print("Embedding shape:", embeddings.shape)
    torch.save(embeddings, save_path)
    print("Saved:", save_path)
