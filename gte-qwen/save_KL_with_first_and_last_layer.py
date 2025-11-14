from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import pickle

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
# Config
# ==============================
max_length = 512

# ✅ Load directly from Hugging Face
pretrained_model_name_or_path = "Qwen/Qwen2-1.5B-Instruct"
which_embedding = "Qwen2-1.5B-Instruct_embedding"

save_dir = f"/content/drive/MyDrive/Text-Fluoroscopy/save/{which_embedding}/"
os.makedirs(save_dir, exist_ok=True)

# ==============================
# Load tokenizer & model
# ==============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
)
model.eval()

# ==============================
# KL Divergence computation
# ==============================
def get_kl(model, input_texts):
    # Tokenize and move to model device
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict, output_hidden_states=True)

        # Compute logits for first, middle, and last layers
        last_logits = model.lm_head(outputs.hidden_states[-1]).squeeze()
        first_logits = model.lm_head(outputs.hidden_states[0]).squeeze()

    kls = []
    for i in range(1, len(outputs.hidden_states) - 1):
        with torch.no_grad():
            middle_logits = model.lm_head(outputs.hidden_states[i]).squeeze()
        kl_first = F.kl_div(
            F.log_softmax(middle_logits, dim=-1),
            F.softmax(first_logits, dim=-1),
            reduction="batchmean",
        )
        kl_last = F.kl_div(
            F.log_softmax(middle_logits, dim=-1),
            F.softmax(last_logits, dim=-1),
            reduction="batchmean",
        )
        kls.append((kl_first + kl_last).item())
    return kls

data_dir = "/content/drive/MyDrive/Text-Fluoroscopy/dataset"

# ==============================
# Process dataset recursively
# ==============================


for root, dirs, files in os.walk(data_dir):  # 👈 walks through all subfolders
    for file_name in files:
        if not file_name.endswith(".json"):
            continue  # skip non-JSON files

        file_path = os.path.join(root, file_name)
        save_path = os.path.join(save_dir, file_name.split('.')[0] + '.pkl')

        if os.path.exists(save_path):
            print(f"Already exists: {save_path}")
            continue

        print(f"\nProcessing: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)

        kls = []
        for text_info in tqdm(data, desc=f"Computing KL for {file_name}"):
            text = text_info["text"]
            kl = get_kl(model, [text])
            kls.append(kl)
            if len(kls) >= 300:  # limit for testing
                break

        print(f"Saving: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(kls, f)