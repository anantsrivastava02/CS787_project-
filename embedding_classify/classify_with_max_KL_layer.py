

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import pickle
# import numpy as np
# from sklearn.metrics import roc_auc_score

# torch.manual_seed(42)
# which_embedding='gte-qwen_all';embedding_dim=4096;kl_path='gte-qwen_KL_with_first_and_last_layer';learning_rate=0.003;droprate=0.4

# which_layer = 'max_kl'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_embeddings           = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_train.pt')[:160]
# valid_embeddings           = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_valid.pt')[:20]
# test_embeddings            = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_test.pt')[:20]
          
# train_labels               = torch.load('dataset/labels/HC3_en_train.pt')[:160].to(device)
# valid_labels               = torch.load('dataset/labels/HC3_en_valid.pt')[:20].to(device)
# test_labels                = torch.load('dataset/labels/HC3_en_test.pt')[:20].to(device)

# train_embeddings = train_embeddings.to(device)
# valid_embeddings = valid_embeddings.to(device)
# test_embeddings = test_embeddings.to(device)
# with open(f'save/{kl_path}/HC3_en_train.pkl', 'rb') as f:
#     train_kl = pickle.load(f)
#     train_kl = np.array(train_kl)
#     idx = train_kl.argmax(axis=1)
#     if which_layer == 'max_kl':
#         train_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings,idx) ]).to(device)
#     if which_layer == 'max_kl_and_last_layer':
#         train_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings,idx) ])
#                                       ,train_embeddings[:,-embedding_dim:]],dim=1).to(device)
# with open(f'save/{kl_path}/HC3_en_test.pkl', 'rb') as f:
#     test_kl = pickle.load(f)
#     test_kl = np.array(test_kl)
#     idx = test_kl.argmax(axis=1)
#     if which_layer == 'max_kl':
#         test_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(test_embeddings,idx) ]).to(device)
#     if which_layer == 'max_kl_and_last_layer':
#         test_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(test_embeddings,idx) ])
#                                       ,test_embeddings[:,-embedding_dim:]],dim=1).to(device)
# with open(f'save/{kl_path}/HC3_en_valid.pkl', 'rb') as f:
#     valid_kl = pickle.load(f)
#     valid_kl = np.array(valid_kl)
#     idx = train_kl.argmax(axis=1)
#     if which_layer == 'max_kl':
#         valid_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(valid_embeddings,idx) ]).to(device)
#     if which_layer == 'max_kl_and_last_layer':
#         valid_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(valid_embeddings,idx) ])
#                                       ,valid_embeddings[:,-embedding_dim:]],dim=1).to(device)
# if which_layer == 'first_layer':
#     train_embeddings = train_embeddings[:,:embedding_dim].to(device)
#     valid_embeddings = valid_embeddings[:,:embedding_dim].to(device)
#     test_embeddings = test_embeddings[:,:embedding_dim].to(device)

# elif which_layer == 'last_layer':
#     train_embeddings = train_embeddings[:,-embedding_dim:].to(device)
#     valid_embeddings = valid_embeddings[:,-embedding_dim:].to(device)
#     test_embeddings = test_embeddings[:,-embedding_dim:].to(device)

# elif which_layer == 'first_and_last_layers':
#     train_embeddings = torch.cat([train_embeddings[:,:embedding_dim],train_embeddings[:,-embedding_dim:]],dim=1).to(device)
#     valid_embeddings = torch.cat([valid_embeddings[:,:embedding_dim],valid_embeddings[:,-embedding_dim:]],dim=1).to(device)
#     test_embeddings = torch.cat([test_embeddings[:,:embedding_dim],test_embeddings[:,-embedding_dim:]],dim=1).to(device)

# elif which_layer.startswith('layer_'):
#     if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
#         layer_num = int(which_layer.split('_')[-1])
#         train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
#         valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
#     elif 'last_layer' in which_layer:
#         layer_num = int(which_layer.split('_')[1])
#         train_embeddings = torch.cat([train_embeddings[:,-embedding_dim:],train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]],dim=1).to(device)
#         valid_embeddings = torch.cat([valid_embeddings[:,-embedding_dim:],valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]],dim=1).to(device)
#     elif 'later_layer' in which_layer:
#         layer_num = int(which_layer.split('_')[1])
#         train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:].to(device)
#         valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:].to(device)    
#     elif 'to' in which_layer:
#         layer_num = int(which_layer.split('_')[1])
#         layer_num2 = int(which_layer.split('_')[3])
#         train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)
#         valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)

# testsets = ['Xsum_gpt3.pt', 'writing_gpt-3.pt','pub_gpt-3.pt','gpt4-Xsum-gpt3.pt','gpt4-writing-gpt3.pt', 'gpt4-pub-gpt3.pt', 'xsum_claude-3-opus-20240229-gpt3.pt','writing_claude-3-opus-20240229-gpt3.pt', 'pub_claude-3-opus-20240229-gpt3.pt',    ]
# num=None
# testset_embeddings = []
# testset_labels = []
# for file_name in testsets:
#     testset_embeddings.append(torch.load(f'save/{which_embedding}_embedding/save_embedding/{file_name}')[:num])
#     testset_labels.append(torch.load(f'dataset/labels/{file_name}').to(device)[:num])
#     with open(f'save/{kl_path}/{file_name.split(".")[0]}.pkl', 'rb') as f:
#         kl = pickle.load(f)
#         kl = np.array(kl)
#         idx = kl.argmax(axis=1)
#         if which_layer == 'max_kl':
#             testset_embeddings[-1] = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(testset_embeddings[-1],idx) ]).to(device)
#         elif which_layer == 'max_kl_and_last_layer':
#             testset_embeddings[-1] = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(testset_embeddings[-1],idx) ])
#                                       ,testset_embeddings[-1][:,-embedding_dim:]],dim=1).to(device)
#         elif which_layer == 'first_layer':
#             testset_embeddings[-1] = testset_embeddings[-1][:, :embedding_dim].to(device)
#         elif which_layer == 'last_layer':
#             testset_embeddings[-1] = testset_embeddings[-1][:, -embedding_dim:].to(device)
#         elif which_layer == 'first_and_last_layers':
#             testset_embeddings[-1] = torch.cat([testset_embeddings[-1][:, :embedding_dim], testset_embeddings[-1][:, -embedding_dim:]], dim=1).to(device)
#         elif which_layer.startswith('layer_'):
#             if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
#                 layer_num = int(which_layer.split('_')[-1])
#                 testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
#             elif 'last_layer' in which_layer:
#                 layer_num = int(which_layer.split('_')[1])
#                 testset_embeddings[-1] = torch.cat([testset_embeddings[-1][:, -embedding_dim:], testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
#             elif 'later_layer' in which_layer:
#                 layer_num = int(which_layer.split('_')[1])
#                 testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:].to(device)
#             elif 'to' in which_layer:
#                 layer_num = int(which_layer.split('_')[1])
#                 layer_num2 = int(which_layer.split('_')[3])
#                 testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)

# def test(model,test_set,test_label,test_acc,testset_name):
#     with torch.no_grad():
#         outputs = model(test_set)
#         probabilities = torch.softmax(outputs, dim=1)[:, 1]
#         auroc = roc_auc_score(test_label.cpu().numpy(), probabilities.cpu().numpy())
#         test_acc.append(auroc)
#     return auroc

# class BinaryClassifier(nn.Module):
#     def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=2, dropout_prob=0.2):
#         super(BinaryClassifier, self).__init__()
#         self.num_labels = num_labels
#         layers = []
#         prev_size = input_size
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Dropout(dropout_prob),
#                 nn.Linear(prev_size, hidden_size),
#                 nn.Tanh(),
#             ])
#             prev_size = hidden_size
#         self.dense = nn.Sequential(*layers)
#         self.classifier = nn.Linear(prev_size, num_labels)
    
#     def forward(self, x):
#         x = self.dense(x)
#         x = self.classifier(x)
#         return x
    

# def train(hidden_sizes,droprate):
#     input_size = train_embeddings.shape[1]
#     model = BinaryClassifier(input_size,hidden_sizes=hidden_sizes,dropout_prob=droprate).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     num_epochs = 10
#     batch_size = 16
#     best_valid_acc = 0
#     for epoch in range(num_epochs):
#         for i in range(0, len(train_embeddings), batch_size):
#             model.train()
#             batch_embeddings = train_embeddings[i:i+batch_size]
#             batch_labels = train_labels[i:i+batch_size]
#             outputs = model(batch_embeddings)
#             loss = criterion(outputs, batch_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         with torch.no_grad():
#             outputs = model(valid_embeddings)
#             _, predicted = torch.max(outputs.data, 1)
#             accuracy = (predicted == valid_labels).sum().item() / len(valid_labels)
#             testset_acc = []
#             for test_embed,test_label in zip(testset_embeddings,testset_labels):
#                 test(model,test_embed ,test_label,testset_acc,f' ')
#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}, Xsum/writing/pub/gpt4Xsum/gpt4writing/gpt4pub/claude3Xsum/claude3writing/claude3pub Test auroc: {','.join([str(round(i,4)) for i in testset_acc])}")
#             if accuracy > best_valid_acc:
#                 best_valid_acc = accuracy
#                 best_test_acc = testset_acc
#     return best_test_acc

# if __name__ == '__main__':
#     best_test_acc = train([1024,512],droprate) 
#     print('='*20)
#     print('best test acc:',','.join([str(round(i,4)) for i in best_test_acc]))
#     print('average test acc:',[sum(best_test_acc[i*3:i*3+3])/3 for i in range(3)])













# final_classifier.py
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import csv

# --------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change these base paths if your files live elsewhere
EMB_BASE = "/content/save/gte-qwen1.5-1.5B_all_embedding/save_embedding"
KL_BASE  = "/content/drive/MyDrive/Text-Fluoroscopy/save/Qwen2-1.5B-Instruct_embedding"
LABEL_BASE = "/content/drive/MyDrive/Text-Fluoroscopy/dataset/labels"

# dataset split (chosen earlier)
train_file = "writing_gpt-3"
valid_file = "writing_claude-3-opus-20240229-gpt3"
test_files = [
    "Xsum_gpt3.5-turbo",
    "writing_gpt-3",
    "pub_gpt-3",
    "gpt4-Xsum-gpt3",
    "gpt4-writing-gpt3",
    "gpt4-pub-gpt3",
    "xsum_claude-3-opus-20240229-gpt3",
    "writing_claude-3-opus-20240229-gpt3",
    "pub_claude-3-opus-20240229-gpt3"
]

# model/hyperparams
which_layer = "max_kl"   # options: max_kl, max_kl_and_last_layer, first_layer, last_layer, first_and_last_layers, layer_X, etc.
learning_rate = 3e-3
batch_size = 16
num_epochs = 10
hidden_sizes = [1024, 512]
dropout_prob = 0.4
max_train_samples = None   # e.g. 160 to limit, or None for all
max_valid_samples = None
max_test_samples = None

log_csv = "training_log.csv"
best_model_path = "best_binary_classifier.pt"

# --------- UTILITIES ----------
def load_embedding(name):
    path = os.path.join(EMB_BASE, f"{name}.pt")
    return torch.load(path)

def load_labels(name):
    path = os.path.join(LABEL_BASE, f"{name}.pt")
    return torch.load(path)

def load_kl(name):
    path = os.path.join(KL_BASE, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def infer_per_layer_dim_from(embedding_tensor, sample_kl):
    # embedding_tensor: [N, D]
    # sample_kl: e.g. train_kl[0] is list length L = num_middle_layers
    L = len(sample_kl)          # L = len(outputs.hidden_states)-2
    total_hidden_states = L + 2 # equals number of hidden_states concatenated
    D = embedding_tensor.shape[1]
    if D % total_hidden_states != 0:
        raise ValueError(f"Cannot infer per-layer dim: D={D} is not divisible by total_hidden_states={total_hidden_states}")
    per_layer = D // total_hidden_states
    return per_layer, total_hidden_states

def select_max_kl_layer_embeddings(embed_tensor, kl_array, per_layer):
    # kl_array: shape (N, L) where L = len(kls) length
    idx = kl_array.argmax(axis=1)   # idx per example
    out = []
    for i, layer in enumerate(idx):
        # layer corresponds to middle layer index i in kls -> maps to embeddings slice:
        # kls are for i in range(1, len(hidden_states)-1) so if chosen layer = r (0-based in kls),
        # actual hidden_state index = r+1, and embedding slice starts at ( (r+1)*per_layer )
        start = (layer + 1) * per_layer
        end = (layer + 2) * per_layer
        out.append(embed_tensor[i, start:end])
    return torch.stack(out, dim=0)

def select_layer_embeddings(embed_tensor, kl_array, per_layer, which_layer_option):
    # embed_tensor is torch tensor on CPU/GPU
    if which_layer_option == "max_kl":
        return select_max_kl_layer_embeddings(embed_tensor, kl_array, per_layer)
    elif which_layer_option == "max_kl_and_last_layer":
        sel = select_max_kl_layer_embeddings(embed_tensor, kl_array, per_layer)
        last = embed_tensor[:, -per_layer:]
        return torch.cat([sel, last.to(sel.device)], dim=1)
    elif which_layer_option == "first_layer":
        return embed_tensor[:, :per_layer]
    elif which_layer_option == "last_layer":
        return embed_tensor[:, -per_layer:]
    elif which_layer_option == "first_and_last_layers":
        return torch.cat([embed_tensor[:, :per_layer], embed_tensor[:, -per_layer:]], dim=1)
    else:
        # supports layer_X (zero-based index of hidden_state block)
        if which_layer_option.startswith("layer_"):
            parts = which_layer_option.split("_")
            # layer_N  -> take that layer only (hidden_state index N)
            # layer_N_last_layer or other variants can be implemented similarly
            try:
                layer_num = int(parts[1])
                return embed_tensor[:, (layer_num)*per_layer : (layer_num+1)*per_layer]
            except Exception as e:
                raise ValueError("Unsupported which_layer option: " + which_layer_option) from e
        raise ValueError("Unsupported which_layer option: " + which_layer_option)

# --------- MODEL ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024,512], num_labels=2, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Dropout(dropout_prob))
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_labels)
    def forward(self, x):
        x = self.dense(x)
        x = self.classifier(x)
        return x

# --------- LOAD RAW EMBEDDINGS & KL to infer sizes ----------
print("Loading sample embeddings to infer per-layer dim...")
sample_embed = load_embedding(train_file)      # e.g. tensor [N, D]
sample_kl = load_kl(train_file)                # list of lists: shape (N, L)
sample_kl = np.array(sample_kl)
per_layer_dim, total_hidden_states = infer_per_layer_dim_from(sample_embed, sample_kl[0])
print(f"Detected embedding D={sample_embed.shape[1]}, total_hidden_states={total_hidden_states}, per_layer_dim={per_layer_dim}")

# --------- LOAD FULL DATA ----------
def prepare_dataset(name, max_samples=None):
    emb = load_embedding(name)
    labels = load_labels(name)
    # optionally trim
    if max_samples is not None:
        emb = emb[:max_samples]
        labels = labels[:max_samples]
    # load kl
    kl = np.array(load_kl(name))
    # select according to which_layer
    emb_sel = select_layer_embeddings(emb, kl, per_layer_dim, which_layer)
    return emb_sel.to(device), labels.to(device)

print("Preparing train/valid/test datasets...")
train_embeddings, train_labels = prepare_dataset(train_file, max_train_samples)
valid_embeddings, valid_labels = prepare_dataset(valid_file, max_valid_samples)

# prepare multiple test sets (list)
testset_embeddings = []
testset_labels = []
for tname in test_files:
    try:
        emb_t, lab_t = prepare_dataset(tname, max_test_samples)
        testset_embeddings.append(emb_t)
        testset_labels.append(lab_t)
    except Exception as e:
        print(f"Warning: couldn't load {tname}: {e}")

print(f"Train shape: {train_embeddings.shape}, Valid shape: {valid_embeddings.shape}")
print(f"Loaded {len(testset_embeddings)} test sets.")

# --------- TRAIN LOOP ----------
def evaluate_model(model, emb, labels):
    model.eval()
    with torch.no_grad():
        logits = model(emb)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds = probs >= 0.5
        labels_np = labels.cpu().numpy()
        try:
            auroc = roc_auc_score(labels_np, probs)
        except Exception:
            auroc = float("nan")
        acc = accuracy_score(labels_np, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary", zero_division=0)
    return {"auroc": auroc, "acc": acc, "prec": p, "rec": r, "f1": f1}

def train_and_evaluate():
    input_size = train_embeddings.shape[1]
    model = BinaryClassifier(input_size, hidden_sizes=hidden_sizes, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = -1.0
    best_stats = None

    # CSV header
    with open(log_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch","train_loss","valid_acc","valid_auroc","testset_names","testset_aurocs"])

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(train_embeddings.size(0))
        for i in range(0, train_embeddings.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            batch_emb = train_embeddings[idx]
            batch_labels = train_labels[idx]
            optimizer.zero_grad()
            outputs = model(batch_emb)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_emb.size(0)
        epoch_loss = epoch_loss / train_embeddings.size(0)

        # validation
        val_stats = evaluate_model(model, valid_embeddings, valid_labels)
        test_aurocs = []
        for emb_t, lab_t in zip(testset_embeddings, testset_labels):
            stats = evaluate_model(model, emb_t, lab_t)
            test_aurocs.append(stats["auroc"])

        print(f"Epoch {epoch+1}/{num_epochs} | loss={epoch_loss:.4f} | valid_acc={val_stats['acc']:.4f} | valid_auroc={val_stats['auroc']:.4f}")
        print(" Test AUROCs:", [round(x,4) if not np.isnan(x) else None for x in test_aurocs])

        # write CSV row
        with open(log_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, epoch_loss, val_stats["acc"], val_stats["auroc"], ";".join(test_files), ";".join([str(x) for x in test_aurocs])])

        if val_stats["acc"] > best_valid_acc:
            best_valid_acc = val_stats["acc"]
            best_stats = {"epoch": epoch+1, "valid": val_stats, "test_aurocs": test_aurocs}
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")

    return best_stats

if __name__ == "__main__":
    best = train_and_evaluate()
    print("==== BEST ====")
    print(best)
    print("Log saved to", log_csv)
    print("Model saved to", best_model_path)
