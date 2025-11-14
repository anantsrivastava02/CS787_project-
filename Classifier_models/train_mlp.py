import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_BASE   ="/kaggle/input/newnewmodels"
KL_BASE    ="/kaggle/input/newnewpickle"
LABEL_BASE ="/kaggle/input/newlabels"

train_file="HC3_en_test"
valid_file="gpt4-Xsum-gpt3"
test_files=["gpt4-pub-gpt3", "gpt4-writing-gpt3"]

which_layer="max_kl"
learning_rate=1e-3
batch_size=16
num_epochs=10
hidden_sizes=[1024, 512]
dropout_prob=0.2

best_model_path="best_binary_classifier.pt"

def load_embedding(name):
    path=os.path.join(EMB_BASE, f"{name}.pt")
    return torch.load(path, map_location="cpu")

def load_labels(name):
    path=os.path.join(LABEL_BASE, f"{name}.pt")
    labels=torch.load(path, map_location="cpu").long()
    if labels.min() < 0 or labels.max() > 1:
        raise ValueError(f"[ERROR] Labels for {name} contain non-binary values.")
    return labels

def load_kl(name):
    path=os.path.join(KL_BASE, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def infer_per_layer_dim(embed, sample_kl):
    L=len(sample_kl)
    total_states=L + 2
    D=embed.shape[1]
    if D % total_states != 0:
        raise ValueError(f"Dimension mismatch: D={D} states={total_states}")
    return D // total_states, total_states

def select_max_kl(embed, kl, per_layer):
    idx=kl.argmax(axis=1)
    selected=[]
    for i, layer in enumerate(idx):
        start=(layer + 1) * per_layer
        end  =(layer + 2) * per_layer
        selected.append(embed[i, start:end])
    return torch.stack(selected, dim=0)

def select_layer_embeddings(embed, kl, per_layer, mode):
    if mode == "max_kl":
        return select_max_kl(embed, kl, per_layer)
    elif mode == "last_layer":
        return embed[:, -per_layer:]
    else:
        raise ValueError("Unknown which_layer:", mode)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob):
        super().__init__()
        layers=[]
        prev=input_size
        for h in hidden_sizes:
            layers += [
                nn.Dropout(dropout_prob),
                nn.Linear(prev, h),
                nn.ReLU(),
            ]
            prev=h
        self.dense=nn.Sequential(*layers)
        self.classifier=nn.Linear(prev, 2)

    def forward(self, x):
        return self.classifier(self.dense(x))

def prepare_dataset(name, max_items=None):
    emb=load_embedding(name)
    labels=load_labels(name)
    kl=np.array(load_kl(name))

    min_size=min(len(emb), len(labels), len(kl))
    if max_items is not None:
        min_size=min(min_size, max_items)

    emb=emb[:min_size]
    labels=labels[:min_size]
    kl=kl[:min_size]

    emb_sel=select_layer_embeddings(
        embed=emb,
        kl=kl,
        per_layer=per_layer_dim,
        mode=which_layer,
    )
    return emb_sel.to(device), labels.to(device)

sample_embed=load_embedding(train_file)
sample_kl=np.array(load_kl(train_file))

per_layer_dim, total_states=infer_per_layer_dim(sample_embed, sample_kl[0])

train_embeddings, train_labels=prepare_dataset(train_file, max_items=3000)
valid_embeddings, valid_labels=prepare_dataset(valid_file, max_items=3000)

test_embeddings=[]
test_labels=[]
for t in test_files:
    try:
        e, l=prepare_dataset(t, max_items=3000)
        test_embeddings.append(e)
        test_labels.append(l)
    except Exception as ex:
        print("Skipping:", t, "| Reason:", ex)

def evaluate(model, emb, labels):
    model.eval()
    with torch.no_grad():
        logits=model(emb)
        probs=torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds=(probs >= 0.5).astype(int)
        labels_np=labels.cpu().numpy()
        return {
            "acc": accuracy_score(labels_np, preds),
            "auroc": roc_auc_score(labels_np, probs),
            "precision": precision_score(labels_np, preds, zero_division=0),
            "recall": recall_score(labels_np, preds, zero_division=0),
            "f1": f1_score(labels_np, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(labels_np, preds).tolist(),
        }

def train_model():
    model=BinaryClassifier(
        input_size=train_embeddings.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_prob=dropout_prob
    ).to(device)

    opt=torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn=nn.CrossEntropyLoss()

    best_acc=-1

    for epoch in range(num_epochs):
        model.train()
        perm=torch.randperm(train_embeddings.shape[0])
        total_loss=0
        for i in range(0, len(perm), batch_size):
            idx=perm[i:i+batch_size]
            emb=train_embeddings[idx]
            lab=train_labels[idx]

            opt.zero_grad()
            loss=loss_fn(model(emb), lab)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        val=evaluate(model, valid_embeddings, valid_labels)

        print(
            f"Epoch {epoch+1:02d} | loss={total_loss:.4f} | "
            f"acc={val['acc']:.4f} | f1={val['f1']:.4f} | "
            f"precision={val['precision']:.4f} | recall={val['recall']:.4f} | "
            f"auroc={val['auroc']:.4f}"
        )

        if val["acc"] > best_acc:
            best_acc=val["acc"]
            torch.save(model.state_dict(), best_model_path)

train_model()

best_model=BinaryClassifier(
    input_size=train_embeddings.shape[1],
    hidden_sizes=hidden_sizes,
    dropout_prob=dropout_prob
).to(device)

best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.eval()

print("\n---- VALIDATION SET METRICS ----")
val_metrics=evaluate(best_model, valid_embeddings, valid_labels)
print(val_metrics)

print("\n---- TEST SET METRICS ----")
for emb, lab in zip(test_embeddings, test_labels):
    metrics=evaluate(best_model, emb, lab)
    print(metrics)
