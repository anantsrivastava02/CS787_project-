import os
import pickle
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_BASE   ="/kaggle/input/newnewmodels"
KL_BASE    ="/kaggle/input/newnewpickle"
LABEL_BASE ="/kaggle/input/newlabels"

train_file="HC3_en_test"
valid_file="gpt4-Xsum-gpt3"
test_files=["gpt4-pub-gpt3", "gpt4-writing-gpt3"]

which_layer="max_kl"

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

X_train=train_embeddings.cpu().numpy()
y_train=train_labels.cpu().numpy()

X_valid=valid_embeddings.cpu().numpy()
y_valid=valid_labels.cpu().numpy()

test_sets_np=[]
for e, l in zip(test_embeddings, test_labels):
    test_sets_np.append((e.cpu().numpy(), l.cpu().numpy()))

def evaluate_classifier(model, X, y):
    probs=model.predict_proba(X)[:, 1]
    preds=(probs >= 0.5).astype(int)
    return {
        "acc": accuracy_score(y, preds),
        "auroc": roc_auc_score(y, probs),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

print("\n======================")
print("Training Random Forest")
print("======================")

rf=RandomForestClassifier(
    n_estimators=600,
    max_depth=40,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

print("\n=== RF Validation ===")
rf_val=evaluate_classifier(rf, X_valid, y_valid)
for k, v in rf_val.items():
    print(f"{k}: {v}")

for (X, y), name in zip(test_sets_np, test_files):
    print(f"\n=== RF Test: {name} ===")
    r=evaluate_classifier(rf, X, y)
    for k, v in r.items():
        print(f"{k}: {v}")

print("\n================")
print("Training SVM RBF")
print("================")

svm=SVC(
    kernel="rbf",
    C=15,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train, y_train)

print("\n=== SVM Validation ===")
svm_val=evaluate_classifier(svm, X_valid, y_valid)
for k, v in svm_val.items():
    print(f"{k}: {v}")

for (X, y), name in zip(test_sets_np, test_files):
    print(f"\n=== SVM Test: {name} ===")
    r=evaluate_classifier(svm, X, y)
    for k, v in r.items():
        print(f"{k}: {v}")
