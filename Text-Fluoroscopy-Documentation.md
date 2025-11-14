# Text Fluoroscopy Repository Documentation

## 📖 Overview
This repository implements "Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features" (EMNLP 2024). The method detects AI-generated text by identifying the model layer with the largest distribution difference from both first and last layers, revealing intrinsic "fingerprints" of AI generation.

**Paper**: [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.885.pdf)  
**Authors**: Yu, Xiao; Chen, Kejiang; Yang, Qi; Zhang, Weiming; Yu, Nenghai

---

## 📂 Directory Structure and File Descriptions

### **Root Directory**

#### `README.md`
- **Purpose**: Main documentation and entry point for the repository
- **Contents**: 
  - Project overview and methodology explanation
  - Installation instructions and prerequisites (Python 3.8+, PyTorch 1.10+, CUDA GPU)
  - Three-step workflow guide (download model → extract features → train classifier)
  - Performance benchmarks and optimization strategies
  - Applicability tests across different LLM architectures (gte-Qwen2-7B, stella_en_1.5B_v5, GPT-neo-2.7B)
  - Citation information for the paper

---

### **`assets/` Directory**
Visual documentation and result figures for the paper.

#### `framework.png`
- **Purpose**: Architectural diagram showing the Text Fluoroscopy methodology
- **Description**: Illustrates how the method identifies the optimal layer by calculating KL divergence between middle layers and first/last layers to extract intrinsic text features

#### `results.png`
- **Purpose**: Performance comparison visualization
- **Description**: Shows detection accuracy results comparing Text Fluoroscopy against baseline methods across different LLM-generated text sources

---

### **`dataset/` Directory**
Contains pre-processed datasets and labels for training and evaluation.

#### `dataset/processed_data/`
JSON files containing text samples with labels (0=human, 1=AI-generated).

**Training/Validation/Test Sets:**
- `HC3_en_train.json` - Training dataset from HC3 benchmark (human vs ChatGPT)
- `HC3_en_valid.json` - Validation dataset for hyperparameter tuning
- `HC3_en_test.json` - Test dataset for general evaluation

**ChatGPT Test Sets (gpt-3.5-turbo):**
- `Xsum_gpt3.5-turbo.json` - ChatGPT-generated text on XSum summarization dataset
- `writing_gpt-3.5-turbo.json` - ChatGPT-generated creative writing samples
- `pub_gpt-3.5-turbo.json` - ChatGPT-generated PubMed medical abstracts

**GPT-4 Test Sets:**
- `gpt4-Xsum-gpt3.5.json` - GPT-4 generated XSum summaries
- `gpt4-writing-gpt3.5.json` - GPT-4 generated creative writing
- `gpt4-pub-gpt3.5.json` - GPT-4 generated medical abstracts

**Claude-3 Test Sets:**
- `xsum_claude-3-opus-20240229-gpt3.5.json` - Claude-3 generated XSum summaries
- `writing_claude-3-opus-20240229-gpt3.5.json` - Claude-3 generated creative writing
- `pub_claude-3-opus-20240229-gpt3.5.json` - Claude-3 generated medical abstracts

**Data Format**: Each JSON file contains an array of objects with:
```json
{
  "text": "the actual text content",
  "result": "0 (human) or 1 (AI-generated)"
}
```

#### `dataset/labels/`
PyTorch tensor files (.pt) containing binary labels (0/1) corresponding to the processed_data JSON files.

**Files mirror processed_data structure:**
- Training/validation/test labels: `HC3_en_train.pt`, `HC3_en_valid.pt`, `HC3_en_test.pt`
- ChatGPT labels: `Xsum_gpt3.pt`, `writing_gpt-3.pt`, `pub_gpt-3.pt`
- GPT-4 labels: `gpt4-Xsum-gpt3.pt`, `gpt4-writing-gpt3.pt`, `gpt4-pub-gpt3.pt`
- Claude-3 labels: `xsum_claude-3-opus-20240229-gpt3.pt`, `writing_claude-3-opus-20240229-gpt3.pt`, `pub_claude-3-opus-20240229-gpt3.pt`

**Purpose**: Pre-extracted labels for faster loading during training/testing (avoids parsing JSON each time)

---

### **`gte-qwen/` Directory**
Scripts for feature extraction and KL divergence calculation using the gte-Qwen1.5-7B-instruct model.

#### `save_KL_with_first_and_last_layer.py`
- **Purpose**: Calculates KL divergence for each hidden layer relative to first and last layers
- **What it does**:
  1. Loads the gte-Qwen1.5-7B-instruct model from HuggingFace
  2. For each text sample, processes it through the model with `output_hidden_states=True`
  3. Computes logits from first layer, last layer, and all middle layers using the language model head
  4. Calculates combined KL divergence: `KL(middle||first) + KL(middle||last)` for each middle layer
  5. Saves KL divergence values as pickle files (`.pkl`) in `save/gte-qwen_KL_with_first_and_last_layer/`
- **Key Function**: `get_kl(model, input_texts)` - returns array of KL divergences for each layer
- **Output**: Pickle files containing KL divergence arrays for each dataset (300 samples max per dataset)
- **Runtime**: Processes up to 300 texts per dataset to identify which layer has maximum divergence

#### `save_embedding.py`
- **Purpose**: Extracts embeddings from all hidden layers for each text sample
- **What it does**:
  1. Loads the gte-Qwen1.5-7B-instruct model (as AutoModel, not CausalLM)
  2. For each text, extracts hidden states from all 29 layers + input embedding
  3. Uses last-token pooling to get a single vector per layer
  4. Concatenates all layer embeddings into one large tensor (embedding_dim × num_layers)
  5. Saves concatenated embeddings as PyTorch tensors (`.pt`) in `save/gte-qwen_all_embedding/save_embedding/`
- **Key Function**: `get_all_embedding(model, input_texts)` - returns concatenated embeddings from all layers
- **Output**: PyTorch tensor files of shape `[num_samples, num_layers * embedding_dim]` (e.g., 300 × 122880 for 30 layers × 4096 dim)
- **Purpose**: These embeddings will be sliced by the classifier to extract features from the optimal layer

---

### **`embedding_classify/` Directory**
Classification model training and evaluation.

#### `classify_with_max_KL_layer.py`
- **Purpose**: Main classification script that trains a neural network to detect AI-generated text
- **What it does**:
  1. **Layer Selection**: Loads KL divergence values from Step 2 and identifies the layer with maximum KL divergence for each sample
  2. **Feature Extraction**: Extracts embeddings from the identified optimal layer for each text
  3. **Model Architecture**: Implements a `BinaryClassifier` - a feedforward neural network with:
     - Input: embeddings from max-KL layer (4096 dimensions)
     - Hidden layers: [1024, 512] with Dropout (0.4) and Tanh activation
     - Output: 2 classes (human vs AI-generated)
  4. **Training**: 
     - Trains on HC3 training set (160 samples)
     - Validates on HC3 validation set (20 samples)
     - Uses Adam optimizer with learning rate 0.003
     - Runs for 10 epochs with batch size 16
     - Uses CrossEntropyLoss
  5. **Evaluation**: Tests on 9 different test sets:
     - ChatGPT: XSum, Writing, PubMed
     - GPT-4: XSum, Writing, PubMed
     - Claude-3: XSum, Writing, PubMed
  6. **Metrics**: Reports AUROC (Area Under ROC Curve) for each test set
- **Key Classes/Functions**:
  - `BinaryClassifier`: Neural network for binary classification
  - `train(hidden_sizes, droprate)`: Main training loop
  - `test(model, test_set, test_label, test_acc, testset_name)`: Evaluation function computing AUROC
- **Configurable Parameters**:
  - `which_layer`: Can be 'max_kl' (default), 'first_layer', 'last_layer', 'first_and_last_layers', etc.
  - `learning_rate`: 0.003
  - `droprate`: 0.4
  - `embedding_dim`: 4096
- **Output**: Prints validation accuracy and AUROC scores for all test sets, tracking best performance

---

## 🔄 Workflow Pipeline

### **Step 1: Model Download**
Download the gte-Qwen1.5-7B-instruct model from HuggingFace:
```bash
huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen1.5-7B-instruct \
  --local-dir ../huggingface_model/gte-Qwen1.5-7B-instruct
```

### **Step 2: Feature Extraction**
```bash
# Calculate KL divergences to find optimal layers
python gte-qwen/save_KL_with_first_and_last_layer.py

# Extract embeddings from all layers
python gte-qwen/save_embedding.py
```

### **Step 3: Training and Evaluation**
```bash
# Train classifier using max-KL layer features
python embedding_classify/classify_with_max_KL_layer.py
```

---

## 🎯 Key Methodology

### **Core Insight**
Most detection methods use either:
- **Last layer** (semantic features) - captures meaning but loses generation patterns
- **First layer** (linguistic features) - captures syntax but misses deep patterns

**Text Fluoroscopy** uses **middle layers** with maximum KL divergence from both extremes, capturing intrinsic AI generation fingerprints.

### **Why It Works**
1. AI models generate text layer-by-layer, transforming input through hidden representations
2. Certain middle layers contain unique patterns that reveal AI authorship
3. These "intrinsic features" are more robust across domains and resistant to paraphrasing attacks
4. By comparing against both first and last layers, we isolate the transformation signatures

### **Performance Optimization**
- **Dynamic method**: Examine all layers, select max-KL layer → 0.52s per text
- **Fixed 30th layer**: Pre-selected optimal layer → 0.08s per text (~6.5× faster, <0.7% accuracy loss)

---

## 📊 Results Summary

### **Cross-LLM Generalization**
The method achieves high detection accuracy across:
- **ChatGPT (gpt-3.5-turbo)**: Avg 0.9189 AUROC
- **GPT-4**: Avg 0.9428 AUROC  
- **Claude-3**: Avg 0.9773 AUROC

### **Cross-Domain Robustness**
Tested on diverse domains:
- **XSum**: News summarization
- **Writing**: Creative writing
- **PubMed**: Medical/scientific abstracts

### **Architectural Flexibility**
Successfully tested with:
- gte-Qwen2-7B (advanced embedding model)
- stella_en_1.5B_v5 (efficient embedding model)
- GPT-neo-2.7B (classical generative model)

---

## 🔧 Technical Requirements

- **Python**: 3.8+
- **PyTorch**: 1.10+
- **Hardware**: CUDA-compatible GPU (7B parameter model requires ~14-16GB VRAM)
- **Dependencies**: transformers, torch, scikit-learn, tqdm, pickle, numpy

---

## 📝 Citation

```bibtex
@inproceedings{yu2024textfluoroscopy,
    title={Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features},
    author={Yu, Xiao and Chen, Kejiang and Yang, Qi and Zhang, Weiming and Yu, Nenghai},
    booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2024},
    url={https://aclanthology.org/2024.emnlp-main.885.pdf}
}
```

---

## 📧 Contact
For questions or issues, contact: yuxiao1217@mail.ustc.edu.cn

---

**Last Updated**: November 11, 2025  
**Documentation Generated**: Comprehensive analysis of Text-Fluoroscopy repository

