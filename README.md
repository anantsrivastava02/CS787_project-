# Statistical Analysis of Human vs AI Generated Text

## 📌 Overview
The Statistical Analysis folder contains a analysis comparing **human-written** text with **AI-generated** text using entropy-based metrics.  
The core objective is to explore whether **Entropy distributions** can help distinguish between human authorship and outputs from modern large language models.

---

## 📂 Data Sources

### **1. Human-Generated Text**
We used the *Human vs AI Text* dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

### **2. Writing Prompts for AI Generation**
Prompts were taken from the HuggingFace WritingPrompts dataset:  
🔗 https://huggingface.co/datasets/euclaise/writingprompts

---

## 🤖 AI Models Used
The following models were used to generate AI-written paragraphs:

- **deepseek-r1:latest**
- **qwen2.5:7b**
- **llama3.2**

---

## 🧪 Methodology

1. Collected human-written text samples from Kaggle  
2. Sampled prompts from WritingPrompts  
3. Generated AI text using the selected models  
4. Computed entropy metrics:
   - Token entropy  
   - Normalized entropy  
   - Sequence-average entropy  
5. Visualized the distributions through histograms + KDE plots  

All plots are saved in the **`figures/`** folder.

---

## 📊 Visualizations

### **Human-written text entropy distribution**
![Human Text Plot](figures/humanText_plot.jpg)

### **LLaMA 3.2 text entropy distribution**
![Llama 3.2 Plot](figures/llama3.2_3b_plots.jpg)

### **Qwen2.5 7B text entropy distribution**
![Qwen Plot](figures/qwen2.5_7b_plots.jpg)

