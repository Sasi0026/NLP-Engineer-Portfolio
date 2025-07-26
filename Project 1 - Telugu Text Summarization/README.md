# 📚 Telugu Text Summarization using mBART + LoRA (PEFT)

This project focuses on building a multilingual text summarization model for **Telugu language**, fine-tuned using **Facebook’s mBART** architecture with **parameter-efficient fine-tuning (PEFT)** using **LoRA**. The project includes everything from research, data collection, model fine-tuning, and evaluation to deployment using a simple Gradio UI.

---

## 🔍 Research Background

- Conducted extensive research on **Telugu text summarization**, reviewing over **20 academic research papers**.
- Studied existing summarization challenges and limitations in Indian regional languages.
- Identified and utilized the **TeSum Dataset**, a benchmark dataset designed specifically for Telugu summarization.

---

## 🗃️ Dataset Preparation

- Used the **TeSum dataset** for primary training.
- Collected an additional **1,000 Telugu news articles** using **Scrapy web scraping framework**.
- Annotated additional data using pretrained summarizers to enhance dataset diversity.
- Final training dataset consisted of approximately **7,000+ samples**.
- Average target summary length: ~30 words.

---

## 🧹 Data Preprocessing

- Scrapy was used for data collection with minimal post-cleaning (handled in spider logic).
- Used Hugging Face's `AutoTokenizer` for tokenizing Telugu text suited for mBART.

---

## 🧠 Model Details

- **Base model**: `facebook/mbart-large-50-many-to-many-mmt`
- Applied **LoRA (Low-Rank Adaptation)** with PEFT to reduce training cost and freeze ~98.5% of model parameters.
- Training performed using Hugging Face’s `transformers` and `peft` libraries.

### 🔧 Training Configuration

| Parameter       | Value             |
|----------------|-------------------|
| Dataset Size    | 7,000 samples     |
| Epochs          | 6                 |
| Optimizer       | AdamW             |
| Tokenizer       | AutoTokenizer     |
| PEFT Type       | LoRA              |
| Frozen Params   | ~98.5%            |

---

## 📈 Evaluation

Model was evaluated using **ROUGE metrics**:

| Metric        | Value (Final Model) |
|---------------|---------------------|
| ROUGE-1       | 0.27                |
| ROUGE-L       | 0.29                |
| Avg Summary Length | ~30 words     |

- Earlier version trained on 1,500 examples for 8 epochs yielded ROUGE-1 score of 0.21.
- Performance improved after increasing dataset size and tuning epochs.

---

## Deployment

- Model saved and reloaded for inference.
- Deployed a simple Gradio UI for real-time Telugu text summarization.

## Future Work & Enhancements

1. **RLHF Fine-Tuning:** Plan to implement Reinforcement Learning from Human Feedback (RLHF) for further fine-tuning and aligning model outputs with human preferences.
2. **DPO Alignment:** Currently working on Direct Preference Optimization (DPO) to improve model alignment.



**Contact:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)]
- [LinkedIn](www.linkedin.com/in/sasi-kiran18)
