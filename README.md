

# 🧠 Fine-Tuning LLMs with Hugging Face

This project demonstrates how to fine-tune a **Large Language Model (LLM)** using **Hugging Face Transformers**, **TRL**, and **PEFT**. It applies **LoRA-based parameter-efficient fine-tuning** with **4-bit quantization** for optimal performance on limited hardware.

---

## 🔍 Features

* Fine-tunes a **LLaMA-2** based model
* Uses **LoRA (Low-Rank Adaptation)** for efficient training
* Supports **4-bit quantization** with `bitsandbytes`
* Implements **Supervised Fine-Tuning (SFT)** via `trl.SFTTrainer`
* Loads and trains on custom datasets from Hugging Face Hub
* Generates text interactively after fine-tuning

---

## 🛠️ Tech Stack

* **Language**: Python
* **Frameworks**: PyTorch, Hugging Face Transformers, TRL
* **Optimization**: PEFT (LoRA), BitsAndBytes (4-bit)
* **Dataset Loader**: Hugging Face Datasets

---

## 📁 Project Structure

```
📦 fine-tuning-llm/
├── fine_tuning_llm.ipynb   # Main training notebook
├── requirements.txt         # Dependencies
├── README.md                # Project documentation

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fine-tuning-llm.git
cd fine-tuning-llm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `fine_tuning_llm.ipynb` in Jupyter or Google Colab and execute the cells step by step.

---

## 🧩 Dataset

This project uses:

```
aboonaji/wiki_medical_terms_llam2_format
```

A medical-domain dataset designed for LLaMA instruction-style fine-tuning.

---

## ⚙️ Training Configuration

* **Model**: `aboonaji/llama2finetune-v2`
* **Quantization**: 4-bit (`nf4`)
* **PEFT Type**: LoRA
* **Training Steps**: 100
* **Batch Size**: 4

---

## 💬 Inference Example

Once training is complete, you can interact with the fine-tuned model:

```python
user_prompt = "Explain the importance of medical data privacy."
response = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(response[0]['generated_text'])
```

---

## ⚠️ Notes

* You must log in to [Weights & Biases](https://wandb.ai/authorize) for experiment tracking.
* Fine-tuning large models may require a GPU (preferably T4, A100, or higher).
* Ensure you have enough VRAM or use quantized models to avoid memory overflow.

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and share it for educational or research purposes.

---

## 🤝 Contributing

Pull requests are welcome!
If you'd like to add improvements (like evaluation scripts or deployment examples), please open an issue to discuss.

---



