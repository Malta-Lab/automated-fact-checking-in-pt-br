# Fact-Checking & Dataset Translator Automation

Welcome to the **Fact-Checking & Dataset Translator Automation** project!  
This repository provides scripts and workflows for translating fact-checking datasets and automating claim classification using large language models (LLMs).

**What this project does:**  
- Translates the original [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) and [Averitec](https://fever.ai/dataset/averitec.html) datasets to Portuguese using LLMs.
- Runs Zero-Shot and Few-Shot classification experiments on both the original English datasets and their Portuguese translations.

---

## 📂 Project Structure

```
averitec/
  main_english.ipynb
  main.ipynb
  dataset/
  results/
  results_english/
  scripts/
  scripts_english/
liar/
  main_english.ipynb
  main.ipynb
  dataset/
  results/
  results_english/
  scripts/
  scripts_english/
```

- **main.ipynb / main_english.ipynb**: Jupyter notebooks for running and analyzing experiments.
- **dataset/**: Contains the translated datasets (Portuguese) and original datasets (English).
- **results/**: Model outputs and evaluation results for Portuguese datasets.
- **results_english/**: Model outputs and evaluation results for English datasets.
- **scripts/**: Processing, translation, and classification scripts for Portuguese.
- **scripts_english/**: Processing and classification scripts for English.

---

## 🚀 Features

- **Dataset Translation**: Translate LIAR and Averitec datasets from English to Portuguese using LLMs.
- **Zero-Shot & Few-Shot Classification**: Run claim classification experiments on both English and Portuguese datasets.
- **Multi-Label Support**: Handles multiple fact-checking labels (e.g., Supported, Refuted, Not Enough Evidence).
- **Automated Evaluation**: Generates classification reports and confusion matrices.
- **Extensible**: Easily add new models or datasets.

---

## 🛠️ Requirements

- Python 3.8+
- [tqdm](https://tqdm.github.io/)
- [scikit-learn](https://scikit-learn.org/)
- [requests](https://docs.python-requests.org/)
- [Jupyter Notebook](https://jupyter.org/)
- An LLM API endpoint (e.g., [Ollama](https://ollama.com/))

Install dependencies:
```sh
pip install -r requirements.txt
```

---

## ⚡ Usage

### 1. Translate Datasets

Translate datasets from English to Portuguese using LLMs:
```sh
python averitec/scripts/translate.py
python liar/scripts/translate.py
```

### 2. Run Classification

**Portuguese:**
```sh
python averitec/scripts/zero_shot_claim_question.py
python averitec/scripts/few_shot_claim_question.py
python liar/scripts/zero_shot_all.py
python liar/scripts/few_shot_all.py
```

**English:**
```sh
python averitec/scripts_english/zero_shot_claim_question.py
python averitec/scripts_english/few_shot_claim_question.py
python liar/scripts_english/zero_shot_all.py
python liar/scripts_english/few_shot_all.py
```

### 3. Evaluate Results

Generate evaluation reports:
```sh
python averitec/scripts_english/report.py ./results_english/zero_shot_claim_question_answers_en.json
python liar/scripts_english/report.py ./results_english/zero_shot_all_en.json
```

Or open the Jupyter notebooks for interactive analysis.

---

## 📊 Example Outputs

- **Classification Reports**: Precision, recall, F1-score for each label.
- **Confusion Matrices**: Visualize model performance.
- **Result Files**: JSON files with model predictions.

---


## 📜 License

This project is licensed under the MIT License.

---

## ✨ Acknowledgements

- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [Averitec Dataset](https://fever.ai/dataset/averitec.html)
- [Ollama](https://ollama.com/)
- OpenAI, Google, and the open-source LLM community

---

Enjoy automating your fact-checking experiments! 🚀
