# 🤖 Automated Fact-Checking in PT-BR

This project focuses on fact-checking using various language models, particularly for English and Brazilian Portuguese (PT-BR) datasets. It includes scripts for zero-shot and few-shot inference, dataset translation, model fine-tuning, and result analysis.

## 🧩 Citation
If you use this repository in your research, please cite it as follows:

- BibTeX:
```
@inproceedings{stil,
 author = {Marcelo Delucis and Lucas Fraga and Otávio Parraga and Christian Mattjie and Rafaela Ravazio and Rodrigo Barros and Lucas Kupssinskü},
 title = { Automated Fact-Checking in Brazilian Portuguese: Resources and Baselines},
 booktitle = {Anais do XVI Simpósio Brasileiro de Tecnologia da Informação e da Linguagem Humana},
 location = {Fortaleza/CE},
 year = {2025},
 pages = {137--148},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/stil.2025.37820},
 url = {https://sol.sbc.org.br/index.php/stil/article/view/37820}
}
```

## 📁 Project Structure

The project is organized into three main directories:

  * **`averitec/`**: Contains scripts and results related to the Averitec fact-checking dataset.
  * **`liar/`**: Contains scripts and results related to the LIAR fact-checking dataset.
  * **`fine-tuning/`**: Houses scripts and configurations for fine-tuning language models on fact-checking datasets.

### Detailed Directory Breakdown:

  * **`averitec/` and `liar/`**

      * **`dataset/`**: Stores the raw fact-checking datasets (`dev.json`, `train.json`, `test.jsonl`, `valid.jsonl`).
      * **`results/`**: Contains subdirectories for different language models (e.g., `gemma3-4b/`, `llama3-8b/`, `deepseek-r1-7b/`) and JSON files with inference results (e.g., `few_shot_all.json`, `zero_shot_only_statement.json`).
      * **`scripts/`**: Includes Python scripts for running inference and generating reports:
          * `few_shot_all.py`: Performs few-shot inference including claim, subjects, speaker, job title, state, party affiliation, and context.
          * `few_shot_only_statement.py`: Performs few-shot inference using only the claim statement.
          * `zero_shot_all.py`: Performs zero-shot inference including all available context fields.
          * `zero_shot_only_statement.py`: Performs zero-shot inference using only the claim statement.
          * `report.py`: Generates classification reports, confusion matrices, and ordinal error metrics from inference results.
          * `translate.py`: Translates datasets using a specified language model.
      * **`results_overview.ipynb`**: Jupyter notebooks for visualizing and analyzing results of zero-shot and few-shot experiments.

  * **`fine-tuning/`**

      * **`configs/`**: Stores JSON configuration files for various fine-tuning experiments, specifying model names, dataset paths, hyperparameters (learning rate, batch size, epochs), and other training settings (e.g., `BERT_LARGE_cased_ENG_LIAR_answer_LR_2e-5_1kepochs.json`, `finetune_mBERT_cased_PTBR_averitec_answers_LR_2e-5_2kepochs.json`).
      * **`datasets/`**: Contains Python scripts for custom dataset classes:
          * `averitec.py`: Dataset class for the Averitec dataset.
          * `liar.py`: Dataset class for the LIAR dataset.
      * **`errors/`**: Stores CSV files detailing errors from specific fine-tuning runs (e.g., `BERT_LARGE_cased_ENG_LIAR_answer_gold_errors.csv`).
      * **`LIAR_LR_hyperparam_search_csvs/`**: Contains CSV files with results from hyperparameter search for learning rates on the LIAR dataset (e.g., `BERT_LARGE_cased_ENG_LIAR_answer_LR_2e-5_1kepochs_optuna_trials.csv`).
      * **`hyperparam_search.py`**: Script for performing hyperparameter optimization using Optuna.
      * **`main.py`**: The main script for running fine-tuning experiments based on a specified configuration file.
      * **`model.py`**: Defines the model architecture and loading functions for pre-trained language models.
      * **`results.ipynb`**: Jupyter notebook for analyzing fine-tuning results.
      * **`training.py`**: Contains functions for training and evaluating models.
      * **`utils.py`**: Provides utility functions such as loading configurations, setting random seeds, saving metadata, and finalizing test results.
      * **`.gitignore`**: Specifies intentionally untracked files to ignore.

  * **`requirements.txt`**: Lists Python dependencies required to run the project.

## 🧠 Datasets

This project utilizes the following fact-checking datasets:

  * **[LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)**: A publicly available dataset for fake news detection, consisting of 12.8K manually fact-checked short statements from PolitiFact.com. Each statement is accompanied by meta-information and a label indicating its veracity.


  * **[Averitec Dataset](https://fever.ai/dataset/averitec.html)**: A dataset used for evidence-based fact-checking, typically involving claims that require external evidence for verification.


## ⚙️ Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation

1.  **Clone the repository (if not already done):**

    ```bash
    git clone https://github.com/lucasfrag/automated-fact-checking-in-pt-br
    cd automated-fact-checking-in-pt-br
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Ollama Setup

This project uses `ollama` for running language models locally. You need to install `ollama` and pull the necessary models.

1.  **Download and install Ollama:**
    Follow the instructions on the [Ollama website](https://ollama.com/download) to install it for your operating system.

2.  **Pull the required models:**
    The scripts frequently use `gemma3:4b`. You can pull them using the following commands:

    ```bash
    ollama pull gemma3:4b
    ```

    Ensure your Ollama server is running before executing the project scripts.

## 🚀 Usage

### Running Inference (Zero-Shot and Few-Shot)

Navigate to either the `averitec/` or `liar/` directory and then into the `scripts/` folder.

#### Zero-Shot Inference:

To run zero-shot inference, execute the respective Python script. For example, for Portuguese LIAR dataset:

```bash
cd liar/pt-br/scripts/
python zero_shot_all.py
# or for only statement:
python zero_shot_only_statement.py
```

For English Averitec dataset:

```bash
cd averitec/english/scripts/
python zero_shot_claim_Youtubes.py
# or for only claim and question:
python zero_shot_claim_question.py
```

#### Few-Shot Inference:

To run few-shot inference, execute the respective Python script. For example, for Portuguese LIAR dataset:

```bash
cd liar/pt-br/scripts/
python few_shot_all.py
# or for only statement:
python few_shot_only_statement.py
```

For English Averitec dataset:

```bash
cd averitec/english/scripts/
python few_shot_claim_Youtubes.py
# or for only claim and question:
python few_shot_claim_question.py
```

### Hyperparameter Search

To perform hyperparameter optimization, execute the `hyperparam_search.py` script with a configuration file.

```bash
python hyperparam_search.py <path_to_config_json>
# Example:
python hyperparam_search.py configs/example.json
```

This script uses Optuna to find optimal learning rates, and results are saved in CSV files within the `LIAR_LR_hyperparam_search_csvs/` directory.

### Dataset Translation

To translate datasets, navigate to the `averitec/pt-br/scripts/` or `liar/pt-br/scripts/` directory and run the `translate.py` script.

```bash
cd averitec/pt-br/scripts/
python translate.py
```

This script will translate specified datasets using the configured Ollama model.

## 📊 Analysis and Results

Jupyter notebooks located in `averitec/english/results_overview.ipynb`, `averitec/pt-br/results_overview.ipynb` and `liar/english/results_overview.ipynb`, `liar/pt-br/results_overview.ipynb` and `fine-tuning/results.ipynb` provide detailed analyses and visualizations of the experimental results. You can open these notebooks to explore:

  * Overall evaluation metrics (exact matches, accuracy).
  * Confusion matrices.
  * Classification reports (precision, recall, f1-score).
  * Ordinal error metrics (mean, median, mode error, Quadratic Weighted Kappa).

To view these notebooks, make sure you have Jupyter installed (`pip install jupyter`) and run:

```bash
jupyter notebook
```

Then navigate to the respective `.ipynb` file.





