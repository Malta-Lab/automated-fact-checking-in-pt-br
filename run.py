import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# CONFIGURAÇÕES
MODELS = ["deepseek-r1:8b", "qwen3:8b"]
DATASET_DIR = "dataset"
RESULTS_DIR = "results"


def translate_text_ollama(model: str, prompt: str) -> str:
    """
    Usa a API do Ollama para traduzir texto do inglês para português.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"Traduza para português com linguagem natural e clara:\n{prompt}",
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"❌ Erro ao traduzir com o modelo '{model}': {e}")
        return ""


def warmup_model(model: str) -> bool:
    """
    Inicializa o modelo no Ollama com uma chamada de teste.
    """
    print(f"\n🚀 Inicializando modelo '{model}'...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "Diga apenas 'ok'.",
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        print(f"✅ Modelo '{model}' está pronto.\n")
        return True
    except Exception as e:
        print(f"❌ Erro ao iniciar o modelo '{model}': {e}")
        return False


def process_file(file_path: Path, model: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translated_data = []
    for item in tqdm(data, desc=f"Traduzindo {file_path.name}", unit="item"):
        if 'claim' in item and item['claim']:
            item['claim'] = translate_text_ollama(model, item['claim'])

        if 'justification' in item and item['justification']:
            item['justification'] = translate_text_ollama(model, item['justification'])

        if 'questions' in item:
            for question_obj in item['questions']:
                if 'question' in question_obj and question_obj['question']:
                    question_obj['question'] = translate_text_ollama(model, question_obj['question'])
                if 'answers' in question_obj:
                    for ans in question_obj['answers']:
                        if 'answer' in ans and ans['answer']:
                            ans['answer'] = translate_text_ollama(model, ans['answer'])

        translated_data.append(item)

    return translated_data


def main():
    for model in MODELS:
        if not warmup_model(model):
            print(f"⚠️ Pulando modelo '{model}' por falha na inicialização.\n")
            continue

        model_dirname = model.replace(":", "-").replace("/", "_")
        output_dir = Path(RESULTS_DIR) / model_dirname
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔁 Iniciando traduções com modelo: {model_dirname}\n")
        for dataset_file in Path(DATASET_DIR).glob("*.json"):
            print(f"📂 Processando arquivo: {dataset_file.name}")
            translated_data = process_file(dataset_file, model)
            output_file = output_dir / dataset_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Tradução salva em: {output_file}\n")


if __name__ == "__main__":
    main()
