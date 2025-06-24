import json
import re
import requests
from pathlib import Path
from tqdm import tqdm

# === CONFIGURAÇÕES ===
HOST = "http://localhost:11434/api/generate"
MODELS = ["gemma3:4b"]
DATASET_DIR = Path("../dataset")
RESULTS_DIR = Path("../results")

# === TRADUÇÃO COM OLLAMA ===
def translate_text_ollama(model: str, prompt: str) -> str:
    try:
        response = requests.post(
            HOST,
            json={
                "model": model,
                "prompt": f"Traduza para português do Brasil. Apenas responda com a tradução, sem pensar, explicar ou comentar. Apenas a tradução:\n\"{prompt}\"\n\nTradução:",
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        return clean_translation(raw)
    except Exception as e:
        print(f"❌ Erro ao traduzir com o modelo '{model}': {e}")
        return ""

def clean_translation(text: str) -> str:
    text = re.sub(r"<.*?>.*?</.*?>", "", text, flags=re.DOTALL)
    text = re.split(r"(?:[Tt]radu[cç][aã]o\s*[:\-–]?\s*|\n{2,})", text)[-1]
    text = text.strip()
    if text.startswith('"') and text.endswith('"') or text.startswith('“') and text.endswith('”'):
        text = text[1:-1].strip()
    return text

def warmup_model(model: str) -> bool:
    print(f"\n🚀 Inicializando modelo '{model}'...")
    try:
        response = requests.post(
            HOST,
            json={"model": model, "prompt": "Diga apenas 'ok'.", "stream": False},
            timeout=60
        )
        response.raise_for_status()
        print(f"✅ Modelo '{model}' está pronto.\n")
        return True
    except Exception as e:
        print(f"❌ Erro ao iniciar o modelo '{model}': {e}")
        return False

# === TRADUÇÃO DE UM ITEM ===
def translate_item(item, model):
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
                    if 'boolean_explanation' in ans and ans['boolean_explanation']:
                        ans['boolean_explanation'] = translate_text_ollama(model, ans['boolean_explanation'])
    return item

# === PROCESSAMENTO DE ARQUIVO ===
def process_file(file_path: Path, model: str, output_file: Path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar {file_path}: {e}")
        return

    translated_data = []
    for item in tqdm(data, desc=f"Traduzindo {file_path.name}"):
        try:
            item_traduzido = translate_item(item, model)
            translated_data.append(item_traduzido)
        except Exception as e:
            print(f"❌ Erro ao traduzir item: {e}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Tradução final salva em: {output_file}\n")

# === EXECUÇÃO PRINCIPAL ===
def main():
    for model in MODELS:
        if not warmup_model(model):
            continue

        model_dirname = model.replace(":", "-").replace("/", "_")
        output_dir = RESULTS_DIR / model_dirname
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset_file in DATASET_DIR.glob("*.json"):
            print(f"📂 Processando arquivo: {dataset_file.name}")
            output_file = output_dir / dataset_file.name
            process_file(dataset_file, model, output_file)

if __name__ == "__main__":
    main()
