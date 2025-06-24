import json
import re
import requests
from pathlib import Path
from tqdm import tqdm

# === CONFIGURAÇÕES ===
HOST = "http://localhost:11434/api/generate"
MODELS = ["gemma3:4b"]
DATASET_DIR = Path("../../english/dataset")
RESULTS_DIR = Path("../dataset/")

# === TRADUÇÃO COM OLLAMA ===
def translate_text_ollama(model: str, prompt: str) -> str:
    try:
        response = requests.post(
            HOST,
            json={
                "model": model,
                "prompt": f"Traduza para português do Brasil. Responda apenas com a tradução:\n\"{prompt}\"\n\nTradução:",
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

def translate_line(entry: dict, model: str) -> dict:
    for key in ["statement", "context", "speaker_job_title", "subjects", "party_affiliation", "context"]:
        if key in entry and entry[key]:
            entry[key] = translate_text_ollama(model, entry[key])
    return entry

def process_jsonl_file(file_path: Path, model: str, output_file: Path):
    translated_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Traduzindo {file_path.name}"):
        try:
            item = json.loads(line)
            item_traduzido = translate_line(item, model)
            translated_lines.append(json.dumps(item_traduzido, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Erro ao processar linha: {e}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for line in translated_lines:
            out_f.write(line + '\n')

    print(f"✅ Tradução salva em: {output_file}\n")

# === EXECUÇÃO PRINCIPAL ===
def main():
    for model in MODELS:
        if not warmup_model(model):
            continue

        model_dirname = model.replace(":", "-").replace("/", "_")
        output_dir = RESULTS_DIR / model_dirname
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset_file in DATASET_DIR.glob("*.jsonl"):
            print(f"📂 Processando arquivo: {dataset_file.name}")
            output_file = output_dir / dataset_file.name
            process_jsonl_file(dataset_file, model, output_file)

if __name__ == "__main__":
    main()
