import json
import re
import requests
from pathlib import Path
from tqdm import tqdm

# === CONFIGURAÇÕES ===
HOST = "http://localhost:11434/api/generate"
MODELS = ["gemma3:4b"]
DATASET_DIR = "dataset"
RESULTS_DIR = "results"

# === TRADUÇÃO COM OLLAMA ===
def translate_text_ollama(model: str, prompt: str) -> str:
    try:
        response = requests.post(
            HOST,
            json={
                "model": model,
                "prompt": f"Traduza para português. Apenas responda com a tradução, sem pensar, explicar ou comentar. Apenas a tradução:\n\"{prompt}\"\n\nTradução:",
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
    if (text.startswith('"') and text.endswith('"')) or (text.startswith('“') and text.endswith('”')):
        text = text[1:-1].strip()
    return text

def warmup_model(model: str) -> bool:
    print(f"\n🚀 Inicializando modelo '{model}'...")
    try:
        response = requests.post(
            HOST,
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

# === PROCESSAMENTO DO .JSONL ===
def process_jsonl(file_path: Path, model: str, output_file: Path, errors_file: Path, cache_file: Path):
    translations_cache = {}
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            translations_cache = json.load(f)

    errors = []
    if errors_file.exists():
        with open(errors_file, 'r', encoding='utf-8') as f:
            errors = json.load(f)

    def cached_translate(text: str) -> str:
        if not text.strip():
            return text
        if text in translations_cache:
            return translations_cache[text]
        translated = translate_text_ollama(model, text)
        if translated:
            translations_cache[text] = translated
        return translated

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Conta o número total de linhas do arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, 'r', encoding='utf-8') as fin, \
         open(output_file, 'a', encoding='utf-8') as fout:

        for idx, line in enumerate(tqdm(fin, total=total_lines, desc=f"Traduzindo {file_path.name}")):
            try:
                item = json.loads(line)
                claim = item.get("claim", "")
                translated_claim = cached_translate(claim)
                item["claim"] = translated_claim  # 🔁 Substitui o campo original
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

                # Atualiza cache a cada linha
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(translations_cache, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"❌ Erro ao traduzir linha {idx}: {e}")
                errors.append({'index': idx, 'line': line.strip(), 'error': str(e)})
                with open(errors_file, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, ensure_ascii=False, indent=2)

# === EXECUÇÃO PRINCIPAL ===
def main():
    for model in MODELS:
        if not warmup_model(model):
            print(f"⚠️ Pulando modelo '{model}' por falha na inicialização.\n")
            continue

        model_dirname = model.replace(":", "-").replace("/", "_")
        output_dir = Path(RESULTS_DIR) / model_dirname
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔁 Iniciando traduções com modelo: {model_dirname}\n")
        for dataset_file in Path(DATASET_DIR).glob("*.jsonl"):
            print(f"📂 Processando arquivo: {dataset_file.name}")
            output_file = output_dir / dataset_file.name
            errors_file = output_dir / (dataset_file.stem + "_errors.json")
            cache_file = output_dir / (dataset_file.stem + "_cache.json")
            process_jsonl(dataset_file, model, output_file, errors_file, cache_file)
            print(f"✅ Tradução salva em: {output_file}\n")

if __name__ == "__main__":
    main()
