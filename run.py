import json
import re
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Union

# CONFIG
HOST = "http://localhost:11434/api/generate"
MODELS = ["gemma3:4b", "qwen3:8b", "deepseek-r1:8b"]
DATASET_DIR = "dataset"
RESULTS_DIR = "results"

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
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith('“') and text.endswith('”'):
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

def process_file(file_path: Path, model: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_dirname = model.replace(":", "-").replace("/", "_")
    output_dir = Path(RESULTS_DIR) / model_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / file_path.name
    cache_file = output_dir / f"{file_path.stem}_cache.json"
    errors_file = output_dir / f"{file_path.stem}_errors.json"

    translated_data = []
    cache = {}
    errors = []

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)
        print(f"📄 Retomando tradução a partir do progresso existente ({len(translated_data)} itens).")
    else:
        translated_data = [None] * len(data)

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    def cached_translate(text: str) -> str:
        if not text.strip():
            return text
        if text in cache:
            return cache[text]
        translated = translate_text_ollama(model, text)
        if translated:
            cache[text] = translated
        return translated

    for idx in tqdm(range(len(data)), desc=f"Traduzindo {file_path.name}", unit="item"):
        if translated_data[idx] is not None:
            continue

        item = data[idx]
        try:
            if 'claim' in item and item['claim']:
                item['claim'] = cached_translate(item['claim'])

            if 'justification' in item and item['justification']:
                item['justification'] = cached_translate(item['justification'])

            if 'questions' in item:
                for question_obj in item['questions']:
                    if 'question' in question_obj and question_obj['question']:
                        question_obj['question'] = cached_translate(question_obj['question'])
                    if 'answers' in question_obj:
                        for ans in question_obj['answers']:
                            if 'answer' in ans and ans['answer']:
                                ans['answer'] = cached_translate(ans['answer'])
                            if 'boolean_explanation' in ans and ans['boolean_explanation']:
                                ans['boolean_explanation'] = cached_translate(ans['boolean_explanation'])

            translated_data[idx] = item

        except Exception as e:
            print(f"⚠️ Erro ao traduzir item {idx}: {e}")
            errors.append({"index": idx, "item": item})

        # Salvar progresso a cada item
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        if errors:
            with open(errors_file, "w", encoding="utf-8") as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)

    return translated_data

def retry_errors_for_model(model: str):
    model_dir = Path(RESULTS_DIR) / model.replace(":", "-").replace("/", "_")
    for errors_file in model_dir.glob("*_errors.json"):
        print(f"\n🔁 Reprocessando erros em: {errors_file.name}")
        
        dataset_name = errors_file.stem.replace("_errors", "")
        output_file = model_dir / f"{dataset_name}.json"
        cache_file = model_dir / f"{dataset_name}_cache.json"

        with open(errors_file, "r", encoding="utf-8") as f:
            errors = json.load(f)
        with open(output_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)

        cache = {}
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)

        updated_errors = []
        for err in errors:
            index = err['index']
            item = err['item']
            print(f"🔄 Tentando item {index}...")

            try:
                def cached_translate(text: str) -> str:
                    if not text.strip():
                        return text
                    if text in cache:
                        return cache[text]
                    translated = translate_text_ollama(model, text)
                    if translated:
                        cache[text] = translated
                    return translated

                if 'claim' in item and item['claim']:
                    item['claim'] = cached_translate(item['claim'])

                if 'justification' in item and item['justification']:
                    item['justification'] = cached_translate(item['justification'])

                if 'questions' in item:
                    for question_obj in item['questions']:
                        if 'question' in question_obj and question_obj['question']:
                            question_obj['question'] = cached_translate(question_obj['question'])
                        if 'answers' in question_obj:
                            for ans in question_obj['answers']:
                                if 'answer' in ans and ans['answer']:
                                    ans['answer'] = cached_translate(ans['answer'])
                                if 'boolean_explanation' in ans and ans['boolean_explanation']:
                                    ans['boolean_explanation'] = cached_translate(ans['boolean_explanation'])

                translated_data[index] = item
                print(f"✅ Corrigido item {index}")

            except Exception as e:
                print(f"❌ Falha novamente no item {index}: {e}")
                updated_errors.append(err)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        if updated_errors:
            with open(errors_file, "w", encoding="utf-8") as f:
                json.dump(updated_errors, f, ensure_ascii=False, indent=2)
            print(f"⚠️ {len(updated_errors)} erros ainda persistem.")
        else:
            errors_file.unlink()
            print(f"✅ Todos os erros corrigidos. Arquivo de erros removido.")

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
            process_file(dataset_file, model)

    print("🔁 Tentando corrigir erros restantes...")
    for model in MODELS:
        retry_errors_for_model(model)
    print("🏁 Processo concluído com verificação de erros.")

if __name__ == "__main__":
    main()
