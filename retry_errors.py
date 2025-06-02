import json
import re
import requests
from pathlib import Path
from typing import Union

# === CONFIG ===
HOST = "http://localhost:11434/api/generate"
RESULTS_DIR = "results"
MODELS = ["gemma3:4b"]  # mesmo nome da pasta usada na tradução original

# === TRADUÇÃO COM CACHE ===
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
        print(f"❌ Erro ao traduzir prompt: {e}")
        return ""

def clean_translation(text: str) -> str:
    text = re.sub(r"<.*?>.*?</.*?>", "", text, flags=re.DOTALL)
    text = re.split(r"(?:[Tt]radu[cç][aã]o\s*[:\-–]?\s*|\n{2,})", text)[-1]
    text = text.strip()
    if text.startswith('"') and text.endswith('"') or text.startswith('“') and text.endswith('”'):
        text = text[1:-1].strip()
    return text

def retry_errors_for_model(model: str):
    model_dir = Path(RESULTS_DIR) / model.replace(":", "-").replace("/", "_")
    for errors_file in model_dir.glob("*_errors.json"):
        print(f"\n🔁 Reprocessando erros em: {errors_file.name}")
        
        dataset_name = errors_file.stem.replace("_errors", "")
        output_file = model_dir / f"{dataset_name}.json"
        cache_file = model_dir / f"{dataset_name}_cache.json"

        # Carregar arquivos existentes
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

                # Atualiza item traduzido no JSON
                translated_data[index] = item
                print(f"✅ Corrigido item {index}")

            except Exception as e:
                print(f"❌ Falha novamente no item {index}: {e}")
                updated_errors.append(err)

        # Salvar resultados atualizados
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
        retry_errors_for_model(model)

if __name__ == "__main__":
    main()
