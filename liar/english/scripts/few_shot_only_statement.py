import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
ARQUIVOS_JSON = [
    Path("../dataset/test.jsonl"),
    Path("../dataset/train.jsonl"),
    Path("../dataset/valid.jsonl"),
]

MODELOS = [
    "gemma3:4b",
    "llama3:8b",
    #"deepseek-r1:7b"
]

HOST = "http://localhost:11434/api/generate"
# ===================================

# Few-shot examples in English
EXAMPLES_FEW_SHOT = [
    {"statement": "Vaccines implant tracking microchips.", "label": "pants-fire"},
    {"statement": "Drinking saltwater cures cancer.", "label": "pants-fire"},
    {"statement": "Barack Obama was born in Kenya.", "label": "false"},
    {"statement": "COVID-19 was caused by 5G towers.", "label": "false"},
    {"statement": "Most refugees are criminals.", "label": "barely-true"},
    {"statement": "Illegal immigrants have full access to Social Security.", "label": "barely-true"},
    {"statement": "Canada offers free healthcare to everyone.", "label": "half-true"},
    {"statement": "Renewable energy is the main source of power in Brazil.", "label": "half-true"},
    {"statement": "California has one of the strongest economies in the U.S.", "label": "mostly-true"},
    {"statement": "The U.S. spends more on healthcare per capita than any other country.", "label": "mostly-true"},
    {"statement": "NASA landed a man on the Moon in 1969.", "label": "true"},
    {"statement": "Hawaii became the 50th U.S. state in 1959.", "label": "true"},
]

# Generate prompt
def gerar_prompt(dado):
    prompt = (
        "You are a fact-checker. Classify the following claim based on the examples provided.\n"
        "Choose only one of the following options:\n"
        "pants-fire\n"
        "false\n"
        "barely-true\n"
        "half-true\n"
        "mostly-true\n"
        "true\n\n"
    )

    for ex in EXAMPLES_FEW_SHOT:
        prompt += f"Claim: \"{ex['statement']}\"\n"
        prompt += f"Label: {ex['label']}\n\n"

    prompt += f"Claim: \"{dado['statement']}\"\n"
    prompt += f"Respond with the label only, without explanations or comments.\n\n"
    prompt += "Label:"
    return prompt

# Send prompt to the model
def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "prompt": prompt},
            timeout=90,
            stream=True
        )
        partes = []
        for linha in response.iter_lines(decode_unicode=True):
            if not linha.strip():
                continue
            try:
                obj = json.loads(linha)
                if "response" in obj:
                    partes.append(obj["response"])
                if obj.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

        resposta_final = "".join(partes).strip()
        return resposta_final if resposta_final else "empty-response"

    except Exception as e:
        print(f"Error during classification: {e}")
        return "connection-error"

# ========== MAIN LOOP FOR EACH MODEL ==========
for MODELO in MODELOS:
    print(f"\n🧠 Running model: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/few_shot_only_statement.json")
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    if CAMINHO_SAIDA.exists():
        with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
            saida_reduzida = json.load(f)
    else:
        saida_reduzida = []

    statements_processados = set(item["statement"] for item in saida_reduzida)

    for arquivo in ARQUIVOS_JSON:
        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in tqdm(f, desc=f"{arquivo.name} ({MODELO})"):
                dado = json.loads(linha)
                statement = dado["statement"]

                if statement in statements_processados:
                    continue

                prompt = gerar_prompt(dado)
                label = classificar_claim(MODELO, prompt)

                saida_reduzida.append({
                    "statement": statement,
                    "label": label.strip().lower().strip('"')
                })
                statements_processados.add(statement)

                with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f_out:
                    json.dump(saida_reduzida, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {CAMINHO_SAIDA}")
