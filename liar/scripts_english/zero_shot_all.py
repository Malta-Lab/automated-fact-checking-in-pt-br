import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
ARQUIVOS_JSON = [
    Path("../dataset/test.jsonl"),
    Path("../dataset/train.jsonl"),
    Path("../dataset/valid.jsonl"),
]
CAMINHO_SAIDA = Path("../results_english/zero_shot_all_en.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

# Carrega classificações já existentes
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

# Coletar frases já processadas
statements_processados = set(item["statement"] for item in saida_reduzida)

# Função para enviar prompt em inglês ao modelo
def classificar_claim_contexto(dado: dict) -> str:
    statement = dado["statement"]
    speaker = dado.get("speaker", "Unknown")
    job = dado.get("speaker_job_title", "")
    party = dado.get("party_affiliation", "")
    state = dado.get("state_info", "")
    subjects = dado.get("subjects", "")
    context = dado.get("context", "")

    # Prompt em inglês
    prompt = f"""You are a fact-checker. Classify the following claim based on the information provided. Answer with only one of the following options:

        "true\n"
        "false\n"
        "half-true\n"
        "pants-fire\n"
        "barely-true\n"
        "mostly-true\n\n"


Claim: "{statement}"

Subject(s): {subjects}
Who made the claim: {speaker} ({job}, {party} - {state})
Context: {context}
"""

    try:
        resposta = requests.post(HOST, json={"model": MODELO, "prompt": prompt}, stream=True)
        resposta.raise_for_status()

        partes = []
        for linha in resposta.iter_lines(decode_unicode=True):
            if not linha.strip():
                continue
            try:
                obj = json.loads(linha)
                if "response" in obj:
                    partes.append(obj["response"])
                if obj.get("done", False):
                    break
            except json.JSONDecodeError:
                print(f"⚠️ [INVALID JSON] {linha}")
                continue

        resposta_final = "".join(partes).strip()
        return resposta_final if resposta_final else "Empty response"

    except Exception as e:
        print(f"🚨 [HTTP ERROR] Failed to classify: {statement}\n{e}")
        return "Connection error"

# Processar os arquivos .jsonl
for arquivo in ARQUIVOS_JSON:
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in tqdm(f, desc=f"📂 Processing {arquivo.name}"):
            dado = json.loads(linha)
            statement = dado["statement"]

            if statement in statements_processados:
                print(f"⏭️ [SKIPPED] Already processed: {statement[:100]}...")
                continue

            print(f"🔍 [PROCESSING] \"{statement}\"")
            label = classificar_claim_contexto(dado)
            print(f"✅ [RESULT] \"{statement[:80]}...\" → {label}\n")

            saida_reduzida.append({
                "statement": statement,
                "label": label.strip().lower().strip('"')
            })
            statements_processados.add(statement)

# Salvar resultados
with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida_reduzida, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to: {CAMINHO_SAIDA}")
