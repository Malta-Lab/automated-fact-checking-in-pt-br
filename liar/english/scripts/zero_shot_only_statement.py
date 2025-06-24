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

# Função para gerar o prompt com apenas o statement
def gerar_prompt(statement: str) -> str:
    return f"""You are a fact-checker. Classify the following claim using only the information provided.
Choose only one of the following options:

pants-fire
false
barely-true
half-true
mostly-true
true

Claim: "{statement}"

Respond with the label only, without explanations or comments.

Label:"""

# Envia o prompt para o modelo
def classificar_claim(modelo: str, claim: str) -> str:
    prompt = gerar_prompt(claim)
    try:
        resposta = requests.post(HOST, json={"model": modelo, "prompt": prompt}, stream=True)
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
        return resposta_final if resposta_final else "empty-response"

    except Exception as e:
        print(f"🚨 [HTTP ERROR] Failed to classify: {claim}\n{e}")
        return "connection-error"

# ========== LOOP PRINCIPAL PARA CADA MODELO ==========
for MODELO in MODELOS:
    print(f"\n🧠 Running model: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/zero_shot_only_statement.json")
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    # Carrega classificações já existentes
    if CAMINHO_SAIDA.exists():
        with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
            saida_reduzida = json.load(f)
    else:
        saida_reduzida = []

    # Coletar frases já processadas
    statements_processados = set(item["statement"] for item in saida_reduzida)

    for arquivo in ARQUIVOS_JSON:
        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in tqdm(f, desc=f"📂 Processing {arquivo.name} ({MODELO})"):
                dado = json.loads(linha)
                statement = dado["statement"]

                if statement in statements_processados:
                    continue

                print(f"🔍 [PROCESSING] \"{statement}\"")
                label = classificar_claim(MODELO, statement)
                print(f"✅ [RESULT] \"{statement[:80]}...\" → {label}\n")

                saida_reduzida.append({
                    "statement": statement,
                    "label": label.strip().lower().strip('"')
                })
                statements_processados.add(statement)

                # Salvar incrementalmente
                with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f_out:
                    json.dump(saida_reduzida, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {CAMINHO_SAIDA}")
