import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========

MODELOS = [
    "gemma3:4b",
    "llama3:8b",
    #"deepseek-r1:7b",
    #"deepseek-r1:70b"
]

ARQUIVOS_JSON_TEMPLATE = lambda modelo: [
    Path(f"../dataset/test.jsonl"),
    Path(f"../dataset/train.jsonl"),
    Path(f"../dataset/valid.jsonl"),
]

HOST = "http://localhost:11434/api/generate"

# ===================================

# 🔁 Função robusta para resposta em streaming
def classificar_claim(modelo: str, claim: str) -> str:
    prompt = f"""Você é um verificador de fatos. Classifique a seguinte alegação com base nas perguntas fornecidas. Use apenas as informações fornecidas. Responda apenas com uma das opções:

- Mentira Descarada
- Falso
- Quase Verdade
- Parcialmente Verdadeiro
- Majoritariamente Verdadeiro
- Verdadeiro

Alegação: "{claim}"

Apenas responda com a classificação, sem pensar, explicar ou comentar.

Classificação:"""

    try:
        resposta = requests.post(HOST, json={"model": modelo, "think": False, "prompt": prompt}, stream=True)
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
                print(f"⚠️ [JSON INVÁLIDO] {linha}")
                continue

        resposta_final = "".join(partes).strip()
        return resposta_final if resposta_final else "Erro de resposta"

    except Exception as e:
        print(f"🚨 [ERRO HTTP] Falha ao classificar: {claim}\n{e}")
        return "Erro de conexão"

# ========== LOOP PARA CADA MODELO ==========
for MODELO in MODELOS:
    print(f"\n🧠 Executando modelo: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    ARQUIVOS_JSON = ARQUIVOS_JSON_TEMPLATE(nome_modelo_pasta)
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/zero_shot_only_statement.json")
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    # Carrega classificações já existentes
    if CAMINHO_SAIDA.exists():
        with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
            saida_reduzida = json.load(f)
    else:
        saida_reduzida = []

    statements_processados = set(item["statement"] for item in saida_reduzida)

    for arquivo in ARQUIVOS_JSON:
        if not arquivo.exists():
            print(f"⚠️ Arquivo não encontrado: {arquivo}")
            continue

        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in tqdm(f, desc=f"📂 Processando {arquivo.name} ({MODELO})"):
                dado = json.loads(linha)
                statement = dado["statement"]

                if statement in statements_processados:
                    continue

                print(f"🔍 [PROCESSANDO] \"{statement}\"")
                label = classificar_claim(MODELO, statement)
                print(f"✅ [RESULTADO] \"{statement[:80]}...\" → {label}\n")

                saida_reduzida.append({
                    "statement": statement,
                    "label": label
                })
                statements_processados.add(statement)

                # Salvamento incremental
                with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f_out:
                    json.dump(saida_reduzida, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Resultados salvos em: {CAMINHO_SAIDA}")
