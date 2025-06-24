# Campos utilizados: "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "context"
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

HOST = "http://localhost:11434/api/generate"

MODELOS = [
    "gemma3:4b",
    "llama3:8b",
    #"deepseek-r1:7b",
    #"deepseek-r1:70b"
]
# ===================================

# ========== FUNÇÃO DE CLASSIFICAÇÃO ==========

def classificar_claim_contexto(dado: dict, modelo: str) -> str:
    statement = dado["statement"]
    speaker = dado.get("speaker", "Desconhecido")
    cargo = dado.get("speaker_job_title", "")
    partido = dado.get("party_affiliation", "")
    estado = dado.get("state_info", "")
    subjects = dado.get("subjects", "")
    contexto = dado.get("context", "")

    prompt = f"""Você é um verificador de fatos. Classifique a seguinte alegação com base nas informações fornecidas. Responda apenas com uma das opções:

- Mentira Descarada
- Falso
- Quase Verdade
- Parcialmente Verdadeiro
- Majoritariamente Verdadeiro
- Verdadeiro

Alegação: "{statement}"

Assunto: {subjects}
Quem fez a afirmação: {speaker} ({cargo}, {partido} - {estado})
Contexto: {contexto}

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
        if "<|think|>" in resposta_final:
            resposta_final = resposta_final.split("<|think|>")[-1].strip()
        return resposta_final if resposta_final else "Erro de resposta"

    except Exception as e:
        print(f"🚨 [ERRO HTTP] Falha ao classificar: {statement}\n{e}")
        return "Erro de conexão"

# ========== LOOP PRINCIPAL POR MODELO ==========

for MODELO in MODELOS:
    print(f"\n🚀 Rodando modelo: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/zero_shot_all.json")
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    # Carrega classificações já existentes
    if CAMINHO_SAIDA.exists():
        with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
            saida_reduzida = json.load(f)
    else:
        saida_reduzida = []

    statements_processados = set(item["statement"] for item in saida_reduzida)

    for arquivo in ARQUIVOS_JSON:
        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in tqdm(f, desc=f"📂 {arquivo.name} ({MODELO})"):
                dado = json.loads(linha)
                statement = dado["statement"]

                if statement in statements_processados:
                    print(f"⏭️ [PULADO] Já processado: {statement[:100]}...")
                    continue

                print(f"🔍 [PROCESSANDO] \"{statement}\"")
                label = classificar_claim_contexto(dado, MODELO)
                print(f"✅ [RESULTADO] \"{statement[:80]}...\" → {label}\n")

                saida_reduzida.append({
                    "statement": statement,
                    "label": label
                })
                statements_processados.add(statement)

                with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f_out:
                    json.dump(saida_reduzida, f_out, indent=2, ensure_ascii=False)

    print(f"💾 Resultados salvos em: {CAMINHO_SAIDA}")
