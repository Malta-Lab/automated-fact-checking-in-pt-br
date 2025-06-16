# Campos utilizados: "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "context"
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
ARQUIVOS_JSON = [
    Path("../results/gemma3-4b/test.jsonl"),
    Path("../results/gemma3-4b/train.jsonl"),
    Path("../results/gemma3-4b/valid.jsonl"),
]
CAMINHO_SAIDA = Path("../results/zero_shot_all.json")
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

# Novo prompt com contexto adicional
def classificar_claim_contexto(dado: dict) -> str:
    statement = dado["statement"]
    speaker = dado.get("speaker", "Desconhecido")
    cargo = dado.get("speaker_job_title", "")
    partido = dado.get("party_affiliation", "")
    estado = dado.get("state_info", "")
    subjects = dado.get("subjects", "")
    contexto = dado.get("context", "")

    # Criar prompt contextualizado
    prompt = f"""Você é um verificador de fatos. Classifique a seguinte alegação com base nas informações fornecidas. Responda apenas com uma das opções:

- Verdadeiro
- Falso
- Parcialmente Verdadeiro
- Mentira Descarada
- Quase Falso
- Majoritariamente Verdadeiro

Alegação: "{statement}"

Assunto: {subjects}
Quem fez a afirmação: {speaker} ({cargo}, {partido} - {estado})
Contexto: {contexto}
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
                print(f"⚠️ [JSON INVÁLIDO] {linha}")
                continue

        resposta_final = "".join(partes).strip()
        return resposta_final if resposta_final else "Erro de resposta"

    except Exception as e:
        print(f"🚨 [ERRO HTTP] Falha ao classificar: {statement}\n{e}")
        return "Erro de conexão"

# Processar os arquivos .jsonl
for arquivo in ARQUIVOS_JSON:
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in tqdm(f, desc=f"📂 Processando {arquivo.name}"):
            dado = json.loads(linha)
            statement = dado["statement"]

            if statement in statements_processados:
                print(f"⏭️ [PULADO] Já processado: {statement[:100]}...")
                continue

            print(f"🔍 [PROCESSANDO] \"{statement}\"")
            label = classificar_claim_contexto(dado)
            print(f"✅ [RESULTADO] \"{statement[:80]}...\" → {label}\n")

            saida_reduzida.append({
                "statement": statement,
                "label": label
            })
            statements_processados.add(statement)

# Salvar resultados
with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida_reduzida, f, indent=2, ensure_ascii=False)

print(f"\n💾 Resultados salvos em: {CAMINHO_SAIDA}")
