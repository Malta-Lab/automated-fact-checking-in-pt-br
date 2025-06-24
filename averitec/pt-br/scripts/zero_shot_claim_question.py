import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
ARQUIVOS_JSON = [
    Path("../results/gemma3-4b/dev.json"),
    Path("../results/gemma3-4b/train.json"),
]
CAMINHO_SAIDA = Path("../results/zero_shot_claim_question.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

rotulos_traduzidos = {
    "Supported": "Apoiada",
    "Refuted": "Refutada",
    "Not Enough Evidence": "Sem Evidência Suficiente",
    "Conflicting Evidence/Cherrypicking": "Evidência Conflitante/Cherrypicking"
}

# Carrega classificações já existentes
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

claims_classificadas = {item["claim"] for item in saida_reduzida}

def gerar_prompt(claim, questions):
    prompt = f"""Você é um verificador de fatos. Classifique a seguinte alegação com base nas perguntas fornecidas. Use apenas as informações fornecidas. Responda apenas com uma das opções:

- Apoiada
- Refutada
- Sem Evidência Suficiente
- Evidência Conflitante/Cherrypicking

Alegação: "{claim}"

"""
    for q in questions:
        prompt += f"Pergunta: {q['question']}\n"
    prompt += "\nClassificação:"
    return prompt

def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "prompt": prompt, "stream": False},
            timeout=90
        )
        resposta = response.json().get("response", "").strip()
        for rotulo in rotulos_traduzidos.values():
            if rotulo.lower() in resposta.lower():
                return rotulo
        return "Sem Evidência Suficiente"
    except Exception as e:
        print(f"Erro ao classificar: {e}")
        return None

# Conta total de claims
total_claims = sum(
    len(json.load(open(arquivo, "r", encoding="utf-8"))) for arquivo in ARQUIVOS_JSON
)
claims_processadas = len(claims_classificadas)
saida_dict = {item["claim"]: item for item in saida_reduzida}

# Processamento principal
for arquivo in ARQUIVOS_JSON:
    with open(arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)

    pbar = tqdm(dados, desc=arquivo.name, unit="claim")

    for item in pbar:
        claim = item["claim"]
        questions = item.get("questions", [])

        if claim in claims_classificadas:
            claims_processadas += 1
            pbar.set_postfix_str(f"{claims_processadas}/{total_claims} ({claims_processadas/total_claims:.1%})")
            continue

        prompt = gerar_prompt(claim, questions)
        classificacao = classificar_claim(MODELO, prompt)

        if classificacao:
            questions_reduzidas = [{"question": q["question"]} for q in questions]

            entrada_reduzida = {
                "claim": claim,
                "questions": questions_reduzidas,
                "label": classificacao
            }

            saida_dict[claim] = entrada_reduzida
            claims_classificadas.add(claim)

            with open(CAMINHO_SAIDA, "w", encoding="utf-8") as out:
                json.dump(list(saida_dict.values()), out, indent=2, ensure_ascii=False)

            claims_processadas += 1
            pbar.set_postfix_str(f"{claims_processadas}/{total_claims} ({claims_processadas/total_claims:.1%})")
