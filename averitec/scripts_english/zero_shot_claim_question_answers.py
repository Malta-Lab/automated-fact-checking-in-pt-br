import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
ARQUIVOS_JSON = [
    Path("../dataset/dev.json"),
    Path("../dataset/train.json"),
]
CAMINHO_SAIDA = Path("../results_english/zero_shot_claim_question_answers_en.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

# Labels in English
ROTULOS_INGLES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking"
]

# Load existing results (reduced structure)
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

claims_classificadas = {item["claim"] for item in saida_reduzida}

def gerar_prompt(claim, questions):
    prompt = f"""You are a fact checker. Classify the following claim based on the provided questions and answers. Use only the information given. Respond with only one of the following options:

- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence/Cherrypicking

Claim: "{claim}"

"""
    for q in questions:
        prompt += f"Question: {q['question']}\n"
        for ans in q.get("answers", []):
            prompt += f"Answer: {ans.get('answer', '').strip()}\n"
    prompt += "\nClassification:"
    return prompt

def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "prompt": prompt, "stream": False},
            timeout=90
        )
        resposta = response.json().get("response", "").strip()
        for rotulo in ROTULOS_INGLES:
            if rotulo.lower() in resposta.lower():
                return rotulo
        return "Not Enough Evidence"
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

# Count total claims
total_claims = sum(
    len(json.load(open(arquivo, "r", encoding="utf-8"))) for arquivo in ARQUIVOS_JSON
)
claims_processadas = len(claims_classificadas)
saida_dict = {item["claim"]: item for item in saida_reduzida}

# Process files
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
            # Reduce questions and answers
            questions_reduzidas = []
            for q in questions:
                respostas_filtradas = [{"answer": a["answer"]} for a in q.get("answers", []) if "answer" in a]
                questions_reduzidas.append({
                    "question": q["question"],
                    "answers": respostas_filtradas
                })

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
