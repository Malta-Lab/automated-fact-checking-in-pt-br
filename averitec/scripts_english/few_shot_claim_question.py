import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
ARQUIVOS_JSON = [
    Path("../dataset/dev.json"),
    Path("../dataset/train.json"),
]
CAMINHO_SAIDA = Path("../results_english/few_shot_claim_question_en.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

ROTULOS_INGLES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking"
]

EXEMPLOS_FEW_SHOT = [
    {"claim": "Brazil is the largest coffee producer in the world.",
     "question": "Does Brazil lead global coffee production?",
     "label": "Supported"},

    {"claim": "Water boils at 100 degrees Celsius at sea level.",
     "question": "What is the boiling point of water at sea level?",
     "label": "Supported"},

    {"claim": "Earth is the hottest planet in the solar system.",
     "question": "What is the hottest planet in the solar system?",
     "label": "Refuted"},

    {"claim": "Albert Einstein won the Nobel Prize for developing the theory of relativity.",
     "question": "Did Einstein win the Nobel Prize for relativity?",
     "label": "Refuted"},

    {"claim": "Dark chocolate improves intelligence.",
     "question": "Is there evidence that dark chocolate improves intelligence?",
     "label": "Not Enough Evidence"},

    {"claim": "Watching horror movies increases creativity.",
     "question": "Do horror movies increase creativity?",
     "label": "Not Enough Evidence"},

    {"claim": "COVID-19 vaccines cause infertility.",
     "question": "Do COVID-19 vaccines cause infertility?",
     "label": "Conflicting Evidence/Cherrypicking"},

    {"claim": "Global warming is caused only by Earth's natural cycles.",
     "question": "Is global warming caused by natural or human factors?",
     "label": "Conflicting Evidence/Cherrypicking"}
]

# Load previous results
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

claims_classificadas = {item["claim"] for item in saida_reduzida}

def gerar_prompt(claim, questions):
    prompt = (
        "You are a fact checker. Classify the claim based on the provided questions.\n"
        "Use **only** the given information. Choose **only one** of the following options:\n"
        "- Supported\n"
        "- Refuted\n"
        "- Not Enough Evidence\n"
        "- Conflicting Evidence/Cherrypicking\n\n"
    )

    # Few-shot examples
    for exemplo in EXEMPLOS_FEW_SHOT:
        prompt += f"CLAIM: \"{exemplo['claim']}\"\n"
        prompt += f"Question: {exemplo['question']}\n"
        prompt += f"Classification: {exemplo['label']}\n\n"

    # Instance to classify
    prompt += f"CLAIM: \"{claim}\"\n"
    for q in questions:
        prompt += f"Question: {q['question']}\n"
    prompt += "Classification:"
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

total_claims = sum(
    len(json.load(open(arquivo, "r", encoding="utf-8"))) for arquivo in ARQUIVOS_JSON
)
claims_processadas = len(claims_classificadas)
saida_dict = {item["claim"]: item for item in saida_reduzida}

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
