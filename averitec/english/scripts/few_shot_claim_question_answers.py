import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
ARQUIVOS_JSON = [
    Path("../dataset/dev.json"),
    Path("../dataset/train.json"),
]
CAMINHO_SAIDA = Path("../results_english/few_shot_claim_question_answers_en.json")
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
    # SUPPORTED
    {
        "claim": "Brazil is the largest coffee producer in the world.",
        "qa": [
            {
                "question": "Does Brazil lead global coffee production?",
                "answers": [{"answer": "Yes, Brazil has been the largest coffee producer in the world for many years."}]
            }
        ],
        "label": "Supported"
    },
    {
        "claim": "Water boils at 100 degrees Celsius at sea level.",
        "qa": [
            {
                "question": "What is the boiling point of water at sea level?",
                "answers": [{"answer": "100°C is the standard boiling point of water at sea level."}]
            }
        ],
        "label": "Supported"
    },

    # REFUTED
    {
        "claim": "Earth is the hottest planet in the solar system.",
        "qa": [
            {
                "question": "What is the hottest planet in the solar system?",
                "answers": [{"answer": "Venus is the hottest planet, not Earth."}]
            }
        ],
        "label": "Refuted"
    },
    {
        "claim": "Albert Einstein won the Nobel Prize for developing the theory of relativity.",
        "qa": [
            {
                "question": "Did Einstein win the Nobel Prize for relativity?",
                "answers": [{"answer": "No. He won the Nobel Prize for his explanation of the photoelectric effect."}]
            }
        ],
        "label": "Refuted"
    },

    # NOT ENOUGH EVIDENCE
    {
        "claim": "Dark chocolate improves intelligence.",
        "qa": [
            {
                "question": "Is there proof that dark chocolate improves intelligence?",
                "answers": [{"answer": "There are no conclusive studies proving this effect."}]
            }
        ],
        "label": "Not Enough Evidence"
    },
    {
        "claim": "Watching horror movies increases creativity.",
        "qa": [
            {
                "question": "Do horror movies increase creativity?",
                "answers": [{"answer": "There is no strong scientific evidence supporting this claim."}]
            }
        ],
        "label": "Not Enough Evidence"
    },

    # CONFLICTING EVIDENCE
    {
        "claim": "COVID-19 vaccines cause infertility.",
        "qa": [
            {
                "question": "Do COVID-19 vaccines cause infertility?",
                "answers": [
                    {"answer": "Reliable studies show no evidence that they cause infertility."},
                    {"answer": "Some posts make this claim, but they are based on misinformation or misinterpreted data."}
                ]
            }
        ],
        "label": "Conflicting Evidence/Cherrypicking"
    },
    {
        "claim": "Global warming is caused only by Earth's natural cycles.",
        "qa": [
            {
                "question": "Is global warming caused by natural or human-made factors?",
                "answers": [
                    {"answer": "There is evidence that natural factors influence climate, but human activity is the main driver today."},
                    {"answer": "Some studies highlight natural influences, but they don’t fully explain the phenomenon."}
                ]
            }
        ],
        "label": "Conflicting Evidence/Cherrypicking"
    }
]

# Load existing classifications
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

claims_classificadas = {item["claim"] for item in saida_reduzida}

def gerar_prompt(claim, questions):
    prompt = (
        "You are a fact checker. Classify the claim based on the provided questions and answers.\n"
        "Use **only** the provided information. Choose **only one** of the following options:\n"
        "- Supported\n"
        "- Refuted\n"
        "- Not Enough Evidence\n"
        "- Conflicting Evidence/Cherrypicking\n\n"
    )

    # Add few-shot examples
    for exemplo in EXEMPLOS_FEW_SHOT:
        prompt += f"CLAIM: \"{exemplo['claim']}\"\n"
        for q in exemplo["qa"]:
            prompt += f"Question: {q['question']}\n"
            for a in q.get("answers", []):
                prompt += f"Answer: {a['answer'].strip()}\n"
        prompt += f"Classification: {exemplo['label']}\n\n"

    # Instance to classify
    prompt += f"CLAIM: \"{claim}\"\n"
    for q in questions:
        prompt += f"Question: {q['question']}\n"
        for a in q.get("answers", []):
            resposta = a.get("answer", "").strip()
            if resposta:
                prompt += f"Answer: {resposta}\n"
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
