import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
ARQUIVOS_JSON = [
    Path("../results/gemma3-4b/dev.json"),
    Path("../results/gemma3-4b/train.json"),
]
CAMINHO_SAIDA = Path("../results/few_shot_claim/question_answers.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

rotulos_traduzidos = {
    "Supported": "Apoiada",
    "Refuted": "Refutada",
    "Not Enough Evidence": "Sem Evidência Suficiente",
    "Conflicting Evidence/Cherrypicking": "Evidência Conflitante/Cherrypicking"
}

EXEMPLOS_FEW_SHOT = [
    # APOIADA
    {
        "claim": "O Brasil é o maior produtor mundial de café.",
        "qa": [
            {
                "question": "O Brasil lidera a produção global de café?",
                "answers": [{"answer": "Sim, o Brasil é o maior produtor mundial de café há muitos anos."}]
            }
        ],
        "label": "Apoiada"
    },
    {
        "claim": "A água ferve a 100 graus Celsius ao nível do mar.",
        "qa": [
            {
                "question": "Qual é a temperatura de ebulição da água ao nível do mar?",
                "answers": [{"answer": "100°C é a temperatura padrão de ebulição da água ao nível do mar."}]
            }
        ],
        "label": "Apoiada"
    },

    # REFUTADA
    {
        "claim": "A Terra é o planeta mais quente do sistema solar.",
        "qa": [
            {
                "question": "Qual é o planeta mais quente do sistema solar?",
                "answers": [{"answer": "Vênus é o planeta mais quente, não a Terra."}]
            }
        ],
        "label": "Refutada"
    },
    {
        "claim": "Albert Einstein ganhou o Prêmio Nobel por desenvolver a teoria da relatividade.",
        "qa": [
            {
                "question": "Einstein ganhou o Nobel pela teoria da relatividade?",
                "answers": [{"answer": "Não. Ele ganhou o Nobel pela explicação do efeito fotoelétrico."}]
            }
        ],
        "label": "Refutada"
    },

    # SEM EVIDÊNCIA SUFICIENTE
    {
        "claim": "O chocolate amargo melhora a inteligência.",
        "qa": [
            {
                "question": "Há provas de que chocolate amargo melhora a inteligência?",
                "answers": [{"answer": "Não há estudos conclusivos sobre esse efeito."}]
            }
        ],
        "label": "Sem Evidência Suficiente"
    },
    {
        "claim": "Assistir filmes de terror aumenta a criatividade.",
        "qa": [
            {
                "question": "Filmes de terror aumentam a criatividade?",
                "answers": [{"answer": "Não há evidências científicas sólidas que comprovem essa afirmação."}]
            }
        ],
        "label": "Sem Evidência Suficiente"
    },

    # EVIDÊNCIA CONFLITANTE / CHERRYPICKING
    {
        "claim": "As vacinas contra COVID-19 causam infertilidade.",
        "qa": [
            {
                "question": "Vacinas de COVID-19 causam infertilidade?",
                "answers": [
                    {"answer": "Estudos confiáveis mostram que não há evidência de que causam infertilidade."},
                    {"answer": "Algumas postagens alegam isso, mas são baseadas em desinformação ou dados mal interpretados."}
                ]
            }
        ],
        "label": "Evidência Conflitante/Cherrypicking"
    },
    {
        "claim": "O aquecimento global é causado apenas por ciclos naturais da Terra.",
        "qa": [
            {
                "question": "O aquecimento global tem causas naturais ou humanas?",
                "answers": [
                    {"answer": "Há evidências de que fatores naturais influenciam, mas o consenso é que a ação humana é o principal fator atual."},
                    {"answer": "Alguns estudos destacam fatores naturais, mas isso não é suficiente para explicar o fenômeno."}
                ]
            }
        ],
        "label": "Evidência Conflitante/Cherrypicking"
    }
]

# Carrega classificações já existentes (estrutura reduzida)
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

# Para checagem de duplicatas
claims_classificadas = {item["claim"] for item in saida_reduzida}

def gerar_prompt(claim, questions):
    prompt = (
        "Você é um verificador de fatos. Classifique a alegação com base nas perguntas e respostas fornecidas.\n"
        "Use **apenas** as informações fornecidas. Escolha **apenas uma** das opções abaixo:\n"
        "- Apoiada\n"
        "- Refutada\n"
        "- Sem Evidência Suficiente\n"
        "- Evidência Conflitante/Cherrypicking\n\n"
    )

    # Incluir exemplos few-shot
    for exemplo in EXEMPLOS_FEW_SHOT:
        prompt += f"ALEGAÇÃO: \"{exemplo['claim']}\"\n"
        for q in exemplo["qa"]:
            prompt += f"Pergunta: {q['question']}\n"
            for a in q.get("answers", []):
                prompt += f"Resposta: {a['answer'].strip()}\n"
        prompt += f"Classificação: {exemplo['label']}\n\n"

    # Nova instância a ser classificada
    prompt += f"ALEGAÇÃO: \"{claim}\"\n"
    for q in questions:
        prompt += f"Pergunta: {q['question']}\n"
        for a in q.get("answers", []):
            resposta = a.get("answer", "").strip()
            if resposta:
                prompt += f"Resposta: {resposta}\n"
    prompt += "Classificação:"

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

# Conta total de claims a processar
total_claims = sum(
    len(json.load(open(arquivo, "r", encoding="utf-8"))) for arquivo in ARQUIVOS_JSON
)
claims_processadas = len(claims_classificadas)

# Reorganiza em dicionário para sobreposição segura
saida_dict = {item["claim"]: item for item in saida_reduzida}

# Processa os arquivos
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
            # Reduz perguntas e respostas
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
