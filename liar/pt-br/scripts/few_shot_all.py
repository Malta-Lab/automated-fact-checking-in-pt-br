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

# ========== EXEMPLOS FEW-SHOT ==========

EXEMPLOS_FEW_SHOT = [
    # MENTIRA DESCARADA
    {
        "statement": "Vacinas implantam microchips de rastreamento.",
        "subjects": "Saúde pública",
        "speaker": "teórico-da-conspiração",
        "speaker_job_title": "Autônomo",
        "state_info": "Flórida",
        "party_affiliation": "Republicano",
        "context": "Vídeo viral",
        "label": "Mentira Descarada"
    },
    {
        "statement": "Beber água salgada cura o câncer.",
        "subjects": "Saúde",
        "speaker": "curandeiro",
        "speaker_job_title": "Alternativo",
        "state_info": "Nevada",
        "party_affiliation": "",
        "context": "Fórum alternativo",
        "label": "Mentira Descarada"
    },
    # FALSO
    {
        "statement": "Barack Obama nasceu no Quênia.",
        "subjects": "Política",
        "speaker": "conspirador-anônimo",
        "speaker_job_title": "Cidadão",
        "state_info": "Texas",
        "party_affiliation": "Republicano",
        "context": "Postagem em rede social",
        "label": "Falso"
    },
    {
        "statement": "O COVID-19 foi causado por torres 5G.",
        "subjects": "Ciência, Saúde",
        "speaker": "desinformador",
        "speaker_job_title": "Influencer",
        "state_info": "Califórnia",
        "party_affiliation": "Independente",
        "context": "Vídeo online",
        "label": "Falso"
    },
    # QUASE VERDADE
    {
        "statement": "A maioria dos refugiados são criminosos.",
        "subjects": "Imigração, Política",
        "speaker": "político-x",
        "speaker_job_title": "Deputado",
        "state_info": "Texas",
        "party_affiliation": "Republicano",
        "context": "Comício",
        "label": "Quase Verdade"
    },
    {
        "statement": "Os imigrantes ilegais têm acesso completo à previdência social.",
        "subjects": "Política, Economia",
        "speaker": "comentarista-debate",
        "speaker_job_title": "Analista",
        "state_info": "Arizona",
        "party_affiliation": "",
        "context": "Debate televisivo",
        "label": "Quase Verdade"
    },   

    # PARCIALMENTE VERDADEIRO
    {
        "statement": "O Canadá tem assistência médica gratuita para todos.",
        "subjects": "Saúde",
        "speaker": "analista-político",
        "speaker_job_title": "Comentarista",
        "state_info": "Ontario",
        "party_affiliation": "",
        "context": "Programa de TV",
        "label": "Parcialmente Verdadeiro"
    },
    {
        "statement": "As energias renováveis representam a maior parte da matriz energética do Brasil.",
        "subjects": "Energia, Meio ambiente",
        "speaker": "engenheiro-ambiental",
        "speaker_job_title": "Consultor",
        "state_info": "São Paulo",
        "party_affiliation": "",
        "context": "Entrevista",
        "label": "Parcialmente Verdadeiro"
    },

    # MAJORITARIAMENTE VERDADEIRO
    {
        "statement": "A Califórnia tem uma das economias mais fortes dos EUA.",
        "subjects": "Economia",
        "speaker": "governador",
        "speaker_job_title": "Governador",
        "state_info": "Califórnia",
        "party_affiliation": "Democrata",
        "context": "Coletiva de imprensa",
        "label": "Majoritariamente Verdadeiro"
    },
    {
        "statement": "Os EUA gastam mais em saúde per capita do que qualquer outro país.",
        "subjects": "Saúde, Economia",
        "speaker": "economista",
        "speaker_job_title": "Pesquisador",
        "state_info": "Nova York",
        "party_affiliation": "",
        "context": "Conferência internacional",
        "label": "Majoritariamente Verdadeiro"
    },

    # VERDADEIRO
    {
        "statement": "A NASA pousou um homem na Lua em 1969.",
        "subjects": "Exploração espacial",
        "speaker": "neil-armstrong",
        "speaker_job_title": "Astronauta",
        "state_info": "Ohio",
        "party_affiliation": "",
        "context": "Entrevista",
        "label": "Verdadeiro"
    },
    {
        "statement": "O Havaí se tornou o 50º estado dos EUA em 1959.",
        "subjects": "História",
        "speaker": "historiador",
        "speaker_job_title": "Professor",
        "state_info": "Havaí",
        "party_affiliation": "",
        "context": "Palestra",
        "label": "Verdadeiro"
    },    
]

# ========== FUNÇÕES AUXILIARES ==========

def gerar_prompt(dado):
    prompt = (
        "Você é um verificador de fatos. Classifique a alegação com base nas informações fornecidas.\n"
        "Use apenas as informações fornecidas. Escolha apenas uma das opções:\n"

        "- Mentira Descarada\n"
        "- Falso\n"
        "- Quase Verdade\n"
        "- Parcialmente Verdadeiro\n"
        "- Majoritariamente Verdadeiro\n"
        "- Verdadeiro\n\n"
    )

    for ex in EXEMPLOS_FEW_SHOT:
        prompt += f"Alegação: \"{ex['statement']}\"\n"
        prompt += f"Assunto: {ex['subjects']}\n"
        prompt += f"Quem fez a afirmação: {ex['speaker']} ({ex['speaker_job_title']}, {ex['party_affiliation']} - {ex['state_info']})\n"
        prompt += f"Contexto: {ex['context']}\n"
        prompt += f"Classificação: {ex['label']}\n\n"

    prompt += f"Alegação: \"{dado['statement']}\"\n"
    prompt += f"Assunto: {dado.get('subjects', '')}\n"
    prompt += f"Quem fez a afirmação: {dado.get('speaker', '')} ({dado.get('speaker_job_title', '')}, {dado.get('party_affiliation', '')} - {dado.get('state_info', '')})\n\n"
    prompt += f"Contexto: {dado.get('context', '')}\n"
    prompt += f"Apenas responda com a classificação, sem pensar, explicar ou comentar.\n"
    prompt += "Classificação:"
    return prompt

def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "think": False, "prompt": prompt},
            timeout=90,
            stream=True
        )
        partes = []
        for linha in response.iter_lines(decode_unicode=True):
            if not linha.strip():
                continue
            try:
                obj = json.loads(linha)
                if "response" in obj:
                    partes.append(obj["response"])
                if obj.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

        resposta_final = "".join(partes).strip()


        return resposta_final if resposta_final else "Erro de resposta"

    except Exception as e:
        print(f"Erro ao classificar: {e}")
        return "Erro de conexão"

# ========== LOOP PRINCIPAL DE CLASSIFICAÇÃO ==========

for MODELO in MODELOS:
    print(f"\n🧠 Rodando modelo: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/few_shot_all.json")
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    if CAMINHO_SAIDA.exists():
        with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
            saida_reduzida = json.load(f)
    else:
        saida_reduzida = []

    statements_processados = set(item["statement"] for item in saida_reduzida)

    for arquivo in ARQUIVOS_JSON:
        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in tqdm(f, desc=f"{arquivo.name} ({MODELO})"):
                dado = json.loads(linha)
                statement = dado["statement"]

                if statement in statements_processados:
                    continue

                prompt = gerar_prompt(dado)
                label = classificar_claim(MODELO, prompt)

                saida_reduzida.append({
                    "statement": statement,
                    "label": label
                })
                statements_processados.add(statement)

                with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f_out:
                    json.dump(saida_reduzida, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Resultados salvos em: {CAMINHO_SAIDA}")
