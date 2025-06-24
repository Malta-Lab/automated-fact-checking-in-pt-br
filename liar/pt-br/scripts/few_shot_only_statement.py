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

EXEMPLOS_FEW_SHOT = [
    {"statement": "Vacinas implantam microchips de rastreamento.", "label": "Mentira Descarada"},
    {"statement": "Beber água salgada cura o câncer.", "label": "Mentira Descarada"},

    {"statement": "Barack Obama nasceu no Quênia.", "label": "Falso"},
    {"statement": "O COVID-19 foi causado por torres 5G.", "label": "Falso"},

    {"statement": "A maioria dos refugiados são criminosos.", "label": "Quase Verdade"},
    {"statement": "Os imigrantes ilegais têm acesso completo à previdência social.", "label": "Quase Verdade"},

    {"statement": "O Canadá tem assistência médica gratuita para todos.", "label": "Parcialmente Verdadeiro"},
    {"statement": "As energias renováveis representam a maior parte da matriz energética do Brasil.", "label": "Parcialmente Verdadeiro"},

    {"statement": "A NASA pousou um homem na Lua em 1969.", "label": "Verdadeiro"},
    {"statement": "O Havaí se tornou o 50º estado dos EUA em 1959.", "label": "Verdadeiro"},

    {"statement": "A Califórnia tem uma das economias mais fortes dos EUA.", "label": "Majoritariamente Verdadeiro"},
    {"statement": "Os EUA gastam mais em saúde per capita do que qualquer outro país.", "label": "Majoritariamente Verdadeiro"}
]

# ========== FUNÇÕES ==========

def gerar_prompt(dado):
    prompt = (
        "Você é um verificador de fatos. Classifique a seguinte alegação com base nos exemplos fornecidos.\n"
        "Use apenas uma das opções:\n"
        "- Mentira Descarada\n"
        "- Falso\n"
        "- Quase Verdade\n"
        "- Parcialmente Verdadeiro\n"
        "- Majoritariamente Verdadeiro\n"
        "- Verdadeiro\n\n"
    )

    for ex in EXEMPLOS_FEW_SHOT:
        prompt += f"Alegação: \"{ex['statement']}\"\n"
        prompt += f"Classificação: {ex['label']}\n\n"

    prompt += f"Alegação: \"{dado['statement']}\"\n"
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

# ========== LOOP PRINCIPAL ==========

for MODELO in MODELOS:
    print(f"\n🔁 Rodando modelo: {MODELO}")

    nome_modelo_pasta = MODELO.replace(":", "-").replace("/", "_")
    CAMINHO_SAIDA = Path(f"../results/{nome_modelo_pasta}/few_shot_only_statement.json")
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
