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
CAMINHO_SAIDA = Path("../results/few_shot_only_statement.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

EXEMPLOS_FEW_SHOT = [
    {"statement": "A NASA pousou um homem na Lua em 1969.", "label": "Verdadeiro"},
    {"statement": "O Havaí se tornou o 50º estado dos EUA em 1959.", "label": "Verdadeiro"},
    {"statement": "Barack Obama nasceu no Quênia.", "label": "Falso"},
    {"statement": "O COVID-19 foi causado por torres 5G.", "label": "Falso"},
    {"statement": "O Canadá tem assistência médica gratuita para todos.", "label": "Parcialmente Verdadeiro"},
    {"statement": "As energias renováveis representam a maior parte da matriz energética do Brasil.", "label": "Parcialmente Verdadeiro"},
    {"statement": "Vacinas implantam microchips de rastreamento.", "label": "Mentira Descarada"},
    {"statement": "Beber água salgada cura o câncer.", "label": "Mentira Descarada"},
    {"statement": "A maioria dos refugiados são criminosos.", "label": "Quase Falso"},
    {"statement": "Os imigrantes ilegais têm acesso completo à previdência social.", "label": "Quase Falso"},
    {"statement": "A Califórnia tem uma das economias mais fortes dos EUA.", "label": "Majoritariamente Verdadeiro"},
    {"statement": "Os EUA gastam mais em saúde per capita do que qualquer outro país.", "label": "Majoritariamente Verdadeiro"}
]

if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

statements_processados = set(item["statement"] for item in saida_reduzida)

def gerar_prompt(dado):
    prompt = (
        "Você é um verificador de fatos. Classifique a seguinte alegação com base nos exemplos fornecidos.\n"
        "Use apenas uma das opções:\n"
        "- Verdadeiro\n"
        "- Falso\n"
        "- Parcialmente Verdadeiro\n"
        "- Mentira Descarada\n"
        "- Quase Falso\n"
        "- Majoritariamente Verdadeiro\n\n"
    )

    for ex in EXEMPLOS_FEW_SHOT:
        prompt += f"Alegação: \"{ex['statement']}\"\n"
        prompt += f"Classificação: {ex['label']}\n\n"

    prompt += f"Alegação: \"{dado['statement']}\"\n"
    prompt += "Classificação:"
    return prompt

def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "prompt": prompt},
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

for arquivo in ARQUIVOS_JSON:
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in tqdm(f, desc=f"Processando {arquivo.name}"):
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

with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida_reduzida, f, indent=2, ensure_ascii=False)

print(f"\n💾 Resultados salvos em: {CAMINHO_SAIDA}")