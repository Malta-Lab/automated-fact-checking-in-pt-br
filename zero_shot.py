import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
PASTA_ENTRADA = "./" 
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "classificados")
HOST = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
# ====================================

os.makedirs(PASTA_SAIDA, exist_ok=True)

def classificar_frase(model: str, frase: str) -> str:
    prompt = (
        f"Avalie a seguinte frase em português. Ela é verdadeira ou falsa?\n\n"
        f"Frase: \"{frase}\"\n\n"
        f"Responda apenas com VERDADEIRO ou FALSO."
    )
    try:
        response = requests.post(
            HOST,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        resultado = response.json().get("response", "").strip().upper()
        if "VERDADEIRO" in resultado:
            return "VERDADEIRO"
        elif "FALSO" in resultado:
            return "FALSO"
        else:
            return "INDEFINIDO"
    except Exception as e:
        print(f"❌ Erro ao processar frase: {frase}\n{e}")
        return "ERRO"

def processar_arquivo_json(caminho_arquivo: Path):
    nome_base = caminho_arquivo.stem
    saida_path = caminho_arquivo.with_name(caminho_arquivo.stem + "_classificado.json")
    saida_path.parent.mkdir(parents=True, exist_ok=True)

    frases_processadas = {}
    if saida_path.exists():
        with open(saida_path, "r", encoding="utf-8") as f:
            for linha in json.load(f):
                frases_processadas[linha["frase"]] = linha["classificacao"]

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)

    if isinstance(dados, dict):
        frases_pt = list(dados.values())
    else:
        frases_pt = dados

    resultados = []
    for frase in tqdm(frases_pt, desc=f"Processando {caminho_arquivo.name}"):
        if frase in frases_processadas:
            classificacao = frases_processadas[frase]
        else:
            classificacao = classificar_frase(MODEL, frase)
        resultados.append({
            "frase": frase,
            "classificacao": classificacao
        })

        with open(saida_path, "w", encoding="utf-8") as out:
            json.dump(resultados, out, ensure_ascii=False, indent=2)

def main():
    arquivos_json = [f for f in Path(PASTA_ENTRADA).glob("*.json") if "cache" in f.name.lower()]
    print(f"🔍 Encontrados {len(arquivos_json)} arquivos com 'cache' no nome.")
    for arquivo in arquivos_json:
        processar_arquivo_json(arquivo)

if __name__ == "__main__":
    main()
