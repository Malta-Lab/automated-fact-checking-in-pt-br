import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
PASTA_ENTRADA = Path("./")
ARQUIVO_SAIDA = PASTA_ENTRADA / "zero_shot.json"
HOST = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
# ===================================

# Carrega classificações já existentes
if ARQUIVO_SAIDA.exists():
    with open(ARQUIVO_SAIDA, "r", encoding="utf-8") as f:
        frases_classificadas = json.load(f)
else:
    frases_classificadas = {}

def classificar_frase(model: str, frase: str) -> int:
    prompt = (
        "Considere a frase a seguir escrita em português.\n"
        "Avalie se ela está de acordo com fatos do mundo real, com base no seu conhecimento.\n"
        "Responda apenas com uma das palavras: VERDADEIRO ou FALSO.\n\n"
        f"Frase: \"{frase}\"\n\n"
        "Resposta:"
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
            return 1
        elif "FALSO" in resultado:
            return 0
        else:
            return -1  # Indefinido
    except Exception as e:
        print(f"❌ Erro ao classificar frase: {frase}\n{e}")
        return -2  # Erro

def coletar_frases_alvo(arquivos):
    frases = []
    for caminho in arquivos:
        try:
            if caminho.suffix == ".jsonl":
                with open(caminho, "r", encoding="utf-8") as f:
                    linhas = [json.loads(l) for l in f]
                for item in linhas:
                    frases.extend(item.values())

            elif caminho.suffix == ".json":
                with open(caminho, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    frases.extend(data.values())

        except Exception as e:
            print(f"❌ Erro ao ler {caminho}: {e}")
    return frases

def main():
    arquivos_alvo = list(PASTA_ENTRADA.rglob("*cache*.json")) + list(PASTA_ENTRADA.rglob("*cache*.jsonl"))
    print(f"🔍 {len(arquivos_alvo)} arquivos encontrados com 'cache' no nome.")

    frases_todas = coletar_frases_alvo(arquivos_alvo)
    frases_unicas = list(set(frases_todas))
    frases_para_classificar = [f for f in frases_unicas if f not in frases_classificadas]

    print(f"📊 Total de frases únicas: {len(frases_unicas)}")
    print(f"🚀 Faltam classificar: {len(frases_para_classificar)}")

    for frase in tqdm(frases_para_classificar, desc="🔎 Classificando frases"):
        classificacao = classificar_frase(MODEL, frase)

        if classificacao in [0, 1]:  # Apenas salva se houver resultado binário claro
            frases_classificadas[frase] = classificacao
            with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as out:
                json.dump(frases_classificadas, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
