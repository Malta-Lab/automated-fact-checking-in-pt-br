import json
import os
from tqdm import tqdm

def extrair_claims_e_classificar(arquivos_json, caminho_zero_shot, caminho_saida):
    # Carrega o zero_shot.json
    with open(caminho_zero_shot, 'r', encoding='utf-8') as f:
        zero_shot_dict = json.load(f)

    claims_classificadas = {}
    total_claims_processadas = 0
    total_claims_adicionadas = 0

    print(f"🔍 Processando {len(arquivos_json)} arquivos...\n")

    # Itera sobre os arquivos fornecidos
    for idx, caminho in enumerate(arquivos_json, 1):
        if not os.path.isfile(caminho):
            print(f"⚠️  Arquivo não encontrado: {caminho}")
            continue

        print(f"📁 [{idx}/{len(arquivos_json)}] Lendo: {caminho}")

        is_jsonl = caminho.endswith(".jsonl")
        with open(caminho, 'r', encoding='utf-8') as f:
            if is_jsonl:
                dados = [json.loads(linha) for linha in f if linha.strip()]
            else:
                dados = json.load(f)
                if isinstance(dados, dict):
                    dados = [dados]

        for item in tqdm(dados, desc=f"→ Processando claims ({os.path.basename(caminho)})", unit="claim"):
            total_claims_processadas += 1
            claim = item.get("claim")
            if claim and claim in zero_shot_dict:
                claims_classificadas[claim] = zero_shot_dict[claim]
                total_claims_adicionadas += 1

    # Salva o novo arquivo com apenas claims e classificações
    with open(caminho_saida, 'w', encoding='utf-8') as f_out:
        json.dump(claims_classificadas, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Fim do processamento!")
    print(f"📌 Claims processadas: {total_claims_processadas}")
    print(f"📌 Claims adicionadas ao arquivo final: {total_claims_adicionadas}")
    print(f"💾 Arquivo salvo em: {caminho_saida}")

# Exemplo de uso:
if __name__ == "__main__":
    arquivos_dados = [
        "./averitec/results/gemma3-4b/dev.json",
        "./averitec/results/gemma3-4b/train.json",
        "./fever/results/gemma3-4b/train.jsonl",
        "./feverous/results/gemma3-4b/feverous_dev_challenges.jsonl",
        "./feverous/results/gemma3-4b/feverous_train_challenges.jsonl",
        "./liar-raw/results/gemma3-4b/test.json",
        "./liar-raw/results/gemma3-4b/train.json",
        "./liar-raw/results/gemma3-4b/val.json",
    ]
    caminho_zero_shot = "./zero_shot.json"
    caminho_saida = "./zero_shot_only_claims.json"

    extrair_claims_e_classificar(arquivos_dados, caminho_zero_shot, caminho_saida)
