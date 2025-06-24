import json
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import numpy as np

# ============================ CONFIGURAÇÕES ============================

ARQUIVO_CLASSIFICADO = Path(sys.argv[1]) 

ARQUIVOS_ORIGINAIS = [
    Path("./dataset/train.jsonl"),
    Path("./dataset/valid.jsonl"),
    Path("./dataset/test.jsonl")
]

CLASSES_ORDINAIS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]

TRADUCAO_INVERTIDA = {
    "Mentira Descarada": "pants-fire",
    "Quase Verdade": "barely-true", # "Quase Falso": "barely-true" 
    "Falso": "false",
    "Parcialmente Verdadeiro": "half-true",
    "Majoritariamente Verdadeiro": "mostly-true",
    "Verdadeiro": "true",
        
}

# =======================================================================

def traduzir_rotulo_pt_para_en(rotulo):
    if not isinstance(rotulo, str):
        return rotulo
    return TRADUCAO_INVERTIDA.get(rotulo.strip().title(), rotulo.strip())

def carregar_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]

def carregar_gold(lista_caminhos):
    dados = {}
    for caminho in lista_caminhos:
        with open(caminho, "r", encoding="utf-8") as f:
            if caminho.suffix == ".jsonl":
                for linha in f:
                    item = json.loads(linha)
                    chave = item.get("statement") or item.get("claim")
                    if chave:
                        dados[chave] = item["label"]
            else:
                lista = json.load(f)
                for item in lista:
                    chave = item.get("claim") or item.get("statement")
                    if chave:
                        dados[chave] = item["label"]
    return dados

def carregar_pred(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
        return {item.get("claim") or item.get("statement"): item["label"] for item in dados}

def imprimir_matriz_confusao(y_true, y_pred, categorias):
    matriz = confusion_matrix(y_true, y_pred, labels=categorias)
    print("\n📊 Matriz de Confusão (label verdadeiro × classificação do modelo):\n")
    cabecalho = f"{'':35}" + "".join(f"{c[:20]:>25}" for c in categorias)
    print(cabecalho)
    for i, linha in enumerate(matriz):
        nome = categorias[i]
        linha_str = f"{nome[:33]:35}" + "".join(f"{n:>25}" for n in linha)
        print(linha_str)

def imprimir_relatorio(y_true, y_pred, categorias):
    print(f"\n📊 Avaliação geral")
    print(f"Claims comparadas: {len(y_true)}")
    acertos = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    print(f"Acertos exatos: {acertos} ({acertos / len(y_true):.1%})\n")
    imprimir_matriz_confusao(y_true, y_pred, categorias)
    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=categorias, digits=3, zero_division=0))

def calcular_erro_ordinal(y_true, y_pred, classes_ordenadas):
    """
    Compute absolute ordinal errors and return:
        - errors: list of absolute differences (including zeros)
        - y_true_ord: list of true ordinal indices
        - y_pred_ord: list of predicted ordinal indices
        - mode_error: mode of errors only over misclassified instances (errors > 0), or None
    """
    erros = []
    y_real_ord, y_pred_ord = [], []
    ord_map = {lbl: i for i, lbl in enumerate(classes_ordenadas)}
    rotulos_invalidos = {}

    for real, pred in zip(y_true, y_pred):
        pred_en = traduzir_rotulo_pt_para_en(pred)
        try:
            idx_real = ord_map[real]
            idx_pred = ord_map[pred_en]
            diff = abs(idx_real - idx_pred)
            erros.append(diff)
            y_real_ord.append(idx_real)
            y_pred_ord.append(idx_pred)
        except KeyError:
            rotulos_invalidos.setdefault(pred_en, 0)
            rotulos_invalidos[pred_en] += 1

    if rotulos_invalidos:
        print("⚠️ Rótulos inválidos encontrados:")
        for rotulo, count in rotulos_invalidos.items():
            print(f"  → '{rotulo}': {count} ocorrência(s)")

    # Compute mode only over misclassified instances
    misclassified_errors = [e for e in erros if e != 0]
    if misclassified_errors:
        mode_error = Counter(misclassified_errors).most_common(1)[0][0]
    else:
        mode_error = None

    return erros, y_real_ord, y_pred_ord, mode_error

# def calcular_erro_ordinal(y_true, y_pred, classes_ordenadas):
#     erros = []
#     y_real_ord, y_pred_ord = [], []
#     ord_map = {lbl: i for i, lbl in enumerate(classes_ordenadas)}
#     rotulos_invalidos = {}

#     for real, pred in zip(y_true, y_pred):
#         pred_en = traduzir_rotulo_pt_para_en(pred)
#         try:
#             idx_real = ord_map[real]
#             idx_pred = ord_map[pred_en]
#             erros.append(abs(idx_real - idx_pred))
#             y_real_ord.append(idx_real)
#             y_pred_ord.append(idx_pred)
#         except KeyError:
#             rotulos_invalidos.setdefault(pred_en, 0)
#             rotulos_invalidos[pred_en] += 1

#     if rotulos_invalidos:
#         print("⚠️ Rótulos inválidos encontrados:")
#         for rotulo, count in rotulos_invalidos.items():
#             print(f"  → '{rotulo}': {count} ocorrência(s)")
#     return erros, y_real_ord, y_pred_ord

# ============================ EXECUÇÃO PRINCIPAL ============================

if __name__ == "__main__":
    gold_data = carregar_gold(ARQUIVOS_ORIGINAIS)
    pred_data = carregar_pred(ARQUIVO_CLASSIFICADO)

    # Emparelhar
    y_true, y_pred = [], []
    claims_faltantes = 0
    for claim, label_real in gold_data.items():
        label_pred_pt = pred_data.get(claim)
        if not label_pred_pt:
            claims_faltantes += 1
            continue
        label_pred_en = traduzir_rotulo_pt_para_en(label_pred_pt)
        y_true.append(label_real)
        y_pred.append(label_pred_en)

    if claims_faltantes:
        print(f"⚠️ Claims não encontrados nas predições: {claims_faltantes}")

    # Relatório padrão
    imprimir_relatorio(y_true, y_pred, CLASSES_ORDINAIS)

    # Erro ordinal
    erros_ord, y_real_ord, y_pred_ord, mode_err = calcular_erro_ordinal(y_true, y_pred, CLASSES_ORDINAIS)
    if erros_ord:
        erros_pos = [e for e in erros_ord if e > 0]
        moda = np.bincount(erros_pos).argmax() if erros_pos else "Indefinida"
        media = np.mean(erros_pos) if erros_pos else 0
        mediana = np.median(erros_pos) if erros_pos else 0
        qwk = cohen_kappa_score(y_real_ord, y_pred_ord, weights="quadratic")

        print("\n📐 Erros Ordinais:")
        print(f"Erro médio: {round(media, 3)}")
        print(f"Erro mediano: {round(mediana, 3)}")
        print(f"Erro moda: {moda}")
        print(f"QWK (Cohen Kappa Quadrático): {round(qwk, 4)}")
        print(f"Total de exemplos avaliados: {len(erros_ord)}")
