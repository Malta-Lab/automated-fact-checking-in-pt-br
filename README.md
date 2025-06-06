# Dataset Translator Ollama

Este projeto automatiza a tradução e classificação binária de datasets de fact-checking para o português, utilizando modelos LLM via [Ollama](https://ollama.com/). Ele suporta múltiplos datasets (Averitec, Fever, Feverous, Liar RAW) e realiza a classificação zero-shot das frases traduzidas.

## Estrutura do Projeto

```
.
├── averitec/
│   └── dataset/    <=== Insira os datasets aqui
│   └── results/    <=== As traduções serão salvas aqui
│   └── run.py
├── fever/
│   └── dataset/
│   └── results/
│   └── run.py
├── feverous/
│   └── dataset/
│   └── results/
│   └── run.py
├── liar-raw/
│   └── dataset/
│   └── results/
│   └── run.py
├── zero_shot.py
├── main.ipynb
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.8+
- [Ollama](https://ollama.com/) rodando localmente (`localhost:11434`)
- Pacotes Python: `requests`, `tqdm`, `notebook`
``

## Como Usar

1. **Prepare os datasets**  
   Coloque os arquivos dos datasets originais (link abaixo) nas pastas `dataset/` dentro de cada subdiretório (`averitec`, `fever`, etc).

2. **Traduza os datasets**  
   Execute os scripts de cada dataset individualmente ou rode todos via o notebook `main.ipynb`.

3. **Classificação Zero-Shot**  
   Após a tradução, execute `zero_shot.py`:

   Os resultados classificados serão salvos na pasta `classificados/`.

## Detalhes dos Scripts

- Cada `run.py` traduz os campos relevantes dos datasets usando modelos definidos na variável `MODELS`.
- O script [`zero_shot.py`](zero_shot.py) faz a classificação binária (VERDADEIRO/FALSO) das frases traduzidas.
- O progresso, erros e cache de traduções são salvos automaticamente para retomar execuções interrompidas.


## Datasets
- AVeriTeC: https://fever.ai/dataset/averitec.html
- FEVER: https://fever.ai/dataset/fever.html
- FEVEROUS: https://fever.ai/dataset/feverous.html
- LIAR-RAW: https://www.kaggle.com/datasets/harry0paul/liar-raw?select=test.json


## Observações

- Os arquivos `.json` e `.jsonl` são ignorados pelo Git (veja [.gitignore](.gitignore)).
- Certifique-se de que o Ollama está rodando e os modelos necessários estão baixados.

