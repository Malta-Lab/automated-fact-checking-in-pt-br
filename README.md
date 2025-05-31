# Dataset Translator Ollama

Este projeto automatiza a tradução de datasets JSON do inglês para o português utilizando modelos de linguagem via API do Ollama.

## Estrutura do Projeto

```
run.py
dataset/
    dev.json
    train.json
results/
    deepseek-r1-8b/
```

- `run.py`: Script principal para tradução dos arquivos.
- `dataset/`: Contém os arquivos JSON originais a serem traduzidos.
- `results/`: Onde os arquivos traduzidos são salvos, organizados por modelo.

## Pré-requisitos

- Python 3.7+
- [Ollama](https://ollama.com/) rodando localmente na porta 11434 com os modelos desejados já baixados.
- Instalar dependências Python:
  ```sh
  pip install requests tqdm
  ```

## Como usar

1. Coloque os arquivos JSON a serem traduzidos na pasta `dataset/`.
2. Certifique-se de que o Ollama está rodando e os modelos listados em `MODELS` no [`run.py`](run.py) estão disponíveis.
3. Execute o script:

   ```sh
   python run.py
   ```

4. Os arquivos traduzidos serão salvos em subpastas dentro de `results/`, uma para cada modelo.

## Configuração

- Para adicionar/remover modelos, edite a lista `MODELS` no [`run.py`](run.py).
- Para alterar o diretório dos datasets ou resultados, modifique as variáveis `DATASET_DIR` e `RESULTS_DIR` no [`run.py`](run.py).

## Observações

- O script faz uma chamada de "aquecimento" para cada modelo antes de iniciar as traduções.
- Traduções são feitas linha a linha, e o progresso é exibido no terminal.

---

Desenvolvido para facilitar a tradução de datasets para tarefas de NLP em português.