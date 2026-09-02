# treecover-cd

Projeto de disciplina para detecção de mudança de cobertura arbórea a partir de séries temporais SAR do Sentinel-1, sobre uma área de estudo no sul da Amazônia (~6,8–7,4°S).

## Sobre o projeto

O modelo prevê, por pixel, um **mapa de recência de mudança**: um valor contínuo normalizado entre 0 e 1 indicando o quão recentemente a cobertura arbórea foi perdida dentro da janela de estudo (2018–2024). A detecção binária de mudança/não-mudança é uma consequência de aplicar um limiar sobre essa saída, não o alvo de treinamento, por isso o modelo é um regressor (saída com `Sigmoid`), não um classificador.

A entrada é uma sequência de 84 composições mensais VV/VH do Sentinel-1 (órbita descendente), e a arquitetura é uma rede convolucional recorrente que processa essa sequência inteira em um único estado oculto antes de gerar o mapa de saída. Duas variantes foram treinadas e comparadas:

- **ConvGRU**: célula com 3 portões (update/reset/candidate).
- **ConvLSTM**: célula com 4 portões (input/forget/output/candidate) e estado de célula separado.

Um segundo dataset, geograficamente distinto da área de treino, é usado como conjunto de teste *held-out* para verificar generalização.

## Estrutura do repositório

```
src/
  data.py        # DataPipeline: download (Kaggle) → pré-processamento → alinhamento de rótulos
  dataset.py     # SARDataset: tiling em patches + normalização z-score
  models.py      # ConvGRURegressor / ConvLSTMRegressor
  train.py       # loop de treinamento (masked MSE, checkpointing, early stopping)
  inference.py   # carregamento de pesos, inferência full-extent, pós-processamento, métricas
  utils.py       # utilitários (leitura de histórico de treino)
main.ipynb       # ponto de entrada único: treino, avaliação, análise e visualização
data/labels/     # rótulos (rastreados no git)
checkpoints/     # pesos treinados, um .pth por combinação de hiperparâmetros (rastreados)
logs/            # histórico de treino por execução, em .parquet (rastreados)
best_model.pth   # cópia dos pesos do melhor modelo (menor MAE no teste), na raiz
```

`data/raw*/`, `data/preprocessed*/`, `data/aggregated/` e `results/` são gerados pelo pipeline e não são versionados, veja abaixo como recriá-los.

## Como rodar

### Pré-requisitos

- Python ≥ 3.12 e [uv](https://docs.astral.sh/uv/) instalados.
- GPU com CUDA é fortemente recomendada, o treinamento completo (16 combinações × 50 épocas) leva ~16h numa T4. É possível rodar em CPU, mas o tempo aumenta consideravelmente.
- Uma conta no Kaggle com um token de API, para baixar os dados brutos (ver abaixo).

### Instalação

```bash
uv sync
```

Em ambientes sem `venv` local (Colab, Kaggle), instale direto contra o `pyproject.toml`:

```bash
uv pip install --system .
```

### Acesso aos dados (Kaggle)

Os rasters brutos (composições mensais VV/VH, já exportadas do Earth Engine) estão hospedados no dataset [`jotasaraiva/treecover-cd-data`](https://www.kaggle.com/datasets/jotasaraiva/treecover-cd-data) e são baixados automaticamente pelo pipeline via `kagglehub`, não é necessário autenticar no Earth Engine para rodar o projeto. É preciso apenas de um token de API do Kaggle:

1. Gere um token em [kaggle.com/settings](https://www.kaggle.com/settings) → *API* → *Create New Token* (baixa um `kaggle.json`).
2. Coloque o arquivo em `~/.kaggle/kaggle.json`, **ou** defina as variáveis de ambiente `KAGGLE_USERNAME`/`KAGGLE_KEY`, **ou** rode `kagglehub.login()` uma vez em Python para configurar interativamente.

O restante do pipeline (reprojeção, normalização, alinhamento de rótulos) roda localmente e não depende de nenhum serviço externo.
