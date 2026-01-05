# InGesture: Gesture Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema flexível e extensível para reconhecimento de gestos usando dados de IMU de pulso. Este projeto suporta modelos de deep learning e machine learning clássico para classificação de gestos.

## 📋 Visão Geral

Este sistema processa dados de acelerômetro e giroscópio para detectar e classificar vários gestos de mão. Ele fornece um pipeline completo, desde o pré-processamento de dados e extração de características até o treinamento do modelo e análise de resultados. A estrutura foi projetada para ser facilmente configurável para diferentes modelos, conjuntos de dados e tipos de experimento.

## 📂 Estrutura do Projeto

Uma breve visão geral dos principais arquivos e diretórios deste projeto:

```
├── src/                     # Código fonte
│   ├── architectures/      # Arquiteturas de modelos de deep learning (ex: CNN, LSTM)
│   ├── experiments/        # (Gerado) Armazena todos os resultados, modelos e logs de experimentos
│   ├── config.py           # ★ Arquivo de configuração principal para experimentos
│   ├── train.py            # ★ Script principal para iniciar um experimento de treinamento
│   ├── datasets.py         # Lógica de carregamento e processamento inicial de dados
│   ├── transforms.py       # Funções de transformação de dados (normalização, janelamento)
│   ├── model_trainer.py    # Lógica de treinamento e avaliação para modelos de deep learning
│   ├── classic_model_trainer.py # Lógica de treinamento e avaliação para modelos de ML clássicos
│   ├── feature_extractor.py # Extrai características estatísticas para modelos clássicos
│   ├── training_utils.py   # Lógica central de validação cruzada leave-one-subject-out
│   ├── analyze_*.py        # Scripts para analisar os resultados dos experimentos
│   └── utils.py            # Funções utilitárias (ex: salvar/carregar progresso)
├── .gitignore
└── README.md
```

## 🛠️ Como Usar

Siga estes passos para configurar e executar seus próprios experimentos.

### 1. Configuração (`src/config.py`)

Este é o arquivo mais importante para configurar um experimento. Todos os parâmetros são controlados a partir daqui.

#### Parâmetros Chave para Configurar:

*   **`DATASET_PATH`**: Caminho absoluto para o diretório do seu conjunto de dados (contendo arquivos `.csv` ou `.pkl`).
*   **`EXPERIMENT_TYPE`**: `'mc'` para classificação multiclasse ou `'bin'` para classificação binária.
*   **`MODEL_TYPE`**:
    *   `'dl'`: Para modelos de deep learning.
    *   `'classic'`: Para modelos de machine learning clássicos (ex: RandomForest, SVM).
*   **`MODEL_NAME`** (para `dl`): Selecione uma arquitetura de deep learning de `src/architectures/`.
    *   Exemplo: `'ignatov_cnn'`, `'laura_cnn'`, `'msconv1d'`.
*   **`CLASSIC_MODEL_NAME`** (para `classic`): Selecione um modelo clássico.
    *   Exemplo: `'RandomForest'`, `'SVM'`, `'KNN'`.
*   **Processamento de Dados (dicionário `config`)**:
    *   `sampling_rate`: A taxa de amostragem dos seus dados (ex: `50` Hz).
    *   `window_size_seconds`: A duração de cada janela de dados em segundos.
    *   `overlap_fraction`: A porcentagem de sobreposição entre janelas consecutivas (ex: `0.5` para 50%).
*   **Hiperparâmetros de Deep Learning (dicionário `config`)**:
    *   `num_epochs`, `batch_size`, `optimizer`, `learning_rate`, etc.
*   **Hiperparâmetros de Modelos Clássicos (dicionário `classic_model_config`)**:
    *   Configure os parâmetros para cada modelo clássico (ex: `n_estimators` para RandomForest).

### 2. Executando um Experimento

Uma vez que `src/config.py` esteja configurado, você pode iniciar o processo de treinamento.

#### Iniciando um Novo Experimento

1.  Em `src/config.py`, certifique-se de que `MANUAL_EXPERIMENT_NAME` está configurado para ser gerado automaticamente.
    ```python
    # Descomente a linha abaixo para gerar um novo nome de experimento
    MANUAL_EXPERIMENT_NAME = get_experiment_name() 
    # Comente a linha abaixo para não usar um nome fixo
    # MANUAL_EXPERIMENT_NAME = "NomeDoExperimentoAnterior"
    ```
2.  Execute o script de treinamento no terminal a partir do diretório raiz do projeto:
    ```bash
    python src/train.py
    ```
    Um novo diretório de experimento será criado em `experiments/` com um nome baseado na sua configuração (ex: `RandomForest_mc_2026-01-04_16-30-00`).

#### Continuando um Experimento

Se um experimento foi interrompido, você pode continuá-lo do ponto onde parou.

1.  Em `src/config.py`, comente a geração automática de nome e defina `MANUAL_EXPERIMENT_NAME` com o nome exato da pasta do experimento que você deseja continuar.
    ```python
    # Comente a linha abaixo para não gerar um novo nome
    # MANUAL_EXPERIMENT_NAME = get_experiment_name()
    # Descomente e defina o nome do experimento para continuar
    MANUAL_EXPERIMENT_NAME = "RandomForest_mc_2026-01-04_16-30-00"
    ```
2.  Execute o script de treinamento novamente:
    ```bash
    python src/train.py
    ```
    O script irá carregar o progresso salvo e continuar o treinamento a partir do último fold não concluído.

### 3. Analisando os Resultados

Após a conclusão de um experimento, você pode usar os scripts de análise para avaliar o desempenho do modelo.

1.  **Importante**: Para que os scripts de análise funcionem, `MANUAL_EXPERIMENT_NAME` em `src/config.py` **deve** estar definido com o nome do experimento que você deseja analisar (o mesmo passo para continuar um experimento).

2.  Execute os scripts de análise:
    *   **`analyze_results.py`**: Gera uma matriz de confusão média de todos os folds e calcula métricas de desempenho agregadas.
        ```bash
        python src/analyze_results.py
        ```
    *   **`analyze_fold_perfomance.py`**: Plota gráficos de acurácia e perda por época para cada fold (apenas para modelos de deep learning).
        ```bash
        python src/analyze_fold_perfomance.py
        ```
    *   **`analyze_confusion_matrices.py`**: Exibe a matriz de confusão para cada fold individualmente.
        ```bash
        python src/analyze_confusion_matrices.py
        ```
    Os resultados e gráficos gerados serão salvos dentro da pasta do respectivo experimento.

## 📚 Citação

Se você usar este código em sua pesquisa, por favor, cite o dataset original:

```
Gohl, Pedro Daniel; Spellen, Amanda Nicole; Queiroz, Laura Isabelle; Souto, Eduardo James (2025), 
"InGesture Dataset", Mendeley Data, V3, doi: 10.17632/fdxst56tcj.3
```

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
