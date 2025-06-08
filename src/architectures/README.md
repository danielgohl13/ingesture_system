# Guia de Implementação de Arquiteturas de Modelos

Este guia descreve como implementar e configurar arquiteturas de modelos para o pipeline de reconhecimento de gestos alimentares. O sistema atual utiliza uma abordagem baseada em aprendizado profundo para classificação de gestos de alimentação a partir de dados de acelerômetro triaxial.

## Estrutura Básica

Cada arquitetura deve ser implementada em um arquivo Python separado na pasta `architectures`. O arquivo deve conter uma função principal chamada `create_model` que segue a assinatura abaixo:

```python
def create_model(input_shape: tuple, num_classes: int, config: dict = None) -> tf.keras.Model:
    """
    Cria e compila o modelo.
    
    Args:
        input_shape (tuple): Formato dos dados de entrada (window_size, n_features)
        num_classes (int): Número de classes de saída
        config (dict, optional): Dicionário de configuração contendo parâmetros do modelo
        
    Returns:
        tf.keras.Model: Modelo compilado
    """
    # Implementação do modelo
    pass
```

## Arquitetura Laura CNN

A arquitetura atual implementa uma rede neural convolucional 1D otimizada para reconhecimento de gestos alimentares. A arquitetura foi projetada para capturar padrões temporais em séries temporais de sensores inerciais.

### Características Principais:
- **Camadas Convolucionais**: 3 camadas convolucionais 1D
- **Filtros**: 16 filtros por camada
- **Tamanho do Kernel**: 25 amostras
- **Pooling**: MaxPooling com stride 1 e tamanho 2
- **Dropout**: 0.5 para regularização
- **Função de Ativação**: ReLU nas camadas ocultas, softmax na saída

### Implementação:

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_model(input_shape, num_classes, config=None):
    """
    Cria e compila o modelo Laura CNN para classificação de gestos alimentares.
    
    Args:
        input_shape (tuple): Formato dos dados de entrada (window_size, n_features)
        num_classes (int): Número de classes de saída
        config (dict, optional): Dicionário de configuração com parâmetros do modelo
        
    Returns:
        tf.keras.Model: Modelo compilado
    """
    # Configurações padrão
    config = config or {}
    filters = config.get('filters', 16)
    kernel_size = config.get('kernel_size', 25)
    dropout_rate = config.get('dropout_rate', 0.5)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Entrada
    inputs = layers.Input(shape=input_shape)
    
    # Bloco Convolucional 1
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    
    # Bloco Convolucional 2
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    
    # Bloco Convolucional 3
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Camadas Densas
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Saída
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Criar e compilar o modelo
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )
    
    return model
```

## Arquitetura MS-Conv1D

A arquitetura MS-Conv1D implementa uma rede neural convolucional 1D multi-escala projetada para capturar padrões temporais em diferentes escalas em séries temporais de sensores inerciais.

### Características Principais:
- **Camadas Convolucionais Múltiplas**: Utiliza múltiplos caminhos com diferentes tamanhos de kernel (3, 5, 7) para capturar padrões em diferentes escalas temporais
- **Batch Normalization**: Normalização em lote após cada camada convolucional para melhor estabilidade do treinamento
- **Pooling Estratégico**: Redução de dimensionalidade progressiva para capturar características hierárquicas
- **Concatenação**: Combinação das saídas dos diferentes caminhos para enriquecer a representação de características

### Implementação:

```python
import tensorflow as tf

def build_multiscale_conv1d(input_shape, num_classes):
    """Constrói o bloco principal da MS-Conv1D."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Convolução e Pooling Iniciais
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bloco Convolucional Multi-Escala
    branch1 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    branch3 = tf.keras.layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.concatenate([branch1, branch2, branch3], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Camada Convolucional Final
    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Camada de Saída
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_model(input_shape, num_classes, config=None):
    """Cria e compila o modelo MS-Conv1D."""
    if config is None:
        config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'use_learning_rate_scheduler': True
        }
    
    # Construir o modelo
    model = build_multiscale_conv1d(input_shape, num_classes)
    
    # Configurar otimizador
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = float(config.get('learning_rate', 0.001))
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:  # rmsprop ou outro
        optimizer = tf.keras.optimizers.get(optimizer_name)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = learning_rate
    
    # Configurar função de perda e métricas
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    if num_classes == 2:
        metrics.append(tf.keras.metrics.AUC(name='auc'))
    
    # Compilar o modelo
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model
```

## Requisitos Obrigatórios

1. **Função `create_model`**: 
   - Deve aceitar `input_shape`, `num_classes` e `config` como parâmetros
   - Deve retornar um modelo Keras compilado
   - Deve ser compatível com o formato de entrada/saída especificado

2. **Formato de Entrada**:
   - Formato: `(batch_size, window_size, n_features)`
   - `window_size`: Definido por `config["window_size"]` (padrão: 200 amostras)
   - `n_features`: Número de canais do sensor (tipicamente 6: 3 eixos do acelerômetro + 3 eixos do giroscópio)

3. **Formato de Saída**:
   - Formato: `(batch_size, num_classes)`
   - Ativação: 'softmax' para classificação multiclasse
   - Número de neurônios: Igual a `num_classes`

4. **Compilação**:
   - Otimizador: Configurável via `config['optimizer']` (padrão: 'adam')
   - Taxa de aprendizado: Configurável via `config['learning_rate']` (padrão: 0.01)
   - Loss: 'sparse_categorical_crossentropy' (para rótulos inteiros)
   - Métricas: ['accuracy', 'sparse_categorical_accuracy']

5. **Callbacks**:
   - `EarlyStopping`: Monitora 'val_loss' com paciência configurável
   - `ModelCheckpoint`: Salva o melhor modelo baseado em 'val_loss'
   - Suporte a métricas personalizadas (ex: F1-Score)

## Boas Práticas e Configurações Recomendadas

1. **Configurações de Treinamento** (em `config.py`):
   ```python
   {
       # Configurações de dados
       "sampling_rate": 50,  # Hz
       "window_size_seconds": 4,  # segundos
       "overlap_fraction": 0.5,  # 50% de sobreposição
       
       # Configurações de treinamento
       "num_epochs": 100,
       "batch_size": 16,
       "patience": 12,  # épocas para early stopping
       "optimizer": "adam",
       "learning_rate": 0.01,
       "use_learning_rate_scheduler": True
   }
   ```

2. **Pré-processamento**:
   - Amostragem para 50Hz
   - Normalização por amostra (zero mean, unit variance)
   - Janelamento com 50% de sobreposição

3. **Regularização**:
   - Dropout (0.5) nas camadas densas
   - Early Stopping baseado em val_loss
   - Batch Normalization quando aplicável

4. **Monitoramento**:
   - Métricas salvas em tempo real durante o treinamento
   - Gráficos de loss e acurácia (treino/validação)
   - Matriz de confusão e relatórios de classificação

5. **Salvamento de Modelos**:
   - Melhor modelo baseado em val_loss (formato SavedModel)
   - Pesos do modelo (formato HDF5)
   - Configuração completa do treinamento

6. **Documentação**:
   - Inclua docstrings detalhadas
   - Documente os hiperparâmetros e suas funções
   - Mantenha um registro de alterações e experimentos

## Fluxo de Trabalho de Desenvolvimento

### 1. Criando uma Nova Arquitetura

1. Crie um novo arquivo em `src/architectures/` (ex: `minha_arquitetura.py`)
2. Implemente a função `create_model` seguindo o padrão:
   ```python
   def create_model(input_shape, num_classes, config=None):
       # Sua implementação aqui
       pass
   ```

### 2. Registrando a Arquitetura

1. Adicione a nova arquitetura em `model_configs.py`:
   ```python
   MODEL_CONFIGS = {
       # ... outras configurações ...
       'minha_arquitetura': {
           'create_fn': create_minha_arquitetura,
           'name': 'Minha Arquitetura',
           'description': 'Descrição da nova arquitetura'
       }
   }
   ```

### 3. Configurando o Treinamento

1. Em `config.py`, defina:
   ```python
   MODEL_NAME = 'minha_arquitetura'
   ```

2. Ajuste os parâmetros de treinamento conforme necessário

### 4. Executando o Treinamento

```bash
python src/train.py
```

### 5. Analisando Resultados

Os resultados serão salvos em:
```
experiments/
└── [nome_do_experimento]/
    ├── models/
    │   └── fold_[n]/
    │       ├── saved_model/
    │       └── best_weights.h5
    └── results/
        └── fold_[n]/
            ├── classification_report.json
            ├── confusion_matrix.png
            ├── training_history.png
            └── metrics_summary.json
```

## Modelos Implementados

### 1. Laura CNN
- **Descrição**: CNN 1D otimizada para reconhecimento de gestos alimentares
- **Arquitetura**: 3 camadas convolucionais + camadas densas
- **Características**:
  - 16 filtros por camada convolucional
  - Kernel size de 25 amostras
  - Dropout de 0.5 para regularização
  - Global Average Pooling antes das camadas densas

### 2. MSCONV1d
- **Descrição**: Implementação baseada em múltiplas escalas de convolução 1D
- **Uso**: Comparação de desempenho

### 3. CNN-LSTM
- **Descrição**: Combinação de camadas convolucionais e LSTM
- **Uso**: Análise de dependências temporais de longo prazo

## Monitoramento e Análise

### Métricas Salvas
- Acurácia e Loss (treino/validação)
- F1-Score (macro, micro, weighted)
- Matriz de confusão
- Relatório de classificação detalhado

### Visualizações Geradas
1. **Histórico de Treinamento**:
   - Acurácia por época
   - Loss por época
   - F1-Score por época

2. **Avaliação**:
   - Matriz de confusão
   - Curvas ROC (para classificação binária)

## Próximos Passos e Melhorias

1. **Experimentação**:
   - Testar diferentes arquiteturas de rede
   - Ajustar hiperparâmetros
   - Experimentar técnicas de aumento de dados

2. **Otimização**:
   - Implementar busca em grade para hiperparâmetros
   - Adicionar suporte a mixed-precision training
   - Otimizar o pipeline de dados para maior desempenho

3. **Documentação**:
   - Adicionar exemplos de uso
   - Incluir guia de solução de problemas
   - Documentar padrões de código

## Suporte

Para suporte, consulte:
- Documentação do [TensorFlow/Keras](https://www.tensorflow.org/guide/keras)
- Repositório do projeto
- Equipe de desenvolvimento

## Licença
