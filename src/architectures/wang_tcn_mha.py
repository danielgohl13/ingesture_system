import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def create_model(input_shape: tuple, num_classes: int, config: dict = None) -> tf.keras.Model:
    """
    Cria e compila o modelo TCN-MHA para classificação de gestos alimentares,
    baseado no artigo "Eating Speed Measurement Using Wrist-Worn IMU Sensors".

    Args:
        input_shape (tuple): Formato dos dados de entrada (window_size, n_features)
        num_classes (int): Número de classes de saída
        config (dict, optional): Dicionário de configuração contendo parâmetros do modelo.

    Returns:
        tf.keras.Model: Modelo Keras compilado.
    """
    # ------------------- Configurações do Modelo -------------------
    # Parâmetros extraídos do artigo e configuráveis.
    config = config or {}
    learning_rate = config.get('learning_rate', 0.0005) # 
    optimizer_name = config.get('optimizer', 'adam')
    
    # Parâmetros da arquitetura TCN do artigo
    tcn_layers = 9 # 
    tcn_kernels = 64 # 
    tcn_dropout = 0.3 # 
    
    # Parâmetros da arquitetura MHA (Multi-Head Attention) do artigo
    mha_heads = 8 # 
    mha_key_dim = 16 # Dimensão de cada cabeça, resultando em d_model=128 
    
    # Parâmetros da arquitetura FCN do artigo
    fcn_units = 64 # 

    # ------------------- Construção da Arquitetura -------------------
    
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # ========= 1. Módulo TCN (Temporal Convolutional Network) =========
    # O artigo descreve um TCN com camadas convolucionais dilatadas empilhadas 
    
    # Camada inicial 1x1 para ajustar a dimensionalidade para tcn_kernels
    x = layers.Conv1D(filters=tcn_kernels, kernel_size=1, padding='same')(x)

    for i in range(tcn_layers):
        residual = x
        dilation_rate = 2**i # Fator de dilação d_l = 2^(l-1) 
        
        # Bloco convolucional dilatado
        x = layers.Conv1D(filters=tcn_kernels, kernel_size=3, padding='same', 
                         dilation_rate=dilation_rate, activation='relu')(x)
        x = layers.Dropout(tcn_dropout)(x)
        
        # Conexão residual para combinar características 
        # Se a dimensionalidade do residual for diferente, ajuste com uma camada 1x1.
        # Neste caso, as dimensões são as mesmas.
        x = layers.Add()([residual, x])

    # ========= 2. Módulo MHA (Multi-Head Attention) =========
    # Para focar em características temporais representativas 
    
    # O artigo não menciona explicitamente a codificação posicional, mas é padrão
    # para MHA e está implícito na Fig. 7c. Para simplificar, pulamos para a camada MHA.
    
    residual_attention = x
    # Normalização de camada antes da atenção, uma prática comum.
    x = layers.LayerNormalization()(x)
    
    # Camada de Atenção Multi-Cabeça
    attention_output = layers.MultiHeadAttention(
        num_heads=mha_heads, 
        key_dim=mha_key_dim, 
        output_shape=tcn_kernels
    )(query=x, key=x, value=x)
    
    attention_output = layers.Dropout(tcn_dropout)(attention_output)
    
    # Conexão residual e normalização após a atenção (padrão em Transformers)
    x = layers.Add()([residual_attention, attention_output])
    x = layers.LayerNormalization()(x)

    # ========= 3. Módulo FCN (Fully Connected Network) =========
    # Reduz a dimensão temporal para obter um vetor de características
    x = layers.GlobalAveragePooling1D()(x)
    
    # Camada densa intermediária
    x = layers.Dense(fcn_units, activation='relu')(x)
    x = layers.Dropout(tcn_dropout)(x)
    
    # Camada de saída final
    # Para classificação binária, usamos 1 neurônio com ativação sigmoid
    # Para multiclasse, usamos softmax com num_classes neurônios
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # ========= 4. Criação e Compilação do Modelo =========
    model = Model(inputs=inputs, outputs=outputs)
    
    # Otimizador Adam foi utilizado no artigo 
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate.assign(learning_rate)

    # Configuração da função de perda e métricas
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model