import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_config(config_path='analysis_config.json'):
    """Carrega as configurações do arquivo JSON."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração '{config_path}' não encontrado.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Erro: Arquivo de configuração '{config_path}' está mal formatado.")
        exit(1)

def load_confusion_matrices(base_path, num_folds):
    """Carrega as matrizes de confusão de todos os folds."""
    confusion_matrices = []
    
    for fold in range(1, num_folds + 1):
        try:
            result_file = os.path.join(base_path, f"fold_{fold}", "training_results.json")
            if not os.path.exists(result_file):
                print(f"Aviso: Arquivo não encontrado: {result_file}")
                continue
                
            with open(result_file, 'r') as f:
                data = json.load(f)
                if "confusion_matrix" in data:
                    cm = np.array(data["confusion_matrix"])
                    confusion_matrices.append(cm)
        except Exception as e:
            print(f"Erro ao processar fold {fold}: {e}")
    
    return confusion_matrices

def plot_confusion_matrix(cm, output_path, title, class_names, config):
    """Plota e salva a matriz de confusão."""
    plt.figure(figsize=config['plot_settings']['figsize'])
    
    # Verifica se é necessário normalizar
    if 'normalize' in title.lower():
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plota a matriz
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=config['plot_settings']['cmap'],
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(title)
    
    # Cria o diretório de saída se não existir
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva nos formatos especificados
    for ext in config['plot_settings']['format']:
        plt.savefig(
            output_path.replace('.png', f'.{ext}'), 
            bbox_inches='tight',
            dpi=config['plot_settings'].get('dpi', 100)
        )
    
    plt.close()

def main():
    # Carrega as configurações
    config = load_config()
    
    # Carrega as matrizes de confusão
    confusion_matrices = load_confusion_matrices(
        config['base_path'], 
        config['num_folds']
    )
    
    if not confusion_matrices:
        print("Nenhuma matriz de confusão encontrada.")
        exit(1)
    
    # Calcula a matriz de confusão média
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    avg_confusion_matrix = np.round(avg_confusion_matrix).astype(int)
    
    # Exibe a matriz de confusão média
    print("Matriz de Confusão Média (valores arredondados para inteiros):")
    print(avg_confusion_matrix)
    
    # Gera os caminhos de saída
    output_dir = config['output_dir']
    avg_cm_path = os.path.join(output_dir, 'average_confusion_matrix.png')
    norm_cm_path = os.path.join(output_dir, 'normalized_average_confusion_matrix.png')
    
    # Plota a matriz de confusão
    plot_confusion_matrix(
        avg_confusion_matrix,
        avg_cm_path,
        'Matriz de Confusão Média',
        config['class_names'],
        config
    )
    
    print(f"\nMatriz de confusão salva em {avg_cm_path}")
    
    # Plota a matriz de confusão normalizada
    plot_confusion_matrix(
        avg_confusion_matrix,
        norm_cm_path,
        'Matriz de Confusão Média Normalizada',
        config['class_names'],
        {**config, 'normalize': True}
    )
    
    print(f"Matriz de confusão normalizada salva em {norm_cm_path}")

if __name__ == "__main__":
    main()
