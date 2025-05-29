# Plotar os sinais com rótulos reais e previstos
def plot_signals_with_labels_separately(time, signal, true_labels, pred_labels, class_names, title='Sinais Inerciais e Rótulos'):
    plt.figure(figsize=(15, 8))
    
    # Subplot 1: Sinal inercial
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Sinal Inercial')
    plt.title('Sinal Inercial')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Subplot 2: Rótulos verdadeiros
    plt.subplot(3, 1, 2)
    plt.step(time, true_labels, where='post', label='Rótulos Verdadeiros')
    plt.title('Rótulos Verdadeiros')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Classe')
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.legend()
    
    # Subplot 3: Rótulos previstos
    plt.subplot(3, 1, 3)
    plt.step(time, pred_labels, where='post', label='Rótulos Previstos', linestyle='--')
    plt.title('Rótulos Previstos')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Classe')
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#working okay with 0.25 window stride
def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    
#working okay with 0.25 window stride
def reconstruct_labels_over_time(y_pred, test_x_length, window_size, overlap_size):
    y_pred_time = np.full(test_x_length, -1)
    start = 0
    idx = 0
    iteration = 0
    step = window_size - overlap_size
    while start < test_x_length and idx < len(y_pred):
        end = start + window_size
        if end > test_x_length:
            end = test_x_length
        y_pred_time[start:end] = y_pred[idx]
        print(f'Iteração {iteration}: idx={idx}, start={start}, end={end}, y_pred[idx]={y_pred[idx]}')
        start += step
        idx += 1
        iteration += 1
    print(f'Total de iterações: {iteration}')
    return y_pred_time