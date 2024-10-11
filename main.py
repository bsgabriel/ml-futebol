import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_fscore_support
import seaborn as sns

### FUNÇÕES DA REDE NEURAL ###

def get_data():
    columns_to_remove = [
        'Unnamed: 0',
        'HomeTeam', 
        'AwayTeam',
        'FTHG',
        'FTAG'
    ]
    
    data = pd.read_csv('futebol.csv', delimiter=';')
    data = data.drop(columns_to_remove, axis=1)
    data = pd.get_dummies(data, columns=['FTR', 'HTR'])
    data = data.astype(int)
    data = (data - data.min()) / (data.max() - data.min())
    return data

# Dividir os dados em treinamento e teste
def split_train_test(data, train_size=0.8):
    np.random.seed(42)
    indices = np.random.permutation(data.index)
    train_count = int(len(data) * train_size)
    
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

# Inicialização dos pesos
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_hidden_input = np.random.rand(input_size, hidden_size) * 0.01
    weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01
    return weights_hidden_input, weights_hidden_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def matrix_mult(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = np.sum(A[i, :] * B[:, j])
    return result

def forward_propagation(inputs, weights_hidden_input, weights_hidden_output):
    hidden_layer_input = matrix_mult(inputs, weights_hidden_input)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = matrix_mult(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output_layer_output

def backward_propagation(inputs, hidden_layer_output, output_layer_output, labels, weights_hidden_input, weights_hidden_output, learning_rate, momentum):
    
    output_error = labels - output_layer_output
    output_delta = output_layer_output * (1 - output_layer_output) * output_error
    
    hidden_error = matrix_mult(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_layer_output * (1 - hidden_layer_output) * hidden_error
    
    weights_hidden_output = weights_hidden_output * momentum + learning_rate * matrix_mult(hidden_layer_output.T, output_delta)
    weights_hidden_input = weights_hidden_input * momentum + learning_rate * matrix_mult(inputs.T, hidden_delta)
    
    return weights_hidden_input, weights_hidden_output, output_error

def train_model(features_train, labels_train, input_size, hidden_size, output_size, epochs, learning_rate, momentum):
    weights_hidden_input, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    error_list = []
    
    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(features_train, weights_hidden_input, weights_hidden_output)
        
        weights_hidden_input, weights_hidden_output, output_error = backward_propagation(
            features_train, hidden_layer_output, output_layer_output, labels_train, 
            weights_hidden_input, weights_hidden_output, learning_rate, momentum
        )

        error = np.mean(np.abs(output_error))
        error_list.append(error)

        if epoch % 10 == 0:
            error = np.mean(np.abs(labels_train - output_layer_output))
            print(f"Epoch {epoch}, Error: {error}")
    
    return weights_hidden_input, weights_hidden_output, error_list

def predict(features, weights_hidden_input, weights_hidden_output):
    _, output_layer_output = forward_propagation(features, weights_hidden_input, weights_hidden_output)
    return output_layer_output

### FUNÇÕES DE MÉTRICAS ###

# Calcular a acurácia a partir da matriz de confusão
def calculate_accuracy(conf_matrix):
    vp_vn = np.trace(conf_matrix)  # Soma dos elementos da diagonal
    total_elements = np.sum(conf_matrix)
    accuracy = vp_vn / total_elements
    return accuracy

def evaluate_model(predictions, labels_test):
    y_true = np.argmax(labels_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Matriz de confusão:\n{conf_matrix}")
    
    accuracy = calculate_accuracy(conf_matrix)
    print(f"Acurácia (usando matriz de confusão): {accuracy:.2f}")
    
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Vitória Casa', 'Empate', 'Vitória Fora'], 
                yticklabels=['Vitória Casa', 'Empate', 'Vitória Fora'])
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.title('Matriz de Confusão')
    plt.show()

    class_report = classification_report(y_true, y_pred, target_names=['Vitória Casa', 'Empate', 'Vitória Fora'])
    print(f"Relatório de classificação:\n{class_report}")
    
    auc = roc_auc_score(labels_test, predictions, multi_class='ovr')
    print(f"AUC: {auc:.2f}")

    return accuracy, auc

def plot_roc_curve(labels_test, predictions):
    plt.figure(figsize=(10, 8))
    
    for i in range(predictions.shape[1]):
        fpr, tpr, _ = roc_curve(labels_test[:, i], predictions[:, i])
        plt.plot(fpr, tpr, label=f'Classe {i}: AUC = {roc_auc_score(labels_test[:, i], predictions[:, i]):.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--')  # linha de referência
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall(labels_test, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(np.argmax(labels_test, axis=1), np.argmax(predictions, axis=1))
    
    classes = ['Vitória Casa', 'Empate', 'Vitória Fora']
    
    x = np.arange(len(classes))
    
    plt.bar(x - 0.2, precision, width=0.2, label='Precisão', color='blue')
    plt.bar(x, recall, width=0.2, label='Revocação', color='orange')
    plt.bar(x + 0.2, f1, width=0.2, label='F1 Score', color='green')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precisão, Revocação e F1 Score por Classe')
    plt.xticks(x, classes)
    plt.legend()
    plt.show()

def plot_error_over_epochs(error_list):
    plt.plot(error_list)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Médio')
    plt.title('Erro Médio ao Longo das Épocas')
    plt.show()

def plot_prediction_distribution(predictions):
    pred_labels = np.argmax(predictions, axis=1)
    counts = np.bincount(pred_labels)
    
    classes = ['Vitória Casa', 'Empate', 'Vitória Fora']
    
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Número de Predições')
    plt.title('Distribuição das Predições do Modelo')
    plt.show()

if __name__ == "__main__":
    # 1. Carregar e preparar os dados
    data = get_data()

    # 2. Dividir os dados em treinamento e teste
    train_data, test_data = split_train_test(data)
    
    # 3. Separar dados de entrada (features) e rótulos (labels)
    ftr_columns = ['FTR_H', 'FTR_D', 'FTR_A']
    features_train = train_data.drop(columns=ftr_columns).values
    labels_train = train_data[ftr_columns].values
    features_test = test_data.drop(columns=ftr_columns).values
    labels_test = test_data[ftr_columns].values

    # 4. Definir hiperparâmetros
    input_size = features_train.shape[1]  # Número de features de entrada
    hidden_size = 20  # Número de neurônios na camada oculta
    output_size = 3  # Número de classes (vitória casa, empate, vitória fora)

    # 5. Treinar o modelo
    epochs = 200
    learning_rate = 0.01
    momentum = 1

    weights_hidden_input, weights_hidden_output, error_list = train_model(
        features_train, labels_train, input_size, hidden_size, output_size, 
        epochs, learning_rate, momentum
    )

    # 5. Fazer previsões
    predictions = predict(features_test, weights_hidden_input, weights_hidden_output)

    # 6. Avaliar o modelo
    accuracy, auc = evaluate_model(predictions, labels_test)

    # 7. Plotar gráficos
    plot_roc_curve(labels_test, predictions)
    plot_precision_recall(labels_test, predictions)
    plot_error_over_epochs(error_list)
    plot_prediction_distribution(predictions)
