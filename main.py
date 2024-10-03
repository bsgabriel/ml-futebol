import pandas as pd
import numpy as np

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
    weights_hidden_input = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_hidden_input, weights_hidden_output

# Propagação para frente
def forward_propagation(features, weights_hidden_input, weights_hidden_output):
    linear_combination_input_hidden = np.dot(features, weights_hidden_input)
    activation_hidden = 1 / (1 + np.exp(-linear_combination_input_hidden))  # Sigmoid
    
    linear_combination_hidden_output = np.dot(activation_hidden, weights_hidden_output)
    activation_output = 1 / (1 + np.exp(-linear_combination_hidden_output))  # Sigmoid
    
    return linear_combination_input_hidden, activation_hidden, linear_combination_hidden_output, activation_output

# Ajuste dos pesos usando retropropagação
def backward_propagation(features, labels, linear_combination_input_hidden, activation_hidden, activation_output, weights_hidden_input, weights_hidden_output, learning_rate=0.01, momentum=0.9, alpha=0.2):
    num_samples = features.shape[0]
    
    # Erro da camada de saída
    error_output = activation_output - labels
    gradient_hidden_output = (1/num_samples) * np.dot(activation_hidden.T, error_output)
    
    # Erro da camada oculta
    error_hidden = np.dot(error_output, weights_hidden_output.T) * (activation_hidden * (1 - activation_hidden))
    gradient_input_hidden = (1/num_samples) * np.dot(features.T, error_hidden)
    
    # Atualização dos pesos
    weights_hidden_input = weights_hidden_input * momentum + alpha * gradient_input_hidden
    weights_hidden_output = weights_hidden_output * momentum + alpha * gradient_hidden_output
    
    return weights_hidden_input, weights_hidden_output

# Perda (cross-entropy)
def compute_cross_entropy_loss(labels, activation_output):
    num_samples = labels.shape[0]
    logprobs = np.multiply(np.log(activation_output), labels)
    loss = -np.sum(logprobs) / num_samples
    return loss

# Treinar o modelo
def train_neural_network(features_train, labels_train, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.01):
    weights_hidden_input, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        linear_combination_input_hidden, activation_hidden, linear_combination_hidden_output, activation_output = forward_propagation(features_train, weights_hidden_input, weights_hidden_output)
        loss = compute_cross_entropy_loss(labels_train, activation_output)
        weights_hidden_input, weights_hidden_output = backward_propagation(features_train, labels_train, linear_combination_input_hidden, activation_hidden, activation_output, weights_hidden_input, weights_hidden_output, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights_hidden_input, weights_hidden_output

# Fazer predições
def predict(features, weights_hidden_input, weights_hidden_output):
    _, _, _, activation_output = forward_propagation(features, weights_hidden_input, weights_hidden_output)
    return np.argmax(activation_output, axis=1)

# Função principal para executar o script
if __name__ == "__main__":
    # 1. Carregar e preparar os dados
    data = get_data()

    # 2. Dividir os dados em treinamento e teste
    train_data, test_data = split_train_test(data)
    
    # 3. Separar dados de entrada (features) e rótulos (labels)
    features_train = train_data.drop(columns=['FTR_H', 'FTR_D', 'FTR_A']).values
    labels_train = train_data[['FTR_H', 'FTR_D', 'FTR_A']].values
    features_test = test_data.drop(columns=['FTR_H', 'FTR_D', 'FTR_A']).values
    labels_test = test_data[['FTR_H', 'FTR_D', 'FTR_A']].values

    # 4. Definir hiperparâmetros
    input_size = features_train.shape[1]  # Número de features de entrada
    hidden_size = 50  # Número de neurônios na camada oculta
    output_size = 3  # Número de classes (vitória casa, empate, vitória fora)

    # 5. Treinar o modelo
    weights_hidden_input, weights_hidden_output = train_neural_network(features_train, labels_train, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.01)

    # 6. Fazer previsões com os dados de teste
    predictions = predict(features_test, weights_hidden_input, weights_hidden_output)
    
    # Mostrar as predições
    print("Predictions:", predictions)
