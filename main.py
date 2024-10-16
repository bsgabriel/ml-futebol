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
    
    return weights_hidden_input, weights_hidden_output

def train_model(features_train, labels_train, input_size, hidden_size, output_size, 
                epochs, learning_rate, momentum):
    weights_hidden_input, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(features_train, weights_hidden_input, weights_hidden_output)
        weights_hidden_input, weights_hidden_output = backward_propagation(
            features_train, hidden_layer_output, output_layer_output, labels_train, 
            weights_hidden_input, weights_hidden_output, learning_rate, momentum
        )
        
        if epoch % 5 == 0:
            error = np.mean(np.abs(labels_train - output_layer_output))
            print(f"Epoch {epoch}, Error: {error}")
    
    return weights_hidden_input, weights_hidden_output

def predict(features, weights_hidden_input, weights_hidden_output):
    _, output_layer_output = forward_propagation(features, weights_hidden_input, weights_hidden_output)
    return output_layer_output

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

    # 5. Treinar o modelo
    weights_hidden_input, weights_hidden_output = train_model(
        features_train, labels_train, input_size, hidden_size, output_size, 
        epochs, learning_rate, momentum
    )

    # 6. Fazer previsões com os dados de teste
    predictions = predict(features_test, weights_hidden_input, weights_hidden_output)

    # 7. Avaliar o modelo
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels_test, axis=1))
    print(f"Acurácia do modelo: {accuracy:.2f}")

    # 8. Mostrar algumas predições
    print("\nAlgumas predições:")
    for i in range(50):
        true_label = np.argmax(labels_test[i])
        predicted_label = np.argmax(predictions[i])
        print(f"Real: {true_label}, Previsto: {predicted_label}")
