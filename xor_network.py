from neural_network import NeuralNetwork
import numpy as np
import os

def evaluate_xor_network(network, arr1, arr2):
    assert len(arr1) == len(arr2), "arr1 and arr2 are not same length"

    labels = 1*np.logical_xor(arr1, arr2)
    return (1/len(arr1))*np.sum([network.cost(np.reshape(np.array([x1, x2]), (2, 1)), np.array([[label]])) for x1, x2, label in zip(arr1, arr2, labels)])

if __name__ == "__main__":
    SAVED_MODELS_DIR = 'saved_models'
    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)

    num_of_neurons_per_layer = [2, 5, 1]
    net = NeuralNetwork(dimension_list=num_of_neurons_per_layer, activation_function='relu')

    a = np.random.randint(2, size=10000)
    b = np.random.randint(2, size=10000)

    cross_validation_a = np.random.randint(2, size=100)
    cross_validation_b = np.random.randint(2, size=100)
    counter = 1
    for x1, x2 in zip(a, b):
        input = np.reshape(np.array([x1, x2]), (2, 1))
        label = 1*np.logical_xor(np.array([[x1]]), np.array([[x2]]))
        net.update(input, label, learning_rate=25)

        if counter % 100 == 0: 
            print("Example: {} Cross validation loss: {}".format(counter, evaluate_xor_network(net, cross_validation_a, cross_validation_b)))
            net.save_model(filename=os.path.join(SAVED_MODELS_DIR, "params_{}.pkl".format(counter)))
        counter += 1