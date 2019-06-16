# Neural Network

## Credits
* [A Step by Step Backpropagation Example by Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
* [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

## Setup/Dependencies
```
pip install -r requirements.txt
```
Dependencies: numpy

## Initializing a neural network
Define a neural network by a list of the number of neurons in each of its layers (```dimension_list```), and optionally activation function ('relu', 'logistic', 'tanh', default: 'logistic'), and preset (perhaps from a saved model) weights (```weights_list```) and biases(```biases_list```):
```
from neural_network import NeuralNetwork

dimension_list = [2, 2, 1]

weights_list = [ [[0.5, 0.8], [-0.2, 0.4]], [[0.3, 0.7]] ]
biases_list = [ [[0.2], [1]], [[0.9]] ]

net = NeuralNetwork(dimension_list=dimension_list, weights_list=weights_list, biases_list=biases_list, activation_function='logistic')
```

### Initializing a neural network from saved parameters
If you have already saved a model using ```save_model(filename)``` method, then you can initialize a neural network using ```NeuralNetwork.load_model(filename)```:
```
from neural-network import NeuralNetwork

net = NeuralNetwork.load_model(filename)
```

## Update parameters (weights, biases) through gradient descent (online learning/single example at a time)
Update a network's weights and biases by the ```NeuralNetwork.update(x, label, learning_rate=0.01)``` method. The input ```x``` must have dimensions ```(number of neurons in first layer, 1)```, or equivalently, ```(dimension_list[0], 1)```. Similarly, the label must have dimensions ```(number of neurons in (last) output layer, 1)```, or equivalently, ```(dimension_list[-1], 1)```:
```
from neural_network import NeuralNetwork

net = NeuralNetwork(dimension_list=[2, 5, 1], activation_function='relu') # randomly init weights and biases

x1, x2 = 0, 1
input = np.reshape(np.array([x1, x2]), (2, 1))
label = 1*np.logical_xor(np.array([[x1]]), np.array([[x2]]))
net.update(input, label)
```

## Save model/parameters
You can save a model's parameters (dimension list, weights, biases) using the ```save_model(filename)``` method and load it into a new ```NeuralNetwork``` object with ```NeuralNetwork.load_model(filename)```:
```
... # Initialize then update neural network, 'net'

MODEL_FILENAME = 'model_final.pkl'
net.save_model(filename=MODEL_FILENAME)

copy_of_net = NeuralNetwork.load_model(filename=MODEL_FILENAME)
```

## XOR Example
```
xor_network.py:
```
```
... # Define cross validation evaluation method for XOR, 'evaluate_xor_network

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
```

Results:
```
Example: 100 Cross validation loss: 0.1960322950522185
Example: 200 Cross validation loss: 0.16367952566242833
Example: 300 Cross validation loss: 0.07394138667542996
Example: 400 Cross validation loss: 0.00028398860038579164
```