import numpy as np
import utils
import pickle

class NeuralNetwork:
    def __init__(self, dimension_list, weights_list=None, biases_list=None, activation_function='logistic'):
        self.dimension_list = dimension_list
        self.activation_function = activation_function
        if weights_list is None and biases_list is None:
            self.layers = [Layer(input_dim, output_dim) for input_dim, output_dim in zip(dimension_list[:-1], dimension_list[1:])]
        else:
            if weights_list is not None:
                weights_np_array = [np.array(layer_weights, dtype=np.float64) for layer_weights in weights_list]
                assert [(output_dim, input_dim) for input_dim, output_dim in zip(dimension_list[:-1], dimension_list[1:])] == \
                        [weights.shape for weights in weights_np_array], \
                        "Dimension list and weights list do not match in dimensions"
            if biases_list is not None:
                biases_np_array = [np.array(layer_bias, dtype=np.float64) for layer_bias in biases_list]
                assert [(output_dim, 1) for output_dim in dimension_list[1:]] == \
                        [bias.shape for bias in biases_np_array], \
                        "Dimension list and bias list do not match in dimensions"
            
            if weights_list is not None and biases_list is not None:
                self.layers = [Layer(input_dim, output_dim, weights=weight, bias=bias, activation_function=activation_function) for 
                            input_dim, output_dim, weight, bias in zip(dimension_list[:-1], dimension_list[1:], weights_np_array, biases_np_array)]
            elif weights_list is not None:
                self.layers = [Layer(input_dim, output_dim, weights=weight, activation_function=activation_function) for 
                            input_dim, output_dim, weight in zip(dimension_list[:-1], dimension_list[1:], weights_np_array)]
            elif biases_list is not None:
                self.layers = [Layer(input_dim, output_dim, bias=bias, activation_function=activation_function) for 
                            input_dim, output_dim, bias in zip(dimension_list[:-1], dimension_list[1:], biases_np_array)]

        self.layers[-1].set_layer_before_output(True)

    def cost(self, x, label):
        return 0.5*((np.linalg.norm(self.forward(x) - label))**2)

    def forward(self, x):
        """
        return the final layer output

        mutates the layer_input, presigs, and oo
        """
        layer_output = x
        for layer_index in range(len(self.layers)):
            layer_output = self.layers[layer_index].forward(layer_output)

        assert layer_output.shape == (self.layers[-1].output_dim, 1), "Layer output has shape {}, but should have shape ({}, 1)".format(layer_output.shape, self.layers[-1].output_dim) 
        return layer_output

    def backward(self, x, label):
        """
        returns deltas, does not modify weights or biases
        """
        self.forward(x)

        deltas = []
        for layer_index in range(len(self.layers)-1, -1, -1):
            if layer_index != len(self.layers)-1:
                delta = self.layers[layer_index].backward(weights_from_next_layer=self.layers[layer_index+1].w,
                                                          delta_from_next_layer=deltas[-1])
                deltas.append(delta)
            else:
                delta = self.layers[layer_index].backward(label=label)
                deltas.append(delta)

        return deltas

    def update(self, x, label, learning_rate=0.01):
        weight_partials = np.array(self.weight_partials(x, label))
        bias_partials = np.array(self.bias_partials(x, label))

        for layer_index in range(1, len(self.layers) + 1):
            layer = self.layers[-layer_index]
            layer.w -= learning_rate*weight_partials[layer_index-1]
            layer.b -= learning_rate*bias_partials[layer_index-1]


    def weight_partials(self, x, label):
        """
        weight partials in backwards order
        """
        deltas = np.array(self.backward(x, label))
        inputs = [layer.layer_input.transpose() for layer in self.layers][::-1]  
        delta_shapes = [delta.shape for delta in deltas]
        input_shapes = [input.shape for input in inputs]

        return [np.matmul(delta, input) for delta, input in zip(deltas, inputs)]

    def bias_partials(self, x, label):
        """
        bias partials in backwards order
        """
        return np.array(self.backward(x, label))

    def gradient_check_weights(self, x, label):
        """
        compute the weight partial derivatives by approximation for each weight one by one
        in backwards order
        """
        weight_partials = []
        for layer_index in range(1, len(self.layers)+1):
            layer = self.layers[-layer_index]

            weight_partial = np.zeros(layer.w.shape)

            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    original_weight = layer.w[i, j]
                    epsilon = 0.0001
                    layer.w[i, j] -= epsilon
                    cost_minus_epsilon = self.cost(x, label)
                    layer.w[i, j] += 2*epsilon
                    cost_plus_epsilon = self.cost(x, label)

                    weight_partial[i, j] = (cost_plus_epsilon - cost_minus_epsilon) / (2*epsilon)

                    layer.w[i, j] = original_weight
            weight_partials.append(weight_partial)

        return weight_partials

    def gradient_check_biases(self, x, label):
        bias_partials = []
        for layer_index in range(1, len(self.layers)+1):
            layer = self.layers[-layer_index]

            bias_partial = np.zeros(layer.b.shape)

            for j in range(layer.b.shape[0]):
                original_bias = layer.b[j, 0]
                epsilon = 0.0001
                layer.b[j, 0] -= epsilon
                cost_minus_epsilon = self.cost(x, label)
                layer.b[j, 0] += 2*epsilon
                cost_plus_epsilon = self.cost(x, label)

                bias_partial[j, 0] = (cost_plus_epsilon - cost_minus_epsilon) / (2*epsilon)

                layer.b[j, 0] = original_bias
            bias_partials.append(bias_partial)

        return bias_partials

    def weights(self):
        return [layer.w for layer in self.layers]

    def biases(self):
        return [layer.b for layer in self.layers]

    def outputs(self):
        return [layer.output for layer in self.layers]

    def inputs(self):
        return [layer.layer_input for layer in self.layers]

    def preactivations(self):
        return [layer.preactivation for layer in self.layers]

    def save_model(self, filename="model.pkl"):
        params = {"dimension_list": self.dimension_list, 
                    "weights_list": self.weights(), 
                    "biases_list": self.biases(),
                    "activation_function": self.activation_function}
        with open(filename, 'wb') as outfile:
            pickle.dump(params, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, filename="model.pkl"):
        with open(filename, 'rb') as infile:
            params = pickle.load(infile)
            loaded_dimension_list = params["dimension_list"]
            loaded_weights_list = params["weights_list"]
            loaded_biases_list = params["biases_list"]
            loaded_activation_function = params["activation_function"]
            return cls(dimension_list=loaded_dimension_list, 
                        weights_list=loaded_weights_list, 
                        biases_list=loaded_biases_list, 
                        activation_function=loaded_activation_function)

class Layer:
    def __init__(self, input_dim, output_dim, layer_before_output=False, weights=None, bias=None, activation_function='logistic'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        if weights is None:
            self.w = np.random.randn(output_dim, input_dim)
        else:
            assert weights.shape == (output_dim, input_dim)
            self.w = np.array(weights, dtype=np.float64)
        if bias is None:
            self.b = np.random.randn(output_dim, 1)
        else:   
            assert bias.shape == (output_dim, 1)
            self.b = np.array(bias, dtype=np.float64)

        self.layer_before_output = layer_before_output

    def set_layer_before_output(self, before_output):
        self.layer_before_output = before_output

    def forward(self, x):
        """
        returns the layer output
        modifies state of layer - layer_input, preactivation, output
        """
        assert x.shape == (self.input_dim, 1), "x.shape is {}, but should be ({}, 1)".format(x.shape, self.input_dim)
        
        self.layer_input = x
        self.preactivation = np.matmul(self.w, x) + self.b
        self.output = self.activation(self.preactivation)
        return self.output

    def backward(self, delta_from_next_layer=None, weights_from_next_layer=None, label=None):
        """
        return the delta, i.e., the cost derivative of the layer's biases
        does not modify state - weights, biases
        """
        if not self.layer_before_output:
            assert delta_from_next_layer is not None

            delta = self.activation_derivative(self.preactivation) * np.matmul(weights_from_next_layer.transpose(), delta_from_next_layer)

        else:
            assert label is not None

            delta = self.activation_derivative(self.preactivation) * (self.output - label)

        #self.b = self.b - learning_rate*delta
        #self.w = self.w - learning_rate*(np.matmul(delta, self.x.transpose())) 

        assert delta.shape == (self.output_dim, 1), "delta.shape is {}, but should be ({}, 1)".format(delta.shape, self.output_dim)
        return delta

    def activation(self, array):
        if self.activation_function == 'tanh':
            return np.tanh(array)
        elif self.activation_function == 'relu':
            return utils.relu(array)
        else:
            # assume logistic
            return utils.logistic(array)

    def activation_derivative(self, array):
        if self.activation_function == 'tanh':
            return utils.tanh_derivative(array)
        elif self.activation_function == 'relu':
            return utils.relu_derivative(array)
        else:
            # assume logistic
            return utils.logistic_derivative(array)

def gradient_check_tests():
    num_of_neurons_per_layer = [2, 5, 3]
    net = NeuralNetwork(dimension_list=num_of_neurons_per_layer)

    a = np.random.randint(2, size=100)
    b = np.random.randint(2, size=100)

    x1 = a[0]
    x2 = b[0]
    input = np.reshape(np.array([x1, x2]), (2, 1))
    label = np.reshape(np.array([1, 2, 3]), (3, 1))

    print("backward | gradient check")
    weight_partials = net.weight_partials(input, label)
    gradient_check_weights = net.gradient_check_weights(input, label)

    for weight_partial, gradient_check in zip(weight_partials, gradient_check_weights):
        print("Weight partial shape: {} | Gradient check shape: {}\n".format(weight_partial.shape, gradient_check.shape))
        print("Weight partial:\n{}\nGradient check:\n{}\n".format(weight_partial, gradient_check))

    biases_partials = net.bias_partials(input, label)
    gradient_check_biases = net.gradient_check_biases(input, label)

    for bias_partial, gradient_check in zip(biases_partials, gradient_check_biases):
        print("Bias partial shape: {} | Gradient check shape: {}\n".format(bias_partial.shape, gradient_check.shape))
        print("Bias partial:\n{}\nGradient check:\n{}\n".format(bias_partial, gradient_check))

if __name__ == "__main__":
    gradient_check_tests()


