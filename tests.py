from neural_network import NeuralNetwork
from utils import logistic, logistic_derivative, relu, relu_derivative, tanh_derivative

import unittest
import numpy as np
import os

class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.dimension_list = [2, 2, 1]
        weights_list = [ [[0.5, 0.8], [-0.2, 0.4]], [[0.3, 0.7]] ]
        biases_list = [ [[0], [0]], [[0]] ]
        self.net = NeuralNetwork(dimension_list=self.dimension_list, 
                                    weights_list=weights_list, 
                                    biases_list=biases_list, 
                                    activation_function='logistic')

    def test_forward(self):
        x1, x2 = 0.1, 0.4
        layer_1_presig = np.reshape(np.array([(0.5*x1 + 0.8*x2), (-0.2*x1 + 0.4*x2)]), (2, 1))
        layer_1_output = logistic(layer_1_presig) # [[0.59145898], [0.53494295]]
        #layer_2_presig = np.reshape(np.array([0.3*0.59145898 + 0.7*0.53494295]), (1, 1))
        #layer_2_output = logistic(layer_2_presig)
        layer_2_output = logistic(np.reshape(np.array([0.3*logistic(0.5*x1 + 0.8*x2) + 0.7*logistic(-0.2*x1 + 0.4*x2)]), (1, 1)))

        print("Expected Output:\nLayer 1:\n{}\nLayer 2:\n{}\n".format(layer_1_output, layer_2_output))

        input = np.reshape(np.array([x1, x2]), (2, 1))
        network_output = self.net.forward(input)

        print("Neural Network Output (layer 2):\n{}\n".format(network_output))

        self.assertEqual(network_output, layer_2_output)

    def test_backward_deltas(self):
        """
        test deltas, not the weight partials
        """
        x1, x2 = 0.1, 0.4
        label = 0.9

        layer_1_presig = np.reshape(np.array([(0.5*x1 + 0.8*x2), (-0.2*x1 + 0.4*x2)]), (2, 1))
        layer_1_output = logistic(layer_1_presig) # [[0.59145898], [0.53494295]]
        layer_2_presig = np.reshape(np.array([0.3*logistic(0.5*x1 + 0.8*x2) + 0.7*logistic(-0.2*x1 + 0.4*x2)]), (1, 1))
        layer_2_output = logistic(layer_2_presig)

        error = layer_2_output-label

        layer_2_delta = error*logistic_derivative(layer_2_presig)
        layer_1_delta = logistic_derivative(layer_1_presig)*np.matmul(np.reshape(np.array([0.3, 0.7]), (2, 1)), layer_2_delta)
        print("Expected Deltas:\nLayer 2:\n{}\nLayer 1:\n{}\n".format(layer_2_delta, layer_1_delta))

        input = np.reshape(np.array([x1, x2]), (2, 1))
        network_deltas = self.net.backward(input, label)
        print("Neural Network Deltas:\nLayer 2:\n{}\nLayer 1:\n{}\n".format(network_deltas[0], network_deltas[1]))

        np.testing.assert_array_equal(network_deltas[0], layer_2_delta, "Layer 2 delta is not equal to expected")
        np.testing.assert_array_equal(network_deltas[1], layer_1_delta, "Layer 1 delta is not equal to expected")

    def test_weight_partials(self):
        x1, x2 = 0.1, 0.4
        label = 0.9

        layer_1_presig = np.reshape(np.array([(0.5*x1 + 0.8*x2), (-0.2*x1 + 0.4*x2)]), (2, 1))
        layer_1_output = logistic(layer_1_presig) # [[0.59145898], [0.53494295]]
        layer_2_presig = np.reshape(np.array([0.3*logistic(0.5*x1 + 0.8*x2) + 0.7*logistic(-0.2*x1 + 0.4*x2)]), (1, 1))
        layer_2_output = logistic(layer_2_presig)

        error = layer_2_output-label

        layer_2_delta = error*logistic_derivative(layer_2_presig)
        layer_1_delta = logistic_derivative(layer_1_presig)*np.matmul(np.reshape(np.array([0.3, 0.7]), (2, 1)), layer_2_delta)

        layer_2_weight_partials = np.matmul(layer_2_delta, layer_1_output.transpose())
        layer_1_weight_partials = np.matmul(layer_1_delta, np.array([[x1, x2]]))
        print("Expected Weight Partial Derivatives:\nLayer 2:\n{}\nLayer 1:\n{}\n".format(layer_2_weight_partials, layer_1_weight_partials))


        input = np.reshape(np.array([x1, x2]), (2, 1))
        network_weight_partials = self.net.weight_partials(input, label)
        print("Neural Network Weight Partial Derivatives:\nLayer 2:\n{}\nLayer 1:\n{}\n".format(network_weight_partials[0], network_weight_partials[1]))

        np.testing.assert_array_equal(network_weight_partials[0], layer_2_weight_partials, "Layer 2 weight partial derivatives is not equal to expected")
        np.testing.assert_array_equal(network_weight_partials[1], layer_1_weight_partials, "Layer 1 weight partial derivatives is not equal to expected")

    def test_save_and_load_model(self):
        """
        Test save_model() and load_model() of a basic XOR ReLu network
        """

        self.net.activation_function = 'relu'

        MODEL_FILENAME = "test_save_and_load_model.pkl"

        a = np.random.randint(2, size=100)
        b = np.random.randint(2, size=100)

        for x1, x2 in zip(a, b):
            input = np.reshape(np.array([x1, x2]), (2, 1))
            label = 1*np.logical_xor(np.array([[x1]]), np.array([[x2]]))
            self.net.update(input, label)

        # save model
        self.net.save_model(filename=MODEL_FILENAME)

        # load model into new network
        test_net = NeuralNetwork.load_model(filename=MODEL_FILENAME)

        print("Saved Model Weights:\n{}\nLoaded Model Weights:\n{}\n".format(self.net.weights(), test_net.weights()))

        # check dimension_list
        self.assertEqual(test_net.dimension_list, self.net.dimension_list, "Saved model dimension list and loaded model dimension list are not the same")

        # check activation function
        self.assertEqual(test_net.activation_function, self.net.activation_function, "Saved model activation function and loaded model activation function are not the same")
        
        # check weights and biases
        layer = 1
        for saved_weight, loaded_weight, saved_bias, loaded_bias in zip(self.net.weights(), test_net.weights(), self.net.biases(), test_net.biases()):
            np.testing.assert_array_equal(loaded_weight, saved_weight, "Layer {} saved weights are not the same as loaded weights".format(layer))
            np.testing.assert_array_equal(loaded_bias, saved_bias, "Layer {} saved biases are no the same as loaded biases".format(layer))
            layer += 1

        # remove model
        os.remove(MODEL_FILENAME)

class ActivationFunctionsTest(unittest.TestCase):
    """
    test logistic and relu activation and derivative functions
    tanh not tested since it is assumed that numpy's function is correct
    """
    def test_logistic(self):
        input = np.array([[0, 3, -3]])
        expected_output = np.array([[0.5, 1/(1+np.exp(-3)), 1/(1+np.exp(3))]])

        np.testing.assert_array_equal(logistic(input), expected_output, "utils.logistic is incorrect")

    def test_logistic_derivative(self):
        input = np.array([[0, 10, 2, -3, -10]])
        expected_output = np.array([[np.exp(0)/((1+np.exp(0))**2), 
                                np.exp(10)/((1+np.exp(10))**2), 
                                np.exp(2)/((1+np.exp(2))**2), 
                                np.exp(-3)/((1+np.exp(-3))**2), 
                                np.exp(-10)/((1+np.exp(-10))**2)]])

        np.testing.assert_almost_equal(logistic_derivative(input), expected_output, decimal=16, err_msg="utils.logistic_derivative is incorrect")

    def test_relu(self):
        input = np.array([[-1, 20, 3, -2, -30, 10, 0]])
        expected_output = np.array([[0, 20, 3, 0, 0, 10, 0]])

        np.testing.assert_array_equal(relu(input), expected_output, "utils.relu is incorrect")

    def test_relu_derivative(self):
        input = np.array([[-1, 20, 3, -2, -30, 10, 0]])
        expected_output = np.array([[0, 1, 1, 0, 0, 1, 0]])

        np.testing.assert_array_equal(relu_derivative(input), expected_output, "utils.relu_derivative is incorrect")


if __name__ == "__main__":
    unittest.main()