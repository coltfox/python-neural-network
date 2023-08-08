from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


def relu(float_arr: NDArray[np.float32]) -> NDArray[np.float32]:
    zeros = np.zeros(float_arr.shape, dtype=np.float32)
    return np.maximum(float_arr, zeros)


def calc_relu_derivative(float_arr: NDArray[np.float32]):
    """For any output of the ReLu function the derivative is 0 for negative values
    and 1 for positive values."""
    result = np.zeros(float_arr.shape, dtype=np.float32)
    result[np.where(float_arr > 0)[0]] = 1

    return result


def he_weights_initialization(num_inputs: int, num_outputs: int):
    """Initialize weights for a particular layer using He Weights Initialization.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

    W = N(0, 2/num_inputs)
    """
    mean = 0
    stand_dev = 2 / num_inputs

    return np.random.normal(mean, stand_dev, (num_outputs, num_inputs)).astype(np.float32)


def adjust_network_param(param: NDArray[np.float32], learning_rate: np.float32, cost_gradient: NDArray[np.float32]):
    """Adjust a network parameter (i.e. bias, weight) based on its cost gradient."""
    return param + (param - learning_rate * cost_gradient)


@dataclass
class NetworkLayer:
    num_inputs: int
    num_outputs: int

    def __post_init__(self):
        self.weights = he_weights_initialization(
            self.num_inputs, self.num_outputs)
        self.biases = np.zeros(self.num_outputs, dtype=np.float32)
        self.activations = np.zeros(self.num_outputs, dtype=np.float32)
        self.pre_activations = np.zeros(self.num_outputs, dtype=np.float32)

    def calculate_activations(self, prev_activations: NDArray[np.float32]):
        self.calculate_pre_activations(prev_activations)
        self.activations = relu(self.pre_activations)

        return self.activations

    def calculate_pre_activations(self, prev_activations: NDArray[np.float32]):
        """Calculate activations before activation function (ReLu) is applied."""
        self.pre_activations = self.weights @ prev_activations + self.biases

        return self.pre_activations

    def calculate_weights_gradient(self, cost_activation_change: NDArray[np.float32]) -> NDArray[np.float32]:
        relu_derivative = calc_relu_derivative(self.pre_activations)

        # Change in current activation with respect to weights * Change in activation with respect to relu(activation) * Change in cost with respect to previous activation
        return relu_derivative * cost_activation_change

    def calculate_biases_gradient(self, cost_activation_change: NDArray[np.float32]) -> NDArray[np.float32]:
        relu_derivative = calc_relu_derivative(self.activations)

        # Change in activation with respect to relu(activation) * Change in cost with respect to previous activation
        return relu_derivative * cost_activation_change

    def adjust_weights(self, learning_rate: np.float32, weights_gradient: NDArray[np.float32]):
        self.weights = adjust_network_param(
            self.weights, learning_rate, weights_gradient)

    def adjust_baises(self, learning_rate: np.float32, biases_gradient: NDArray[np.float32]):
        self.biases = adjust_network_param(
            self.biases, learning_rate, biases_gradient)


@dataclass
class FeedforwardNeuralNetwork:
    network_layers: list[NetworkLayer]

    def __post_init__(self) -> None:
        if len(self.network_layers) == 0:
            raise ValueError("Cannot instantiate a network with no layers!")

    def validate_input_shape(self, input: NDArray[np.float32]):
        """Ensure input shape matches number of connections of first hidden network layer."""
        first_layer = self.network_layers[0]

        if input.ndim > 1 or input.size != first_layer.num_inputs:
            raise ValueError(
                f"Expected input size to be equal to the number of inputs in the first layer! Input is size '{input.size}' when it should be of size '{first_layer.num_inputs}'")

    def feedforward(self, input: NDArray[np.float32]):
        """Feed input through neural network. Returns resulting output layer."""
        self.validate_input_shape(input)

        for layer in self.network_layers:
            input = layer.calculate_activations(input)

        return input

    def backpropogate(self, input: NDArray[np.float32], desired_output: NDArray[np.float32]):
        """Calculate weight gradients and bias gradients by backpropogating."""
        output = self.feedforward(input)
        num_layers = len(self.network_layers)

        layers_reversed = list(reversed(self.network_layers))

        last_layer = layers_reversed[0]
        second_to_last_layer = layers_reversed[1]

        weights_gradients: list[NDArray[np.float32]] = [None] * num_layers
        bias_gradients: list[NDArray[np.float32]] = [None] * num_layers

        cost_derivative = 2 * (output - desired_output)
        relu_derivative = calc_relu_derivative(last_layer.pre_activations)
        delta = cost_derivative * relu_derivative

        bias_gradients[-1] = delta
        weights_gradients[-1] = np.outer(delta,
                                         second_to_last_layer.activations)

        for i, layer in enumerate(layers_reversed[0:]):
            pre_activs = layer.pre_activations
            relu_derivative = calc_relu_derivative(pre_activs)

            previous_weights = layers_reversed[i + 1].weights
            previous_activs = layers_reversed[i + 1].activations

            # delta = (previous_weights.transpose() @ delta) * relu_derivative
            delta = np.outer(delta, previous_weights) * relu_derivative

            weights_gradients[-i] = np.outer(delta, previous_activs)
            bias_gradients[-i] = delta

        return weights_gradients, bias_gradients

    def calculate_cost(self, input: NDArray[np.float32], desired_output: NDArray[np.float32]):
        """Calculate cost for a single input."""
        output = self.feedforward(input)
        return np.sum((output - desired_output)**2)

    def calculate_avg_cost(self, inputs: NDArray[np.float32], desired_outputs: NDArray[np.float32]):
        """Calculate average cost across a range of inputs."""
        if len(inputs) != len(desired_outputs):
            raise ValueError(
                "Expected same number of inputs as desired outputs!")

        total_cost = sum(self.calculate_cost(i, o)
                         for i, o in zip(inputs, desired_outputs))

        return total_cost / len(inputs)
