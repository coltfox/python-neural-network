from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


def relu(float_arr: NDArray[np.float32]) -> NDArray[np.float32]:
    zeros = np.zeros(float_arr.shape, dtype=np.float32)
    return np.maximum(float_arr, zeros)


def he_weights_initialization(num_inputs: int, num_outputs: int):
    """Initialize weights for a particular layer using He Weights Initialization.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

    W = N(0, 2/num_inputs)
    """
    mean = 0
    stand_dev = 2 / num_inputs

    return np.random.normal(mean, stand_dev, (num_outputs, num_inputs)).astype(np.float32)


@dataclass
class NetworkLayer:
    num_inputs: int
    num_outputs: int

    def __post_init__(self):
        self.weights = he_weights_initialization(
            self.num_inputs, self.num_outputs)
        self.biases = np.zeros(self.num_outputs, dtype=np.float32)

    def calculate_activations(self, previous_activations: NDArray[np.float32]):
        return relu(self.weights @ previous_activations + self.biases)


class FeedforwardNeuralNetwork:
    def __init__(self, input: NDArray[np.float32], network_layers: list[NetworkLayer]) -> None:
        self.input = input
        self.network_layers = network_layers

    def feedforward(self):
        """Feed input through neural network. Returns resulting output layer."""
        output = self.input

        for layer in self.network_layers:
            output = layer.calculate_activations(output)

        return output
