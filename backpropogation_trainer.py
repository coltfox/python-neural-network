import os
from time import time
import numpy as np
import pickle
from dataclasses import dataclass
from numpy.typing import NDArray
from tqdm import tqdm

from feedforward_neural_network import FeedforwardNeuralNetwork


@dataclass
class BackpropogationTrainer:
    network: FeedforwardNeuralNetwork
    inputs: NDArray[np.float32]
    desired_outputs: NDArray[np.float32]
    learning_rate: float
    make_backups: bool = True
    backups_path: str = "./training_backups/"

    def __post_init__(self) -> None:
        num_network_outputs = self.network.network_layers[-1].num_outputs

        if self.desired_outputs.shape[1] != num_network_outputs:
            raise ValueError(
                f"Desired outputs must have the same number of outputs as the network's output layer! Got {self.desired_outputs.shape[1]} outputs, expected {num_network_outputs} outputs!")

        if self.inputs.shape[0] != self.desired_outputs.shape[0]:
            raise ValueError(
                f"There must be the same number of inputs as desired outputs! Got {self.inputs.shape[0]} inputs and {self.desired_outputs.shape[0]} desired outputs!")

    def train_in_batches(self, batches: int):
        inputs_split = np.array_split(self.inputs, batches)
        outputs_split = np.array_split(self.desired_outputs, batches)

        for inputs_batch, outputs_batch in tqdm(zip(inputs_split, outputs_split), desc="Training batch"):
            self.train(inputs_batch, outputs_batch)

        return self.network

    def train(self, inputs: NDArray[np.float32], desired_outputs: NDArray[np.float32]):
        all_weight_gradients: list[NDArray[np.float32]] = []
        all_bias_gradients: list[NDArray[np.float32]] = []

        for input, desired_output in tqdm(zip(inputs, desired_outputs), desc="Training sample"):
            weight_gradients, bias_gradients = self.network.backpropogate(
                input, desired_output)
            all_weight_gradients.extend(weight_gradients)
            all_bias_gradients.extend(bias_gradients)

        total_weight_gradient = np.mean(all_weight_gradients, axis=0)
        total_bias_gradient = np.mean(all_bias_gradients, axis=0)

        for layer in self.network.network_layers:
            layer.adjust_weights(self.learning_rate, total_weight_gradient)
            layer.adjust_biases(self.learning_rate, total_bias_gradient)

        return self.network

    def backup_current_network_state(self):
        filename = f"{self.network.__class__.__name__}.{time()}"
        full_path = os.path.join(self.backups_path, filename)

        with open(full_path, "w") as f:
            pickle.dump(self.network, f)

    def load_backup(self, filepath: str):
        if not os.path.exists(filepath):
            raise ValueError(
                f"Failed to load training backup: {filepath} is not a valid path!")

        with open(filepath, "r") as f:
            self.network = pickle.load(f)

        return self.network
