import math
import numpy as np
from mnist_idx_reader import IDX1Reader, IDX3Reader
from feedforward_neural_network import FeedforwardNeuralNetwork, NetworkLayer
from backpropogation_trainer import BackpropogationTrainer
from config import Config
from network import Network
from utils import flatten_images, get_mnist_desired_outputs_arr

CONFIG = Config(
    TRAINING_IMGS_PATH="./data/train-images.idx3-ubyte",
    TRAINING_LABELS_PATH="./data/train-labels.idx1-ubyte",
    TEST_IMGS_PATH="./data/t10k-images.idx3-ubyte",
    TEST_LABELS_PATH="./data/t10k-labels.idx1-ubyte",
    LEARNING_RATE=0.1
)

labels = IDX1Reader.read_from_file(CONFIG.TRAINING_LABELS_PATH)
imgs = IDX3Reader.read_from_file(CONFIG.TRAINING_IMGS_PATH)

inputs = flatten_images(imgs)
num_pixels = imgs[0].size
num_neurons = math.floor(num_pixels * (2 / 3))

num_outputs = 10
desired_outputs = get_mnist_desired_outputs_arr(labels)

# network = Network([784, num_neurons, num_neurons, 10])
# network.SGD(
#     training_data=[(inputs[i], desired_outputs[i])
#                    for i in range(len(inputs))],
#     epochs=1,
#     num_batches=100,
#     eta=CONFIG.LEARNING_RATE,
# )

network = FeedforwardNeuralNetwork(
    network_layers=[
        NetworkLayer(num_pixels, num_neurons),
        NetworkLayer(num_neurons, num_neurons),
        NetworkLayer(num_neurons, num_outputs)
    ]
)


trainer = BackpropogationTrainer(
    network=network,
    inputs=inputs,
    desired_outputs=desired_outputs,
    learning_rate=CONFIG.LEARNING_RATE
)

trainer.train_in_batches(100)
