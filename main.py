import math
from mnist_idx_reader import IDX1Reader, IDX3Reader
from feedforward_neural_network import FeedforwardNeuralNetwork, NetworkLayer
from config import Config

CONFIG = Config(
    TRAINING_IMGS_PATH="./data/train-images.idx3-ubyte",
    TRAINING_LABELS_PATH="./data/train-labels.idx1-ubyte",
    TEST_IMGS_PATH="./data/t10k-images.idx3-ubyte",
    TEST_LABELS_PATH="./data/t10k-labels.idx1-ubyte",
)

labels = IDX1Reader.read_from_file(CONFIG.TRAINING_LABELS_PATH)
imgs = IDX3Reader.read_from_file(CONFIG.TRAINING_IMGS_PATH)

pixels = imgs[0].flatten()
num_pixels = pixels.size
num_neurons = math.floor(num_pixels * (2 / 3))

num_outputs = 10

test_network = FeedforwardNeuralNetwork(
    input=pixels,
    network_layers=[
        NetworkLayer(num_pixels, num_neurons),
        NetworkLayer(num_neurons, num_neurons),
        NetworkLayer(num_neurons, num_outputs)
    ]
)

output = test_network.feedforward()

print(f"Actual: {labels[0]}\n")
for i in range(10):
    print(f"Confidence {i}: {output[i]}")

# display_number(imgs[0], labels[0])
