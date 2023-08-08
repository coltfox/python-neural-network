import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def display_number(pixel_buffer: NDArray[np.uint8], label: np.uint8):
    plt.imshow(pixel_buffer, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()


def get_mnist_desired_output(expected_num: int):
    """Get desired output layer of a particular training example based on the expected number.
    Returns a float array of length 10 with the expected num having full activation (value of 1.0)."""
    if 0 > expected_num > 9:
        raise ValueError(
            "MNIST dataset output values must be between 0 and 9 inclusive!")

    output = np.zeros(10, dtype=np.float32)
    output[expected_num] = 1.0

    return output


def get_mnist_desired_outputs_arr(labels: NDArray[np.uint8]):
    """Create a numpy array of desired outputs for each label. Each sub-array is a float array
    with 10 items where the item with index ``label`` is fully activated (activation of 1.0)."""
    desired_outputs = np.eye(10, dtype=np.float32)[labels]

    return desired_outputs


def flatten_images(imgs: NDArray[np.float32]) -> NDArray[np.float32]:
    num_pixles = imgs[0].size

    return imgs.reshape((len(imgs), num_pixles))
