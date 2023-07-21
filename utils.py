import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def display_number(pixel_buffer: NDArray[np.uint8], label: np.uint8):
    plt.imshow(pixel_buffer, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()
