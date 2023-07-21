from mnist_idx_reader import IDX1Reader, IDX3Reader
from utils import display_number
from config import Config

CONFIG = Config(
    TRAINING_IMGS_PATH="./data/train-images.idx3-ubyte",
    TRAINING_LABELS_PATH="./data/train-labels.idx1-ubyte",
    TEST_IMGS_PATH="./data/t10k-images.idx3-ubyte",
    TEST_LABELS_PATH="t10k-labels.idx1-ubyte",
)

labels = IDX1Reader.read_from_file(CONFIG.TRAINING_LABELS_PATH)
imgs = IDX3Reader.read_from_file(CONFIG.TRAINING_IMGS_PATH)

display_number(imgs[0], labels[0])
