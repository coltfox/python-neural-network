from dataclasses import dataclass


@dataclass
class Config:
    TRAINING_IMGS_PATH: str
    TRAINING_LABELS_PATH: str
    TEST_IMGS_PATH: str
    TEST_LABELS_PATH: str
    LEARNING_RATE: float
