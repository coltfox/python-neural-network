"""Handles reading of idx label and image files from the MNIST database."""
from abc import ABC, abstractmethod
from typing import BinaryIO
import numpy as np
from numpy.typing import NDArray


class BinaryReader(ABC):
    @classmethod
    def read_from_file(cls, filepath: str):
        cls.validate_filepath(filepath)

        with open(filepath, "r") as f:
            parser = cls(f.buffer)
            return parser.buffer_to_arr()

    def __init__(self, buffer: BinaryIO) -> None:
        self.buffer = buffer

    @classmethod
    @abstractmethod
    def validate_filepath(cls, filepath: str):
        ...

    @abstractmethod
    def buffer_to_arr(self) -> NDArray[np.uint8]:
        ...

    def read_idx_int32(self):
        return int.from_bytes(self.buffer.read(4), byteorder="big")


class IDX1Reader(BinaryReader):
    LABEL_OFFSET = 8

    @classmethod
    def validate_filepath(cls, filepath: str):
        return filepath.endswith("idx3-ubyte")

    def buffer_to_arr(self) -> NDArray[np.uint8]:
        return np.frombuffer(self.buffer.read(), dtype=np.ubyte, offset=self.LABEL_OFFSET)


class IDX3Reader(BinaryReader):
    ROWS_OFFSET = 8
    COLS_OFFSET = 12
    PIXEL_START_OFFSET = 16

    @classmethod
    def validate_filepath(cls, filepath: str):
        return filepath.endswith("idx1-ubyte")

    def buffer_to_arr(self) -> NDArray[np.uint8]:
        pixel_grid_dt = self.get_idx3_img_dt()

        self.buffer.seek(self.PIXEL_START_OFFSET)

        return np.frombuffer(self.buffer.read(), dtype=pixel_grid_dt)

    def get_idx3_img_dt(self):
        """Get the numpy data type of the MNIST image from the buffer."""
        self.buffer.seek(self.ROWS_OFFSET)
        num_rows = self.read_idx_int32()

        self.buffer.seek(self.COLS_OFFSET)
        num_cols = self.read_idx_int32()

        return np.dtype((np.ubyte, (num_rows, num_cols)))
