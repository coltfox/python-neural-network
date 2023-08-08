from dataclasses import dataclass


@dataclass
class TestingResults:
    cost: float
    success_rate: float
    samples_trained: int

    def __repr__(self) -> str:
        success_rate_percent = str(self.success_rate).format(":.3%")
        return f"""Cost {self.cost}\nSuccess Rate: {success_rate_percent}\nSamples Trained{self.samples_trained}"""
