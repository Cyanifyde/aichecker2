from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_version: str = "v0.1"
    tau_ai: float = 0.90
    tau_ood: float = 0.60
    tau_unknown: float = 0.55


class ModelStore:
    def __init__(self, config: ModelConfig):
        self.config = config

    def info(self) -> ModelConfig:
        return self.config
