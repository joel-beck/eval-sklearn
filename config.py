from dataclasses import dataclass


@dataclass
class PublicConfig:
    NUM_SAMPLES_NOTEBOOKS: int = 1000
    NUM_SAMPLES_TESTING: int = 50
    NUM_FEATURES: int = 20
    NUM_TARGETS: int = 2
    NUM_CLUSTERS: int = 3
    SEED: int = 123
    TARGET_COL: str = "y"
