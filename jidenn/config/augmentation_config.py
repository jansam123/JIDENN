from dataclasses import dataclass


@dataclass
class AugmentationBase:
    prob: float
    name: str


@dataclass
class DropSoft(AugmentationBase):
    skew: float
    center_location: float
    min_number_consti: int


@dataclass
class Rotation(AugmentationBase):
    """Augmentation for rotation of the jets.

    Args:
        max_angle (int): Maximum angle in **degrees** to rotate the jets.
    """
    max_angle: int


@dataclass
class Boost(AugmentationBase):
    max_beta: float


@dataclass
class CollinearSplit(AugmentationBase):
    splitting_amount: float


@dataclass
class SoftSmear(AugmentationBase):
    energy_scale: float
