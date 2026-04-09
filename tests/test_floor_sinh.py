import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.floor_sinh_activation import FloorSinhActivation


def test_floor_sinh_regularizer() -> None:
    """Validates Enhancement #28: Ensures that tensors track onto Hyperbolic Sine limits natively
    while structurally locking severe negative anomalies smoothly onto the physical floor limit.
    """
    # Create an arbitrary tensor matrix passing through Positive, Neutral, and severely Negative zones
    raw_tensor = torch.tensor(
        [
            [2.0, 0.5, 0.0],
            [
                -0.5,
                -2.0,
                -100.0,
            ],  # The -100 is an anomaly to hit the physical structural floor bounds
        ]
    )

    physical_floor = -1.0

    layer = FloorSinhActivation(physical_floor=physical_floor)

    activated = layer(raw_tensor)

    # Validation A: Positive bounds must equal math.sinh natively
    assert torch.isclose(activated[0, 0], torch.tensor(math.sinh(2.0))), (
        "Positive Sinh mathematical trajectory corrupted natively."
    )

    # Validation B: Negative bounds structurally lock strictly to the boundary constraint floor (-1.0)
    # math.sinh(-100.0) is effectively -infinity, but the layer must catch it safely on the boundary.
    assert activated[1, 2] == physical_floor, (
        "Hyperbolic floor parameter logically failed to catch severe anomaly vectors."
    )

    # Validation C: Structural shape fidelity
    assert activated.shape == raw_tensor.shape, (
        "Activation map fractured underlying tensor structural shapes."
    )

    print(
        "Test passed: Floor-Sinh Regularizer passed natural tensors geometrically and firmly gripped anomalous matrices dynamically onto the floor boundary."
    )


if __name__ == "__main__":
    test_floor_sinh_regularizer()
