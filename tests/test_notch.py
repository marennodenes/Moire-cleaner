import matplotlib.pyplot as plt

from moire.notch import (
    peak_to_coordinates,
    gaussian_notch
)


class DummyPeak:
    def __init__(self, fy, fx):
        self.fy = fy
        self.fx = fx


def main():
    peak = DummyPeak(10, 15)

    p1, p2 = peak_to_coordinates(
        (64, 64),
        peak
    )

    print("Peak:", p1)
    print("Mirror peak:", p2)

    assert p1 == (42, 47)
    assert p2 == (22, 17)

    filter_mask = gaussian_notch(
        (64, 64),
        p1,
        sigma=5
    )

    assert filter_mask.shape == (64, 64)
    plt.imshow(filter_mask, cmap="gray")
    plt.colorbar()
    plt.title("Gaussian Notch Filter")
    plt.show()

    print("All tests passed!")
    
if __name__ == "__main__":
    main()