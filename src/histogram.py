import numpy as np


def build_histogram(
    feature,
    gradients,
    hessians,
    max_bins=16
):

    bins = np.linspace(
        np.min(feature),
        np.max(feature),
        max_bins + 1
    )

    bin_ids = np.digitize(
        feature,
        bins
    )

    histogram = []

    for b in range(
        1,
        max_bins + 1
    ):

        mask = (
            bin_ids == b
        )

        G = np.sum(
            gradients[mask]
        )

        H = np.sum(
            hessians[mask]
        )

        histogram.append(
            {
                "G": G,
                "H": H,
                "count": np.sum(mask)
            }
        )

    return histogram, bins