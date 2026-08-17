import numpy as np


def goss_sampling(
    X,
    gradients,
    hessians,
    top_rate=0.2,
    other_rate=0.1
):

    n = len(X)

    abs_grad = np.abs(
        gradients
    )

    sorted_idx = np.argsort(
        -abs_grad
    )

    top_n = int(
        n * top_rate
    )

    top_idx = sorted_idx[:top_n]

    rest_idx = sorted_idx[top_n:]

    sample_n = int(
        len(rest_idx)
        * other_rate
    )

    sampled_idx = np.random.choice(
        rest_idx,
        sample_n,
        replace=False
    )

    weight = (
        (1-top_rate)
        /
        other_rate
    )

    gradients_sampled = (
        gradients[sampled_idx]
        * weight
    )

    hessians_sampled = (
        hessians[sampled_idx]
        * weight
    )

    final_idx = np.concatenate(
        [
            top_idx,
            sampled_idx
        ]
    )

    final_gradients = np.concatenate(
        [
            gradients[top_idx],
            gradients_sampled
        ]
    )

    final_hessians = np.concatenate(
        [
            hessians[top_idx],
            hessians_sampled
        ]
    )

    return (
        X[final_idx],
        final_gradients,
        final_hessians
    )