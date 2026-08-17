import numpy as np


def sigmoid(x):

    return 1/(1+np.exp(-x))


def compute_gradient(
    y,
    prob
):

    return prob-y


def compute_hessian(
    prob
):

    return prob*(1-prob)


def calculate_gain(
    GL,
    HL,
    GR,
    HR,
    lamda=1,
    gamma=0
):

    G = GL+GR
    H = HL+HR

    gain = 0.5 * (
        (GL**2)/(HL+lamda)
        +
        (GR**2)/(HR+lamda)
        -
        (G**2)/(H+lamda)
    ) - gamma

    return gain


def leaf_weight(
    G,
    H,
    lamda=1
):

    return -G/(H+lamda)