import numpy as np


def accuracy(
    y_true,
    y_pred
):

    return np.mean(
        y_true==y_pred
    )


def confusion_matrix(
    y_true,
    y_pred
):

    tp = np.sum(
        (y_true==1)
        &
        (y_pred==1)
    )

    tn = np.sum(
        (y_true==0)
        &
        (y_pred==0)
    )

    fp = np.sum(
        (y_true==0)
        &
        (y_pred==1)
    )

    fn = np.sum(
        (y_true==1)
        &
        (y_pred==0)
    )

    return tp,tn,fp,fn