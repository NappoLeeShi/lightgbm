class Node:

    def __init__(
        self,
        X=None,
        gradients=None,
        hessians=None,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        gain=-float("inf")
    ):

        self.X = X

        self.gradients = gradients
        self.hessians = hessians

        self.feature = feature
        self.threshold = threshold

        self.left = left
        self.right = right

        self.value = value

        self.gain = gain

    def is_leaf(self):

        return (
            self.left is None
            and
            self.right is None
        )