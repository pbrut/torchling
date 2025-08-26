class MSE:
    """Mean Squared Error Loss"""
    def __call__(self, logits, Y):
        return ((logits - Y) ** 2).mean()

class CCE:
    """Categorical Cross Entropy Loss"""
    def __call__(self, logits, Y):
        return logits.cce(Y)
