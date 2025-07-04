class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.learning_rate * g
