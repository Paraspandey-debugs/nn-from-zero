class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.01):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_params(self, layer):
        if hasattr(layer, 'dweights') and hasattr(layer, 'dbiases'):
            layer.weights += -self.current_learning_rate * layer.dweights
            layer.biases += -self.current_learning_rate * layer.dbiases

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1