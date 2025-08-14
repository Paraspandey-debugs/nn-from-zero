import numpy as np
from data.mnist_loader import load_mnist
from layers.dense import Layer_Dense
from layers.activation import Activation_ReLU, Activation_Softmax
from loss.categorical_crossentropy import Loss_CategoricalCrossentropy
from optimizers.sgd import Optimizer_SGD
from optimizers.adam import Optimizer_Adam

def train():
    # Load the MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist('data/mnist')

    # Normalize the data
    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Initialize layers
    dense1 = Layer_Dense(X_train.shape[1], 128)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(128, 64)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(64, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Initialize optimizer
    optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-5)

    # Training parameters
    EPOCHS = 90
    BATCH_SIZE = 128

    for epoch in range(EPOCHS):
        # Shuffle the training data
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Training loop
        for step in range(0, X_train.shape[0], BATCH_SIZE):
            batch_X = X_train[step:step + BATCH_SIZE]
            batch_y = y_train[step:step + BATCH_SIZE]

            # Forward pass
            dense1.forward(batch_X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)

            # Calculate loss
            loss = loss_activation.forward(dense3.output, batch_y)

            # Backward pass
            loss_activation.backward(loss_activation.output, batch_y)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # Update parameters
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
            optimizer.post_update_params()

        # Print epoch results
        print(f'Epoch {epoch + 1}/{EPOCHS} completed.')

if __name__ == '__main__':
    train()