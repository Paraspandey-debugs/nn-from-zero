import os
import numpy as np

def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(-1, 28, 28)

def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(8)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data

def create_data_mnist(path):
    train_images_path = os.path.join(path, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte')

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    return X_train, y_train, X_test, y_test