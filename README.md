# nn-from-zero
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python-based implementation of a neural network developed from first principles, inspired by
**"Neural Networks from Scratch in Python"** by Harrison Kinsley and Daniel Kukieła. This project provides a transparent, educational exploration of neural network mechanics without relying on high-level deep learning frameworks such as TensorFlow or PyTorch.

---
![1_2n62cyhUZNqYql_eBJIO5Q](https://github.com/user-attachments/assets/73fb4089-1909-4557-9aed-32d51d8c060d)


## Table of Contents
1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Example Architecture](#example-architecture)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## About the Project
The objective of this repository is to present a clear, modular, and well-documented implementation of a feedforward neural network. It emphasizes understanding over abstraction, ensuring that each computational step—from forward propagation to backpropagation—is explicitly defined and traceable. The implementation uses the MNIST dataset as an illustrative example for model training and evaluation.

---

## Features
- **Dense Layers:** Fully connected layers with explicit weight and bias initialization.
- **Activation Functions:**
  - ReLU (Rectified Linear Unit) for hidden layers.
  - Softmax for output layers in classification tasks.
- **Loss Functions:**
  - Categorical Cross-Entropy for multi-class classification.
- **Optimizers:**
  - Stochastic Gradient Descent (SGD).
  - Adam optimizer with learning rate decay.
- **Training Algorithm:**
  - Complete backpropagation implementation for parameter updates.
- **Data Handling Utilities:**
  - Functions for acquisition, extraction, and preprocessing of the MNIST dataset.

---

## Project Structure
```text
nn-from-zero/
├── data/                # MNIST dataset storage
├── layers.py            # Dense layers
├── activations.py       # Activation functions (ReLU, Softmax)
├── losses.py            # Loss functions
├── optimizers.py        # Optimizers (SGD, Adam)
├── main.py              # Training script
├── utils.py             # Data preprocessing helpers
└── README.md
```

---

## Getting Started

### Prerequisites
Ensure Python 3 is installed along with the required dependencies:
```bash
pip install numpy matplotlib opencv-python
```

### Clone the Repository
```bash
git clone https://github.com/your_username/nn-from-zero.git
cd nn-from-zero
```

### Run the Main Script
```bash
python main.py
```

---

## Usage

### Model Definition
```python
# --- Model Definition ---
dense1 = Layer_Dense(X.shape[1], 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# --- Optimizer Configuration ---
optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-5)
```

### Training Loop
```python
for epoch in range(EPOCHS):
    # Forward Pass
    dense1.forward(batch_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, batch_y)

    # Backward Pass
    loss_activation.backward(loss_activation.output, batch_y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Parameter Updates
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
```

---

## Example Architecture
```text
Input Layer (784 neurons - MNIST pixels)
        ↓
Dense Layer (128) -> ReLU
        ↓
Dense Layer (64) -> ReLU
        ↓
Dense Layer (10) -> Softmax
        ↓
Categorical Cross-Entropy Loss
```

---

## Acknowledgments
- *Neural Networks from Scratch in Python* by Harrison Kinsley and Daniel Kukieła.
- MNIST handwritten digits dataset.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
