# README.md

# Neural Network from Scratch — MNIST Digit Classification

A fully connected deep neural network implementation built from scratch using only NumPy (no TensorFlow or PyTorch).
Trained and tested on the MNIST handwritten digits dataset for classification (0–9).

---

## Features

- Fully vectorized forward and backward propagation
- He initialization for stable weight distribution
- ReLU and Softmax activations
- Cross-Entropy Loss for multi-class classification
- Mini-batch gradient descent training
- Custom implementation of accuracy tracking
- Clean, minimal code — ideal for learning how neural nets work internally

---

## Network Architecture

Input (784) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)

You can easily modify the number of layers or neurons by changing `layer_dims`:

nn = NeuralNetwork([784, 128, 128, 64, 10])

---

## Dependencies

Install dependencies:

pip install numpy pandas

---

## Dataset

Uses the MNIST dataset in CSV format.
If you don’t have it yet, download from:

https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Place the files in the same directory as your script:

mnist_train.csv
mnist_test.csv

---

## Training

nn.train(x_train, y_train, epochs=50, batch_size=32, lr=0.01)

This will:
- Shuffle and batch the data
- Run forward and backward propagation
- Update weights using gradient descent
- Print training loss and accuracy per epoch

Example output:

Epoch 1/50, Loss: 1.9653, Acc: 61.20%
Epoch 2/50, Loss: 1.3248, Acc: 74.88%
...
Epoch 50/50, Loss: 0.2421, Acc: 93.67%

---

## Testing Accuracy

accuracy = nn.accuracy(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

---

## Code Structure

neural_network.py        # Core neural net class implementation
mnist_train.csv          # Training data
mnist_test.csv           # Testing data
README.md                # Documentation

---

## Key Components Explained

| Method | Purpose |
|--------|----------|
| forward_pass(X) | Computes activations layer by layer |
| backward_pass(y_true) | Calculates gradients and updates weights |
| cross_entropy_loss() | Measures prediction error |
| relu() / softmax() | Activation functions |
| train() | Training loop with batching and accuracy logging |
| accuracy() | Evaluates classification accuracy |

---

## Learning Outcome

This project helps you understand:
- How forward and backward propagation really work
- How gradients flow through deep layers
- How neural networks learn using loss and gradient descent

Perfect for anyone who wants to build neural networks from scratch and grasp the math behind deep learning.

---

## Example Results

| Dataset | Accuracy |
|----------|-----------|
| Training | ~93–95% |
| Testing  | ~92–94% |

*(May vary based on random initialization and learning rate.)*

---

## Future Improvements

- Add Adam optimizer
- Add dropout and L2 regularization
- Implement save/load model functionality
- Visualize training curves with Matplotlib

---

## Author

Your Name
GitHub: https://github.com/yourusername
Contributions and feedback are welcome!
