## Small-World Dynamic Convolutional Neural Network
This project proposes a novel CNN architecture that integrates dynamic convolution with a Watts–Strogatz small-world topology. Instead of sequential layers, the model uses a graph-based DAG where each node is a convolutional unit. Dynamic convolution is applied within these nodes, enabling adaptive computation on input features while preserving efficiency.

## Key Highlights:

- Small-world CNN modules with graph-based connectivity.

- Channel-wise dynamic masking for adaptive convolution.

- Achieved 85.45% accuracy on CIFAR-10 in 50 epochs with only ~49.7% average channel usage.
