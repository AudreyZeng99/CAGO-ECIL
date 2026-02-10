import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(MyMLP, self).__init__()
        # First, flatten the input in the forward method instead of using a layer here
        # Define the first dense layer (128 units, relu activation)
        self.fc1 = nn.Linear(28 * 28 * input_channels, 128)  # 28*28 from image dimensions, *input_channels is typically 1 for MNIST
        # Define dropout layer
        self.dropout = nn.Dropout(0.2)
        # Define the output layer (10 units for the 10 classes of MNIST)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28*28)  # Flatten the 28x28 image into a 784-dimensional vector
        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Output layer (no activation; logits are returned)
        x = self.fc2(x)
        return x
