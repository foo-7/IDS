import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        """
            Kernel size (AKA filter size)
            - most important hyperparameters
            - a small matrix that convolves over the input data to detect patterns like edges, textures, or shapes
            - the number correlates of how many input features it will look at a time as it slides across the input
        
            Conv1d params:
                1. in_channels      - number of input channels
                2. out_channels     - number of filters/features learned
                        > more gives more capacity to learn complex features, but more computation and risk of overfitting
                3. kernel_size      - size of the filter sliding over input
        """
        self.convLayer1 = nn.Conv1d(1, 16, kernel_size=3)
        self.convLayer2 = nn.Conv1d(16, 32, kernel_size=3)

        # To prevent overfitting by randomly disabling some neurons
        # 20% dropout.
        self.conv2_drop = nn.Dropout1d(0.2)

        # Fully connected layer that takes 1st param input and outputs 2nd param.
        """
            nn.Linear is a fully connected (dense) layer in PyTorch
                - It connects every input neuron to every output neuron
                - It performs a linear transformation on the input data
                - It learns weights and biases during training to map inputs to outputs
            Linear() params:
                1. in_features      - number of input features (size of each input sample)
                2. out_features     - number of output features (number of neurons in this layer)

            How to determine? (ASK PROFESSOR AS WELL)
            After conv + pooling, suppose output shape is (batch_size, 64, 10)
            So, flatten size = 64 * 10 = 640?
        """

        # Input: 83 features
        # conv1: 83 - 3 + 1 = 81, pool: 81/2 = 40
        # conv2: 40 - 3 + 1 = 38, pool: 38/2 = 19
        # Flattened: 32 * 19 lenght = 608
        #self.fc1 = nn.Linear(608, 128)
        self.fc1 = None # will be set in forward pass
        self.fc2 = nn.Linear(128, 1)
        """
            Difference between nn.Linear() and nn.Conv#d
                nn.Linear performs matrix multiplication
                    - basic NN structure, MLP
                    - represents a dense layer

                nn.Conv#d performs convolution: slides a filter/kernel across the input
                    - Uses local connectivity - each output connects to a small region of input
                    - Learns filters/kernels that are much smaller than the full input
                    CONVOLUTION: combines two functions to produce a third function
                        > pattern matching with a sliding window
        """   

    """
        Forward function uses two different activation functions:
            1. Rectified Linear Unit Activation Function
                - Nonlinear function:
                    IF      x <= 0,     then y = 0
                    ELSE                then y = x
            2. Sigmoid Activation Function
                - Should use over SoftMax? No, it is used for multi-class classification
                    > Binary classification great for sigmoid
    """
    def forward(self, x):
        x = x.unsqueeze(1)
        # Find out the correct kernel size for the params
        """
            MaxPool allows the following:
                1. Prevents overfitting
                2. Keeps the stronngest signal, less sensitive to exact feature positions
                3. Computational efficiency
                4. Better generalization
                    - Downsampling helps model focus on important patterns
        """
        x = F.relu(F.max_pool1d(self.convLayer1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.convLayer2(x)), 2))

        # Unconventional, but I don't know what input to have for the first dense layer.
        fc1_input_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(fc1_input_size, 128)
        self.fc1 = self.fc1.to(x.device)

        # view() flattens your tensor from multi-dimensional to 2D
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))


        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)