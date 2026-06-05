import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym # Assuming Gymnasium is available in the environment

class DQN(nn.Module):
    """
    Nature-style Convolutional Neural Network architecture for Deep Q-Learning.
    Accepts a state tensor (stacked frames) and outputs action Q-values.

    Input Shape: (batch_size, 4, 84, 84)
    Output Units: Matches the number of discrete actions in the environment.
    """
    def __init__(self, action_space_size):
        """
        Initializes the DQN model.

        Args:
            action_space_size (int): The total number of discrete actions available 
                                     in the target environment's action space.
        """
        super(DQN, self).__init__()
        self.action_space_size = action_space_size

        # 1. Conv2d: 32 filters, 8x8 kernel, stride 4
        self.conv1 = nn.Conv2d(
            in_channels=4, 
            out_channels=32, 
            kernel_size=8, 
            stride=4
        )

        # 2. Conv2d: 64 filters, 4x4 kernel, stride 2
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=4, 
            stride=2
        )

        # 3. Conv2d: 64 filters, 3x3 kernel, stride 1
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=3, 
            stride=1
        )

        # Calculate the input size for the first fully connected layer.
        # Based on standard DQN architecture with 84x84 input: (7 * 7 * 64)
        self.linear_input_size = 64 * 7 * 7 

        # 5. Linear (Hidden): 512 hidden units.
        self.fc1 = nn.Linear(self.linear_input_size, 512)

        # 6. Linear (Output): Match action space size.
        self.fc2 = nn.Linear(512, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input state tensor. Expected shape (B, 4, H, W).

        Returns:
            torch.Tensor: Q-values for all actions. Shape (B, A), where A is action space size.
        """
        # Conv Layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        # Flatten and Linear Layers
        batch_size = x.size(0)
        x = torch.flatten(x, 1) # flatten all dimensions except batch dimension
        
        x = self.fc1(x)
        x = F.relu(x)
        
        q_values = self.fc2(x)
        return q_values