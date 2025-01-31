import torch
import torch.nn as nn
import torch.nn.functional as F

class Flexi3DCNN(nn.Module):
    def __init__(self, in_channels, conv_channels, conv_kernel_sizes, num_classes, activation):
        super(Flexi3DCNN, self).__init__()
        self.num_conv_layers = len(conv_channels)

        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_conv_layers):
            if i == 0:
                conv_layer = nn.Conv3d(in_channels, conv_channels[i], kernel_size=conv_kernel_sizes[i], padding=1)
            else:
                conv_layer = nn.Conv3d(conv_channels[i-1], conv_channels[i], kernel_size=conv_kernel_sizes[i], padding=1)
            self.conv_layers.append(conv_layer)

        
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[-1] * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function
        self.act = nn.ReLU() if activation == 'ReLU' else nn.LeakyReLU()

    def forward(self, x, latent_f, weight_f):
        # Process input through convolutional layers
        for conv_layer in self.conv_layers:
            x = self.act(conv_layer(x))  # Apply convolution + activation
            x = self.pool(x)  # Use adaptive pooling to ensure size stability

        # Ensure latent_f matches spatial dimensions of x
        latent_f = F.interpolate(latent_f, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = weight_f * latent_f + x  # Element-wise addition

        # Flatten and pass through FC layers
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x