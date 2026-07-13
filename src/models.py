import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Update gate
        self.conv_z = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # Reset gate
        self.conv_r = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # Candidate hidden state
        self.conv_h = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor
    ):
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)

        # Update gate
        z = torch.sigmoid(self.conv_z(combined))

        # Reset gate
        r = torch.sigmoid(self.conv_r(combined))

        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_reset))

        # New hidden state
        h = (1 - z) * h_prev + z * h_tilde

        return h
                
class ConvGRU(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = ConvGRUCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

    def forward(
        self,
        x: torch.Tensor
    ):
        assert x.ndim == 5
        B, T, C, H, W = x.shape

        # Initialize hidden state
        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        for t in range(T):
            h = self.cell(x[:, t], h)

        return h

class ConvGRURegressor(nn.Module):
    
    def __init__(
        self, 
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        head_channels: list[int],
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()

        self.activation = activation

        self.encoder = ConvGRU(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

        layers = []
        in_channels = hidden_channels

        for out_channels in head_channels:

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(self.activation())
            in_channels = out_channels

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1
            )
        )

        layers.append(nn.Sigmoid())

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
    