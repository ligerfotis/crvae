from torch import nn


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, color_channels, output_size):
        super().__init__()
        out_channel = color_channels
        self.decoder_lin_1 = nn.Sequential(nn.Linear(encoded_space_dim, 4 * 4 * 256), nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))
        if output_size == 32:
            output_padding = [1, 1, 1]
            padding = [1, 1, 1]
        else:
            output_padding = [0, 0, 1]
            padding = [1, 1, 0]
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=padding[0],
                               output_padding=output_padding[0]),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=padding[1],
                               output_padding=output_padding[1]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=padding[2],
                               output_padding=output_padding[2]),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channel, 3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.decoder_lin_1(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.sigmoid(x)
        return x
