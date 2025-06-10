import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
    

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # (3, 128, 128) -> (64, 64, 64)
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (64, 64, 64) -> (128, 32, 32)
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (128, 32, 32) -> (256, 16, 16)
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # (256, 16, 16) -> (512, 8, 8)
            nn.ReLU(inplace=True),
            ResidualBlock(512),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), # (512, 8, 8) -> (512, 4, 4)
            nn.ReLU(inplace=True),
            ResidualBlock(512),
        )

        self.flatten = nn.Flatten() # (512, 4, 4) -> (512 * 4 * 4) = (8192,)
        self.fc = nn.Linear(512 * 4 * 4, latent_dim) # (8192,) -> (latent_dim,)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4) # (latent_dim,) -> (512 * 4 * 4,) = (8192,)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), # (512, 4, 4) -> (512, 8, 8)
            nn.ReLU(inplace=True),
            ResidualBlock(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # (512, 8, 8) -> (256, 16, 16)
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (256, 16, 16) -> (128, 32, 32)
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (128, 32, 32) -> (64, 64, 64)
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # (64, 64, 64) -> (3, 128, 128)
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        x = self.tanh(x)
        return x
    

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        emb = self.encoder(x)
        x_reconstructed = self.decoder(emb)
        return x_reconstructed, emb


if __name__ == "__main__":
    model = Autoencoder(latent_dim=128)
    print(model)

    x = torch.randn(1, 3, 128, 128)  # Example input
    reconstructed, embedding = model(x)
    print(reconstructed.shape, embedding.shape)  # Should print: torch.Size([1, 3, 128, 128]) torch.Size([1, 128])
