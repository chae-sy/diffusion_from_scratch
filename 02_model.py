# 02_model.py

import math
import torch
import torch.nn as nn


# ============================================================
# 1. Sinusoidal timestep embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Convert timestep t into a vector embedding.

    Input:
        t: [B]

    Output:
        embedding: [B, dim]

    Similar idea to Transformer positional encoding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device

        half_dim = self.dim // 2

        # Frequencies:
        # exp(-log(10000) * i / (half_dim - 1))
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=device,
                dtype=torch.float32
            )
            / (half_dim - 1)
        )

        # t: [B]
        # frequencies: [half_dim]
        #
        # -> [B, half_dim]
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1
        )

        return embedding


# ============================================================
# 2. Basic convolution block
# ============================================================

class ConvBlock(nn.Module):
    """
    Simple block:

        Conv
        -> GroupNorm
        -> SiLU
        -> add time embedding
        -> Conv
        -> GroupNorm
        -> SiLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.act = nn.SiLU()

        # Project timestep embedding
        # [B, time_emb_dim]
        # ->
        # [B, out_channels]
        self.time_proj = nn.Linear(
            time_emb_dim,
            out_channels,
        )

    def forward(self, x, time_emb):
        # ----------------------------------------------------
        # First conv
        # ----------------------------------------------------

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # ----------------------------------------------------
        # Add timestep information
        #
        # [B, out_channels]
        # ->
        # [B, out_channels, 1, 1]
        #
        # Then broadcast over H, W
        # ----------------------------------------------------

        t = self.time_proj(time_emb)

        t = t[:, :, None, None]

        h = h + t

        # ----------------------------------------------------
        # Second conv
        # ----------------------------------------------------

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h


# ============================================================
# 3. Tiny U-Net
# ============================================================

class TinyUNet(nn.Module):
    """
    Input:
        x_t : [B, 1, 28, 28]
        t   : [B]

    Output:
        predicted noise epsilon_theta:
        [B, 1, 28, 28]
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Timestep embedding
        # ----------------------------------------------------

        time_dim = 128

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        # 28x28
        self.enc1 = ConvBlock(
            in_channels=1,
            out_channels=32,
            time_emb_dim=time_dim,
        )

        # 28x28 -> 14x14
        self.down1 = nn.Conv2d(
            32,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 14x14
        self.enc2 = ConvBlock(
            in_channels=64,
            out_channels=64,
            time_emb_dim=time_dim,
        )

        # 14x14 -> 7x7
        self.down2 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        # 7x7
        self.bottleneck = ConvBlock(
            in_channels=128,
            out_channels=128,
            time_emb_dim=time_dim,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        # 7x7 -> 14x14
        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Skip connection:
        #
        # up1: 64 channels
        # enc2: 64 channels
        #
        # concat -> 128 channels
        self.dec1 = ConvBlock(
            in_channels=128,
            out_channels=64,
            time_emb_dim=time_dim,
        )

        # 14x14 -> 28x28
        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Skip connection:
        #
        # up2: 32 channels
        # enc1: 32 channels
        #
        # concat -> 64 channels
        self.dec2 = ConvBlock(
            in_channels=64,
            out_channels=32,
            time_emb_dim=time_dim,
        )

        # ----------------------------------------------------
        # Final noise prediction
        #
        # [B, 32, 28, 28]
        # ->
        # [B, 1, 28, 28]
        # ----------------------------------------------------

        self.out = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(self, x, t):

        # ====================================================
        # Time embedding
        # ====================================================

        time_emb = self.time_embedding(t)

        # ====================================================
        # Encoder
        # ====================================================

        # [B, 1, 28, 28]
        x1 = self.enc1(
            x,
            time_emb,
        )

        # [B, 32, 28, 28]
        #
        # ->
        #
        # [B, 64, 14, 14]
        x2 = self.down1(x1)

        x2 = self.enc2(
            x2,
            time_emb,
        )

        # [B, 64, 14, 14]
        #
        # ->
        #
        # [B, 128, 7, 7]
        x3 = self.down2(x2)

        # ====================================================
        # Bottleneck
        # ====================================================

        x3 = self.bottleneck(
            x3,
            time_emb,
        )

        # ====================================================
        # Decoder
        # ====================================================

        # [B, 128, 7, 7]
        #
        # ->
        #
        # [B, 64, 14, 14]
        h = self.up1(x3)

        # Skip connection
        #
        # [B, 64, 14, 14]
        # +
        # [B, 64, 14, 14]
        #
        # concat:
        #
        # [B, 128, 14, 14]
        h = torch.cat(
            [h, x2],
            dim=1,
        )

        h = self.dec1(
            h,
            time_emb,
        )

        # [B, 64, 14, 14]
        #
        # ->
        #
        # [B, 32, 28, 28]
        h = self.up2(h)

        # Skip connection
        #
        # [B, 32, 28, 28]
        # +
        # [B, 32, 28, 28]
        #
        # ->
        #
        # [B, 64, 28, 28]
        h = torch.cat(
            [h, x1],
            dim=1,
        )

        h = self.dec2(
            h,
            time_emb,
        )

        # ====================================================
        # Predict epsilon
        # ====================================================

        eps_pred = self.out(h)

        return eps_pred


# ============================================================
# 4. Test model
# ============================================================

if __name__ == "__main__":

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available."
        )

    device = torch.device("mps")

    model = TinyUNet().to(device)

    # Fake batch
    batch_size = 8

    x = torch.randn(
        batch_size,
        1,
        28,
        28,
        device=device,
    )

    # Random timesteps
    t = torch.randint(
        low=0,
        high=200,
        size=(batch_size,),
        device=device,
    )

    with torch.no_grad():
        eps_pred = model(x, t)

    print("Device:", device)
    print("Input shape :", x.shape)
    print("t shape     :", t.shape)
    print("Output shape:", eps_pred.shape)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters  : {num_params:,}")