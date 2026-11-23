# 03_train.py

import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from importlib import import_module


# ============================================================
# 1. Import model from 02_model.py
# ============================================================

model_module = import_module("02_model")
TinyUNet = model_module.TinyUNet


# ============================================================
# 2. Device
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available. "
        "Make sure you are using Apple Silicon "
        "and an MPS-enabled PyTorch build."
    )

DEVICE = torch.device("mps")


# ============================================================
# 3. Hyperparameters
# ============================================================

T = 200

BETA_START = 1e-4
BETA_END = 2e-2

BATCH_SIZE = 128
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10

CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "ddpm_mnist.pt"
)


# ============================================================
# 4. Diffusion schedule
# ============================================================

betas = torch.linspace(
    BETA_START,
    BETA_END,
    T,
    device=DEVICE,
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0,
)


# ============================================================
# 5. Forward diffusion
# ============================================================

def q_sample(x0, t, noise=None):
    """
    Sample x_t directly from x_0.

    x_t =
        sqrt(alpha_bar_t) * x_0
        +
        sqrt(1 - alpha_bar_t) * epsilon

    Parameters
    ----------
    x0 : Tensor
        [B, C, H, W]

    t : LongTensor
        [B]

    noise : Tensor or None
        Gaussian noise epsilon.

    Returns
    -------
    xt : Tensor
        Noisy image.

    noise : Tensor
        The exact epsilon used to generate xt.
    """

    if noise is None:
        noise = torch.randn_like(x0)

    # alpha_bar_t:
    #
    # [B]
    # ->
    # [B, 1, 1, 1]
    alpha_bar_t = alpha_bars[t].view(
        -1,
        1,
        1,
        1,
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar_t
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar_t
    )

    xt = (
        sqrt_alpha_bar * x0
        +
        sqrt_one_minus_alpha_bar * noise
    )

    return xt, noise


# ============================================================
# 6. Dataset
# ============================================================

transform = transforms.Compose(
    [
        transforms.ToTensor(),

        # [0, 1] -> [-1, 1]
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,),
        ),
    ]
)

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)


# ============================================================
# 7. Model
# ============================================================

model = TinyUNet().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 8. Print setup
# ============================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("=" * 60)
print("DDPM Training")
print("=" * 60)

print(f"Device         : {DEVICE}")
print(f"Dataset size   : {len(train_dataset)}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Epochs         : {NUM_EPOCHS}")
print(f"Timesteps T    : {T}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Parameters     : {num_params:,}")

print("=" * 60)


# ============================================================
# 9. Training loop
# ============================================================

model.train()

for epoch in range(NUM_EPOCHS):

    total_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for x0, _ in progress_bar:

        # ----------------------------------------------------
        # Step 1:
        # Clean image x_0
        # ----------------------------------------------------

        x0 = x0.to(DEVICE)

        batch_size = x0.shape[0]


        # ----------------------------------------------------
        # Step 2:
        # Sample random timestep for EACH image
        #
        # t ~ Uniform{0, ..., T-1}
        #
        # Shape:
        # [B]
        # ----------------------------------------------------

        t = torch.randint(
            low=0,
            high=T,
            size=(batch_size,),
            device=DEVICE,
            dtype=torch.long,
        )


        # ----------------------------------------------------
        # Step 3:
        # Generate ground-truth Gaussian noise
        #
        # epsilon ~ N(0, I)
        # ----------------------------------------------------

        noise = torch.randn_like(x0)


        # ----------------------------------------------------
        # Step 4:
        # Construct x_t directly
        #
        # x_t =
        # sqrt(alpha_bar_t) x_0
        # +
        # sqrt(1-alpha_bar_t) epsilon
        # ----------------------------------------------------

        xt, noise = q_sample(
            x0=x0,
            t=t,
            noise=noise,
        )


        # ----------------------------------------------------
        # Step 5:
        # Predict the noise
        #
        # epsilon_theta(x_t, t)
        # ----------------------------------------------------

        noise_pred = model(
            xt,
            t,
        )


        # ----------------------------------------------------
        # Step 6:
        # DDPM simple loss
        #
        # || epsilon - epsilon_theta ||^2
        # ----------------------------------------------------

        loss = F.mse_loss(
            noise_pred,
            noise,
        )


        # ----------------------------------------------------
        # Step 7:
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # End of epoch
    # ========================================================

    avg_loss = (
        total_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d} "
        f"| Average Loss: {avg_loss:.6f}"
    )


# ============================================================
# 10. Save model
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "T": T,

        "beta_start": BETA_START,
        "beta_end": BETA_END,
    },
    CHECKPOINT_PATH,
)

print()
print("=" * 60)
print("Training finished.")
print(f"Checkpoint saved to:")
print(CHECKPOINT_PATH)
print("=" * 60)