# 01_forward_diffusion.py

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. Device: Apple Silicon MPS
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available. "
        "Make sure you are using Apple Silicon and an MPS-enabled PyTorch build."
    )

DEVICE = torch.device("mps")


# ============================================================
# 2. Hyperparameters
# ============================================================

T = 200

BETA_START = 1e-4
BETA_END = 2e-2


# ============================================================
# 3. Diffusion schedule
# ============================================================

# q(x_t | x_{t-1})
#
# = N(
#     sqrt(1 - beta_t) * x_{t-1},
#     beta_t * I
#   )

betas = torch.linspace(
    BETA_START,
    BETA_END,
    T,
    device=DEVICE,
)

# alpha_t = 1 - beta_t
alphas = 1.0 - betas

# alpha_bar_t
# = alpha_1 * alpha_2 * ... * alpha_t
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
# 4. Forward diffusion
# ============================================================

def q_sample(x0, t, noise=None):
    """
    Generate x_t directly from x_0.

    x_t
    =
    sqrt(alpha_bar_t) * x_0
    +
    sqrt(1 - alpha_bar_t) * epsilon

    epsilon ~ N(0, I)

    Parameters
    ----------
    x0 : Tensor
        Clean images.
        Shape: [B, C, H, W]

    t : LongTensor
        Timestep for each image.
        Shape: [B]

    noise : Tensor or None
        Gaussian noise epsilon.
        If None, generate fresh noise.

    Returns
    -------
    xt : Tensor
        Noisy image x_t.

    noise : Tensor
        Noise used to generate x_t.
    """

    if noise is None:
        noise = torch.randn_like(x0)

    # Pick alpha_bar_t for each sample
    #
    # [B] -> [B, 1, 1, 1]
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)

    signal_scale = torch.sqrt(alpha_bar_t)
    noise_scale = torch.sqrt(1.0 - alpha_bar_t)

    xt = (
        signal_scale * x0
        +
        noise_scale * noise
    )

    return xt, noise


# ============================================================
# 5. Load MNIST
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

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)


# ============================================================
# 6. Pick one clean image x_0
# ============================================================

x0, label = dataset[0]

# x0:
# [1, 28, 28]
#
# Add batch dimension:
# [1, 1, 28, 28]
x0 = x0.unsqueeze(0).to(DEVICE)


print(f"Device       : {DEVICE}")
print(f"Label        : {label}")
print(f"x0 shape     : {x0.shape}")
print(f"x0 device    : {x0.device}")

print()
print("Diffusion schedule")
print("-" * 50)

print(f"T                = {T}")
print(f"beta[0]          = {betas[0].item():.6f}")
print(f"beta[-1]         = {betas[-1].item():.6f}")
print(f"alpha_bar[0]     = {alpha_bars[0].item():.6f}")
print(f"alpha_bar[-1]    = {alpha_bars[-1].item():.6f}")


# ============================================================
# 7. Visualize forward diffusion
# ============================================================

timesteps = [
    0,
    25,
    50,
    100,
    150,
    199,
]


# Use one fixed epsilon so that only the effect of timestep
# is being compared.
fixed_noise = torch.randn_like(x0)


fig, axes = plt.subplots(
    1,
    len(timesteps),
    figsize=(15, 3),
)


for ax, t_value in zip(axes, timesteps):

    # Shape: [1]
    t = torch.tensor(
        [t_value],
        dtype=torch.long,
        device=DEVICE,
    )

    xt, _ = q_sample(
        x0=x0,
        t=t,
        noise=fixed_noise,
    )

    # matplotlib expects CPU data.
    #
    # MPS tensor -> CPU tensor
    image = xt[0].detach().cpu()

    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0

    image = image.clamp(
        min=0.0,
        max=1.0,
    )

    # [1, 28, 28] -> [28, 28]
    image = image.squeeze(0)

    alpha_bar = alpha_bars[t_value].item()

    signal_scale = alpha_bar ** 0.5
    noise_scale = (1.0 - alpha_bar) ** 0.5

    ax.imshow(
        image,
        cmap="gray",
    )

    ax.set_title(
        f"t={t_value}\n"
        f"signal={signal_scale:.2f}\n"
        f"noise={noise_scale:.2f}"
    )

    ax.axis("off")


plt.suptitle(
    f"MNIST Forward Diffusion "
    f"(label={label})"
)

plt.tight_layout()

plt.savefig(
    "forward_diffusion.png",
    dpi=200,
)

plt.show()


# ============================================================
# 8. Print signal / noise coefficients
# ============================================================

print()
print("Forward diffusion coefficients")
print("-" * 70)

for t_value in timesteps:

    alpha_bar = alpha_bars[t_value].item()

    signal_scale = alpha_bar ** 0.5
    noise_scale = (1.0 - alpha_bar) ** 0.5

    print(
        f"t={t_value:3d} | "
        f"alpha_bar={alpha_bar:.6f} | "
        f"signal={signal_scale:.4f} | "
        f"noise={noise_scale:.4f}"
    )