# 05_visualize.py

import torch
import matplotlib.pyplot as plt
from importlib import import_module


# ============================================================
# 1. Import model
# ============================================================

model_module = import_module("02_model")
TinyUNet = model_module.TinyUNet


# ============================================================
# 2. Device
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available.")

DEVICE = torch.device("mps")


# ============================================================
# 3. Load checkpoint
# ============================================================

CHECKPOINT_PATH = "./checkpoints/ddpm_mnist.pt"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

T = checkpoint["T"]
BETA_START = checkpoint["beta_start"]
BETA_END = checkpoint["beta_end"]


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
# 5. Load trained model
# ============================================================

model = TinyUNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 6. One reverse step
# ============================================================

@torch.no_grad()
def p_sample(model, xt, t):
    """
    Perform:

        x_t -> x_{t-1}
    """

    batch_size = xt.shape[0]

    # timestep tensor
    t_batch = torch.full(
        (batch_size,),
        t,
        device=DEVICE,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Predict epsilon
    #
    # epsilon_theta(x_t, t)
    # --------------------------------------------------------

    eps_pred = model(
        xt,
        t_batch,
    )


    # --------------------------------------------------------
    # Current diffusion coefficients
    # --------------------------------------------------------

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]


    # --------------------------------------------------------
    # Reverse mean
    #
    # mu_theta(x_t, t)
    #
    # =
    #
    # 1 / sqrt(alpha_t)
    #
    # *
    #
    # (
    #   x_t
    #
    #   -
    #
    #   beta_t / sqrt(1-alpha_bar_t)
    #
    #   * epsilon_theta
    # )
    # --------------------------------------------------------

    model_mean = (
        1.0 / torch.sqrt(alpha_t)
    ) * (
        xt
        -
        (
            beta_t
            / torch.sqrt(1.0 - alpha_bar_t)
        )
        * eps_pred
    )


    # --------------------------------------------------------
    # Final step
    # --------------------------------------------------------

    if t == 0:
        return model_mean, eps_pred


    # --------------------------------------------------------
    # Gaussian noise for reverse sampling
    # --------------------------------------------------------

    z = torch.randn_like(xt)

    sigma_t = torch.sqrt(beta_t)


    # --------------------------------------------------------
    # x_{t-1}
    # --------------------------------------------------------

    x_prev = (
        model_mean
        +
        sigma_t * z
    )

    return x_prev, eps_pred


# ============================================================
# 7. Generate full trajectory
# ============================================================

@torch.no_grad()
def generate_trajectory(model):
    """
    Generate a single MNIST sample and save
    every reverse diffusion step.
    """

    # --------------------------------------------------------
    # Start from pure Gaussian noise
    #
    # x_T ~ N(0, I)
    # --------------------------------------------------------

    x = torch.randn(
        1,
        1,
        28,
        28,
        device=DEVICE,
    )


    trajectory = []
    predicted_noises = []


    # Save initial x_T
    trajectory.append(
        x.detach()
        .cpu()
        .clone()
    )


    # --------------------------------------------------------
    # Reverse diffusion
    #
    # T-1 -> ... -> 0
    # --------------------------------------------------------

    for t in reversed(range(T)):

        x, eps_pred = p_sample(
            model,
            x,
            t,
        )

        trajectory.append(
            x.detach()
            .cpu()
            .clone()
        )

        predicted_noises.append(
            eps_pred.detach()
            .cpu()
            .clone()
        )


    return trajectory, predicted_noises


# ============================================================
# 8. Run sampling
# ============================================================

trajectory, predicted_noises = generate_trajectory(
    model
)


# ============================================================
# 9. Select timesteps to visualize
# ============================================================

# trajectory[0] corresponds approximately to x_T
#
# trajectory[1] corresponds to x_{T-1}
# ...
#
# trajectory[-1] corresponds to x_0

num_images = 10

indices = torch.linspace(
    0,
    len(trajectory) - 1,
    num_images,
).long()


# ============================================================
# 10. Plot denoising trajectory
# ============================================================

fig, axes = plt.subplots(
    1,
    num_images,
    figsize=(18, 3),
)


for ax, idx in zip(
    axes,
    indices,
):

    image = trajectory[idx][0]

    # [-1, 1] -> [0, 1]
    image = (
        image + 1.0
    ) / 2.0

    image = image.clamp(
        0.0,
        1.0,
    )

    image = image.squeeze(0)


    # Map trajectory index to approximate t
    if idx == 0:
        timestep = T
    else:
        timestep = T - idx.item()


    ax.imshow(
        image,
        cmap="gray",
    )

    ax.set_title(
        f"t={timestep}"
    )

    ax.axis("off")


plt.suptitle(
    "Reverse Diffusion Trajectory"
)

plt.tight_layout()

plt.savefig(
    "denoising_trajectory.png",
    dpi=200,
)

plt.show()


# ============================================================
# 11. Visualize predicted epsilon
# ============================================================

noise_indices = torch.linspace(
    0,
    len(predicted_noises) - 1,
    num_images,
).long()


fig, axes = plt.subplots(
    1,
    num_images,
    figsize=(18, 3),
)


for ax, idx in zip(
    axes,
    noise_indices,
):

    noise = predicted_noises[idx][0]

    noise = noise.squeeze(0)

    timestep = (
        T - 1 - idx.item()
    )

    ax.imshow(
        noise,
        cmap="gray",
    )

    ax.set_title(
        f"eps_pred\n"
        f"t={timestep}"
    )

    ax.axis("off")


plt.suptitle(
    "Predicted Noise During Reverse Diffusion"
)

plt.tight_layout()

plt.savefig(
    "predicted_noise_trajectory.png",
    dpi=200,
)

plt.show()


# ============================================================
# 12. Plot image statistics over time
# ============================================================

means = []
stds = []

for x in trajectory:

    means.append(
        x.mean().item()
    )

    stds.append(
        x.std().item()
    )


plt.figure(
    figsize=(8, 4)
)

plt.plot(
    means,
    label="mean"
)

plt.plot(
    stds,
    label="std"
)

plt.xlabel(
    "Reverse diffusion step"
)

plt.ylabel(
    "Value"
)

plt.title(
    "Image Statistics During Reverse Diffusion"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "trajectory_statistics.png",
    dpi=200,
)

plt.show()