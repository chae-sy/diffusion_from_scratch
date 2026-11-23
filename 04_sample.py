# 04_sample.py

import math
import torch
import matplotlib.pyplot as plt
from importlib import import_module


# ============================================================
# 1. Import model
# ============================================================

model_module = import_module("02_model")
TinyUNet = model_module.TinyUNet


# ============================================================
# 2. Device: Apple Silicon MPS
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available.")

DEVICE = torch.device("mps")


# ============================================================
# 3. Checkpoint
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

print("=" * 60)
print("DDPM Sampling")
print("=" * 60)
print(f"Device      : {DEVICE}")
print(f"T           : {T}")
print(f"Checkpoint  : {CHECKPOINT_PATH}")
print("=" * 60)


# ============================================================
# 6. One reverse diffusion step
# ============================================================

@torch.no_grad()
def p_sample(model, xt, t):
    """
    One reverse diffusion step:

        x_t -> x_{t-1}

    DDPM mean:

        mu_theta(x_t, t)
        =
        1 / sqrt(alpha_t)
        *
        (
            x_t
            -
            beta_t / sqrt(1 - alpha_bar_t)
            * epsilon_theta(x_t, t)
        )

    Then sample:

        x_{t-1}
        =
        mu_theta
        +
        sigma_t * z

    Here we use:

        sigma_t^2 = beta_t

    for simplicity.
    """

    batch_size = xt.shape[0]

    # --------------------------------------------------------
    # Model predicts epsilon
    # --------------------------------------------------------

    t_batch = torch.full(
        (batch_size,),
        t,
        device=DEVICE,
        dtype=torch.long,
    )

    eps_pred = model(
        xt,
        t_batch,
    )


    # --------------------------------------------------------
    # Get scalar coefficients for current timestep
    # --------------------------------------------------------

    beta_t = betas[t]

    alpha_t = alphas[t]

    alpha_bar_t = alpha_bars[t]


    # --------------------------------------------------------
    # Reverse mean
    #
    # mu_theta =
    #
    # 1 / sqrt(alpha_t)
    #
    # *
    #
    # (
    #     x_t
    #     -
    #     beta_t / sqrt(1-alpha_bar_t)
    #     * epsilon_theta
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
    # At t = 0, there is no x_{-1}
    #
    # So just return the mean.
    # --------------------------------------------------------

    if t == 0:
        return model_mean


    # --------------------------------------------------------
    # Otherwise sample Gaussian noise
    #
    # z ~ N(0, I)
    # --------------------------------------------------------

    z = torch.randn_like(xt)


    # --------------------------------------------------------
    # Fixed variance:
    #
    # sigma_t^2 = beta_t
    #
    # therefore
    #
    # sigma_t = sqrt(beta_t)
    # --------------------------------------------------------

    sigma_t = torch.sqrt(beta_t)


    # --------------------------------------------------------
    # Sample x_{t-1}
    #
    # x_{t-1}
    # =
    # mu_theta
    # +
    # sigma_t * z
    # --------------------------------------------------------

    x_prev = (
        model_mean
        +
        sigma_t * z
    )

    return x_prev


# ============================================================
# 7. Full sampling loop
# ============================================================

@torch.no_grad()
def sample(model, num_samples=16):
    """
    Start from:

        x_T ~ N(0, I)

    and repeatedly perform:

        x_T
        -> x_{T-1}
        -> ...
        -> x_1
        -> x_0
    """

    # --------------------------------------------------------
    # Start from pure Gaussian noise
    #
    # MNIST shape:
    #
    # [B, 1, 28, 28]
    # --------------------------------------------------------

    x = torch.randn(
        num_samples,
        1,
        28,
        28,
        device=DEVICE,
    )


    # Save intermediate states for visualization
    trajectory = {}

    save_timesteps = {
        T - 1,
        int(T * 0.75),
        int(T * 0.50),
        int(T * 0.25),
        0,
    }


    # --------------------------------------------------------
    # Reverse diffusion
    #
    # T-1 -> T-2 -> ... -> 0
    # --------------------------------------------------------

    for t in reversed(range(T)):

        x = p_sample(
            model,
            x,
            t,
        )

        if t in save_timesteps:
            trajectory[t] = (
                x.detach()
                .cpu()
                .clone()
            )

        if (
            t % 20 == 0
            or t == T - 1
        ):
            print(f"Sampling timestep: {t}")


    return x, trajectory


# ============================================================
# 8. Generate samples
# ============================================================

NUM_SAMPLES = 16

samples, trajectory = sample(
    model,
    num_samples=NUM_SAMPLES,
)


# ============================================================
# 9. Convert [-1, 1] -> [0, 1]
# ============================================================

samples = samples.cpu()

samples = (
    samples + 1.0
) / 2.0

samples = samples.clamp(
    0.0,
    1.0,
)


# ============================================================
# 10. Plot final generated samples
# ============================================================

grid_size = int(math.sqrt(NUM_SAMPLES))

fig, axes = plt.subplots(
    grid_size,
    grid_size,
    figsize=(6, 6),
)

for i, ax in enumerate(axes.flatten()):

    image = samples[i].squeeze(0)

    ax.imshow(
        image,
        cmap="gray",
    )

    ax.axis("off")


plt.suptitle("Generated MNIST Samples")

plt.tight_layout()

plt.savefig(
    "generated_samples.png",
    dpi=200,
)

plt.show()


# ============================================================
# 11. Visualize one denoising trajectory
# ============================================================

sorted_timesteps = sorted(
    trajectory.keys(),
    reverse=True,
)

fig, axes = plt.subplots(
    1,
    len(sorted_timesteps),
    figsize=(15, 3),
)


for ax, t in zip(
    axes,
    sorted_timesteps,
):

    # First generated sample
    image = trajectory[t][0]

    # [-1, 1] -> [0, 1]
    image = (
        image + 1.0
    ) / 2.0

    image = image.clamp(
        0.0,
        1.0,
    )

    image = image.squeeze(0)

    ax.imshow(
        image,
        cmap="gray",
    )

    ax.set_title(
        f"t={t}"
    )

    ax.axis("off")


plt.suptitle(
    "Reverse Diffusion Trajectory"
)

plt.tight_layout()

plt.savefig(
    "reverse_trajectory.png",
    dpi=200,
)

plt.show()