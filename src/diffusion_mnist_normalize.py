import torch
from torchvision import transforms
from tqdm import tqdm

from src.simple_unet import SimpleUnetWithTime, CondSimpleUnet


class Diffuser:
    """Unconditional DDPM/DDIM helper for images normalized to [-1, 1]."""

    def __init__(
        self,
        model: SimpleUnetWithTime,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule_type: str = "linear",
        type: str | None = None,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule_type = type or beta_schedule_type
        self.device = device

        self.betas = make_beta_schedule(
            self.num_timesteps,
            self.beta_start,
            self.beta_end,
            beta_schedule_type=self.beta_schedule_type,
            device=self.device,
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.unet_model = model.to(self.device)

    def add_noise(self, x_0, timestep) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample q(x_t | x_0). x_0 is expected to be normalized to [-1, 1]."""
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1
        alpha_bar = self.alpha_bars[time_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise_ddpm(self, x, timestep):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1
        alpha = self.alphas[time_idx]
        alpha_bar = self.alpha_bars[time_idx]
        alpha_bar_prev = torch.where(
            timestep > 1,
            self.alpha_bars[(time_idx - 1).clamp(min=0)],
            torch.ones_like(alpha_bar),
        )

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        self.unet_model.eval()
        with torch.no_grad():
            eps = self.unet_model(x, timestep)
        self.unet_model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[timestep == 1] = 0

        x_0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        # Training images are normalized to [-1, 1], so keep predicted x_0
        # in the same range to prevent saturation from accumulating.
        x_0_pred = x_0_pred.clamp(-1, 1)

        mu = (
            torch.sqrt(alpha_bar_prev) * (1 - alpha) * x_0_pred
            + torch.sqrt(alpha) * (1 - alpha_bar_prev) * x
        ) / (1 - alpha_bar)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        return mu + noise * std

    def ddpm_sampling(self, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            timestep = torch.tensor(
                [i] * batch_size,
                device=self.device,
                dtype=torch.long,
            )
            x = self.denoise_ddpm(x, timestep)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def _reverse_to_img(self, x):
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = x * 255
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)


class CondDiffuser:
    """Conditional DDPM/DDIM helper for images normalized to [-1, 1]."""

    def __init__(
        self,
        model: CondSimpleUnet,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        gamma: float = 3.0,
        beta_schedule_type: str = "linear",
        type: str | None = None,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gamma = gamma
        self.beta_schedule_type = type or beta_schedule_type
        self.device = device

        self.betas = make_beta_schedule(
            self.num_timesteps,
            self.beta_start,
            self.beta_end,
            beta_schedule_type=self.beta_schedule_type,
            device=self.device,
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.unet_model = model.to(self.device)

    def add_noise(self, x_0, timestep) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample q(x_t | x_0). x_0 is expected to be normalized to [-1, 1]."""
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1
        alpha_bar = self.alpha_bars[time_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise_ddpm(self, x, timestep, labels):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1
        alpha = self.alphas[time_idx]
        alpha_bar = self.alpha_bars[time_idx]
        alpha_bar_prev = torch.where(
            timestep > 1,
            self.alpha_bars[(time_idx - 1).clamp(min=0)],
            torch.ones_like(alpha_bar),
        )

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        self.unet_model.eval()
        with torch.no_grad():
            eps_cond = self.unet_model(x, timestep, labels)
            eps_uncond = self.unet_model(x, timestep)
            eps = eps_uncond + self.gamma * (eps_cond - eps_uncond)
        self.unet_model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[timestep == 1] = 0

        x_0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        # Training images are normalized to [-1, 1], so keep predicted x_0
        # in the same range to prevent saturation from accumulating.
        x_0_pred = x_0_pred.clamp(-1, 1)

        mu = (
            torch.sqrt(alpha_bar_prev) * (1 - alpha) * x_0_pred
            + torch.sqrt(alpha) * (1 - alpha_bar_prev) * x
        ) / (1 - alpha_bar)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        return mu + noise * std

    def denoise_ddim(self, x, timestep, timestep_prev, labels, eta: float = 0):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1
        alpha_bar = self.alpha_bars[time_idx]
        alpha_bar_prev = torch.where(
            timestep_prev > 0,
            self.alpha_bars[(timestep_prev - 1).clamp(min=0)],
            torch.ones_like(alpha_bar),
        )

        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        self.unet_model.eval()
        with torch.no_grad():
            eps_cond = self.unet_model(x, timestep, labels)
            eps_uncond = self.unet_model(x, timestep)
            eps = eps_uncond + self.gamma * (eps_cond - eps_uncond)
        self.unet_model.train()

        x_0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        # Training images are normalized to [-1, 1], so keep predicted x_0
        # in the same range to prevent saturation from accumulating.
        x_0_pred = x_0_pred.clamp(-1, 1)

        std = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(
            1 - alpha_bar / alpha_bar_prev
        )
        sigma = eta * std

        noise = torch.randn_like(x, device=self.device)
        noise[timestep == 1] = 0

        direction_scale = (1 - alpha_bar_prev - sigma**2).clamp(min=0)
        mu = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(direction_scale) * eps

        return mu + noise * sigma

    def ddpm_sampling(self, x_shape=(20, 1, 28, 28), labels=None):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            timestep = torch.tensor(
                [i] * batch_size,
                device=self.device,
                dtype=torch.long,
            )
            x = self.denoise_ddpm(x, timestep, labels)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def ddim_sampling(self, x_shape=(20, 1, 28, 28), labels=None, ddim_timestep: int = 50, eta: float = 0):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        T = self.num_timesteps
        ddim_timesteps = torch.linspace(1, T, ddim_timestep, device=self.device, dtype=torch.long)

        for i in tqdm(range(ddim_timestep, 0, -1)):
            timestep = torch.tensor(
                [ddim_timesteps[i - 1]] * batch_size,
                device=self.device,
                dtype=torch.long,
            )
            if i == 1:
                timestep_prev = torch.tensor(
                    [0] * batch_size,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                timestep_prev = torch.tensor(
                    [ddim_timesteps[i - 2]] * batch_size,
                    dtype=torch.long,
                    device=self.device,
                )
            x = self.denoise_ddim(x, timestep, timestep_prev, labels, eta=eta)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def _reverse_to_img(self, x):
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = x * 255
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)


def make_beta_schedule(
    num_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule_type="linear",
    device="cpu",
):
    if beta_schedule_type == "linear":
        return _make_beta_schedule_linear(num_timesteps, beta_start, beta_end, device)
    if beta_schedule_type == "cosine":
        return _make_beta_schedule_cosine(num_timesteps, s=0.008, device=device)

    raise ValueError(f"unknown beta schedule type: {beta_schedule_type}")


def _make_beta_schedule_linear(num_timesteps, beta_start=0.0001, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, num_timesteps, device=device)


def _make_beta_schedule_cosine(num_timesteps, s=0.008, device="cpu"):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, device=device)
    alpha_bars = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return betas.clamp(max=0.999)
