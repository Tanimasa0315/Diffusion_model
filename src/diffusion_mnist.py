import torch
from torchvision import transforms
from tqdm import tqdm

from src.simple_unet import SimpleUnetWithTime, CondSimpleUnet

# 以下は非推奨
class Diffuser:
    def __init__(self, model: SimpleUnetWithTime, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end=0.02,
                 device='cpu'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # ハイパーパラメータの設定
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # U-netを定義
        self.noise_pred_model = model.to(self.device)

    def add_noise(self, x_0, timestep) -> tuple[torch.Tensor, torch.Tensor]:
        """x_0に指定したタイムステップの時刻でのノイズ添加画像を出力"""
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1  # リストが0から始まるため

        alpha_bar = self.alpha_bars[time_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise_ddpm(self, x, timestep):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1  # リストが0から始まるため
        alpha = self.alphas[time_idx]
        alpha_bar = self.alpha_bars[time_idx]
        alpha_bars_prev = self.alpha_bars[time_idx-1]

        # ブロードキャストが正しく行われるための設定
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bars_prev.view(N, 1, 1, 1)

        # ノイズの予測
        self.noise_pred_model.eval()
        with torch.no_grad():
            eps = self.noise_pred_model(x, timestep)
        self.noise_pred_model.train()

        noise = torch.randn_like(x, device=self.device)
        # 時刻tではノイズを加えない
        noise[timestep == 1] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        return mu + noise * std

    def ddpm_sampling(self, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            timestep = torch.tensor(
                [i] * batch_size, device=self.device,
                dtype=torch.long,
            )
            x = self.denoise_ddpm(x, timestep)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def _reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)


class CondDiffuser:
    def __init__(self, model: CondSimpleUnet, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end=0.02,
                 gamma: float = 3.0, beta_schedule_type: str = "linear", device='cpu'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gamma = gamma
        self.beta_schedule_type = beta_schedule_type
        self.device = device

        # ハイパーパラメータの設定
        self.betas = make_beta_schedule(self.num_timesteps, self.beta_start, self.beta_end, device=self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # U-netを定義
        self.noise_pred_model = model.to(self.device)

    def add_noise(self, x_0, timestep) -> tuple[torch.Tensor, torch.Tensor]:
        """x_0に指定したタイムステップの時刻でのノイズ添加画像を出力"""
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1  # リストが0から始まるため

        alpha_bar = self.alpha_bars[time_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise_ddpm(self, x, timestep, labels):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1  # リストが0から始まるため
        alpha = self.alphas[time_idx]
        alpha_bar = self.alpha_bars[time_idx]
        alpha_bars_prev = self.alpha_bars[time_idx-1]

        # ブロードキャストが正しく行われるための設定
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bars_prev.view(N, 1, 1, 1)

        # ノイズの予測
        self.noise_pred_model.eval()
        with torch.no_grad():
            eps_cond = self.noise_pred_model(x, timestep, labels)
            eps_uncond = self.noise_pred_model(x, timestep)
            eps = eps_uncond + self.gamma * (eps_cond - eps_uncond)
        self.noise_pred_model.train()

        noise = torch.randn_like(x, device=self.device)
        # 時刻tではノイズを加えない
        noise[timestep == 1] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        return mu + noise * std

    def denoise_ddim(self, x, timestep, timestep_prev, labels, eta: float = 0):
        T = self.num_timesteps
        assert (timestep >= 1).all() and (timestep <= T).all()

        time_idx = timestep - 1  # リストが0から始まるため
        alpha = self.alphas[time_idx]
        alpha_bar = self.alpha_bars[time_idx]
        # timestep_prev = 0の時にalpha_bar_prev=1.0にする処置
        alpha_bar_prev = torch.where(
            timestep_prev > 0,
            self.alpha_bars[(timestep_prev - 1).clamp(min=0)],
            torch.ones_like(alpha_bar)
        )

        # ブロードキャストが正しく行われるための設定
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        # ノイズの予測
        self.noise_pred_model.eval()
        with torch.no_grad():
            eps_cond = self.noise_pred_model(x, timestep, labels)
            eps_uncond = self.noise_pred_model(x, timestep)
            eps = eps_uncond + self.gamma * (eps_cond - eps_uncond)
        self.noise_pred_model.train()

        # x_0の予測
        x_0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        # sigmaの計算
        std = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        sigma = eta * std

        noise = torch.randn_like(x, device=self.device)
        # 時刻tではノイズを加えない
        noise[timestep == 1] = 0

        mu = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

        return mu + noise * sigma

    def ddpm_sampling(self, x_shape=(20, 1, 28, 28), labels=None):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            timestep = torch.tensor(
                [i] * batch_size, device=self.device,
                dtype=torch.long,
            )
            x = self.denoise_ddpm(x, timestep, labels)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def ddim_sampling(self, x_shape=(20, 1, 28, 28), labels=None, ddim_timestep: int = 50, eta: float = 0):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        T = self.num_timesteps
        # DDIMでの1ステップをDDPMのステップに変換
        ddim_timesteps = torch.linspace(1, T, ddim_timestep, device=self.device, dtype=torch.long)

        for i in tqdm(range(ddim_timestep, 0, -1)):
            timestep = torch.tensor(
                [ddim_timesteps[i - 1]] * batch_size, device=self.device,
                dtype=torch.long,
            )
            if i == 1:
                timestep_prev = torch.tensor(
                    [0] * batch_size,
                    dtype=torch.long,
                )
            else:
                timestep_prev = torch.tensor(
                    [ddim_timesteps[i - 2]] * batch_size,
                    dtype=torch.long,
                )
            x = self.denoise_ddim(x, timestep, timestep_prev, labels, eta=eta)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]

        return images

    def _reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)


def make_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02, beta_schedule_type="linear", device="cpu"):
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
