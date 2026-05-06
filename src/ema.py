import copy
import torch


class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device

        # EMAモデルを作成（deepcopy）
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # 勾配不要
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        # device移動（任意）
        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model):
        """
        学習モデルからEMAモデルを更新
        """
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_model"])