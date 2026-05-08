import copy

import torch


class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device

        # Keep a detached copy for inference.
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters and copy buffers such as BatchNorm stats."""
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for key, model_value in model_params.items():
            ema_value = ema_params[key]
            model_value = model_value.detach().to(device=ema_value.device)
            ema_value.mul_(self.decay).add_(model_value, alpha=1 - self.decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for key, model_value in model_buffers.items():
            ema_value = ema_buffers[key]
            ema_value.copy_(model_value.detach().to(device=ema_value.device))

        self.ema_model.eval()

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_model"])
