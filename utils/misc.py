import torch
import torch.distributed as dist

def calc_grad_norm(parameters, norm_type=2):

    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch._six.inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]), norm_type)

    return total_norm


def cycle(dl):
    while True:
        for data in dl:
            yield data

unwrap_module = lambda m: m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, ma_model, current_model):
        for ma_params, current_params in zip(ma_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_model_lst(self, ma_models, current_models):
        for ma_model, current_model in zip(ma_models, current_models):
            self.update_model_average(ma_model, current_model)

    def reset_model_lst(self, ma_models, current_models):
        for ma_model, current_model in zip(ma_models, current_models):
            ma_model.load_state_dict(unwrap_module(current_model).state_dict())

    def broadcast(self, *ma_models):
        if not dist.is_available():
            return

        if not dist.is_initialized():
            return

        for ma_model in ma_models:
            for current_params in ma_model.parameters():
                dist.broadcast(current_params, 0)
