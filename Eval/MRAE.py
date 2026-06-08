import torch
def mrae_loss(pred, target):

    return torch.mean(
        torch.abs(pred - target)
        / (target + 1e-6)
    )
