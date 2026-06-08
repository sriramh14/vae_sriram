import torch
def sam_loss(pred, target):

    B, C, H, W = pred.shape

    pred = pred.reshape(B, C, -1)
    target = target.reshape(B, C, -1)

    dot = (pred * target).sum(dim=1)

    pred_norm = torch.norm(pred, dim=1)
    target_norm = torch.norm(target, dim=1)

    cos = dot / (
        pred_norm * target_norm + 1e-8
    )

    cos = torch.clamp(cos, -1, 1)

    return torch.mean(torch.acos(cos))
