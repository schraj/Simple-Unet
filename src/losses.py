
from typing import Optional, Sequence, Union
import torch

def _channelwise_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)

def dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
    eps: float = 0.0001,
    smooth: float = 0.0,
):
    tsh, psh = target.shape, probs.shape
    print('tsh', tsh)
    print('psh', psh)
    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif (
        tsh[0] == psh[0] and tsh[1:] == psh[2:]
    ):  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        print('scattering')
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f"Target shape {target.shape} is not compatible with output shape {probs.shape}."
        )
    print('probs:', probs)
    intersection = probs * onehot_target  # (N, C, ...)
    print('intersection:', intersection)
    numerator = 2 * _channelwise_sum(intersection) + smooth  # (C,)
    denominator = probs + onehot_target  # (N, C, ...)
    denominator = _channelwise_sum(denominator) + smooth + eps  # (C,)
    loss_per_channel = 1 - (numerator / denominator)  # (C,)
    weighted_loss_per_channel = weight * loss_per_channel  # (C,)
    return weighted_loss_per_channel.mean()  # ()


class DiceLoss(torch.nn.Module):
    def __init__(
        self,
        apply_softmax: bool = True,
        weight: Optional[torch.Tensor] = None,
        smooth: float = 0.0,
    ):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer("weight", weight)
        self.smooth = smooth

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = self.softmax(output)
        return self.dice(
            probs=probs, target=target, weight=self.weight, smooth=self.smooth
        )
