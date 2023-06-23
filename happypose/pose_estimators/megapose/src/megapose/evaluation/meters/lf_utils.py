import torch
from torch.nn import functional as F


def normalize(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.

    Args:
    ----
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
    ------
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    """
    if not isinstance(quaternion, torch.Tensor):
        msg = f"Input type is not a torch.Tensor. Got {type(quaternion)}"
        raise TypeError(
            msg,
        )

    if not quaternion.shape[-1] == 4:
        msg = f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}"
        raise ValueError(
            msg,
        )
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def angular_distance(q1, q2, eps: float = 1e-7):
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = q1 @ q2.t()
    dist = 2 * acos_safe(dot.abs(), eps=eps)
    return dist


@torch.jit.script
def acos_safe(t, eps: float = 1e-7):
    return torch.acos(torch.clamp(t, min=-1.0 + eps, max=1.0 - eps))
