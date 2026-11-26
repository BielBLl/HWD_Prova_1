# We need to define the sparsemax function

import torch

def sparsemax(logits, dim=-1):
    # Sort values in descending order
    z_sorted, _ = torch.sort(logits, dim=dim, descending=True)

    # Cumulative sum
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    # Indices 1, 2, ..., number of components
    size = logits.size(dim)
    shape = [1] * logits.dim()
    shape[dim] = size
    r = torch.arange(
        1, size + 1, device=logits.device, dtype=logits.dtype
    ).view(shape)

    # Find the support of the sparsemax distribution
    support = 1 + r * z_sorted > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)

    # Calculate the threshold
    tau = (
        z_cumsum.gather(dim, k - 1) - 1
    ) / k.to(logits.dtype)

    # Project onto the probability simplex
    return torch.clamp(logits - tau, min=0)