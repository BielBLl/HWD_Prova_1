# Create dfn throught the procedure on the first pdf of Marc

import torch

def create_dfn(u, v):
    # u: [batch_size, i + 1]
    # v: [batch_size, n - i]

    # Cumulative sums from left to right
    u_cumsum = torch.cumsum(u, dim=1)

    # Reverse cumulative sums for v
    v_cumsum = torch.flip(
        torch.cumsum(torch.flip(v, dims=[1]), dim=1),
        dims=[1]
    )

    # Exclude v[0] to obtain v_1 + ... + v_final, etc.
    v_part = v_cumsum[:, 1:] if v.shape[1] > 1 else v[:, :0]

    # Concatenate both parts
    dfn = torch.cat([u_cumsum, v_part], dim=1)

    return dfn