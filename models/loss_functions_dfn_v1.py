import torch
import torch.nn.functional as F

# True dfn is a one hot encoded label.

def pi_loss(pi, true_dfn):
    # true_labels is a multi-hot vector: 1 for each true class
    mask = (true_dfn == 1)

    loss = -torch.log(pi[mask]).sum()

    return loss


def dfn_loss(dfn_vectors, true_dfn):
    # Find indices where the true label equals 1
    indexes = torch.where(true_dfn == 1)[0]

    # Calculate MSE only for those indices
    losses = []

    for i in indexes:
        predicted_dfn = dfn_vectors[i]

        loss_i = F.mse_loss(predicted_dfn, true_dfn.float())
        losses.append(loss_i)

    # Average the losses over the selected DFNs
    return torch.stack(losses).mean()

def total_loss(
    pi,
    dfn_vectors,
    true_dfn,
    lambda_pi=1.0,
    lambda_dfn=1.0
):
    loss_pi = pi_loss(pi, true_dfn)
    loss_dfn = dfn_loss(dfn_vectors, true_dfn)

    loss = lambda_pi * loss_pi + lambda_dfn * loss_dfn

    return loss, loss_pi, loss_dfn