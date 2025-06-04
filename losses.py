import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def create_image_mask(X: torch.Tensor, pct_mask: float = 0.5):
    N, C, H, W = X.shape

    mask = torch.zeros_like(X)

    i = int(W * pct_mask)
    j = i + (i * 2)
    mask[:, :, i:j, i:j] = 1

    return mask


class SimCLRLoss(nn.Module):
    def __init__(self):
        super(SimCLRLoss, self).__init__()
        self.temperature = 1

    def forward(self, feats):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None], feats[None, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device="mps")
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.sum()
        
        return nll

class VAELoss(nn.Module):

    def __init__(
        self,
        reconstruction_loss_fn,
        kld_weight: float = 1,
        use_cnn: bool = True,
        mask: bool = False,
    ):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.loss_fn = reconstruction_loss_fn
        self.use_cnn = use_cnn
        self.mask = mask

    def forward(
        self,
        orig: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ):

        # compute per-pixel MSE loss
        recons_loss = self.loss_fn(
            recon,
            orig,
            reduction="none",
        )

        if self.mask:
            recons_loss *= create_image_mask(recon, pct_mask=0.25)

        # compute average of the per-pixel total loss for each image
        if self.use_cnn:
            recons_loss = torch.mean(torch.sum(recons_loss, dim=(1, 2, 3)), dim=0)
        else:
            recons_loss = torch.mean(torch.sum(recons_loss, dim=1), dim=0)

        # compute average per-image KL loss across the batch
        kld_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + log_var - torch.square(mu) - torch.exp(log_var),
                dim=1,
            )
        )
        loss = recons_loss #+ self.kld_weight * kld_loss
        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(
        self,
        orig_preds: torch.Tensor,
        recon_preds: torch.Tensor,
    ):

        orig_labels = torch.ones_like(orig_preds)
        orig_loss = torch.nn.functional.binary_cross_entropy(
            orig_labels,
            orig_preds,
        )

        recon_labels = torch.zeros_like(recon_preds)
        recon_loss = torch.nn.functional.binary_cross_entropy(
            recon_labels,
            recon_preds,
        )

        return (orig_loss + recon_loss) / 2.0
