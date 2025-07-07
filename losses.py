import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

DEVICE = torch.device("cuda")

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, projection_dim: int = 32, lmbda: float = 0.0051):
        super(BarlowTwinsLoss, self).__init__()
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)
        self.lmbda = lmbda

    def forward(self, z1, z2):
        # z1 and z2 are the projections from each view, which
        # should be batchnormed
        B, D = z1.shape

        # empirical cross-correlation matrix
        c = z1.T @ z2

        # divide by batch size
        c = torch.divide(c, B)

        # sum the cross-correlation matrix between all gpus
        c_diff = (c - torch.eye(D).to(DEVICE)).pow(2)
        
        on_diag = torch.diagonal(c_diff).sum()
        off_diag = off_diagonal(c_diff).sum()
        loss = on_diag + self.lmbda * off_diag

        return loss

class SpectrumLoss(nn.Module):
    def __init__(self, freq):
        super(SpectrumLoss, self).__init__()
        self.freq = freq

    def forward(self, preds, true):
        # weight the loss by inverse of class frequency
        # convert true mutations to one hot encodings
        # preds = preds.permute(0, 2, 1)
        weights = 1 / self.freq
        # if self.per_batch:
        #     _true = torch.flatten(true)
        #     u, c = torch.unique(_true, return_counts=True)
        #     weights = c / torch.sum(c)
        #     weights = 1 / weights
        # weights = weights / torch.sum(weights)
        loss = nn.CrossEntropyLoss(reduction="mean")#, weight=weights.to(DEVICE))
        # loss = nn.BCEWithLogitsLoss(reduction="mean", weight=weights.to(DEVICE))
        l = loss(preds, true.to(DEVICE))
        
        return l
        
class PoissonLoss(nn.Module):
    def __init__(self, n_snps: int = 32):
        super(PoissonLoss, self).__init__()
        self.n_snps = n_snps
        
    def forward(self, preds, true):
        loss = nn.PoissonNLLLoss(log_input=False)
        # figure out the total number of sites in the 
        # true image, across batch
        preds_norm = torch.multiply(preds, self.n_snps).to(DEVICE)
        
        return loss(preds_norm, true.to(DEVICE))

class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feats):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None], feats[None, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=DEVICE)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        acc = (sim_argsort < 5).float().mean()
        
        return nll, acc

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
