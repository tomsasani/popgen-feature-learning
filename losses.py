import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

DEVICE = torch.device("cuda")

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    b, n, m = x.shape
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


def total_covariance_loss(batch):
    """get the sum of covariance losses for each
    unique view within the batch

    Args:
        batch (_type_): _description_
    """
    B, V, D = batch.shape
    covariance_loss = 0
    for vi in torch.arange(V):
        v_batch = batch[:, vi, :]
        # Covariance
        v_centered = v_batch - v_batch.mean(dim=0, keepdim=True)
        cov = (v_centered.T @ v_centered) / (B - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        covariance_loss += (off_diag ** 2).sum() / D
    return covariance_loss

def total_invariance_loss(batch):
    """for each pair of views, get the MSE loss between
    their batch embedding vectors

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    invariance_loss = 0
    B, V, D = batch.shape
    for vi in torch.arange(V):
        for vj in torch.arange(V):
            if vi == vj: continue
            vi_batch = batch[:, vi, :]
            vj_batch = batch[:, vj, :]
            # invariance
            invariance = F.mse_loss(vi_batch, vj_batch)
            invariance_loss += invariance# / B
    return invariance_loss

def total_variance_regularization(batch, eps=1e-4):
    """_summary_

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_variance_reg = 0
    # get regularized standard deviation
    B, V, D = batch.shape
    for vi in torch.arange(V):
        v_batch = batch[:, vi, :]
        std = torch.sqrt(v_batch.var(dim=0) + eps)
        variance_reg = torch.mean(F.relu(1.0 - std))
        total_variance_reg += variance_reg

    return total_variance_reg

class SpectrumVICReg(nn.Module):
    def __init__(self):
        super(SpectrumVICReg, self).__init__()

    def forward(
        self,
        z,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
    ):
        """
        VICReg loss for N views per sample.

        Args:
            z: Tensor of shape (B, N, D), embeddings of N views for each sample.
            sim_coeff: weight for invariance (similarity) loss
            std_coeff: weight for variance loss
            cov_coeff: weight for covariance loss
            eps: numerical stability

        Returns:
            Scalar VICReg loss
        """
        B, N, D = z.shape

        invariance_loss = total_invariance_loss(z)
        variance_regularization = total_variance_regularization(z)
        covariance_loss = total_covariance_loss(z)

        loss = (
            sim_coeff * invariance_loss
            + std_coeff * variance_regularization
            + cov_coeff * covariance_loss
        )
        return loss


class SpectrumViewLoss(nn.Module):
    def __init__(self, lmbda: float = 0.0051):
        super(SpectrumViewLoss, self).__init__()
        self.lmbda = lmbda

    def forward(self, z, keep_batch: bool = False):
        # z is a tensor of shape (B, V, D), where B is the batch size
        # V is the number of views (6) and D is the dimensionality
        B, V, D = z.shape
        # compute the average embedding across views
        mean_per_batch = z.mean(dim=1).unsqueeze(dim=1)
        # compute distance to centroid
        dist = F.mse_loss(z, mean_per_batch.expand_as(z), reduction="none")
        if keep_batch:
            return dist.sum(dim=(1, 2))
        else:
            return dist.sum()

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
