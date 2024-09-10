import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):

    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(
        self,
        orig: torch.Tensor,
        recon: torch.Tensor,
        mu,
        log_var,
    ):
        N, C, H, W = orig.shape

        kld_weight = 0.5 # N * H * W

        # NOTE: is this being calculated correctly?
        recons_loss = nn.functional.mse_loss(
            recon,
            orig,
            reduction="sum",
        )
         #recons_loss *= N

        # compute KL loss
        kld_loss = torch.sum(
            -0.5
            * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(),
                dim=1,
            ),
            dim=0,
        )

        loss = recons_loss + kld_weight * kld_loss
        return loss


class CVAELoss(nn.Module):

    def __init__(self):
        super(CVAELoss, self).__init__()

    def forward(
        self,
        tg_inputs,
        tg_outputs,
        bg_inputs,
        bg_outputs,
        tg_s_mean,
        tg_s_log_var,
        tg_z_mean,
        tg_z_log_var,
        bg_z_mean,
        bg_z_log_var,
    ):
        H, W = tg_inputs.shape
        input_dim = H * W

        reconstruction_loss = F.mse_loss(
            tg_inputs,
            tg_outputs,
            reduction="sum",
        )
        reconstruction_loss += F.mse_loss(
            bg_inputs,
            bg_outputs,
            reduction="sum",
        )
        # reconstruction_loss *= input_dim

        kl_loss = 1 + tg_z_log_var - tg_z_mean.pow(2) - torch.exp(tg_z_log_var)
        kl_loss += 1 + tg_s_log_var - tg_s_mean.pow(2) - torch.exp(tg_s_log_var)
        kl_loss += 1 + bg_z_log_var - bg_z_mean.pow(2) - torch.exp(bg_z_log_var)
        kl_loss = torch.sum(kl_loss, dim=0)
        kl_loss *= -0.5

        cvae_loss = torch.mean(reconstruction_loss) + torch.mean(kl_loss)
        return cvae_loss
