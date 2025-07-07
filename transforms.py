import torch
import numpy as np


class MajorMinorTransform(torch.nn.Module):

    def __init__(self):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        pass

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        # convert image so that 0s are -1 and 1s stay the same, but
        # only in the first channel
        new_x = torch.clone(x)
        new_x[new_x == 0] = -1
        new_x[1:, :, :] = x[1:, :, :]

        return new_x


class RandomSiteMaskingTransform(torch.nn.Module):
    def __init__(self, rng: np.random.default_rng, mask_ratio=0.5):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        # draw from uniform distribution to figure out how many sites
        # to mask, a la
        self.mask_ratio = rng.uniform(0.1, mask_ratio)

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape
        n_sites_W = int(self.mask_ratio * W)

        # Randomly select pixels to mask
        mask_sites_W = torch.randperm(W, dtype=int)[:n_sites_W]
        mask = torch.ones((H, W), dtype=torch.int8)
        
        mask[:, mask_sites_W] = 0
        masked_x = x * mask  # Broadcast across channels

        return masked_x

class RandomRepolarizationTransform(torch.nn.Module):
    def __init__(self, rng: np.random.default_rng, frac_sites: float = 0.5):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        self.frac_sites = rng.uniform(0.1, frac_sites)

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape

        # randomly select sites to repolarize
        n_sites = int(W * self.frac_sites)
        mask_sites = torch.randperm(W, dtype=int)[:n_sites]
       
        # # create copy of the input array
        x_copy = torch.clone(x)
        x_copy[0, :, mask_sites] = 1 - x[0, :, mask_sites]
    
        return x_copy


class ChannelSubset(torch.nn.Module):
    def __init__(self, n: int = 2):
        self.n = n

    def __call__(self, x):
        if self.n == 1:
            return torch.unsqueeze(x[0, :, :], dim=0)
        else:
            return x[:self.n, :, :]


class Permute(torch.nn.Module):
    def __init__(self, ):
        pass

    def __call__(self, x):
        C, H, W = x.shape
        return x.permute((0, 2, 1))


class ShuffleHaplotypes(torch.nn.Module):
    def __init__(self, ):
        pass

    def __call__(self, x):
        C, H, W = x.shape
        #perms = torch.stack([torch.randperm(H) for _ in range(B)])  # shape: (B, H)

        # Expand indices to support advanced indexing
        # batch_indices = torch.arange(B).unsqueeze(1).expand(B, H)  # shape: (B, H)

        # Apply the shuffling
        # x is (B, C, H, W)
        # We want to index as: x[batch_idx, :, permuted_rows, :]
        return x[:, torch.randperm(H), :]  # shape: (B, C, H, W)
        # return x.permute((0, 2, 1))


class RandomlySizedCrop(torch.nn.Module):
    def __init__(self, min_width: int = 32):
        self.min_width = min_width

    def __call__(self, x):
        
        B, C, H, W = x.shape
        # get random x coordinate
        xi = torch.randint(W - self.min_width, size=(1,))[0]
        return x[:, :, :, xi:]


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

