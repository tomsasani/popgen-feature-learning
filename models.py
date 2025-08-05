import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional
import math
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


DEVICE = torch.device("cuda")

class ColumnTokenizer(torch.nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        input_width: int = 36,
        input_height: int = 200,
        hidden_size: int = 128,
        how: str = "row",
    ):
        super().__init__()

        self.how = how

        self.input_dim = input_channels * (
            input_width if how == "row" else input_height
        )
        self.hidden_size = hidden_size
        # simple linear projection
        self.proj = torch.nn.Linear(self.input_dim, hidden_size)

    def forward(self, x):
        # shape should be (B, C, H, W) where H is the number
        # of haplotypes and W is the number of SNPs
        B, C, H, W = x.shape
        if self.how == "row":
            # permute to (B, H, C, W)
            x = x.permute(0, 2, 1, 3)
            # then, flatten each "patch" of C * W such that
            # each patch is 1D and size (C * W).
            x = x.reshape(B, H, -1)
        else:
            # permute to (B, W, C, H)
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(B, W, -1)
        # embed "patches" of size (C * W, effectively a 1d
        # array equivalent to the number of SNPs)
        tokens = self.proj(x)
        return tokens

class LabelTokenizer(torch.nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 128,
        how="row",
    ):
        super().__init__()
        
        self.embed = torch.nn.Linear(num_classes, hidden_size)
        self.num_classes = num_classes
        self.how = how


    def forward(self, x):
        # shape should be (B, W, N) where W is the number of SNPs and 
        # N is number of classes (i.e., should be one-hot already)
        B, W, N = x.shape
        # embed each site to (B, W, D)
        embeddings = self.embed(x)
        if self.how == "row":
            # take a mean pool over W to (B, 1, D)
            embeddings = torch.mean(embeddings, dim=1).unsqueeze(dim=1)
        return embeddings


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        scores = self.attn_proj(hidden_states).squeeze(-1)  # [batch_size, seq_len]

        if attention_mask is not None:
            # Apply a large negative value to masked positions before softmax
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = nn.functional.softmax(scores, dim=1)  # [batch_size, seq_len]
        attn_weights = attn_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Weighted sum of hidden states
        pooled = torch.sum(hidden_states * attn_weights, dim=1)  # [batch_size, hidden_dim]
        return pooled

class Transformer(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 1,
        mlp_hidden_dim_ratio: int = 2,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_hidden_dim_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_hidden_dim_ratio, embed_dim),
        )

    def forward(self, x):
        # layernorm initial embeddings
        x_norm = self.norm(x)
        # self-attention on normalized embeddings
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  
        # residual connection + layernorm
        x = self.norm(x + attn_out)
        # mlp on each haplotype token
        x_mlp = self.mlp(x)
        # final residual connection + layernorm
        return self.norm(x + x_mlp)


class TinyTransformer(torch.nn.Module):
    def __init__(
        self,
        width: int = 36,
        in_channels: int = 1,
        num_heads: int = 1,
        hidden_size: int = 128,
        num_classes: int = 2,
        depth: int = 1,
        mlp_ratio: int = 2,
        agg: str = "max",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.agg = agg

        # we use a "custom" tokenizer that takes
        # patches of size (C * W), where W is the
        # number of SNPs
        
        self.tokenizer = ColumnTokenizer(
            input_channels=in_channels,
            input_width=width,
            hidden_size=hidden_size,
        )

        self.attention = nn.Sequential(
            *[
                Transformer(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    mlp_hidden_dim_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # linear classifier head
        self.classifier = torch.nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.tokenizer(x)  # (B, H, hidden_size)
        # pass through transformer encoder
        x = self.attention(x)
        if self.agg == "max":
            cls_output = torch.amax(x, dim=1)
        elif self.agg == "mean":
            cls_output = torch.mean(x, dim=1)
        logits = self.classifier(cls_output)

        return logits
    

class MAE(nn.Module):
    def __init__(
        self,
        width: int = 36,
        height: int = 200,
        in_channels: int = 1,
        num_heads: int = 1,
        hidden_size: int = 128,
        num_classes: int = 2,
        depth: int = 1,
        mlp_ratio: int = 2,
        agg: str = "max",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.agg = agg
        self.input_channels = in_channels
        self.width = width

        # we use a "custom" tokenizer that takes
        # patches of size (C * W), where W is the
        # number of SNPs
        self.tokenizer = ColumnTokenizer(
            input_channels=in_channels,
            input_width=width,
            hidden_size=hidden_size,
            how="row",
        )

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.pos_embed = nn.Parameter(torch.randn(1, height, hidden_size))  # fixed sin-cos embedding
        # self.pos_embed = nn.Embedding(height, hidden_size)  # fixed sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(height**.5), cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.decoder_pos_embed = nn.Parameter(torch.randn(1, height, hidden_size))  # fixed sin-cos embedding
        # self.decoder_pos_embed = nn.Embedding(height, hidden_size)  # fixed sin-cos embedding
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(height**.5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))



        self.encoder = nn.Sequential(
            *[
                Transformer(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    mlp_hidden_dim_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.decoder = nn.Sequential(
            *[
                Transformer(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    mlp_hidden_dim_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.sigmoid = nn.Sigmoid()

        # predict pixel values for each patch
        self.decoder_pred = nn.Linear(hidden_size, width * in_channels)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        from https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, produce_embedding: bool = False):
        B, C, H, W = x.shape
        x = self.tokenizer(x)  # (B, H, hidden_size)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # pass through transformer encoder
        x = self.encoder(x)
        if produce_embedding:
            x = torch.amax(x, dim=1)
        return x, mask, ids_restore
    

    
    def forward_decoder(self, x, ids_restore):
        B, H, D = x.shape # batch_size, length, dimensionality
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - H, 1)
        x_ = torch.cat([x, mask_tokens], dim=1) 
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        # x = torch.cat([x, x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed

        x = self.decoder(x)

        # predictor projection
        x = self.decoder_pred(x)
        B, H, CW = x.shape
        # x = x.reshape((B, H, self.input_channels, self.width))
        return x# .permute((0, 2, 1, 3))
    
    def forward(self, imgs, mask_ratio=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask

class DeFinetti(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        kernel: Union[int, Tuple[int]] = (1, 5),
        hidden_dims: List[int] = [32, 64],
        agg: str = "max",
        width: int = 36,
        num_classes: int = 2,
        fc_dim: int = 128,
        padding: int = 0,
        stride: int = 1,
        pool: bool = False,
    ) -> None:

        super(DeFinetti, self).__init__()

        self.agg = agg
        self.width = width

        _stride = (1, stride)
        _padding = (0, padding)

        out_W = width

        conv = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = [
                nn.Conv2d(
                    in_channels,
                    h_dim,
                    kernel_size=kernel,
                    stride=_stride,
                    padding=_padding,
                ),
                nn.ReLU(),
            ]
            if pool:
                block.append(
                    nn.MaxPool2d(
                        kernel_size=(1, 2),
                        stride=(1, 2),
                    ),
                )
            
            out_W = (
                math.floor((out_W - kernel[1] + (2 * (padding))) / _stride[1]) + 1
            )

            # account for max pooling
            if pool:
                out_W //= 2
            in_channels = h_dim
            conv.extend(block)

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] * out_W, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
        )
        # two projection layers with batchnorm
        self.project = nn.Sequential(
            nn.Linear(fc_dim, num_classes),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        # take the average across haplotypes
        if self.agg == "mean":
            x = torch.mean(x, dim=2)
        elif self.agg == "max":
            x = torch.amax(x, dim=2)
        elif self.agg is None:
            pass
        # flatten, but ignore batch
        x = self.flatten(x)
        encoded = self.fc(x)
        projection = self.project(encoded)

        return projection


class SimpleDeFinetti(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
    ) -> None:

        super(SimpleDeFinetti, self).__init__()

        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(1, 5),
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 5),
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )
        self.conv = conv
        self.relu = nn.ReLU()

        self.representer = nn.Sequential(
            nn.Linear(32 * 64, 128),
        )
        # two projection layers with batchnorm
        # linear classifier head
        self.projector = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.flatten = nn.Flatten()

    def forward(self, x, return_embeds: bool = False):
        x = self.conv(x)
        x = torch.amax(x, dim=2)
        encoded = self.flatten(x)
        encoded = self.representer(encoded)
        # print (encoded.shape)
        projection = self.projector(encoded)
        if return_embeds:
            return encoded, projection
        else:
            return projection


if __name__ == "__main__":

    DEVICE = torch.device("cpu")

    
    model = MAE(
        in_channels=1,
        agg="max",
        width=32,
    )

    print (model)
    model = model.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    x = torch.rand(size=(100, 1, 200, 32)).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        a, b = model(x)
        print (a.shape, b.shape)
        e, _, _ = model.forward_encoder(x, 0.)
        print (e.shape)
    
