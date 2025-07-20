import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional
import math


DEVICE = torch.device("cuda")

class ColumnTokenizer(torch.nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        input_width: int = 36,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.input_dim = input_channels * input_width
        self.hidden_size = hidden_size
        # simple linear projection
        self.proj = torch.nn.Linear(self.input_dim, hidden_size)

    def forward(self, x):
        # shape should be (B, C, H, W) where H is the number
        # of haplotypes and W is the number of SNPs
        B, C, H, W = x.shape
        # permute to (B, H, C, W)
        x = x.permute(0, 2, 1, 3)
        # then, flatten each "patch" of C * W such that
        # each patch is 1D and size (C * W).
        x = x.reshape(B, H, -1)
        # embed "patches" of size (C * W, effectively a 1d
        # array equivalent to the number of SNPs)
        tokens = self.proj(x)
        return tokens
    
class SiteTokenizer(torch.nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        input_width: int = 36,
        input_height: int = 200,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.input_dim = input_channels * input_height
        self.hidden_size = hidden_size
        # simple linear projection
        self.proj = torch.nn.Linear(self.input_dim, hidden_size)

    def forward(self, x):
        # shape should be (B, C, H, W) where H is the number
        # of haplotypes and W is the number of SNPs
        B, C, H, W = x.shape
        # permute to (B, W, C, H)
        x = x.permute(0, 3, 1, 2)
        # then, flatten each "patch" of C * H such that
        # each patch is 1D and size (C * H).
        x = x.reshape(B, W, -1)
        # embed "patches" of size (C * W, effectively a 1d
        # array equivalent to the number of SNPs)
        tokens = self.proj(x)
        return tokens



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


class SpectrumTransformer(torch.nn.Module):
    def __init__(
        self,
        height: int = 200,
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

        self.pos_emb = nn.Parameter(torch.randn(1, width, hidden_size))

        # we use a "custom" tokenizer that takes
        # patches of size (C * W), where W is the
        # number of SNPs
        
        self.tokenizer = SiteTokenizer(
            input_channels=in_channels,
            input_height=height,
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

        x = self.tokenizer(x)  # (B, W, hidden_size)
        x = x + self.pos_emb[:, :x.size(1), :]
        # pass through transformer encoder
        x = self.attention(x)
        
        logits = self.classifier(x)

        return logits


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

    def forward(self, x, return_embeds: bool = False):
        B, C, H, W = x.shape
        x = self.tokenizer(x)  # (B, H, hidden_size)
        # pass through transformer encoder
        x = self.attention(x)
        if self.agg == "max":
            cls_output = torch.amax(x, dim=1)
        elif self.agg == "mean":
            cls_output = torch.mean(x, dim=1)
        logits = self.classifier(cls_output)
        if return_embeds:
            return cls_output, logits
        else:
            return logits


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


if __name__ == "__main__":

    DEVICE = torch.device("cpu")

    model = ColumnTokenizer(
        input_channels=1,
        input_width=36,
        hidden_size=128
        
    )
    model = DeFinetti(
        in_channels=1,
        width=36,
        fc_dim=128,
        hidden_dims=[32, 64],
        stride=1,
        padding=0,
        pool=True,
        
    )

    
    print (model)
    model = model.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    test = torch.rand(size=(100, 1, 200, 36)).to(DEVICE)
    proj = model(test)
    print (proj.shape)
