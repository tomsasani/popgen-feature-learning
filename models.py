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
        input_width: int = 32,
        hidden_size: int = 192,
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
        # print (x.shape)
        # then, flatten each "patch" of C * W such that
        # each patch is 1D and size (C * W).
        x = x.reshape(B, H, -1)
        # print (x.shape)
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

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 1, mlp_hidden_dim_ratio: int = 2):
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
            nn.Dropout(0.),
            nn.Linear(embed_dim * mlp_hidden_dim_ratio, embed_dim),
            nn.Dropout(0.),
        )


    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # Self-attention: Q = K = V = x
        x = x + attn_out
        x_norm = self.norm(x)
        return x + self.mlp(x_norm)


class BabyTransformer(torch.nn.Module):
    def __init__(
        self,
        width: int = 32,
        in_channels: int = 1,
        num_heads: int = 1,
        hidden_size: int = 128,
        num_classes: int = 2,
        attention_pool: bool = False,
        depth: int = 4,
        mlp_ratio: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pooling = AttentionPooling(hidden_size)
        self.attention_pool = attention_pool
        self.norm = nn.LayerNorm(hidden_size)

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
                SelfAttention(
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
        # print (x.shape)
        x = self.tokenizer(x)  # (B, H, hidden_size)
        # print (x.shape)
        # pass through transformer encoder
        x = self.attention(x)
        # classification head on the average or attention-pooled
        # final embeddings (no CLS token)
        x = self.norm(x)
        if self.attention_pool:
            cls_output = self.pooling(x)
        else:
            cls_output = torch.mean(x, dim=1)
        logits = self.classifier(cls_output)  

        return logits

class BasicPredictor(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        kernel: Union[int, Tuple[int]] = 3,
        hidden_dims: List[int] = None,
        agg: Union[str, None] = "max",
        width: int = 32,
        projection_dim: int = 32,
        encoding_dim: int = 256,
        pool: bool = False,
        batch_norm: bool = False,
        padding: int = 0,
    ) -> None:

        super(BasicPredictor, self).__init__()

        self.agg = agg
        self.width = width

        _stride = (1, 2)
        _padding = (0, padding)

        out_W = width
        out_H = 1

        encoder_blocks = []
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

            out_W = math.floor((out_W - kernel[1] + (2 * (_padding[1]))) / _stride[1]) + 1

            if batch_norm:
                block.append(nn.BatchNorm2d(h_dim))
            if pool:
                block.append(
                    nn.MaxPool2d(
                        kernel_size=(1, 2),
                        stride=(1, 2),
                    )
                )
                out_W //= 2

            in_channels = h_dim
            encoder_blocks.extend(block)

        self.encoder_conv = nn.Sequential(*encoder_blocks)
        self.encode = nn.Sequential(
            nn.Linear(hidden_dims[-1] * out_W * out_H, encoding_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(encoding_dim),
            nn.Dropout(0.),
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(encoding_dim),
            nn.Dropout(0.),
            
        )
        # two projection layers with batchnorm
        self.project = nn.Sequential(
            nn.Linear(encoding_dim, projection_dim),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder_conv(x)
        # take the average across haplotypes
        if self.agg == "mean":
            x = torch.mean(x, dim=2)
        elif self.agg == "max":
            x = torch.amax(x, dim=2)
        elif self.agg is None:
            pass
        # flatten, but ignore batch
        x = self.flatten(x)
        encoded = self.encode(x)
        projection = self.project(encoded)

        return projection


if __name__ == "__main__":

    DEVICE = torch.device("cpu")


    model = BasicPredictor(
        in_channels=1,
        kernel=(1, 5),
        hidden_dims=[32, 64],
        agg="mean",
        width=36,
        encoding_dim=192,
        projection_dim=3,
        pool=True,
        batch_norm=False,
        padding=0,
        
    )
    print (model)
    model = model.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    test = torch.rand(size=(100, 1, 200, 36)).to(DEVICE)
    proj = model(test)
    print (proj.shape)
