import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        batch_norm: bool = True,
        activation: bool = True,
        bias: bool = False,
        pool: bool = True,
    ):
        super(ConvBlock2D, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        relu = nn.ReLU()
        norm = nn.BatchNorm2d(out_channels)
        pooling = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        layers = [conv]
        if pool:
            layers.append(pooling)
        if activation:
            layers.append(relu)
        if batch_norm:
            layers.append(norm)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        output_padding: Union[int, Tuple[int]],
        batch_norm: bool = True,
        activation: bool = True,
        bias: bool = False,
    ):
        super(ConvTransposeBlock2D, self).__init__()
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        relu = nn.ReLU()
        norm = nn.BatchNorm2d(out_channels)
        layers = [conv]
        if activation:
            layers.append(relu)
        if batch_norm:
            layers.append(norm)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder1D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        enc_dim: int,
        proj_dim: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        in_HW: Tuple[int] = (16, 16),
        collapse: bool = False,
    ) -> None:

        super(Encoder1D, self).__init__()

        self.latent_dim = latent_dim
        self.collapse = collapse

        bias = True
        batch_norm = False
        pool = True

        # figure out final size of image after convolutions.
        # NOTE: we assume square convolutional kernels
        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)
        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)
            if pool:
                out_H //= 2 ** len(hidden_dims)

        if collapse:
            out_H = 1
        if pool:
            out_W //= 2 ** len(hidden_dims)

        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=batch_norm,
                activation=True,
                bias=bias,
                pool=pool,
            )
            encoder_blocks.append(block)
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        # if we collapse by haplotype to be permutation-invariant,
        # need to adjust expected output dims
        self.fc = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            enc_dim,
            bias=bias,
        )
        self.proj = nn.Linear(
            enc_dim,
            proj_dim,
            bias=bias,
        )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder_conv(x)
        if self.collapse:
            # take the average across haplotypes
            x = torch.mean(x, dim=2).unsqueeze(dim=2)
        # flatten, but ignore batch
        # x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        # split the result into mu and var components
        # of the latent Gaussian distribution
        enc = self.fc(x)
        enc = self.relu(enc)
        proj = self.proj(enc)
        return proj

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        in_HW: Tuple[int] = (16, 16),
    ) -> None:

        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        bias = True
        batch_norm = True

        # figure out final size of image after convolutions.
        # NOTE: we assume square convolutional kernels
        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)
        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)

        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=batch_norm,
                activation=True,
                bias=bias,
            )
            encoder_blocks.append(block)
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc_mu = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            latent_dim,
            bias=bias,
        )
        self.fc_var = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            latent_dim,
            bias=bias,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.encoder_conv(x)
        # flatten, but ignore batch
        # x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        # split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        output_padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        in_HW: Tuple[int] = (32, 32),
    ) -> None:
        super(Decoder, self).__init__()

        bias = True
        batch_norm = True

        # figure out final dimension to which we
        # need to reshape our filters before decoding
        self.final_dim = hidden_dims[-1]

        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)

        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)

        self.out_H = out_H
        self.out_W = out_W

        self.decoder_input = nn.Linear(
            latent_dim,
            hidden_dims[-1] * out_H * out_W,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dims[-1] * out_H * out_W)

        decoder_blocks = []
        hidden_dims.reverse()
        # loop over hidden dims in reverse
        for i in range(len(hidden_dims) - 1):
            block = ConvTransposeBlock2D(
                in_channels=hidden_dims[i],
                out_channels=hidden_dims[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                batch_norm=batch_norm,
                activation=True,
                bias=bias,
            )
            decoder_blocks.append(block)

        self.decoder_conv = nn.Sequential(*decoder_blocks)
        self.relu = nn.ReLU()

        # NOTE: no batch norm and no ReLu in final block
        final_block = [
            ConvTransposeBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                batch_norm=batch_norm,
                activation=True,
                bias=bias,
            ),
            ConvBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                batch_norm=batch_norm,
                activation=False,
                bias=bias,
            ),
        ]

        self.final = nn.Sequential(*final_block)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor):
        # fc from latent to intermediate
        x = self.decoder_input(z)
        # x = self.relu(x)

        # reshape
        x = x.view((-1, self.final_dim, self.out_H, self.out_W))

        decoded = self.decoder_conv(x)
        decoded = self.final(decoded)
        decoded = self.sigmoid(decoded)

        return decoded
class VAE(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        discriminator,
    ) -> None:
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        return [decoded, mu, log_var, z]


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        in_HW: Tuple[int] = (16, 16),
    ) -> None:

        super(Discriminator, self).__init__()

        bias = True
        batch_norm = True

        # figure out final size of image after convolutions.
        # NOTE: we assume square convolutional kernels
        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)
        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)

        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=batch_norm,
                activation=True,
                bias=bias,
            )
            encoder_blocks.append(block)
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            1,
            bias=bias,
        )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder_conv(x)
        # flatten, but ignore batch
        # x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        # split the result into mu and var components
        # of the latent Gaussian distribution
        x = self.fc(x)
        return x

if __name__ == "__main__":

    model = Encoder1D(
        in_channels=1,
        latent_dim=2,
        kernel_size=(1, 3),
        stride=(1, 2),
        padding=(0, 1),
        in_HW=(64, 64),
        hidden_dims=[8, 16],
        collapse=True,
    )

    model = model.to("mps")

    test = torch.rand(size=(100, 1, 64, 64)).to("mps")
    out = model(test)
    print (out)
