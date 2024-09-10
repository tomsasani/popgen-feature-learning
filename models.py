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
        stride: int,
        padding: int,
    ):
        super(ConvBlock2D, self).__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        relu = nn.LeakyReLU(0.2)
        norm = nn.BatchNorm2d(out_channels)
        layers = [conv, norm, relu]
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
        stride: int,
        padding: int,
        output_padding: int,
    ):
        super(ConvTransposeBlock2D, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        relu = nn.LeakyReLU(0.2)
        norm = nn.BatchNorm2d(out_channels)
        layers = [conv, norm, relu]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1,
        hidden_dims: List[int] = None,
        in_H: int = 32,
    ) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # figure out final size of image after convs
        out_H = int(in_H / (2 ** len(hidden_dims)))

        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            encoder_blocks.append(block)

            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.relu = nn.LeakyReLU(0.2)

        self.fc_mu = nn.Linear(
            hidden_dims[-1] * out_H * out_H,
            latent_dim,
        )
        self.fc_var = nn.Linear(
            hidden_dims[-1] * out_H * out_H,
            latent_dim,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, x):
        x = self.encoder_conv(x)

        # flatten, but ignore batch
        x = torch.flatten(x, start_dim=1)

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
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        hidden_dims: List[int] = None,
        in_H: int = 32,
    ) -> None:
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # figure out final size of image after convs
        out_H = int(in_H / (2 ** len(hidden_dims)))
        self.out_H = out_H

        self.decoder_input = nn.Linear(
            latent_dim,
            hidden_dims[-1] * out_H * out_H,
        )

        decoder_blocks = []

        # loop over hidden dims in reverse
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):

            block = ConvTransposeBlock2D(
                in_channels=hidden_dims[i],
                out_channels=hidden_dims[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            decoder_blocks.append(block)

        self.decoder_conv = nn.Sequential(*decoder_blocks)
        self.relu = nn.LeakyReLU(0.2)

        final_block = [
            ConvTransposeBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Sigmoid(),
        ]
        self.final_block = nn.Sequential(*final_block)

    def forward(self, z: torch.Tensor):

        # fc from latent to intermediate
        x = self.decoder_input(z)
        x = self.relu(x)

        # reshape

        x = x.view((-1, 256, self.out_H, self.out_H))
        x = self.decoder_conv(x)
        x = self.final_block(x)
        return x


class VAE(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
    ) -> None:
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

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

# contrastive-vae-no-bias
class CVAE(nn.Module):
    def __init__(self, s_encoder, z_encoder, decoder):
        super(CVAE, self).__init__()

        # instantiate two encoders q_s and q_z
        self.s_encoder = s_encoder
        self.z_encoder = z_encoder
        # instantiate one shared decoder
        self.decoder = decoder

    def forward(self, tg_inputs, bg_inputs) -> torch.Tensor:

        # step 1: pass target features through salient
        # encoder
        tg_z_mu, tg_z_log_var, tg_z = self.z_encoder(tg_inputs)
        tg_s_mu, tg_s_log_var, tg_s = self.s_encoder(tg_inputs)
        # step 2: pass background features through irrelevant
        # encoder
        bg_z_mu, bg_z_log_var, bg_z = self.z_encoder(bg_inputs)

        # step 3: decode

        # we decode the target outputs using both the salient and
        # irrelevant features
        tg_outputs = self.decoder(torch.cat([tg_s, tg_z], dim=-1))
        zeros = torch.zeros_like(tg_s)
        # we decode the background outputs using only the irrelevant
        # features
        bg_outputs = self.decoder(torch.cat([bg_z, zeros], dim=-1))
        # fg_outputs = self.decoder(torch.cat([tg_z, zeros], dim=0))

        return (
            tg_outputs,
            bg_outputs,
            tg_s_mu,
            tg_s_log_var,
            tg_z_mu,
            tg_z_log_var,
            bg_z_mu,
            bg_z_log_var,
        )


# class VAE(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(VAE, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def reparameterize(
#         self,
#         mu: torch.Tensor,
#         log_var: torch.Tensor,
#     ) -> torch.Tensor:

#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)

#         return eps * std + mu

#     def forward(self, x) -> torch.Tensor:
#         mu, log_var, x_, encoded_shape = self.encoder(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decoder(z, encoded_shape), mu, log_var, z


if __name__ == "__main__":

    encoder = Encoder(
        in_channels=1,
        latent_dim=2,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    decoder = Decoder(
        latent_dim=2,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    )

    model = VAE(encoder=encoder, decoder=decoder,)
    model = model.to("mps")

    test = torch.rand(size=(100, 1, 64, 64)).to("mps")

    out, mu, log_var, z = model(test)
    print (out.shape)
