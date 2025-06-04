from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import models
import losses


KERNEL_SIZE = (3, 3)
PADDING = (1, 1)
OUTPUT_PADDING = (1, 1)
STRIDE = (2, 2)

BATCH_SIZE = 128
LATENT_DIM = 64
H, W = 32, 32
CHANNELS = 1
INPUT_DIM = H * W
LR = 1e-3
EPOCHS = 10
HIDDEN_DIMS = [32, 64]
DEVICE = torch.device("mps")


def create_image_mask(X: np.ndarray, pct_mask: float = 0.5):
    N, C, H, W = X.shape

    mask = torch.zeros_like(X)

    i = int(W * pct_mask)
    j = i + (i * 2)
    mask[:, :, i:j, i:j] = 1

    return mask


def train_loop(
    model,
    dataloader,
    loss_fn,
    optimizer,
    mask_input: bool = False,
):

    model.train()

    n_batches = len(dataloader)
    total_loss = 0
    batch_size = None
    for batch_idx, (x, _) in tqdm.tqdm(enumerate(dataloader)):
        if batch_size is None:
            batch_size = x.shape[0]

        x_in = x.to(DEVICE)

        if mask_input:
            mask = create_image_mask(x_in, pct_mask=0.25).to(DEVICE)
            x_out = x_in * mask
            x_in = x_in * (1 - mask)

        x_hat, mean, log_var, z = model(x_in)
        if mask_input:
            loss = loss_fn(x_out, x_hat, mean, log_var)
        else:
            loss = loss_fn(x_in, x_hat, mean, log_var)

        optimizer.zero_grad()

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / (n_batches * batch_size)


def test_loop(model, dataloader, loss_fn, mask_input: bool = False):
    model.eval()

    n_batches = len(dataloader)
    total_loss = 0

    with torch.no_grad():
        batch_size = None
        for batch_idx, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            if batch_size is None:
                batch_size = x.shape[0]

            x_in = x.to(DEVICE)

            if mask_input:
                mask = create_image_mask(x_in, pct_mask=0.25).to(DEVICE)

                x_out = x_in * mask
                x_in = x_in * (1 - mask)

            x_hat, mean, log_var, z = model(x_in)
            if mask_input:
                loss = loss_fn(x_out, x_hat, mean, log_var)
            else:
                loss = loss_fn(x_in, x_hat, mean, log_var)
            total_loss += loss.item()

    return total_loss / (n_batches * batch_size)


def plot_example(
    model,
    dataloader,
    plot_name: str,
    mask_input: bool = False,
):

    f, axarr = plt.subplots(5, 2, figsize=(8, 4))

    dataloader_iter = iter(dataloader)

    model.eval()
    with torch.no_grad():

        for i in range(5):
            # grab first example
            xs, ys = next(dataloader_iter)

            x_in = torch.unsqueeze(xs[0], dim=0).to(DEVICE)
            mask = None
            if mask_input:
                mask = create_image_mask(x_in, pct_mask=0.25)

            x_hat, mu, log_var, z = model(
                x_in * (1 - mask) if mask is not None else x_in
            )

            x_in = x_in.view(1, CHANNELS, H, W).cpu().numpy()
            x_hat = x_hat.view(1, CHANNELS, H, W).cpu().numpy()

            x_in = np.transpose(x_in, (0, 2, 3, 1))
            x_hat = np.transpose(x_hat, (0, 2, 3, 1))

            axarr[i, 0].imshow(x_in[0])
            if mask_input:
                mask_np = np.transpose(mask.cpu().numpy(), (0, 2, 3, 1))
                axarr[i, 1].imshow(
                    (x_in[0] * (1 - mask_np[0])) + (x_hat[0] * mask_np[0])
                )
            else:
                axarr[i, 1].imshow(x_hat[0])

            for j in (0, 1):
                axarr[i, j].set_xticks([])
                axarr[i, j].set_yticks([])

    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig(plot_name, dpi=200)
    plt.close()


img_tfms = []
if CHANNELS == 1:
    img_tfms.append(transforms.Grayscale())
img_tfms.extend(
    [
        transforms.Resize(size=(H, W)),
        transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(
                    torch.float32,
                    scale=True,
                ),
            ]
        ),
    ]
)
img_tfms = transforms.Compose(img_tfms)

train = CIFAR10(
    "data/",
    transform=img_tfms,
    train=True,
    download=True,
)
test = CIFAR10(
    "data/",
    transform=img_tfms,
    train=False,
    download=True,
)

train = torchvision.datasets.ImageFolder(
    "data/simulated/train/",
    transform=img_tfms,
)
test = torchvision.datasets.ImageFolder(
    "data/simulated/test/",
    transform=img_tfms,
)

train_loader = DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    in_HW=(H, W),
)

decoder = models.Decoder(
    out_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    output_padding=OUTPUT_PADDING,
    hidden_dims=HIDDEN_DIMS,
    in_HW=(H, W),
)

discriminator = models.Discriminator(
    in_channels=CHANNELS,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    in_HW=(H, W),
)

model = models.VAE(
    encoder=encoder,
    decoder=decoder,
    discriminator=discriminator,
)
model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = losses.VAELoss(
    torch.nn.functional.binary_cross_entropy,
    kld_weight=1.0,
    mask=MASK,
)

print("Start training VAE...")

res = []

for epoch in range(EPOCHS):
    plot_example(
        model,
        test_loader,
        plot_name=f"fig/reconstructions/{epoch}.png",
        mask_input=MASK,
    )

    train_loss = train_loop(
        model,
        train_loader,
        loss_fn,
        optimizer,
        mask_input=MASK,
    )
    test_loss = test_loop(
        model,
        test_loader,
        loss_fn,
        mask_input=MASK,
    )

    for loss, loss_name in zip((train_loss, test_loss), ("train", "test")):
        res.append(
            {
                "epoch": epoch,
                "loss_kind": loss_name,
                "loss_val": loss,
            }
        )

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Train Loss: ",
        train_loss,
        "\tAverage Test Loss: ",
        test_loss,
    )

res_df = pd.DataFrame(res)

f, ax = plt.subplots()
sns.lineplot(data=res_df, x="epoch", y="loss_val", hue="loss_kind", ax=ax)
f.tight_layout()
sns.despine(ax=ax)
f.savefig("vae.loss.png", dpi=200)

print("Finish!!")

model.eval()

f, ax = plt.subplots()

reps = []
labels = []
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):

        if MASK:
            mask = create_image_mask(x)
            x = x * (1 - mask)

        x = x.to(DEVICE)
        x_hat, mu, log_var, z = model(x)
        z = z.cpu().numpy()
        reps.append(z)
        labels.append(y)
reps = np.concatenate(reps)
labels = np.concatenate(labels)


if LATENT_DIM > 2:
    clf = PCA(n_components=2)
    reps = clf.fit_transform(reps)
f, ax = plt.subplots()
ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
f.savefig("vae.coords.png", dpi=200)


def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=8):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            x_hat = model.decoder(z)
            x_hat = x_hat.to('cpu').detach().numpy()[0, :, :]
            axarr[i, j].imshow(np.transpose(x_hat, (1, 2, 0)))
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("vae.recons.png")

if LATENT_DIM == 2:
    plot_reconstructed(model)
