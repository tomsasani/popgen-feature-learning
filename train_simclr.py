from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import models
import losses

BATCH_SIZE = 128
LATENT_DIM = 16
H, W = 32, 32
CHANNELS = 1
INPUT_DIM = H * W
LR = 5e-4
EPOCHS = 10
HIDDEN_DIMS = [32, 64]
DEVICE = torch.device("mps")


class RandomSiteMaskingTransform(torch.nn.Module):
    def __init__(self, mask_ratio=0.25):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape
        n_sites = int(self.mask_ratio * W)

        # Randomly select pixels to mask
        mask_sites = torch.randperm(W, dtype=int)[:n_sites]
        mask = torch.ones((H, W), dtype=torch.float32)
        mask[:, mask_sites] = 0

        masked_x = x * mask  # Broadcast across channels
        return masked_x
    
class RandomRepolarizationTransform(torch.nn.Module):
    def __init__(self, repol_prob=0.25):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        self.repol_prob = repol_prob

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape
        n_sites = int(self.repol_prob * W)

        # randomly select sites to repolarize
        mask_sites = torch.randperm(W, dtype=int)[:n_sites]

        # create copy of the input array
        # NOTE: won't work if we're dealing with distance channel
        x_copy = x
        x_copy[:, :, mask_sites] = 1 - x[:, :, mask_sites]

        return x_copy

class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def train_loop(
    model,
    dataloader,
    loss_fn,
    optimizer,
):

    model.train()

    n_batches = len(dataloader)
    total_loss = 0
    batch_size = None
    for batch_idx, (batch, _) in tqdm.tqdm(enumerate(dataloader)):

        imgs = torch.cat(batch, dim=0)

        if batch_size is None:
            batch_size = imgs.shape[0] // 2

        x_in = imgs.to(DEVICE)

        feats = model(x_in)
        loss = loss_fn(feats)

        optimizer.zero_grad()

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / (n_batches * batch_size)


def test_loop(model, dataloader, loss_fn):
    model.eval()

    n_batches = len(dataloader)
    total_loss = 0

    with torch.no_grad():
        batch_size = None
        for batch_idx, (batch, _) in tqdm.tqdm(enumerate(dataloader)):

            imgs = torch.cat(batch, dim=0)

            if batch_size is None:
                batch_size = imgs.shape[0] // 2

            x_in = imgs.to(DEVICE)

            feats = model(x_in)
            loss = loss_fn(feats)
            total_loss += loss.item()

    return total_loss / (n_batches * batch_size)


def plot_example(
    model,
    dataloader,
    plot_name: str,
):

    f, axarr = plt.subplots(5, 2, figsize=(8, 4))

    dataloader_iter = iter(dataloader)

    model.eval()
    with torch.no_grad():

        for i in range(5):
            # grab first example
            xs, ys = next(dataloader_iter)

            x_view_1, x_view_2 = xs[0][0], xs[1][0]

            x_view_1 = x_view_1.view(1, CHANNELS, H, W).cpu().numpy()
            x_view_2 = x_view_2.view(1, CHANNELS, H, W).cpu().numpy()

            x_view_1 = np.transpose(x_view_1, (0, 2, 3, 1))
            x_view_2 = np.transpose(x_view_2, (0, 2, 3, 1))

            axarr[i, 0].imshow(x_view_1[0])
            axarr[i, 1].imshow(x_view_2[0])

            for j in (0, 1):
                axarr[i, j].set_xticks([])

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
        RandomRepolarizationTransform(repol_prob=0.5),
        RandomSiteMaskingTransform(mask_ratio=0.5),
    ]
)


img_tfms = transforms.Compose(img_tfms)

train = FashionMNIST(
    "data/",
    transform=ContrastiveTransformations(img_tfms, n_views=2),
    train=True,
    download=True,
)
test = FashionMNIST(
    "data/",
    transform=ContrastiveTransformations(img_tfms, n_views=2),
    train=False,
    download=True,
)

# train = torchvision.datasets.ImageFolder(
#     "data/simulated/train/",
#     transform=ContrastiveTransformations(img_tfms, n_views=2),
# )
# test = torchvision.datasets.ImageFolder(
#     "data/simulated/test/",
#     transform=ContrastiveTransformations(img_tfms, n_views=2),
# )

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


KERNEL_SIZE = (1, 5)
PADDING = (0, 2)
OUTPUT_PADDING = (0, 1)
STRIDE = (1, 2)

model = models.Encoder1D(
    in_channels=CHANNELS,
    enc_dim=128,
    proj_dim=32,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    in_HW=(H, W),
    collapse=True,
)

model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = losses.SimCLRLoss()

print("Start training VAE...")

res = []

for epoch in range(EPOCHS):
    plot_example(
        model,
        test_loader,
        plot_name=f"fig/reconstructions/{epoch}.png",
    )

    train_loss = train_loop(
        model,
        train_loader,
        loss_fn,
        optimizer,
    )
    test_loss = test_loop(
        model,
        test_loader,
        loss_fn,
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


test = FashionMNIST(
    "data/",
    transform=img_tfms,
    train=False,
    download=True,
)

test_loader = DataLoader(
    dataset=test,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

f, ax = plt.subplots()


model.eval()

reps = []
labels = []
with torch.no_grad():
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm.tqdm(test_loader)):

        batch_x = batch_x.to(DEVICE)
        feats = model(batch_x)
        feats = feats.cpu().numpy()
        reps.append(feats)
        labels.append(batch_y)
reps = np.concatenate(reps)
labels = np.concatenate(labels)

X_train, X_test, y_train, y_test = train_test_split(reps, labels)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print (clf.score(X_test, y_test))
# print (reps)
# print (np.sum(np.argmax(reps, axis=1) == labels) / reps.shape[0])

if LATENT_DIM > 2:
    clf = PCA(n_components=2)
    reps = clf.fit_transform(reps)
f, ax = plt.subplots()
ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
f.savefig("vae.coords.png", dpi=200)
