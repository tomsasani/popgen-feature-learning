from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import models
import losses


BATCH_SIZE = 100
LATENT_DIM = 128
H, W = 64, 64
CHANNELS = 1
INPUT_DIM = H * W
LR = 1e-3
EPOCHS = 25
HIDDEN_DIMS = [16, 32, 64, 128, 256]
# HIDDEN_DIMS = [h * 2 for h in HIDDEN_DIMS]
DEVICE = torch.device("mps")


mnist_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
])

train = CIFAR10(
    "data/",
    transform=mnist_transform,
    train=True,
    download=True,
)
test = CIFAR10(
    "data/",
    transform=mnist_transform,
    train=False,
    download=True,
)

train = torchvision.datasets.ImageFolder(
    "data/simulated/train/",
    transform=mnist_transform,
)
test = torchvision.datasets.ImageFolder(
    "data/simulated/test/",
    transform=mnist_transform,
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
    kernel_size=3,
    stride=2,
    padding=1,
    hidden_dims=HIDDEN_DIMS,
    in_H=H,
)

decoder = models.Decoder(
    out_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1,
    hidden_dims=HIDDEN_DIMS,
    in_H=H,
)

model = models.VAE(encoder=encoder, decoder=decoder,)
model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = losses.VAELoss()

print("Start training VAE...")
model.train()

for epoch in range(EPOCHS):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var, z = model(x)
        loss = loss_fn(x, x_hat, mean, log_var)
        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Loss: ",
        overall_loss / (batch_idx * BATCH_SIZE),
    )

print("Finish!!")

model.eval()

f, ax = plt.subplots()

reps = []
labels = []
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
        x = x.to(DEVICE)

        x_hat, mu, log_var, z = model(x)
        z = z.cpu().numpy()
        reps.append(z)
        labels.append(y)
reps = np.concatenate(reps)
labels = np.concatenate(labels)

# clf = TSNE(n_components=2)
# X_new = clf.fit_transform(reps)
ax.scatter(reps[:, 0], reps[:, 1], c=labels, cmap="tab10")

f.savefig("coords.png", dpi=200)

x = x.view(BATCH_SIZE, CHANNELS, H, W).cpu().numpy()
x_hat = x_hat.view(BATCH_SIZE, CHANNELS, H, W).cpu().numpy()

x = np.transpose(x, (0, 2, 3, 1))
x_hat = np.transpose(x_hat, (0, 2, 3, 1))

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(x[0])
ax2.imshow(x_hat[0])
f.savefig("o.png")


# def plot_reconstructed(model, r0=(-5, 5), r1=(-5, 5), n=12):

#     img = np.zeros((n*W, n*W))
#     f, ax = plt.subplots()
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             z = torch.Tensor([[x, y]]).to(DEVICE)
#             x_hat = model.decoder(z)
#             print (x_hat.shape)
#             x_hat = x_hat.to('cpu').detach().numpy()[0, :, :]
#             img[(n - 1 - i) * W:(n - 1 - i + 1) * W, j * W:(j + 1) * W] = x_hat
#     ax.imshow(img, extent=[*r0, *r1])
#     f.savefig("recons.png")


# plot_reconstructed(model)
