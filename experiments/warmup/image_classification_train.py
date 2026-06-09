#---------------------------
#
#---------------------------


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm

#Pfadfestlegung
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


from upt.models.approximator import Approximator
from upt.models.decoder_classifier import DecoderClassifier
from upt.models.encoder_image import EncoderImage
from upt.models.upt_image_classifier import UPTImageClassifier


# Gerät festlegen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")






# initialize dataset
transform = ToTensor()
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

# hyperparameters
dim = 384  # ~6M parameter model
num_heads = 6
epochs = 50
batch_size = 256

# initialize model
model = UPTImageClassifier(
    encoder=EncoderImage(
        # CIFAR has 3 channels (RGB)
        input_dim=3,
        # CIFAR has 32x32 images -> patch_size=4 results in 64 patch tokens
        resolution=32,
        patch_size=4,
        # ViT-T latent dimension
        enc_dim=dim,
        enc_num_heads=num_heads,
        # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
        enc_depth=4,
        # the perceiver is optional, it changes the size of the latent space to NUM_LATENT_TOKENS tokens
        # perc_dim=dim,
        # perc_num_heads=num_heads,
        # num_latent_tokens=32,
    ),
    approximator=Approximator(
        # tell the approximator the dimension of the input (perc_dim or enc_dim of encoder)
        input_dim=dim,
        # as in ViT-T
        dim=dim,
        num_heads=num_heads,
        # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
        depth=4,
    ),
    decoder=DecoderClassifier(
        # tell the decoder the dimension of the input (dim of approximator)
        input_dim=dim,
        # CIFAR10 has 10 classes
        num_classes=10,
        # as in ViT-T
        dim=dim,
        num_heads=num_heads,
        # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
        depth=4,
    ),
)
model = model.to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# setup dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# initialize optimizer and learning rate schedule (linear warmup for first 10% -> linear decay)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
total_updates = len(train_dataloader) * epochs
warmup_updates = int(total_updates * 0.1)
lrs = torch.concat(
    [
        # linear warmup
        torch.linspace(0, optim.defaults["lr"], warmup_updates),
        # linear decay
        torch.linspace(optim.defaults["lr"], 0, total_updates - warmup_updates),
    ],
)

# train model
update = 0
pbar = tqdm(total=total_updates)
pbar.update(0)
pbar.set_description("train_loss: ????? train_accuracy: ????% test_accuracy: ????%")
test_accuracy = 0.0
train_losses = []
train_accuracies = []
test_accuracies = []
loss = None
train_accuracy = None
for _ in range(epochs):
    # train for an epoch
    for x, y in train_dataloader:
        # prepare forward pass
        x = x.to(device)
        y = y.to(device)

        # schedule learning rate
        for param_group in optim.param_groups:
            param_group["lr"] = lrs[update]

        # forward pass
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)

        # backward pass
        loss.backward()

        # update step
        optim.step()
        optim.zero_grad()

        # status update
        train_accuracy = ((y_hat.argmax(dim=1) == y).sum() / y.numel()).item()
        update += 1
        pbar.update()
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}%"
        )
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)

    # evaluate
    num_correct = 0
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_hat = model(x)
        num_correct += (y_hat.argmax(dim=1) == y).sum().item()
    test_accuracy = num_correct / len(test_dataset)
    test_accuracies.append(test_accuracy)
    pbar.set_description(
        f"train_loss: {loss.item():.4f} "
        f"train_accuracy: {train_accuracy * 100:4.1f}% "
        f"test_accuracy: {test_accuracy * 100:4.1f}%"
    )
pbar.close()


checkpoint_name = (
    f"upt_cifar10_dim{dim}_heads{num_heads}_ep{epochs}.pth"
)

torch.save(
    {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "test_accuracy": test_accuracy,
        "dim": dim,
        "num_heads": num_heads,
    },
    checkpoint_name,
)

print(f"Checkpoint gespeichert: {checkpoint_name}")





import matplotlib.pyplot as plt
_, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(range(len(train_losses)), train_losses)
axes[0].set_xlabel("Updates")
axes[0].set_ylabel("Train Loss")
axes[0].set_title("Train Loss")
axes[0].grid(True)
axes[1].plot(range(len(train_accuracies)), train_accuracies)
axes[1].set_xlabel("Updates")
axes[1].set_ylabel("Train Accuracy")
axes[1].set_title("Train Accuracy")
axes[1].grid(True)
axes[2].plot(range(len(test_accuracies)), test_accuracies, marker="o")
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Test Accuracy")
axes[2].set_title("Test Accuracy")
axes[2].grid(True)
plt.tight_layout()
plt.show()