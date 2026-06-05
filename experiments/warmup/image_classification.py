#Pakete
import torch


#Pfadfestlegung
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


# initialize CIFAR10
from torchvision.datasets import CIFAR10
train_dataset = CIFAR10(root="./data", train=True, download=True)

print(f"Anzahl Trainingsbilder: {len(train_dataset)}")


# visualize image
image, label = train_dataset[0]
print(f"label: {label}")
image.show()

print(type(image))
print(image.size)




from upt.models.encoder_image import EncoderImage
encoder = EncoderImage(
    # CIFAR has 3 channels (RGB)
    input_dim=3,
    # CIFAR has 32x32 images -> patch_size=4 results in 64 patch tokens
    resolution=32,
    patch_size=4,
    # ViT-T latent dimension
    enc_dim=192,
    enc_num_heads=3,
    # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
    enc_depth=4,
    # the perceiver is optional, it changes the size of the latent space to NUM_LATENT_TOKENS tokens
    # perc_dim=dim,
    # perc_num_heads=num_heads,
    # num_latent_tokens=32,
)

# we can now encode images
image, label = train_dataset[0]
# convert image to a tensor
from torchvision.transforms import ToTensor
tensor = ToTensor()(image).unsqueeze(0)
print(f"tensor.shape: {tensor.shape}")
encoded_image = encoder(tensor)
print(f"encoded_image.shape: {encoded_image.shape}")


from upt.models.approximator import Approximator
approximator = Approximator(
    # tell the approximator the dimension of the input (perc_dim or enc_dim of encoder)
    input_dim=192,
    # as in ViT-T
    dim=192,
    num_heads=3,
    # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
    depth=4,
)

approximator_output = approximator(encoded_image)
print(f"approximator_output.shape: {approximator_output.shape}")


from upt.models.decoder_classifier import DecoderClassifier
decoder = DecoderClassifier(
    # tell the decoder the dimension of the input (dim of approximator)
    input_dim=192,
    # CIFAR10 has 10 classes
    num_classes=10,
    # as in ViT-T
    dim=192,
    num_heads=3,
    # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
    depth=4,
)
prediction = decoder(approximator_output)
print(f"prediction.shape: {prediction.shape}")
print(f"decoder predicted class: {prediction.argmax(dim=1)}")



