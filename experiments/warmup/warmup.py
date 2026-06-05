# create pointclouds
import torch
import numpy
import matplotlib.pyplot as plt

#Pfadfestlegung
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))




point_clouds = [
    torch.randn(10, 3),             #Punktwolke 1 mit 10 Punkten und pro Punkt 3 Dimensionen
    torch.randn(100, 3),            #Punktwolke 2 mit 100 Punkten  und pro Punkt 3 Dimensionen
    torch.randn(1000, 3),           #Punktwolke 3 mit 1000 Punkten  und pro Punkt 3 Dimensionen
    torch.randn(10000, 3),          #Punktwolke 4 mit 10000 Punkten  und pro Punkt 3 Dimensionen
    torch.randn(100000, 3),         #Punktwolke 5 mit 100000 Punkten  und pro Punkt 3 Dimensionen
]
print(point_clouds[1])

#-----------------------------------------------------------------------
#Exkurs zum besseren Verständnis was hier passiert
# point_cloud = torch.randn(100, 3)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# ax.scatter(
#     point_cloud[:,0],
#     point_cloud[:,1],
#     point_cloud[:,2]
# )

# plt.show()
# auskommtentieren/einkommentieren mit Strg und #
#----------------------------------------------------------------------------------

# make a dense tensor representing of all point clouds
from torch.nn.utils.rnn import pad_sequence
dense = pad_sequence(point_clouds, batch_first=True)
print(f"dense.shape: {dense.shape}")
# Welche Punktwolke besitzt die meisten Punkte und alle kleineren werden mittels 0 Einträgen
# auf die gleiche größe gebracht

# make a sparse tensor representation of all point clouds
sparse = torch.concat(point_clouds)
print(f"sparse.shape: {sparse.shape}")
# Alle Punktewolken werden hintereinander gespeichert.
# Dadurch können Punktwolken unterschiedlicher Größe gemeinsam
# verarbeitet werden, ohne sie auf eine feste Länge aufzufüllen.


# create batch_idx tensor to assign indices of the sparse tensor to indices of the pointcloud
batch_idx = []
for i in range(len(point_clouds)):
    batch_idx += [i] * len(point_clouds[i])
batch_idx = torch.tensor(batch_idx)
print(f"batch_idx.shape: {batch_idx.shape}")
print(f"the first 10 samples belong to the first pointcloud (i.e. point_clouds[0]): {batch_idx[:10]}")
print(f"the 11th point in the sparse tensor belongs to the second pointcloud (i.e. point_clouds[1]): {batch_idx[10]}")
# Jeder Punkt wird durch ein weiteres Feature beschrieben, das angibt
# aus welcher Punktewolke dieser ist. Somit ist die Information wieder vollständig


#---------------------------
#Position Encoding
#---------------------------
import torch
import matplotlib.pyplot as plt
from kappamodules.layers.continuous_sincos_embed import ContinuousSincosEmbed

# positions in range [0, 200]
pos = torch.linspace(0, 200, 201)       #201 Punkte im Bereich von 0 bis 200

# the embeddings have dimension 192
embed = ContinuousSincosEmbed(dim=192, ndim=1)
# embed the positions
posembed = embed(pos.unsqueeze(0).unsqueeze(2)).squeeze(2).squeeze(0)
# plot it
plt.rcParams['figure.figsize'] = (9, 6)
plt.imshow(posembed, aspect="auto")
plt.xlabel("dim")
plt.ylabel("position")
plt.title(f"positions in [{int(pos.min().item())}, {int(pos.max().item())}]")
plt.show()

#Ergebnis  dieses Vorgangs
# Position | Embedding Dim 1 | Embedding Dim 2 | Embedding Dim 3 | ... | Embedding Dim 192 
#    1     |
#   .      |
#   .      |
#   .      |
#   200    |


#----------------------
#Encoder
#----------------------


# create some input features and positions
import torch
# create 16 points with 4 features each
input_feat = torch.randn(16, 4)
# 3D coordinates (scaled to be in [0, 200])
input_pos = torch.rand(16, 3) * 200
# create batch_idx (we assume 2 point clouds with length 6 and 10)
batch_idx = torch.tensor([0] * 6 + [1] * 10)
# select 2 supernodes per pointcloud
supernode_idxs = torch.tensor([0, 2, 9, 13])



from upt.modules.supernode_pooling import SupernodePooling
supernode_pooling = SupernodePooling(
    # use a large radius because we dont have a lot of points
    radius=100,
    # max_degree is not relevant here because we have too little points
    max_degree=32,
    # same as dimension of input_feat
    input_dim=4,
    # we use a small hidden dimension for the MLP here
    hidden_dim=8,
    # same as dimension of input_pos
    ndim=3,
)

supernodes = supernode_pooling(
    input_feat=input_feat,
    input_pos=input_pos,
    supernode_idxs=supernode_idxs,
    batch_idx=batch_idx,
)
print(f"supernodes.shape: {supernodes.shape}")


#----------------------
#
#----------------------



# create 4 transformer blocks
from kappamodules.transformer import PrenormBlock
blocks = [PrenormBlock(dim=8, num_heads=2) for _ in range(4)]

# process supernodes with transformer
transformed_supernodes = supernodes
for block in blocks:
    transformed_supernodes = block(transformed_supernodes)
print(f"transformed_supernodes.shape: {transformed_supernodes.shape}")


#------------------------------
#
#------------------------------

# create perceiver block
from kappamodules.transformer import PerceiverBlock
# create query (this is a learnable vector later on)
# we use 2 latent tokens here
query = torch.randn(1, 2, 8)
encoder_perceiver = PerceiverBlock(dim=8, num_heads=2)
latent_tokens = encoder_perceiver(q=query, kv=transformed_supernodes)
print(f"latent_tokens.shape: {latent_tokens.shape}")

# the same thing can be done in one go
from kappamodules.transformer import PerceiverPoolingBlock
encoder_perceiver_onego = PerceiverPoolingBlock(num_query_tokens=2, dim=8, num_heads=2)
latent_tokens_onego = encoder_perceiver_onego(kv=transformed_supernodes)
print(f"latent_tokens_onego.shape: {latent_tokens_onego.shape}")


#-------------------------------
#Approximator
#--------------------------------
from kappamodules.transformer import PrenormBlock
approximator = torch.nn.Sequential(*[PrenormBlock(dim=8, num_heads=2) for _ in range(4)])
latent_tokens_next_timestep = approximator(latent_tokens)
print(f"latent_tokens_next_timestep.shape: {latent_tokens_next_timestep.shape}")


#-------------------------------
#Decoder
#--------------------------------
# create some output positions (we query at 4 positions per pointcloud)
import torch
# (2=number of pointclouds, 4=number of output positions, 3=3D positions)
output_pos = torch.randn(2, 4, 3)

# some transformer blocks
from kappamodules.transformer import PrenormBlock
decoder_transformer = torch.nn.Sequential(*[PrenormBlock(dim=8, num_heads=2) for _ in range(4)])

# create perceiver
from kappamodules.transformer import PerceiverBlock
decoder_perceiver = PerceiverBlock(dim=8, num_heads=2)

# create positional encoding and MLP to encode positions
from kappamodules.layers import ContinuousSincosEmbed
pos_embed = ContinuousSincosEmbed(dim=8, ndim=3)
output_pos_mlp = torch.nn.Sequential(
    torch.nn.Linear(8, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 8),
)


# apply transformer
decoder_perceiver_kv = decoder_transformer(latent_tokens_next_timestep)

# encode output positions into query vector
query = output_pos_mlp(pos_embed(output_pos))

# apply perceiver decoder
pred = decoder_perceiver(q=query, kv=decoder_perceiver_kv)

print(f"decoder_perceiver_kv.shape: {decoder_perceiver_kv.shape}")
print(f"query.shape: {query.shape}")
print(f"pred.shape: {pred.shape}")
