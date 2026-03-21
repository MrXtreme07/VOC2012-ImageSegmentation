import torch
import matplotlib.pyplot as plt
import numpy as np

from models.unet import UNet


# -------- LOAD MODEL --------

model = UNet(num_classes=21)
checkpoint = torch.load('checkpoints/model_epoch_150.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# -------- EXTRACT FIRST CONVOLUTION KERNELS --------

for name, param in model.named_parameters():

    if "weight" in name and len(param.shape) == 4:
        kernels = param.detach().numpy()
        print("Using layer:", name)
        break


# kernels shape:
# [out_channels, in_channels, k, k]

num_kernels = min(16, kernels.shape[0])

plt.figure(figsize=(8,8))

for i in range(num_kernels):

    kernel = kernels[i, 0]   # take first input channel

    plt.subplot(4,4,i+1)
    plt.imshow(kernel, cmap="gray")
    plt.axis("off")

plt.suptitle("Learned Convolution Kernels")
plt.show()


# -------- LOSS PLOT --------

losses = []

with open("losses.txt") as f:
    for line in f:
        losses.append(float(line.strip()))

epochs = np.arange(1, len(losses)+1)

plt.figure()

plt.plot(epochs[24:150], losses[24:150])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.grid(True)
plt.show()