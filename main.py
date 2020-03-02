import matplotlib.pyplot as plt
import torch
import numpy as np

from TorchTest.Utilities import *

input_shape = (512, 512, 3)

model = load_model("TorchTest/best_model.pt")
model = append_heatmap(model.cuda(), input_shape, 5)

image = readImage("TorchTest/bad2.jpg", resize=(512, 512))

image = torch.from_numpy(image)
image = image.unsqueeze(0)
model = model.cuda()
out = model(image.cuda())

heatG = out[1][0][0]
heatB = out[1][0][1]


heatG = heatG.squeeze().cpu()
plt.figure()
plt.imshow(image.squeeze())
plt.imshow(heatG, cmap="jet", alpha=.25)
plt.show()

heatB = heatB.squeeze().cpu()
plt.figure()
plt.imshow(image.squeeze())
plt.imshow(heatB, cmap="jet", alpha=.25)
plt.show()

