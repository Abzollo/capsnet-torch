
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from training_configs import *
from capsnet import CapsNet, spread_loss

# Set random seed if given
torch.manual_seed(RANDOM_SEED or torch.initial_seed())

### Prepare training and test sets ###
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Initialize capsnet
capsnet = CapsNet()
# Initialize optimizer and loss function
optimizer = torch.optim.Adam(capsnet.get_params())
# Using spread loss for capsnet
# loss_function = lambda x, y: spread_loss(x, y, 1)
loss_function = nn.CrossEntropyLoss()
loss_margin = 0.2

# Start training
start_time = time.time(); print('Training...')
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # Zero gradients
        optimizer.zero_grad()

        # Unpack input/output and turn them into variables
        images, labels = data
        images, labels = Variable(images), Variable(labels)

        # Do a forward pass, compute loss, then do a backward pass
        activations = capsnet(images)
        loss = loss_function(activations, labels)
        loss.backward()

        # Update parameters using the optimizer
        optimizer.step()

        # Print report when we reach a checkpoint
        running_loss += loss.item()
        if (i + 1) % CHECKPOINT == 0:
            average_loss = running_loss / CHECKPOINT
            print("[%d, %d] Loss = %.3f" % (epoch+1, i+1, average_loss))
            print("Time elapsed = %ds" % (time.time() - start_time))
            running_loss = 0.0

    print('\n[%d/%d] Loss = %.3f\n' % (epoch+1, epochs, loss.data.mean()))
    if (EPOCHS > 1): loss_margin += (0.9 - loss_margin) / (EPOCHS - 1)

print('Finished training (time elapsed = %d seconds)' % (time.time() - start))



### Test ###
start = time.time(); print('Testing...', end='')
correct = 0
total = 0
for i, data in enumerate(testloader):
    images, labels = data
    # Predict
    activations = capsnet(Variable(images, requires_grad=False))
    _, pred_class = torch.max(activations, 1)
    # Check predition
    total += BATCH_SIZE
    correct += (pred_class == labels).sum()

print('\nFinished testing (time elapsed = %d seconds)' % (time.time() - start))
print('Correct = %d, Total = %d, Error = %.2f%%' % (
    correct, total, 100 * (1 - correct / total)))






