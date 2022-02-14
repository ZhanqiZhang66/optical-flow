import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from __future__ import print_function, division
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#%% Define dataset https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class MouseOpenField(Dataset):
    def __init__(self, home_dir, transform=None):
        self.home_dir = home_dir

        self.flow_dir = self.home_dir + "/flow"
        self.mask_dir = self.home_dir + "/mask"
        self.frame_dir = self.home_dir + "/img"
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.flow_dir))

    def __getitem__(self, name):
        flow_name = name + 'flow.npy'
        mask_name = name + 'animal_mask.npy'
        frame_name = name + '.npy'
        flow = np.load(os.path.join(self.flow_dir,
                                    flow_name))
        # print(flow.shape)
        mask = np.load(os.path.join(self.mask_dir,
                                    mask_name))
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        frame = np.load(os.path.join(self.frame_dir,
                                     frame_name))

        flow = flow * mask
        frame = frame * mask
        sample = {'frame': frame, 'mask': mask, 'flow': flow}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        flow, mask, frame = sample['flow'], sample['mask'], sample['frame']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W

        flow = np.transpose(flow, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        frame = np.transpose(frame, (2, 0, 1))
        return {'flow': torch.from_numpy(flow),
                'mask': torch.from_numpy(mask),
                'frame': torch.from_numpy(frame)}
#%%
transformed_dataset = MouseOpenField(home_dir='G://OF',
                                     transform=transforms.Compose([
                                            ToTensor()
                                           ]))
i = 0
for file in os.listdir(transformed_dataset.frame_dir):

    i += 1
    file_name = file[:-4]
    sample = transformed_dataset[file_name]

    print(file, sample['flow'].size(), sample['mask'].size(), sample['frame'].size())

    if i == 3:
        break
#%%
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
#%%
train_size = int(0.8 * len(transformed_dataset))
test_size = len(transformed_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(transformed_dataset, [train_size, test_size])

bs = 100
train_loader = DataLoader(train_dataset, batch_size=bs,
                        shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs,
                        shuffle=True)
#%%
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

#%%
# build model
vae = VAE(x_dim=2073600, h_dim1=512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

vae
#%%
optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
#%%
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    #%%