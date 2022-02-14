
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


#%%
class MouseOpenField(Dataset):
    def __init__(self, home_dir,transform=None):
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
#%%
face_dataset = MouseOpenField(home_dir='G://OF')

#%%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        flow, mask, frame = sample['flow'], sample['mask'], sample['frame']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W

        flow = np.transpose(flow, (2, 0, 1))
        #mask = np.transpose(mask, (2, 0, 1))
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
