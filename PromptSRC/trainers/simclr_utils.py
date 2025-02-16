# simclr_utils.py
import random
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class GaussianBlur(object):
    """SimCLR 논문에서 제안된 Gaussian Blur transform"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(sigma=[0.1, 2.0]),
    transforms.ToTensor(),
])

class SimCLRDataset(Dataset):
    """
    (img, label) 튜플을 받아, SimCLR을 위해 2개의 서로 다른 aug 결과를 [img1, img2]로 반환.
    """
    def __init__(self, base_dataset, transform):
        super().__init__()
        self.base_dataset = base_dataset  # (PIL_image, label) 형태
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return [img1, img2], label

def simclr_collate_fn(batch):
    """
    batch: list of ( [img1, img2], label ) 튜플
    => { "img1":Tensor, "img2":Tensor, "label":Tensor }
    """
    img1_list, img2_list, labels = [], [], []
    for (img_pair, lbl) in batch:
        img1_list.append(img_pair[0])
        img2_list.append(img_pair[1])
        labels.append(lbl)
    img1 = torch.stack(img1_list, dim=0)
    img2 = torch.stack(img2_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"img1": img1, "img2": img2, "label": labels}

class NTXentLoss(nn.Module):
    """
    SimCLR의 NT-Xent(NT-Xentropic) Loss.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.matmul(z, z.t()) / self.temperature

        N = z1.shape[0]
        mask = ~torch.eye(2*N, dtype=bool, device=z.device)
        sim = sim[mask].view(2*N, -1)

        pos_idx = torch.arange(N, 2*N, device=z.device)
        neg_idx = torch.arange(0, N, device=z.device)
        targets = torch.cat([pos_idx, neg_idx], dim=0)  # [2N]
        loss = self.ce(sim, targets)
        return loss
