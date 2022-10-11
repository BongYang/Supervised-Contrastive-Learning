import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, opt, transform = None):
        pass
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return 0
    
class TestDataset(Dataset):
    def __init__(self, opt, transform = None):
        pass
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return 0
    
def set_loader(opt, is_pretrain=True):
    # construct data loader
    
    # TODO: Implement mean std
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ValueError('dataset not supported: {}'.format(opt.dataset))
    
    if is_pretrain:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform = TwoCropTransform(train_transform)
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(TrainDataset(opt, transform=train_transform),
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(TestDataset(opt, transform=val_transform), 
                            batch_size=8, 
                            shuffle=False,
                            num_workers=8, pin_memory=True)

    return train_loader, val_loader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]