import torchvision.transforms as T
from torch.utils.data import DataLoader
from data_utils.dataset import SpatialScene, TemporalScene


def get_transforms(split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    else:
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])


def get_data_loaders(cfg, mode):
    root = cfg["data"]["root"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = 4

    if mode == "spatial":
        DatasetClass = SpatialScene
    elif mode == "temporal":
        DatasetClass = TemporalScene
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Loading {mode} Data from: {root}")

    train_ds = DatasetClass(root, split='train', transform=get_transforms('train'))
    val_ds = DatasetClass(root, split='val', transform=get_transforms('val'))
    test_ds = DatasetClass(root, split='test', transform=get_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader