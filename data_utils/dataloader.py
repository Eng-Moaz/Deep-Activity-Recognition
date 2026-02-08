from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.dataset import VolleyballSceneDataset, VolleyballPlayerDataset

def get_data_loaders(cfg):

    # Standard ImageNet Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract settings from Config Class
    data_root = cfg.data_root
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    exp_name = cfg.experiment_name

    # Automatic Mode Selection
    if "baseline1" in exp_name:
        mode = "scenefull"      # B1: Whole Image
        task = "scene"
    elif "baseline3_stage1" in exp_name:
        mode = "action_train"   # B3 Step A: Single Player Crops
        task = "scene"
    elif "baseline3_stage2" in exp_name:
        mode = "scenecrops"     # B3 Step B: Stack of 12 Players
        task = "scene"
    elif "baseline4" in exp_name:
        mode = "temporal"       # B4: Sequence of 9 Frames
        task = "scene"
    elif "baseline5" in exp_name:
        mode = "temporal"       # B5: Sequence of Tracking Crops
        task = "player"
    else:
        # Fallback (Safety)
        print(f"Warning: Unknown experiment name '{exp_name}'. Defaulting to B3 Scene Crops.")
        mode = "scenecrops"
        task = "scene"

    print(f"Loading Data for: {exp_name} | Task: {task} | Mode: {mode}")

    # Instantiate Datasets
    if task == "scene":
        train_ds = VolleyballSceneDataset(data_root, "train", mode=mode, transform=transform)
        val_ds   = VolleyballSceneDataset(data_root, "val",   mode=mode, transform=transform)
        test_ds  = VolleyballSceneDataset(data_root, "test",  mode=mode, transform=transform)

    elif task == "player":
        train_ds = VolleyballPlayerDataset(data_root, "train", mode=mode, transform=transform)
        val_ds   = VolleyballPlayerDataset(data_root, "val",   mode=mode, transform=transform)
        test_ds  = VolleyballPlayerDataset(data_root, "test",  mode=mode, transform=transform)

    else:
        raise ValueError(f"Unknown task: {task}")

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader