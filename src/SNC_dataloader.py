import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from easydict import EasyDict

class SNCVoxelDataset(Dataset):
    def __init__(self, dataroot, phase):
        self.root_dir = os.path.join(dataroot, phase)
        self.pt_files = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(self.root_dir)
            for file in files if file.endswith('.pt')
        ])
        if len(self.pt_files) == 0:
            raise RuntimeError(f"No .pt files found in {self.root_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        try:
            data = torch.load(pt_path , weights_only=True)
            voxel = data['Volume'].float().unsqueeze(0)          # (1, 32, 32, 32)
            sample = data['surfaceSamples'].float()              # (N, 3)
            cp = data['closestPoints'].float().reshape(-1, 3)    # (32768, 3)
        except Exception as e:
            print(f"Error reading file {pt_path}: {e}")
            return None

        return {
            'voxel': voxel,
            'sample': sample,
            'cp': cp,
            'path': pt_path
        }


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def create_dataloader(config, split='train'):
    assert hasattr(config, 'dataset'), "Missing 'dataset' section in config"

    # 根据 split 使用正确的 phase 目录
    if split == 'train':
        phase_dir = config.dataset.train_split
    elif split == 'val':
        phase_dir = config.dataset.val_split
    elif split == 'test':
        phase_dir = config.dataset.test_split
    else:
        raise ValueError(f"Unsupported split type: {split}")

    dataset = SNCVoxelDataset(config.dataset.dataroot, phase_dir)

    batch_size = config.training.batch_size if hasattr(config, 'training') else 16
    num_workers = getattr(config, 'num_workers', 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        drop_last=True,
        collate_fn=custom_collate
    )
    return dataloader


if __name__ == '__main__':
    config = load_config('config.yaml')
    train_loader = create_dataloader(config, split='train')

    print("Dataset and DataLoader initialized.")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    for i, batch in enumerate(train_loader):
        print(f"\n📦 Batch {i}")
        print("  Voxel shape:", batch['voxel'].shape)
        print("  Sample shape:", batch['sample'].shape)
        print("  CP shape:", batch['cp'].shape)
        break
