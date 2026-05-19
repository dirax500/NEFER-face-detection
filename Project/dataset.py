from __future__ import annotations
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset

def events_to_voxel_grid(events, T=5, height=360, width=640):
    if len(events) == 0:
        return torch.zeros(T * 2, height, width, dtype=torch.float32)
    voxel = np.zeros((T, 2, height, width), dtype=np.float32)
    t0 = events[0,0]; t1 = events[-1,0]
    dt = max(t1 - t0, 1.0)
    xs   = np.clip(events[:,1].astype(np.int32), 0, width  - 1)
    ys   = np.clip(events[:,2].astype(np.int32), 0, height - 1)
    ps   = np.clip(events[:,3].astype(np.int32), 0, 1)
    bins = np.clip(((events[:,0] - t0) / dt * T).astype(np.int32), 0, T-1)
    np.add.at(voxel, (bins, ps, ys, xs), 1.0)
    voxel = voxel.reshape(T * 2, height, width)
    for c in range(T * 2):
        ch = voxel[c]
        if ch.max() > 0:
            pos = ch[ch > 0]
            p99 = float(np.percentile(pos, 99)) if len(pos) > 0 else 1.0
            voxel[c] = np.clip(ch, 0, p99) / (p99 + 1e-6)
    return torch.from_numpy(voxel)

class NEFERDataset(Dataset):
    def __init__(self, raw_dir, ann_dir, split='train', window_ms=50.0,
                 T=5, height=360, width=640, min_events=50,
                 augment=True, val_fraction=0.15, seed=42):
        self.T       = T
        self.H       = height
        self.W       = width
        self.augment = augment and (split == 'train')

        sample_dir = '/raid/didar_rakhimbay/data/samples'
        all_files  = sorted(glob.glob(os.path.join(sample_dir, '*.npz')))

        if not all_files:
            raise FileNotFoundError(
                'No .npz files in ' + sample_dir +
                ' — run the pre-extraction script first.')

        # Split by index (deterministic)
        random.seed(seed)
        indices = list(range(len(all_files)))
        random.shuffle(indices)
        n_val = max(1, int(len(all_files) * val_fraction))

        if split == 'val':
            chosen = [all_files[i] for i in indices[:n_val]]
        elif split == 'train':
            chosen = [all_files[i] for i in indices[n_val:]]
        else:
            chosen = all_files

        self.files = chosen
        print('[NEFERDataset] split=%s  samples=%d  (loading from .npz)' % (split, len(self.files)))

    def _augment(self, voxel, bbox):
        bbox = bbox.copy()
        if random.random() < 0.5:
            voxel = torch.flip(voxel, dims=[-1])
            bbox[0], bbox[2] = 1.0 - bbox[2], 1.0 - bbox[0]
        if random.random() < 0.4:
            voxel = voxel + torch.randn_like(voxel) * 0.03
        if random.random() < 0.35:
            voxel = voxel * (torch.rand_like(voxel) > 0.25).float()
        return torch.clamp(voxel, 0., 1.), bbox

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        events = data['events']
        bbox   = data['bbox'].astype(np.float32)
        voxel  = events_to_voxel_grid(events, T=self.T, height=self.H, width=self.W)
        if self.augment:
            voxel, bbox = self._augment(voxel, bbox)
        return voxel, torch.from_numpy(bbox)
