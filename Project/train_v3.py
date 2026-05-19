from __future__ import annotations
import os, sys, glob, random, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

class VoxelDataset(Dataset):
    def __init__(self, split='train', val_fraction=0.15, seed=42, augment=True):
        voxel_dir = '/raid/didar_rakhimbay/data/voxels'
        files = sorted(glob.glob(os.path.join(voxel_dir, '*.pt')))
        if not files:
            raise FileNotFoundError('No .pt files in ' + voxel_dir)
        random.seed(seed)
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        if split == 'val':     self.files = files[:n_val]
        elif split == 'train': self.files = files[n_val:]
        else:                  self.files = files
        self.augment = augment and split == 'train'
        print('[Dataset] split=%s  n=%d' % (split, len(self.files)))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data  = torch.load(self.files[idx], weights_only=True)
        voxel = data['voxel']
        bbox  = data['bbox']
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                voxel = torch.flip(voxel, dims=[-1])
                bbox  = bbox.clone()
                bbox[0], bbox[2] = 1.0-bbox[2].item(), 1.0-bbox[0].item()
            # Gaussian noise
            if random.random() < 0.4:
                voxel = torch.clamp(voxel + torch.randn_like(voxel)*0.03, 0, 1)
            # Event dropout (simulate sparse events)
            if random.random() < 0.3:
                mask  = torch.rand_like(voxel) > 0.3
                voxel = voxel * mask
            # Random brightness
            if random.random() < 0.3:
                voxel = torch.clamp(voxel * random.uniform(0.7, 1.3), 0, 1)
        return voxel, bbox

class ResBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class ImprovedFaceDetector(nn.Module):
    """
    v3 — Deeper backbone with residual connections.
    Better feature extraction → higher IoU.
    """
    def __init__(self, in_ch=10):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),

            # Stage 1
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),

            # Stage 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            ResBlock(128),

            # Stage 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256),
            ResBlock(256),

            # Stage 4
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(),
            ResBlock(512),

            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,   4), nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))

def iou_loss(pred, gt):
    px1,py1,px2,py2 = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    gx1,gy1,gx2,gy2 = gt[:,0],  gt[:,1],  gt[:,2],  gt[:,3]
    ix1=torch.max(px1,gx1); iy1=torch.max(py1,gy1)
    ix2=torch.min(px2,gx2); iy2=torch.min(py2,gy2)
    inter=(ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
    pa=(px2-px1).clamp(0)*(py2-py1).clamp(0)
    ga=(gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)
    union=pa+ga-inter+1e-6
    iou=inter/union
    ex1=torch.min(px1,gx1); ey1=torch.min(py1,gy1)
    ex2=torch.max(px2,gx2); ey2=torch.max(py2,gy2)
    c2=(ex2-ex1)**2+(ey2-ey1)**2+1e-6
    pcx=(px1+px2)/2; pcy=(py1+py2)/2
    gcx=(gx1+gx2)/2; gcy=(gy1+gy2)/2
    d2=(pcx-gcx)**2+(pcy-gcy)**2
    return (1-iou+d2/c2).mean()

def compute_iou(pred, gt):
    ix1=max(pred[0],gt[0]); iy1=max(pred[1],gt[1])
    ix2=min(pred[2],gt[2]); iy2=min(pred[3],gt[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    pa=(pred[2]-pred[0])*(pred[3]-pred[1])
    ga=(gt[2]-gt[0])*(gt[3]-gt[1])
    return inter/(pa+ga-inter+1e-6)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ious=[]; lats=[]
    for voxel, gt in loader:
        voxel=voxel.to(device); gt=gt.to(device)
        t0=time.time()
        pred=model(voxel)
        lats.append((time.time()-t0)*1000/voxel.shape[0])
        for i in range(pred.shape[0]):
            ious.append(compute_iou(pred[i].cpu().tolist(),
                                    gt[i].cpu().tolist()))
    return {
        'mean_iou': float(np.mean(ious)),
        'ap50':     float(np.mean([v>=0.5 for v in ious])),
        'ap40':     float(np.mean([v>=0.4 for v in ious])),
        'lat_ms':   float(np.mean(lats)),
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[v3] device:', device)

    train_ds     = VoxelDataset('train', augment=True)
    val_ds       = VoxelDataset('val',   augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=4, pin_memory=True)

    model    = ImprovedFaceDetector(in_ch=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('[v3] params: %d (%.2fM)' % (n_params, n_params/1e6))

    EPOCHS    = 80
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                           total_steps=EPOCHS*len(train_loader),
                           pct_start=0.1, anneal_strategy='cos')
    scaler    = torch.cuda.amp.GradScaler()

    os.makedirs('checkpoints', exist_ok=True)
    best_iou = 0.0
    history  = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for i, (voxel, gt) in enumerate(train_loader):
            voxel=voxel.to(device); gt=gt.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(voxel)
                loss = iou_loss(pred, gt) + F.mse_loss(pred, gt)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print('  [ep%03d %4d/%d] loss=%.4f lr=%.2e' % (
                      epoch, i+1, len(train_loader),
                      loss.item(), optimizer.param_groups[0]['lr']))

        ep_loss = total_loss / len(train_loader)
        ep_time = time.time() - t0

        if epoch % 5 == 0:
            m = validate(model, val_loader, device)
            print('[Epoch %d] loss=%.4f  mIoU=%.3f  AP@50=%.3f  AP@40=%.3f  lat=%.1fms  time=%.0fs' % (
                  epoch, ep_loss, m['mean_iou'], m['ap50'], m['ap40'],
                  m['lat_ms'], ep_time))
            if m['mean_iou'] > best_iou:
                best_iou = m['mean_iou']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {'in_channels':10,'img_h':360,'img_w':640},
                    'architecture': 'ImprovedFaceDetector_v3',
                    'best_iou': best_iou,
                    'epoch': epoch,
                }, 'checkpoints/best_v3.pth')
                print('  -> New best mIoU=%.3f  saved to checkpoints/best_v3.pth' % best_iou)
            history.append({'epoch':epoch,'loss':ep_loss,**m})
            with open('checkpoints/history_v3.json','w') as f:
                json.dump(history, f, indent=2)
        else:
            print('[ep%03d] loss=%.4f  time=%.0fs' % (epoch, ep_loss, ep_time))

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'in_channels':10,'img_h':360,'img_w':640},
        'architecture': 'ImprovedFaceDetector_v3',
        'best_iou': best_iou,
    }, 'final_model_v3.pth')
    print('Done! Best mIoU=%.3f  Saved -> final_model_v3.pth' % best_iou)

if __name__ == '__main__':
    os.chdir('/raid/didar_rakhimbay/Project')
    main()
