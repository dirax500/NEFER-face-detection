from __future__ import annotations
import os, sys, glob, random, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# ── Dataset ───────────────────────────────────────────────────────────────────
class VoxelDataset(Dataset):
    def __init__(self, split='train', val_fraction=0.15, seed=42, augment=True):
        voxel_dir = '/raid/didar_rakhimbay/data/voxels'
        npz_dir   = '/raid/didar_rakhimbay/data/samples'

        # Try precomputed .pt first, fall back to .npz
        files = sorted(glob.glob(os.path.join(voxel_dir, '*.pt')))
        self.use_pt = len(files) > 100
        if not self.use_pt:
            files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        if not files:
            raise FileNotFoundError('No voxel or sample files found!')

        random.seed(seed)
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        if split == 'val':   self.files = files[:n_val]
        elif split == 'train': self.files = files[n_val:]
        else:                self.files = files

        self.augment = augment and split == 'train'
        print('[Dataset] split=%s  n=%d  use_pt=%s' % (split, len(self.files), self.use_pt))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        if self.use_pt:
            data  = torch.load(self.files[idx], weights_only=True)
            voxel = data['voxel']   # [10, H, W]
            bbox  = data['bbox']
        else:
            data   = np.load(self.files[idx])
            events = data['events']
            bbox   = torch.from_numpy(data['bbox'].astype(np.float32))
            voxel  = self._to_voxel(events)

        if self.augment and random.random() < 0.5:
            voxel = torch.flip(voxel, dims=[-1])
            bbox  = bbox.clone()
            bbox[0], bbox[2] = 1.0 - bbox[2].item(), 1.0 - bbox[0].item()

        return voxel, bbox

    def _to_voxel(self, events, T=5, H=360, W=640):
        if len(events) == 0:
            return torch.zeros(T*2, H, W)
        voxel = np.zeros((T, 2, H, W), dtype=np.float32)
        t0=events[0,0]; t1=events[-1,0]; dt=max(t1-t0,1.0)
        xs   = np.clip((events[:,1]*W/1280).astype(np.int32), 0, W-1)
        ys   = np.clip((events[:,2]*H/720).astype(np.int32),  0, H-1)
        ps   = np.clip(events[:,3].astype(np.int32), 0, 1)
        bins = np.clip(((events[:,0]-t0)/dt*T).astype(np.int32), 0, T-1)
        np.add.at(voxel, (bins, ps, ys, xs), 1.0)
        voxel = voxel.reshape(T*2, H, W)
        for c in range(T*2):
            ch = voxel[c]
            if ch.max() > 0:
                voxel[c] = ch / (ch.max() + 1e-6)
        return torch.from_numpy(voxel)

# ── Simple Model ──────────────────────────────────────────────────────────────
class SimpleFaceDetector(nn.Module):
    """Direct bbox regression — no anchors, no FPN, just predict [x1,y1,x2,y2]."""
    def __init__(self, in_ch=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,  64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,   4), nn.Sigmoid(),  # outputs [x1,y1,x2,y2] in [0,1]
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))

# ── IoU Loss ──────────────────────────────────────────────────────────────────
def iou_loss(pred, gt):
    """CIoU loss between predicted and GT boxes [B,4] xyxy normalised."""
    px1,py1,px2,py2 = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    gx1,gy1,gx2,gy2 = gt[:,0],  gt[:,1],  gt[:,2],  gt[:,3]

    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    pa = (px2-px1).clamp(0) * (py2-py1).clamp(0)
    ga = (gx2-gx1).clamp(0) * (gy2-gy1).clamp(0)
    union = pa + ga - inter + 1e-6
    iou   = inter / union

    ex1=torch.min(px1,gx1); ey1=torch.min(py1,gy1)
    ex2=torch.max(px2,gx2); ey2=torch.max(py2,gy2)
    c2 = (ex2-ex1)**2 + (ey2-ey1)**2 + 1e-6
    pcx=(px1+px2)/2; pcy=(py1+py2)/2
    gcx=(gx1+gx2)/2; gcy=(gy1+gy2)/2
    d2 = (pcx-gcx)**2 + (pcy-gcy)**2

    return (1 - iou + d2/c2).mean()

# ── Evaluation ────────────────────────────────────────────────────────────────
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
            ious.append(compute_iou(pred[i].cpu().tolist(), gt[i].cpu().tolist()))
    mean_iou = float(np.mean(ious))
    ap50 = float(np.mean([iou >= 0.5 for iou in ious]))
    lat  = float(np.mean(lats))
    return {'mean_iou': mean_iou, 'ap50': ap50, 'lat_ms': lat}

# ── Training ──────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[train] device:', device)

    train_ds = VoxelDataset('train', augment=True)
    val_ds   = VoxelDataset('val',   augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = SimpleFaceDetector(in_ch=10).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print('[train] params: %d (%.2fM)' % (n_params, n_params/1e6))

    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=3e-3,
                           total_steps=50*len(train_loader),
                           pct_start=0.1, anneal_strategy='cos')
    scaler    = torch.cuda.amp.GradScaler()
    os.makedirs('checkpoints', exist_ok=True)

    best_iou = 0.0
    history  = []

    for epoch in range(1, 51):
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
            if i % 50 == 0:
                print('  [ep%03d %4d/%d] loss=%.4f  lr=%.2e' % (
                      epoch, i+1, len(train_loader),
                      loss.item(), optimizer.param_groups[0]['lr']))

        ep_loss = total_loss / len(train_loader)
        ep_time = time.time() - t0

        if epoch % 5 == 0:
            m = validate(model, val_loader, device)
            print('[Epoch %d] loss=%.4f  mIoU=%.3f  AP@50=%.3f  lat=%.1fms  time=%.0fs' % (
                  epoch, ep_loss, m['mean_iou'], m['ap50'], m['lat_ms'], ep_time))
            if m['mean_iou'] > best_iou:
                best_iou = m['mean_iou']
                torch.save({'model_state_dict': model.state_dict(),
                            'config': {'in_channels':10,'img_h':360,'img_w':640},
                            'best_iou': best_iou, 'epoch': epoch},
                           'checkpoints/best_simple.pth')
                print('  -> New best mIoU=%.3f saved' % best_iou)
            history.append({'epoch':epoch, 'loss':ep_loss, **m})
            with open('checkpoints/history_simple.json','w') as f:
                json.dump(history, f, indent=2)
        else:
            print('[ep%03d] loss=%.4f  time=%.0fs' % (epoch, ep_loss, ep_time))

    # Save final model
    torch.save({'model_state_dict': model.state_dict(),
                'config': {'in_channels':10,'img_h':360,'img_w':640},
                'architecture': 'SimpleFaceDetector',
                'best_iou': best_iou},
               'final_model.pth')
    print('Done! Best mIoU=%.3f  Saved -> final_model.pth' % best_iou)

if __name__ == '__main__':
    os.chdir('/raid/didar_rakhimbay/Project')
    main()
